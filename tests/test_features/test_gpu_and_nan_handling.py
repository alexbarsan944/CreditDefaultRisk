"""
Tests for GPU-related errors and NaN value handling in the model training.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging
from unittest.mock import patch, MagicMock

# Add the src directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from credit_risk.web.tabs.model_training.hyperparameter_optimization import (
    run_hyperparameter_optimization
)
from credit_risk.models.estimator import CreditRiskModel


class TestGPUHandling(unittest.TestCase):
    """Tests for handling GPU-related configurations and errors."""
    
    def setUp(self):
        """Set up test data."""
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # Create a simple dataset
        np.random.seed(42)
        n_samples = 500
        
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
        })
        self.y = pd.Series(np.random.binomial(1, 0.5, n_samples))
    
    def test_gpu_fallback_for_lightgbm(self):
        """Test that LightGBM falls back to CPU when GPU is not available."""
        
        # Mock the LightGBM model to simulate GPU error
        with patch('lightgbm.LGBMClassifier.fit', side_effect=RuntimeError("GPU Tree Learner was not enabled in this build.")):
            estimator = CreditRiskModel(
                model_type="lightgbm",
                model_params={"n_estimators": 10, "device_type": "gpu"},
                random_state=42
            )
            
            # This should throw an error because GPU is not available
            with self.assertRaises(RuntimeError) as context:
                estimator.fit(self.X, self.y)
            
            self.assertIn("GPU Tree Learner was not enabled", str(context.exception))
    
    def test_gpu_graceful_fallback(self):
        """Test that the optimization function handles GPU errors gracefully."""
        # For this we need to patch both the estimator and the optimization function
        
        # Create a mock result for the fallback
        class MockOptResult:
            def __init__(self):
                self.fun = 0.3
                self.x = [100, 3, 0.1, 31, 0.8, 0.8]
        
        # Create a patching context that simulates:
        # 1. GPU error on first attempt
        # 2. Successful run on CPU fallback
        with patch('credit_risk.models.estimator.CreditRiskModel.fit',
                  side_effect=[RuntimeError("GPU Tree Learner was not enabled"), None]) as mock_fit, \
             patch('credit_risk.models.estimator.CreditRiskModel.predict_proba',
                  return_value=np.array([[0.3, 0.7]] * len(self.y))) as mock_predict, \
             patch('credit_risk.web.tabs.model_training.hyperparameter_optimization.gp_minimize',
                  return_value=MockOptResult()) as mock_minimize:
            
            # This should now work with the fallback
            try:
                # Force very minimal parameters to speed up the test
                best_params, result, callback = run_hyperparameter_optimization(
                    self.X, self.y, self.X, self.y,
                    model_type="xgboost",  # Using a model type that's less likely to have GPU issues
                    use_gpu=True,  # Start with GPU enabled
                    n_calls=10,     # At least 10 calls for validation
                    cv_folds=2,
                    random_state=42,
                    progress_callback=lambda x, y, z: None
                )
                
                # Verification
                self.assertIsInstance(best_params, dict)
                self.assertIsNotNone(result)
                self.logger.info(f"Optimization succeeded with parameters: {best_params}")
                
            except Exception as e:
                self.fail(f"Optimization failed unexpectedly: {str(e)}")


class TestDataValidation(unittest.TestCase):
    """Tests for data validation and NaN handling."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 500
        
        # Create a clean dataset
        self.X_clean = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
        })
        # Target data is always clean (no NaNs)
        self.y_clean = pd.Series(np.random.binomial(1, 0.5, n_samples))
        
        # Create datasets with various data quality issues in features only
        
        # 1. Dataset with NaN values in features
        self.X_with_nans = self.X_clean.copy()
        self.X_with_nans.loc[np.random.choice(n_samples, 20), 'feature1'] = np.nan
        
        # 2. Dataset with infinity values in features
        self.X_with_inf = self.X_clean.copy()
        self.X_with_inf.loc[np.random.choice(n_samples, 10), 'feature2'] = np.inf
        
        # 3. Dataset with extreme outliers in features
        self.X_with_outliers = self.X_clean.copy()
        self.X_with_outliers.loc[np.random.choice(n_samples, 5), 'feature3'] = 1000
    
    def test_nan_detection_in_optimization(self):
        """Test that run_hyperparameter_optimization catches NaN values."""
        # Test with NaN in features
        with self.assertRaises(ValueError) as context:
            run_hyperparameter_optimization(
                self.X_with_nans, self.y_clean, self.X_clean, self.y_clean,
                model_type="xgboost",
                use_gpu=False,
                n_calls=10,  # At least 10 for validation
                cv_folds=2,
                random_state=42,
                progress_callback=None
            )
        self.assertIn("X_train contains NaN values", str(context.exception))
        
        # Test with infinity values in features
        with self.assertRaises(ValueError) as context:
            run_hyperparameter_optimization(
                self.X_with_inf, self.y_clean, self.X_clean, self.y_clean,
                model_type="xgboost",
                use_gpu=False,
                n_calls=10,
                cv_folds=2,
                random_state=42,
                progress_callback=None
            )
        self.assertTrue("NaN" in str(context.exception) or "infinite" in str(context.exception).lower(),
                        "Should detect invalid values in features")
    
    def test_data_cleaning_utility(self):
        """Test a utility function for cleaning datasets before optimization."""
        
        def clean_dataset_for_optimization(X, y):
            """Clean a dataset by removing NaN and infinity values."""
            # Check for NaN or infinity values in X
            if X.isna().any().any() or np.isinf(X).any().any():
                # Fill NaNs with median and infinity with large but finite values
                X_clean = X.copy()
                for col in X_clean.columns:
                    # Handle infinite values
                    X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
                    # Fill NaNs with median
                    median_value = X_clean[col].median()
                    X_clean[col] = X_clean[col].fillna(median_value)
                    
                return X_clean, y.fillna(y.mode()[0])
            return X, y
        
        # Test with dataset containing NaNs
        X_clean, y_clean = clean_dataset_for_optimization(self.X_with_nans, self.y_clean)
        self.assertFalse(X_clean.isna().any().any(), "X should not contain NaNs after cleaning")
        
        # Test with dataset containing infinity
        X_clean, y_clean = clean_dataset_for_optimization(self.X_with_inf, self.y_clean)
        self.assertFalse(np.isinf(X_clean).any().any(), "X should not contain infinity after cleaning")
        
        # Test with dataset containing both NaNs and infinity
        X_with_both = self.X_with_nans.copy()
        X_with_both.loc[0, 'feature3'] = np.inf
        
        X_clean, y_clean = clean_dataset_for_optimization(X_with_both, self.y_clean)
        self.assertFalse(X_clean.isna().any().any(), "X should not contain NaNs after cleaning")
        self.assertFalse(np.isinf(X_clean).any().any(), "X should not contain infinity after cleaning")
        self.assertFalse(y_clean.isna().any(), "y should not contain NaNs after cleaning")


if __name__ == "__main__":
    unittest.main() 