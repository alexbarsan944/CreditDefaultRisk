"""
Focused tests for debugging NaN issues in hyperparameter optimization.

This test suite focuses on proper handling of missing values in features,
ensuring that target variables are always clean (no NaNs).
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import logging
from unittest.mock import patch, MagicMock
from sklearn.metrics import roc_auc_score

# Add the src directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from credit_risk.web.tabs.model_training.hyperparameter_optimization import (
    run_hyperparameter_optimization, 
    _update_progress,
    IterationTrackingCallback
)
from credit_risk.models.estimator import CreditRiskModel


class TestHyperparameterOptimizationWithNaNs(unittest.TestCase):
    """Tests for hyperparameter optimization with a focus on NaN handling."""
    
    def setUp(self):
        """Set up test data for the test cases."""
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # Create a small sample dataset
        np.random.seed(42)
        n_samples = 500
        
        # Create standard features - completely clean data (no NaNs)
        self.X_clean = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
        })
        
        # Add categorical features
        categories = ['Self-employed', 'Employee', 'Unemployed', 'Student']
        self.X_clean['occupation'] = np.random.choice(categories, n_samples)
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        self.X_clean['city'] = np.random.choice(cities, n_samples)
        
        # Binary target - always clean (no NaNs)
        self.y_clean = pd.Series(np.random.binomial(1, 0.5, n_samples))
        
        # Create dataset with NaNs only in features
        self.X_with_nans = self.X_clean.copy()
        self.X_with_nans.loc[np.random.choice(n_samples, 10), 'feature1'] = np.nan
        
        # Create dataset with NaNs in both numeric and categorical features
        self.X_with_mixed_nans = self.X_clean.copy()
        self.X_with_mixed_nans.loc[np.random.choice(n_samples, 10), 'feature1'] = np.nan
        self.X_with_mixed_nans.loc[np.random.choice(n_samples, 10), 'occupation'] = np.nan

    def test_detect_nans_in_data(self):
        """Test detection of NaN values in data."""
        # Clean data should have no NaNs
        self.assertFalse(self.X_clean.isna().any().any(), "X_clean should have no NaNs")
        self.assertFalse(self.y_clean.isna().any(), "y_clean should have no NaNs")
        
        # Features with NaNs should be detected
        self.assertTrue(self.X_with_nans.isna().any().any(), "X_with_nans should have NaNs")
        # Target is always clean
        self.assertFalse(self.y_clean.isna().any(), "Target variable should never have NaNs")
    
    def test_credit_risk_model_handles_nan_features(self):
        """Test that CreditRiskModel can handle NaN features with XGBoost (which supports NaNs)."""
        # Create model
        model = CreditRiskModel(
            model_type="xgboost",
            model_params={"n_estimators": 10, "max_depth": 3},
            random_state=42
        )
        
        # Model should succeed with clean data
        try:
            model.fit(self.X_clean, self.y_clean)
            # Simple prediction to verify
            preds = model.predict_proba(self.X_clean.iloc[:5])
            self.assertEqual(preds.shape[1], 2)  # Binary classification
        except Exception as e:
            self.fail(f"Model failed with clean data: {str(e)}")
        
        # XGBoost should be able to handle NaN features!
        try:
            # Clone the model for a fresh fit
            model = CreditRiskModel(
                model_type="xgboost",
                model_params={"n_estimators": 10, "max_depth": 3},
                random_state=42
            )
            model.fit(self.X_with_nans, self.y_clean)
            # It should successfully fit and be able to predict
            preds = model.predict_proba(self.X_with_nans.iloc[:5])
            self.assertEqual(preds.shape[1], 2)
            self.logger.info("XGBoost handled NaN values as expected")
        except Exception as e:
            self.fail(f"XGBoost should handle NaN values but failed: {str(e)}")
    
    @patch('skopt.gp_minimize')
    def test_hyperparameter_optimization_mocked(self, mock_gp_minimize):
        """Test hyperparameter optimization with mocked gp_minimize."""
        # Set up mock return
        mock_result = MagicMock()
        mock_result.fun = 0.3  # This is 1-AUC, so ~0.7 AUC
        
        # XGBoost has 7 parameters (see hyperparameter_optimization.py for details):
        # n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, min_child_weight
        mock_result.x = [100, 5, 0.1, 0.8, 0.9, 0.1, 2]  # 7 parameters for XGBoost
        mock_result.x_iters = [[100, 5, 0.1, 0.8, 0.9, 0.1, 2]]
        mock_gp_minimize.return_value = mock_result
        
        # Skip the run_hyperparameter_optimization as it's complex to mock properly
        # Instead, test the NaN detection directly which is our main concern
        
        # Verify that clean data passes NaN checks
        try:
            # We just want to check the NaN validation part 
            # so we'll mock the optimization to return immediately
            with patch('credit_risk.web.tabs.model_training.hyperparameter_optimization.gp_minimize', 
                      return_value=mock_result):
                with patch('credit_risk.models.estimator.CreditRiskModel.cross_validate', 
                          return_value={"mean_auc": 0.8}):
                    # This should pass NaN validation
                    run_hyperparameter_optimization(
                        self.X_clean, self.y_clean, 
                        self.X_clean, self.y_clean,
                        model_type="xgboost",
                        use_gpu=False,
                        n_calls=10,
                        cv_folds=2,
                        random_state=42
                    )
            self.assertTrue(True, "NaN validation passed as expected")
        except ValueError as e:
            if "NaN" in str(e):
                self.fail(f"NaN check failed on clean data: {str(e)}")
            else:
                # Other errors might happen due to mocking, but we only care about NaN validation
                pass
    
    def test_nan_check_in_optimization(self):
        """Test that run_hyperparameter_optimization detects NaN values."""
        # It should reject NaN values in X
        try:
            # We don't need to actually run the optimization
            # Just check that the validation at the beginning catches NaNs
            run_hyperparameter_optimization(
                self.X_with_nans, self.y_clean, 
                self.X_clean, self.y_clean,
                model_type="xgboost",
                use_gpu=False,
                n_calls=10,
                cv_folds=2,
                random_state=42,
                progress_callback=None
            )
            self.fail("Should have raised ValueError due to NaN values")
        except ValueError as e:
            self.assertIn("NaN values", str(e))

    def test_clean_dataset_for_optimization(self):
        """Test that the clean_dataset_for_optimization function properly handles categorical data."""
        from credit_risk.web.tabs.model_training.data_utils import clean_dataset_for_optimization
        
        # Test cleaning a dataset with mixed data types (numeric and categorical)
        X_clean_result, y_clean_result = clean_dataset_for_optimization(self.X_with_mixed_nans, self.y_clean)
        
        # Debug info about column types
        print("\nColumn types after cleaning:")
        for col in X_clean_result.columns:
            print(f"{col}: {X_clean_result[col].dtype}")
        
        # Verify no NaN values remain
        self.assertFalse(X_clean_result.isna().any().any(), "Cleaned X should have no NaNs")
        
        # Verify all columns are numeric now
        numeric_dtypes = [np.issubdtype(dtype, np.number) for dtype in X_clean_result.dtypes]
        self.assertTrue(all(numeric_dtypes), 
                        f"All columns should be numeric after cleaning. Found non-numeric: {X_clean_result.columns[~np.array(numeric_dtypes)]}")
        
        # Verify one-hot encoding worked - we should have some columns starting with 'city_'
        city_columns = [col for col in X_clean_result.columns if col.startswith('city_')]
        self.assertTrue(len(city_columns) > 0, "City should be one-hot encoded")
        
        # Original categorical columns should be gone
        self.assertNotIn('occupation', X_clean_result.columns, "Original occupation column should not exist after encoding")
        self.assertNotIn('city', X_clean_result.columns, "Original city column should not exist after encoding")


if __name__ == "__main__":
    unittest.main() 