"""
Unit tests for model training functionality with a focus on hyperparameter optimization.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pytest
import os
import sys
import logging
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from credit_risk.web.tabs.model_training.hyperparameter_optimization import (
    run_hyperparameter_optimization,
    _update_progress,
    IterationTrackingCallback
)
from credit_risk.models.estimator import CreditRiskModel


class TestHyperparameterOptimization(unittest.TestCase):
    """Test cases for hyperparameter optimization functions."""
    
    def setUp(self):
        """Set up test data for the test cases."""
        # Create a small sample dataset with no NaNs
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
        })
        # Binary target (0 or 1) - no NaNs in target
        self.y_train = pd.Series(np.random.binomial(1, 0.5, 100))
        
        # Create validation data
        self.X_val = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50),
            'feature3': np.random.normal(0, 1, 50),
        })
        self.y_val = pd.Series(np.random.binomial(1, 0.5, 50))
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
    
    def test_input_data_has_no_nans(self):
        """Test that input data does not contain NaN values."""
        assert not self.X_train.isna().any().any(), "X_train contains NaN values"
        assert not self.y_train.isna().any(), "y_train contains NaN values"
        assert not self.X_val.isna().any().any(), "X_val contains NaN values"
        assert not self.y_val.isna().any(), "y_val contains NaN values"
    
    def test_progress_callback(self):
        """Test that progress callback is correctly called."""
        mock_callback = MagicMock()
        
        # Create a mock optimization result
        class MockResult:
            def __init__(self):
                self.fun = 0.2  # 1 - AUC, so AUC would be 0.8
                self.x_iters = [1, 2, 3]  # 3 iterations completed
        
        result = MockResult()
        _update_progress(result, mock_callback, 10)
        
        # Ensure callback was called with expected values
        mock_callback.assert_called_once_with(3, 10, 0.8)
    
    @patch('credit_risk.web.tabs.model_training.hyperparameter_optimization.gp_minimize')
    def test_optimization_with_mock_minimize(self, mock_gp_minimize):
        """Test hyperparameter optimization using a mocked scikit-optimize function."""
        # Set up the mock result
        mock_result = MagicMock()
        mock_result.fun = 0.2  # 1 - AUC, so AUC would be 0.8
        mock_result.x = [100, 5, 0.1, 31, 0.9, 0.9]  # Mock best parameters
        mock_gp_minimize.return_value = mock_result
        
        # Mock callback for progress tracking
        mock_callback = MagicMock()
        
        try:
            best_params, result, callback = run_hyperparameter_optimization(
                self.X_train, self.y_train, self.X_val, self.y_val,
                model_type="lightgbm",
                use_gpu=False,
                n_calls=10,  # Use at least 10 for validation requirement
                cv_folds=2,
                random_state=42,
                progress_callback=mock_callback
            )
            
            # Verify the callback was used
            mock_callback.assert_called()
            
            # Verify the parameters were returned
            self.assertIsInstance(best_params, dict)
            self.assertEqual(result, mock_result)
            
        except Exception as e:
            self.fail(f"Hyperparameter optimization failed with error: {str(e)}")
    
    @pytest.mark.slow
    def test_real_optimization_minimal(self):
        """Run an actual optimization with minimal iterations to test integration."""
        try:
            # Skip if test is run in CI environment
            if os.environ.get('CI') == 'true':
                pytest.skip("Skipping slow test in CI environment")
            
            # A very small optimization run, just to verify it works end-to-end
            best_params, result, callback = run_hyperparameter_optimization(
                self.X_train, self.y_train, self.X_val, self.y_val,
                model_type="xgboost",  # Using XGBoost as it doesn't require GPU
                use_gpu=False,
                n_calls=10,  # At least 10 for validation
                cv_folds=2,
                random_state=42,
                progress_callback=lambda trial, total, best: None
            )
            
            # Verify results are returned
            self.assertIsInstance(best_params, dict)
            self.assertGreater(len(best_params), 0)
            
        except Exception as e:
            self.fail(f"Real optimization failed with error: {str(e)}")


class TestModelTraining(unittest.TestCase):
    """Test cases for model training functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample dataset with features known to have potential issues
        np.random.seed(42)
        n_samples = 200
        
        # Create features with different distributions
        self.X = pd.DataFrame({
            'numeric_feature1': np.random.normal(0, 1, n_samples),
            'numeric_feature2': np.random.normal(10, 5, n_samples),
            'sparse_feature': np.random.choice([0, 1, 2], size=n_samples, p=[0.8, 0.15, 0.05]),
            'binary_feature': np.random.binomial(1, 0.5, n_samples),
            'uniform_feature': np.random.uniform(0, 1, n_samples),
            'log_normal_feature': np.exp(np.random.normal(0, 1, n_samples)),
        })
        
        # Introduce a few potential issues (but still valid data)
        self.X['extreme_values'] = np.random.normal(0, 1, n_samples)
        self.X.loc[np.random.choice(n_samples, 5), 'extreme_values'] = 100  # Some outliers
        
        # Binary target with slight imbalance
        self.y = pd.Series(np.random.binomial(1, 0.4, n_samples))
    
    def test_check_data_quality(self):
        """Test checking data quality before training."""
        # Ensure no NaNs in the data
        assert not self.X.isna().any().any(), "Features contain NaN values"
        assert not self.y.isna().any(), "Target contains NaN values"
        
        # Check feature ranges
        for col in self.X.columns:
            assert self.X[col].min() != self.X[col].max(), f"Feature {col} has no variance"
    
    def test_data_leakage(self):
        """Test that there's no unexpected data leakage in the features."""
        # Check correlation between features (high correlation might indicate leakage)
        corr_matrix = self.X.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.9:  # Threshold for high correlation
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        assert len(high_corr_pairs) == 0, f"Found highly correlated features: {high_corr_pairs}"
    
    def test_create_estimator(self):
        """Test creation of a CreditScoringEstimator."""
        try:
            # Create a basic estimator
            estimator = CreditRiskModel(
                model_type="xgboost", 
                model_params={"n_estimators": 10, "max_depth": 3},
                random_state=42
            )
            
            # Verify it has the expected properties
            self.assertEqual(estimator.model_type, "xgboost")
            
        except Exception as e:
            self.fail(f"Failed to create estimator: {str(e)}")


if __name__ == "__main__":
    unittest.main() 