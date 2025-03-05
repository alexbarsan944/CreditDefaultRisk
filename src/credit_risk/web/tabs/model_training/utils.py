"""
Utility functions for the model training module.
"""

import logging
import pandas as pd
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def check_gpu_availability(model_type: str) -> bool:
    """
    Check if GPU is available for a specific model type.
    
    Parameters:
    -----------
    model_type : str
        The model type to check (xgboost, lightgbm, catboost)
        
    Returns:
    --------
    bool
        True if GPU is available for the specified model, False otherwise
    """
    try:
        if model_type == "xgboost":
            import xgboost as xgb
            # Modern XGBoost (2.0+) uses 'device' parameter with 'tree_method'='hist'
            param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cuda'}
            try:
                # Check if GPU works with a small dummy model
                dtrain = xgb.DMatrix(np.array([[0, 1], [1, 0]]), label=np.array([0, 1]))
                xgb.train(param, dtrain, num_boost_round=1)
                return True
            except Exception as e:
                logging.warning(f"GPU not available for XGBoost: {str(e)}")
                return False
                
        elif model_type == "lightgbm":
            import lightgbm as lgb
            # Check if lightgbm supports GPU
            try:
                # Try creating a small model with GPU
                train_data = lgb.Dataset(np.array([[0, 1], [1, 0]]), label=np.array([0, 1]))
                params = {'objective': 'binary', 'device_type': 'gpu'}
                lgb.train(params, train_data, num_boost_round=1)
                return True
            except Exception as e:
                logging.warning(f"GPU not available for LightGBM: {str(e)}")
                return False
                
        elif model_type == "catboost":
            # Check if catboost supports GPU
            import catboost as cb
            try:
                # Try creating a small model with GPU
                train_data = cb.Pool(np.array([[0, 1], [1, 0]]), label=np.array([0, 1]))
                params = {'task_type': 'GPU'}
                cb.train(pool=train_data, params=params, iterations=1)
                return True
            except Exception as e:
                logging.warning(f"GPU not available for CatBoost: {str(e)}")
                return False
                
        return False
    except ImportError:
        return False


def validate_feature_data(X_train, X_test, selected_features) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Validate feature data to ensure it's suitable for model training.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    selected_features : list
        List of selected feature names
        
    Returns:
    --------
    tuple
        Cleaned X_train, X_test, and selected_features
    """
    logger.info(f"Validating feature data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    
    # Check that selected features match columns in X_train
    missing_features = [f for f in selected_features if f not in X_train.columns]
    if missing_features:
        logger.warning(f"Some selected features are missing from X_train: {missing_features}")
        # Filter out missing features
        selected_features = [f for f in selected_features if f in X_train.columns]
        logger.info(f"Filtered selected features to {len(selected_features)} available features")
    
    # Verify X_train has data and doesn't have all-NaN columns
    all_nan_cols = [col for col in X_train.columns if X_train[col].isna().all()]
    if all_nan_cols:
        logger.warning(f"Found {len(all_nan_cols)} columns with all NaN values in X_train")
        logger.warning(f"First 10 all-NaN columns: {all_nan_cols[:10]}")
        
        # Remove all-NaN columns
        X_train = X_train.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)
        logger.info(f"Removed all-NaN columns. New shapes: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # Update selected features
        selected_features = [f for f in selected_features if f not in all_nan_cols]
    
    return X_train, X_test, selected_features 