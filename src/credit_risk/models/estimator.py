import pandas as pd
import numpy as np
import logging
import joblib
import os
import re  # Add re module for regex operations
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

# Import model types
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

class CreditRiskModel:
    """
    A unified interface for credit risk prediction models.
    
    This class supports various model types including XGBoost, LightGBM, and CatBoost,
    with methods for training, cross-validation, evaluation, and prediction.
    """
    
    SUPPORTED_MODELS = ["xgboost", "lightgbm", "catboost"]
    
    def __init__(self, model_type: str = "lightgbm", model_params: Optional[Dict] = None, random_state: int = 42):
        """
        Initialize the credit risk model.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('xgboost', 'lightgbm', or 'catboost')
        model_params : dict, optional
            Parameters for the model. If None, default parameters will be used.
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None
        self.feature_names_ = None
        self.early_stopping_rounds = 0
        
        # Use different imputers for numerical and categorical data
        self.numeric_imputer = SimpleImputer(strategy="mean")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.categorical_columns = None
        self.numeric_columns = None
        
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported. Choose from: {self.SUPPORTED_MODELS}")
        
        # Check if required package is available
        if self.model_type == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install it with 'pip install xgboost'.")
        if self.model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install it with 'pip install lightgbm'.")
        if self.model_type == "catboost" and not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install it with 'pip install catboost'.")
        
        # Set default parameters based on model type
        self.model_params = self._get_default_params()
        
        # Update with user-provided parameters if any
        if model_params:
            self.model_params.update(model_params)
            
    def _get_default_params(self) -> Dict:
        """
        Get default parameters for the selected model type.
        
        Returns:
        --------
        dict
            Default parameters for the model
        """
        if self.model_type == "xgboost":
            return {
                "colsample_bylevel": 0.7566586325362956,
                "colsample_bynode": 1.0,
                "colsample_bytree": 0.8699617922932693,
                "gamma": 0,
                "learning_rate": 0.05,
                "max_depth": 4,
                "min_child_weight": 1,
                "n_estimators": 1200,
                "reg_alpha": 1,
                "reg_lambda": 1.0,
                "subsample": 0.91,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": self.random_state,
                "tree_method": "hist",  # Using 'hist' as safer default than 'gpu_hist'
            }
        elif self.model_type == "lightgbm":
            return {
                "n_estimators": 5000,
                "learning_rate": 0.005,
                "num_leaves": 45,
                "max_depth": 7,
                "subsample_for_bin": 240000,
                "reg_alpha": 1.0,
                "reg_lambda": 1.0,
                "feature_fraction": 0.8,  # Use feature_fraction instead of colsample_bytree
                "min_split_gain": 0.04,
                "subsample": 1.0,
                "is_unbalance": False,
                "random_state": self.random_state,
                "verbose": -1,  # Set to -1 to avoid warnings
                "n_jobs": -1,   # Use all available CPU cores
            }
        elif self.model_type == "catboost":
            return {
                "bootstrap_type": "Bernoulli",
                "depth": 6,
                "iterations": 2000,
                "l2_leaf_reg": 10.0,
                "learning_rate": 0.068,
                "random_seed": self.random_state,
                "verbose": 0,
                "task_type": "CPU",  # Default to CPU as safer option
            }
        return {}
    
    def _create_model(self) -> Any:
        """
        Create a new model instance based on model type and parameters.
        
        Returns:
        --------
        model
            The initialized model
        """
        model_params = self.model_params.copy()
        
        if self.model_type == "xgboost":
            # Remove early_stopping_rounds since it's not a valid XGBoost parameter
            if 'early_stopping_rounds' in model_params:
                self.early_stopping_rounds = model_params.pop('early_stopping_rounds')
            else:
                self.early_stopping_rounds = 0
                
            # Remove model_type parameter
            if 'model_type' in model_params:
                model_params.pop('model_type')
                
            return XGBClassifier(**model_params)
        elif self.model_type == "lightgbm":
            # Remove early_stopping_rounds since it should be passed to fit
            if 'early_stopping_rounds' in model_params:
                self.early_stopping_rounds = model_params.pop('early_stopping_rounds')
            else:
                self.early_stopping_rounds = 0
                
            # Remove model_type parameter
            if 'model_type' in model_params:
                model_params.pop('model_type')
                
            # Handle feature_fraction and colsample_bytree conflict
            if 'colsample_bytree' in model_params:
                # Always convert colsample_bytree to feature_fraction for consistency
                if 'feature_fraction' not in model_params:
                    feature_fraction = model_params.pop('colsample_bytree')
                    model_params['feature_fraction'] = feature_fraction
                    logger.info(f"Converted colsample_bytree={feature_fraction} to feature_fraction={feature_fraction}")
                else:
                    # If both are present, remove colsample_bytree to avoid the warning
                    colsample_value = model_params.pop('colsample_bytree')
                    logger.warning(f"Both feature_fraction ({model_params['feature_fraction']}) and colsample_bytree ({colsample_value}) were provided. Using only feature_fraction.")
                
            # Fix silent and verbose parameters
            if 'silent' in model_params:
                # Remove silent parameter since it's deprecated
                silent_value = model_params.pop('silent')
                if 'verbose' not in model_params:
                    model_params['verbose'] = -1 if silent_value else 0
                logger.info(f"Removed deprecated 'silent' parameter, using verbose={model_params['verbose']}")
            
            # Set verbosity to minimal to avoid warnings
            if 'verbose' not in model_params:
                model_params['verbose'] = -1
            
            # Ensure parallel processing uses all available cores
            model_params['n_jobs'] = -1
            
            # Log all parameter values for debugging
            logger.debug(f"LightGBM parameters after cleaning: {model_params}")
            
            return LGBMClassifier(**model_params)
        elif self.model_type == "catboost":
            try:
                return CatBoostClassifier(**model_params)
            except (NameError, ImportError):
                logger.error("CatBoost is not installed. Please install it with 'pip install catboost'.")
                raise ImportError("CatBoost is not installed. Please install it with 'pip install catboost'.")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CreditRiskModel':
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Target variable
            
        Returns:
        --------
        CreditRiskModel
            Trained model instance (self)
        """
        start_time = time.time()
        
        # Clean feature names to avoid LightGBM errors with special characters
        logger.info("Cleaning feature names to ensure compatibility with models")
        X = self._clean_feature_names(X)
        
        # Identify categorical and numerical columns if not already set
        if not hasattr(self, 'categorical_columns') or self.categorical_columns is None:
            self.categorical_columns = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype == 'category']
            self.numeric_columns = [col for col in X.columns if col not in self.categorical_columns]
            
            logger.info(f"Identified {len(self.categorical_columns)} categorical columns and {len(self.numeric_columns)} numeric columns")
            
        # Store original feature names
        self.feature_names_ = X.columns.tolist()
        
        # Preprocess data
        X_processed = self._preprocess_data(X)
        
        # Save the column names after preprocessing for prediction consistency
        self.processed_columns_ = X_processed.columns.tolist()
        logger.info(f"Processed data has {len(self.processed_columns_)} columns")
        
        # Create the model if it doesn't exist
        if self.model is None:
            logger.info(f"Creating new {self.model_type} model")
            self.model = self._create_model()
            if self.model is None:
                logger.error(f"Failed to create model of type {self.model_type}")
                raise ValueError(f"Failed to create model of type {self.model_type}")
        
        # Train the model
        logger.info(f"Starting {self.model_type} model training with {X_processed.shape[1]} features")
        
        try:
            # Handle model-specific training
            if self.model_type == "xgboost":
                # For XGBoost, handle early stopping
                if self.early_stopping_rounds > 0:
                    # Create a validation set for early stopping
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_processed, y, test_size=0.2, random_state=self.random_state, stratify=y
                    )
                    logger.info(f"Created validation set for early stopping: {X_val.shape}")
                    self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=self.early_stopping_rounds, verbose=False)
                else:
                    # Train without early stopping
                    self.model.fit(X_processed, y, verbose=False)
            elif self.model_type == "lightgbm":
                # For LightGBM, handle early stopping
                if self.early_stopping_rounds > 0:
                    # Create a validation set for early stopping
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_processed, y, test_size=0.2, random_state=self.random_state, stratify=y
                    )
                    logger.info(f"Created validation set for early stopping: {X_val.shape}")
                    self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=self.early_stopping_rounds, verbose=False)
                else:
                    # Train without early stopping
                    self.model.fit(X_processed, y, verbose=False)
            elif self.model_type == "catboost":
                # Handle categorical features for CatBoost
                cat_features = [i for i, col in enumerate(X_processed.columns) if col in self.categorical_columns]
                if cat_features:
                    logger.info(f"Using {len(cat_features)} categorical features for CatBoost training")
                    self.model.fit(X_processed, y, cat_features=cat_features, verbose=False)
                else:
                    self.model.fit(X_processed, y, verbose=False)
            else:
                # For other models, just train directly
                self.model.fit(X_processed, y)
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
            
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for model training/prediction.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed features
        """
        if X is None or X.empty:
            logger.warning("Empty DataFrame passed to preprocess_data")
            return X
        
        # Create a working copy
        result = X.copy()
        
        # Extra protection against special characters in column names
        # This ensures any columns created during preprocessing also have clean names
        for col in result.columns:
            if not all(c.isalnum() or c == '_' for c in col):
                # Replace any remaining non-alphanumeric chars with underscores
                new_name = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                # Ensure no duplicate column names
                count = 1
                original_new_name = new_name
                while new_name in result.columns and new_name != col:
                    new_name = f"{original_new_name}_{count}"
                    count += 1
                
                if new_name != col:
                    logger.warning(f"Found column with special characters after initial cleaning: '{col}' -> '{new_name}'")
                    result = result.rename(columns={col: new_name})
        
        # Check for all-NA columns and remove them
        all_na_cols = [col for col in result.columns if result[col].isna().all()]
        if all_na_cols:
            logger.warning(f"Removing {len(all_na_cols)} columns with all NA values")
            result = result.drop(columns=all_na_cols)
        
        # Split data into categorical and numerical
        cat_cols = self.categorical_columns or [col for col in result.columns if result[col].dtype == 'object' or result[col].dtype == 'category']
        num_cols = self.numeric_columns or [col for col in result.columns if col not in cat_cols]
        
        # Filter to keep only columns that exist in the DataFrame
        cat_cols = [col for col in cat_cols if col in result.columns]
        num_cols = [col for col in num_cols if col in result.columns]
        
        logger.info(f"Processing {len(num_cols)} numeric columns and {len(cat_cols)} categorical columns")
        
        # Handle numeric columns
        if num_cols:
            # Get numeric data as a copy for imputation
            num_data = result[num_cols].copy()
            # Convert all numeric columns to float64
            for col in num_data.columns:
                try:
                    num_data[col] = pd.to_numeric(num_data[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Error converting {col} to numeric: {str(e)}")
            
            # First time fitting the imputer
            if not hasattr(self, 'numeric_imputer_fitted') or not self.numeric_imputer_fitted:
                try:
                    logger.info("Fitting numeric imputer for the first time")
                    self.numeric_imputer = self.numeric_imputer.fit(num_data)
                    self.numeric_imputer_fitted = True
                    # Store the column names the imputer was fitted on
                    self.numeric_imputer_columns = num_data.columns.tolist()
                    logger.info(f"Numeric imputer fitted on {len(self.numeric_imputer_columns)} columns")
                except Exception as e:
                    logger.warning(f"Error fitting numeric imputer: {str(e)}. Using SimpleImputer with strategy='median'")
                    from sklearn.impute import SimpleImputer
                    self.numeric_imputer = SimpleImputer(strategy='median')
                    self.numeric_imputer = self.numeric_imputer.fit(num_data)
                    self.numeric_imputer_fitted = True
                    self.numeric_imputer_columns = num_data.columns.tolist()
            
            # Align columns with what the imputer was trained on
            if hasattr(self, 'numeric_imputer_columns'):
                # Check for feature name mismatch with imputer columns
                missing_cols = set(self.numeric_imputer_columns) - set(num_data.columns)
                extra_cols = set(num_data.columns) - set(self.numeric_imputer_columns)
                
                if missing_cols:
                    logger.warning(f"Missing {len(missing_cols)} numeric columns required by imputer. Adding with zeros.")
                    logger.debug(f"Missing columns: {list(missing_cols)[:5]}{'...' if len(missing_cols) > 5 else ''}")
                    
                    # Create a new DataFrame with zeros for missing columns
                    missing_df = pd.DataFrame(0, index=num_data.index, columns=list(missing_cols))
                    
                    # Instead of adding columns one by one, concat all at once to avoid fragmentation
                    num_data = pd.concat([num_data, missing_df], axis=1)
                
                if extra_cols:
                    logger.warning(f"Found {len(extra_cols)} extra numeric columns not seen during imputer fitting.")
                    # Save extra columns and drop them for imputation
                    extra_data = num_data[list(extra_cols)].copy()
                    num_data = num_data.drop(columns=list(extra_cols))
                
                # Ensure columns are in the same order as during fit
                num_data = num_data[self.numeric_imputer_columns]
            
            try:
                # Transform the aligned data
                num_data_imputed_array = self.numeric_imputer.transform(num_data)
                
                # Create DataFrame with the imputed data
                num_data_imputed = pd.DataFrame(
                    num_data_imputed_array,
                    columns=self.numeric_imputer_columns,
                    index=num_data.index
                )
                
                # Add back any extra columns that were removed before imputation
                if 'extra_data' in locals():
                    num_data_imputed = pd.concat([num_data_imputed, extra_data], axis=1)
                
                # Create a new result DataFrame instead of modifying columns one by one
                # First, drop all numeric columns from the original result
                result_non_numeric = result.drop(columns=num_cols, errors='ignore')
                
                # Then create a new DataFrame by concatenating non-numeric and imputed numeric data
                result = pd.concat([result_non_numeric, num_data_imputed], axis=1)
                
            except Exception as e:
                logger.warning(f"Error during numeric imputation: {str(e)}. Filling with median values directly.")
                # Fallback to direct median imputation
                for col in num_cols:
                    median_val = num_data[col].median()
                    result[col] = num_data[col].fillna(median_val)
        
        # Handle categorical columns
        if cat_cols:
            cat_data = result[cat_cols].copy()
            
            # First time fitting the categorical imputer
            if not hasattr(self, 'categorical_imputer_fitted') or not self.categorical_imputer_fitted:
                try:
                    logger.info("Fitting categorical imputer for the first time")
                    self.categorical_imputer = self.categorical_imputer.fit(cat_data)
                    self.categorical_imputer_fitted = True
                    # Store the column names the imputer was fitted on
                    self.categorical_imputer_columns = cat_data.columns.tolist()
                    logger.info(f"Categorical imputer fitted on {len(self.categorical_imputer_columns)} columns")
                except Exception as e:
                    logger.warning(f"Error fitting categorical imputer: {str(e)}. Using SimpleImputer with strategy='most_frequent'")
                    from sklearn.impute import SimpleImputer
                    self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                    self.categorical_imputer = self.categorical_imputer.fit(cat_data)
                    self.categorical_imputer_fitted = True
                    self.categorical_imputer_columns = cat_data.columns.tolist()
            
            # Align columns with what the imputer was trained on
            if hasattr(self, 'categorical_imputer_columns'):
                missing_cols = set(self.categorical_imputer_columns) - set(cat_data.columns)
                extra_cols = set(cat_data.columns) - set(self.categorical_imputer_columns)
                
                if missing_cols:
                    logger.warning(f"Missing {len(missing_cols)} categorical columns required by imputer. Adding with placeholders.")
                    # Create a new DataFrame with placeholders for missing columns
                    missing_df = pd.DataFrame('MISSING', index=cat_data.index, columns=list(missing_cols))
                    
                    # Concat all at once to avoid fragmentation
                    cat_data = pd.concat([cat_data, missing_df], axis=1)
                
                if extra_cols:
                    logger.warning(f"Found {len(extra_cols)} extra categorical columns not seen during imputer fitting.")
                    # Save extra columns and drop them for imputation
                    extra_data = cat_data[list(extra_cols)].copy()
                    cat_data = cat_data.drop(columns=list(extra_cols))
                
                # Ensure columns are in the same order as during fit
                cat_data = cat_data[self.categorical_imputer_columns]
            
            try:
                # Transform the aligned data
                cat_data_imputed_array = self.categorical_imputer.transform(cat_data)
                
                # Create DataFrame with the imputed data
                cat_data_imputed = pd.DataFrame(
                    cat_data_imputed_array,
                    columns=self.categorical_imputer_columns,
                    index=cat_data.index
                )
                
                # Add back any extra columns that were removed before imputation
                if 'extra_data' in locals() and not extra_data.empty:
                    cat_data_imputed = pd.concat([cat_data_imputed, extra_data], axis=1)
                
                # Update result with imputed categorical data (using concat to avoid fragmentation)
                result_non_categorical = result.drop(columns=cat_cols, errors='ignore')
                result = pd.concat([result_non_categorical, cat_data_imputed], axis=1)
                
            except Exception as e:
                logger.warning(f"Error during categorical imputation: {str(e)}. Filling with most frequent values directly.")
                # Fallback to direct most frequent imputation
                for col in cat_cols:
                    most_freq = cat_data[col].mode().iloc[0] if not cat_data[col].mode().empty else "MISSING"
                    result[col] = cat_data[col].fillna(most_freq)
        
        # Convert categorical columns to numeric using one-hot encoding
        # Only do this for actual object/category columns
        object_cols = [col for col in result.columns if result[col].dtype == 'object' or result[col].dtype == 'category']
        if object_cols:
            logger.warning(f"There are still {len(object_cols)} object/string columns: {object_cols}")
            # As a last resort, drop these columns to avoid LightGBM errors
            result = result.drop(columns=object_cols)
        
        # Ensure no NaN values remain
        if result.isna().any().any():
            logger.warning("NaN values remain after imputation. Filling remaining NaNs with 0.")
            result = result.fillna(0)
        
        # Defragment the DataFrame to optimize memory usage
        result = result.copy()
        
        return result
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Samples to predict
            
        Returns:
        --------
        np.ndarray
            Predicted class labels
        """
        if self.model is None:
            logger.error("Model not trained. Call fit() before predict().")
            raise ValueError("Model not trained. Call fit() before predict().")
        
        # Clean feature names to ensure compatibility with the model
        if hasattr(self, 'feature_name_mapping') and self.feature_name_mapping:
            logger.info("Cleaning feature names for prediction to match training data")
            X = self._clean_feature_names(X)
        
        # Process the data according to column types
        try:
            X_processed = self._preprocess_data(X)
            
            # Ensure all columns from training are present
            if hasattr(self, 'processed_columns_'):
                missing_cols = set(self.processed_columns_) - set(X_processed.columns)
                if missing_cols:
                    logger.warning(f"Missing {len(missing_cols)} columns from training data. Adding them with zeros.")
                    for col in missing_cols:
                        X_processed[col] = 0
                
                # Ensure correct column order
                X_processed = X_processed[self.processed_columns_]
            
            # Get probability predictions
            proba = self.predict_proba(X_processed)
            
            # Extract class 1 probabilities
            if len(proba.shape) == 2:
                # If we have a 2D array, take the second column (class 1 probabilities)
                class_1_proba = proba[:, 1]
            else:
                # If 1D, assume these are already class 1 probabilities
                class_1_proba = proba
            
            # Convert to class labels (>= 0.5 => 1, otherwise 0)
            predictions = (class_1_proba >= 0.5).astype(int)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict(): {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        np.ndarray
            Class probabilities
        """
        try:
            # Skip feature cleaning if we've already done it in predict()
            if not hasattr(X, '_feature_names_cleaned') and hasattr(self, 'feature_name_mapping') and self.feature_name_mapping:
                logger.info("Cleaning feature names for probability prediction to match training data")
                X = self._clean_feature_names(X)
                # Mark X as already having cleaned feature names
                X._feature_names_cleaned = True
            
            # Preprocess data
            X_processed = self._preprocess_data(X)
            
            # Check feature mismatch
            if hasattr(self, 'processed_columns_'):
                missing_cols = set(self.processed_columns_) - set(X_processed.columns)
                extra_cols = set(X_processed.columns) - set(self.processed_columns_)
                
                if missing_cols:
                    logger.warning(f"Test data is missing {len(missing_cols)} columns that were in training data.")
                    for col in missing_cols:
                        X_processed[col] = 0  # Add missing columns with zeros
                
                if extra_cols:
                    logger.warning(f"Test data has {len(extra_cols)} extra columns that were not in training data. These will be dropped.")
                    X_processed = X_processed.drop(columns=list(extra_cols))
                
                # Ensure column order matches
                X_processed = X_processed[self.processed_columns_]
            
            # Make predictions
            if self.model_type in ["xgboost", "lightgbm"]:
                # Use best iteration if early stopping was used
                if hasattr(self.model, 'best_iteration_'):
                    probas = self.model.predict_proba(X_processed, num_iteration=self.model.best_iteration_)
                else:
                    probas = self.model.predict_proba(X_processed)
            else:
                probas = self.model.predict_proba(X_processed)
            
            return probas
        except Exception as e:
            logger.error(f"Error in predict_proba(): {str(e)}", exc_info=True)
            # Raise a more descriptive error message
            raise ValueError(f"Error making predictions: {str(e)}")
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate the model on the given data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            True labels
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Get predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y, y_pred_proba)
        accuracy = accuracy_score(y, y_pred)
        
        # Get classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Create a results dictionary
        results = {
            "auc": auc,
            "accuracy": accuracy,
            "classification_report": report
        }
        
        logger.info(f"Model evaluation: AUC = {auc:.4f}, Accuracy = {accuracy:.4f}")
        
        return results
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5, stratified: bool = True) -> Dict:
        """
        Perform cross-validation on the model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        n_folds : int, optional
            Number of cross-validation folds, by default 5
        stratified : bool, optional
            Whether to use stratified sampling, by default True
            
        Returns:
        --------
        Dict
            Cross-validation results including AUC scores
        """
        if self.model is None:
            self.model = self._create_model()
        
        # Clean feature names to ensure compatibility with LightGBM
        logger.info(f"Cleaning feature names before cross-validation")
        X = self._clean_feature_names(X)
        
        # Check for all-NA columns and remove them before preprocessing
        all_na_cols = [col for col in X.columns if X[col].isna().all()]
        if all_na_cols:
            logger.warning(f"Removing {len(all_na_cols)} columns with all NA values before cross-validation")
            logger.debug(f"All-NA columns: {all_na_cols[:5]}{'...' if len(all_na_cols) > 5 else ''}")
            X = X.drop(columns=all_na_cols)
        
        # Preprocess the data - this handles imputation, etc.
        try:
            logger.info(f"Preprocessing data for cross-validation with {X.shape[1]} features")
            X_processed = self._preprocess_data(X)
            logger.info(f"Successfully preprocessed data with {X_processed.shape[1]} features")
            
            # Log column count mismatch if any
            if X_processed.shape[1] != X.shape[1]:
                logger.warning(f"Column count changed during preprocessing: {X.shape[1]} -> {X_processed.shape[1]}")
        except Exception as e:
            logger.error(f"Error preprocessing data for cross-validation: {str(e)}")
            raise ValueError(f"Failed to preprocess data for cross-validation: {str(e)}")
        
        # Store the processed columns for prediction
        logger.info(f"Storing processed column names for future reference: {len(X_processed.columns)} columns")
        self.processed_columns_ = X_processed.columns.tolist()
        
        # Create stratified k-fold
        if stratified and len(y.unique()) > 1:
            # Only use stratified if we have at least 2 classes
            folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        else:
            # Fall back to regular k-fold if only one class
            logger.warning("Using regular KFold cross-validation because there is only one class.")
            folds = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            
        # Initialize results
        fold_scores = []
        fold_predictions = []
        all_predictions = np.zeros(len(y))
        
        # Cross-validation loop
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_processed, y)):
            fold_num = n_fold + 1
            logger.info(f"Training fold {fold_num}/{n_folds}")
            
            # Get train and validation sets
            X_fold_train, X_fold_valid = X_processed.iloc[train_idx].copy(), X_processed.iloc[valid_idx].copy()
            y_fold_train, y_fold_valid = y.iloc[train_idx].copy(), y.iloc[valid_idx].copy()
            
            # Check for any NaN values and handle them
            if X_fold_train.isna().any().any():
                logger.warning(f"NaN values found in fold {fold_num} training data. Filling with 0.")
                X_fold_train = X_fold_train.fillna(0)
            
            if X_fold_valid.isna().any().any():
                logger.warning(f"NaN values found in fold {fold_num} validation data. Filling with 0.")
                X_fold_valid = X_fold_valid.fillna(0)
            
            # Check if we only have one class in the training set
            if len(y_fold_train.unique()) < 2:
                logger.warning(f"Fold {fold_num} has only one class in training. Skip this fold.")
                continue
                
            # Check if we only have one class in the validation set
            if len(y_fold_valid.unique()) < 2:
                logger.warning(f"Fold {fold_num} has only one class in validation. Skip this fold.")
                continue
            
            # Create and train model
            fold_model = self._create_model()
            
            # Start timing for this fold
            fold_start_time = time.time()
            
            # Fit the model - handle different model types
            try:
                fold_model.fit(X_fold_train, y_fold_train, verbose=False)
            except Exception as e:
                logger.error(f"Error fitting fold {fold_num}: {str(e)}")
                continue
            
            # Make predictions
            y_fold_pred = fold_model.predict_proba(X_fold_valid)[:, 1]
            
            # Get AUC score
            try:
                fold_auc = roc_auc_score(y_fold_valid, y_fold_pred)
                fold_scores.append(fold_auc)
                
                # Store predictions
                all_predictions[valid_idx] = y_fold_pred
                fold_predictions.append((valid_idx, y_fold_pred))
                
                # Log fold results
                logger.info(f"Fold {fold_num} completed in {(time.time() - fold_start_time):.2f} seconds. AUC: {fold_auc:.4f}")
            except Exception as e:
                logger.error(f"Error calculating AUC for fold {fold_num}: {str(e)}")
                continue
                
        # Calculate overall AUC
        try:
            logger.info(f"Calculating overall AUC from {len(y)} predictions")
            overall_auc = roc_auc_score(y, all_predictions)
            logger.info(f"Successfully calculated overall AUC: {overall_auc:.4f}")
        except Exception as e:
            logger.error(f"Error calculating overall AUC: {str(e)}")
            if len(fold_scores) > 0:
                # Use mean of fold scores if overall AUC fails
                overall_auc = np.mean(fold_scores)
                logger.info(f"Using mean of fold AUCs instead: {overall_auc:.4f}")
            else:
                # Default to 0.5 if no fold scores available
                overall_auc = 0.5
                logger.warning("No valid fold scores. Using default AUC of 0.5.")
        
        # Create results dictionary
        cv_results = {
            "oof_predictions": fold_predictions,
            "fold_metrics": fold_scores,
            "mean_auc": np.mean(fold_scores),
            "std_auc": np.std(fold_scores),
            "overall_auc": overall_auc,
            "feature_importances": None,
            "train_time": 0,  # Assuming train_time is not available in this method
            "inference_time": 0  # Assuming inference_time is not available in this method
        }
        
        # Make sure the model is initialized for subsequent training
        if self.model is None:
            logger.info(f"Initializing {self.model_type} model after cross-validation")
            self.model = self._create_model()
            if self.model is None:
                logger.error(f"Failed to create model of type {self.model_type} after cross-validation")
                raise ValueError(f"Failed to create model of type {self.model_type}")
            
        logger.info(f"Cross-validation completed. Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        
        return cv_results
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model and metadata
        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "model_params": self.model_params,
            "feature_names": self.feature_names_,
            "numeric_imputer": self.numeric_imputer,
            "categorical_imputer": self.categorical_imputer,
            "categorical_columns": self.categorical_columns,
            "numeric_columns": self.numeric_columns,
            "random_state": self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CreditRiskModel':
        """
        Load a model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        CreditRiskModel
            The loaded model
        """
        model_data = joblib.load(filepath)
        
        # Create a new instance without initialization
        instance = cls.__new__(cls)
        
        # Set attributes
        instance.model = model_data["model"]
        instance.model_type = model_data["model_type"]
        instance.model_params = model_data["model_params"]
        instance.feature_names_ = model_data["feature_names"]
        instance.numeric_imputer = model_data["numeric_imputer"]
        instance.categorical_imputer = model_data["categorical_imputer"]
        instance.categorical_columns = model_data["categorical_columns"]
        instance.numeric_columns = model_data["numeric_columns"]
        instance.random_state = model_data["random_state"]
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def plot_feature_importance(self, top_n: int = 20) -> Any:
        """
        Plot feature importance for the model using Plotly.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to show, by default 20
            
        Returns:
        --------
        plotly.graph_objects.Figure
            The feature importance plot
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not have feature importances")
        
        # Get feature importance
        importances = self.model.feature_importances_
        
        # If we have a feature names attribute, use it, otherwise create generic names
        if hasattr(self, "feature_names_") and len(self.feature_names_) == len(importances):
            feature_names = self.feature_names_
        else:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Check for length mismatch and fix if needed
        if len(importances) != len(feature_names):
            logging.warning(f"Feature importance length mismatch: {len(importances)} importances vs {len(feature_names)} features")
            # Use the minimum length of both arrays
            min_length = min(len(importances), len(feature_names))
            importances = importances[:min_length]
            feature_names = feature_names[:min_length]
        
        # Create dataframe for sorting
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        # Create Plotly bar chart with dark theme
        fig = go.Figure(go.Bar(
            x=top_features["importance"][::-1],
            y=top_features["feature"][::-1],
            orientation="h",
            marker=dict(
                color=top_features["importance"][::-1],
                colorscale="Blues",
                showscale=True
            )
        ))
        
        # Update layout with dark theme
        fig.update_layout(
            template="plotly_dark",
            title=f"Top {top_n} Feature Importances - {self.model_type.upper()}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600
        )
        
        return fig
    
    def plot_confusion_matrix(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Plot confusion matrix for the model using Plotly.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            True labels
            
        Returns:
        --------
        plotly.graph_objects.Figure
            The confusion matrix plot
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Get predictions
        y_pred = self.predict(X)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Create a plotly figure
        labels = ["Not Default", "Default"]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            hoverongaps=False,
            showscale=True
        ))
        
        # Update layout with dark theme
        fig.update_layout(
            template="plotly_dark",
            title=f"Confusion Matrix - {self.model_type.upper()}",
            xaxis=dict(title="Predicted Label"),
            yaxis=dict(title="True Label"),
            height=500,
            width=600
        )
        
        return fig

    def _clean_feature_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean feature names to ensure they're compatible with model requirements.
        Especially important for LightGBM which doesn't support special characters in feature names.
        
        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame with features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with cleaned feature names
        """
        if X is None or X.empty:
            return X
        
        logger.info(f"Cleaning feature names for compatibility with JSON (required by LightGBM)")
        logger.debug(f"Original column count: {len(X.columns)}")
        
        # Create a copy to avoid modifying the original
        X_cleaned = X.copy()
        
        # Build mapping of original to cleaned names
        name_mapping = {}
        problematic_chars = []
        
        for col in X_cleaned.columns:
            # Save original for logging
            original_name = col
            
            # STEP 1: Remove any leading/trailing whitespace
            cleaned_name = col.strip()
            
            # STEP 2: More aggressive cleaning - keep ONLY alphanumeric and underscores
            # This is stricter than before but ensures LightGBM compatibility
            old_name = cleaned_name
            cleaned_name = re.sub(r'[^\w\d]+', '_', cleaned_name)
            
            # Log any found problematic characters
            if old_name != cleaned_name:
                diff_chars = set(char for char in old_name if not char.isalnum() and char != '_')
                problematic_chars.extend(diff_chars)
            
            # STEP 3: Ensure the name doesn't start with a number or underscore
            if cleaned_name and (cleaned_name[0].isdigit() or cleaned_name[0] == '_'):
                cleaned_name = 'f' + cleaned_name
            
            # STEP 4: Handle empty or all-special-char names
            if not cleaned_name or cleaned_name.isspace():
                cleaned_name = f"feature_{len(name_mapping)}"
            
            # STEP 5: Avoid duplicate names by adding a suffix if needed
            count = 1
            orig_cleaned_name = cleaned_name
            while cleaned_name in name_mapping.values() and cleaned_name != col:
                cleaned_name = f"{orig_cleaned_name}_{count}"
                count += 1
            
            name_mapping[col] = cleaned_name
            
            # DEBUG: Log each transformation
            if original_name != cleaned_name:
                logger.debug(f"Feature name transformed: '{original_name}' -> '{cleaned_name}'")
        
        # Log changes summary
        changed_names = {orig: new for orig, new in name_mapping.items() if orig != new}
        if changed_names:
            logger.warning(f"Cleaned {len(changed_names)} feature names to ensure compatibility")
            logger.warning(f"Problematic characters found: {set(problematic_chars)}")
            for orig, new in list(changed_names.items())[:5]:  # Log first 5 changed names as examples
                logger.warning(f"Renamed feature: '{orig}' -> '{new}'")
            if len(changed_names) > 5:
                logger.warning(f"... and {len(changed_names) - 5} more renamed features")
        
        # Rename the columns
        X_cleaned.columns = [name_mapping.get(col, col) for col in X_cleaned.columns]
        
        # Store the mapping for future reference
        self.feature_name_mapping = name_mapping
        
        return X_cleaned 