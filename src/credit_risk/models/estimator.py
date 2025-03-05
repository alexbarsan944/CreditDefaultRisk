import pandas as pd
import numpy as np
import logging
import joblib
import os
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
                "colsample_bytree": 0.4,
                "min_split_gain": 0.04,
                "subsample": 1.0,
                "is_unbalance": False,
                "random_state": self.random_state,
                "verbose": -1,
                "silent": True,
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
            # For XGBoost 2.1.x, set early_stopping_rounds in the constructor
            if 'early_stopping_rounds' not in model_params:
                model_params['early_stopping_rounds'] = 100
            return XGBClassifier(**model_params)
        elif self.model_type == "lightgbm":
            # For LightGBM, set early_stopping_rounds in the constructor
            if 'early_stopping_rounds' not in model_params:
                model_params['early_stopping_rounds'] = 100
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
                if 'early_stopping_rounds' in self.model_params and self.model_params['early_stopping_rounds'] > 0:
                    # Create a validation set for early stopping
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_processed, y, test_size=0.2, random_state=self.random_state, stratify=y
                    )
                    logger.info(f"Created validation set for early stopping: {X_val.shape}")
                    self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    # Train without early stopping
                    self.model.fit(X_processed, y, verbose=False)
            elif self.model_type == "lightgbm":
                # For LightGBM, handle early stopping
                if 'early_stopping_rounds' in self.model_params and self.model_params['early_stopping_rounds'] > 0:
                    # Create a validation set for early stopping
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_processed, y, test_size=0.2, random_state=self.random_state, stratify=y
                    )
                    logger.info(f"Created validation set for early stopping: {X_val.shape}")
                    self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
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
        Preprocess data by handling categorical and numerical features separately.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Processed features
        """
        if X.empty:
            return X
            
        result = X.copy()
        
        # Initialize imputers if not already done
        if not hasattr(self, 'numeric_imputer'):
            self.numeric_imputer = SimpleImputer(strategy='median')
        if not hasattr(self, 'categorical_imputer'):
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Handle categorical features based on model type
        if self.categorical_columns:
            if self.model_type == "catboost":
                # CatBoost handles categorical features natively, just impute missing values
                if len(self.categorical_columns) > 0:
                    # Get only the categorical columns that exist in the dataset
                    existing_cat_cols = [col for col in self.categorical_columns if col in result.columns]
                    if existing_cat_cols:
                        result[existing_cat_cols] = self.categorical_imputer.fit_transform(result[existing_cat_cols])
            else:
                # For other models, need to one-hot encode categorical features
                # First impute missing values
                if len(self.categorical_columns) > 0:
                    # Get only the categorical columns that exist in the dataset
                    existing_cat_cols = [col for col in self.categorical_columns if col in result.columns]
                    if existing_cat_cols:
                        result[existing_cat_cols] = self.categorical_imputer.fit_transform(result[existing_cat_cols])
                        
                        # For non-CatBoost models, encode categorical features
                        if self.model_type != "catboost":
                            # Store the original column names before encoding
                            current_cols = set(result.columns)
                            
                            # One-hot encode categorical features
                            for col in existing_cat_cols:
                                # Convert to string to handle potential non-string categorical data
                                result[col] = result[col].astype(str)
                                
                                try:
                                    # Create dummy variables
                                    dummies = pd.get_dummies(result[col], prefix=col, drop_first=False)
                                    
                                    # Only include dummy columns that were in the training data
                                    if hasattr(self, 'processed_columns_'):
                                        expected_dummy_cols = [c for c in self.processed_columns_ 
                                                               if c.startswith(f"{col}_") and c not in result.columns]
                                        
                                        # Add missing dummy columns (with zeros)
                                        for dummy_col in expected_dummy_cols:
                                            if dummy_col not in dummies.columns:
                                                dummies[dummy_col] = 0
                                    
                                    # Add dummy columns to result
                                    result = pd.concat([result, dummies], axis=1)
                                except Exception as e:
                                    logger.error(f"Error encoding categorical column {col}: {str(e)}")
                            
                            # Drop original categorical columns
                            result = result.drop(columns=existing_cat_cols)
                            
                            # Log added columns for debugging
                            added_cols = set(result.columns) - current_cols
                            logger.debug(f"Added {len(added_cols)} columns during one-hot encoding")
        
        # Handle numerical features
        if self.numeric_columns:
            # Get only the numeric columns that exist in the dataset
            existing_num_cols = [col for col in self.numeric_columns if col in result.columns]
            if existing_num_cols:
                # Fix for "Columns must be same length as key" error
                try:
                    logger.info(f"Imputing {len(existing_num_cols)} numeric columns")
                    
                    # First check for columns with all NaN values
                    all_nan_cols = []
                    for col in existing_num_cols:
                        if result[col].isna().all():
                            all_nan_cols.append(col)
                    
                    if all_nan_cols:
                        logger.warning(f"Found {len(all_nan_cols)} columns with all NaN values. Filling with zeros instead of imputing.")
                        for col in all_nan_cols:
                            result[col] = 0
                        
                        # Remove these columns from imputation
                        impute_cols = [col for col in existing_num_cols if col not in all_nan_cols]
                    else:
                        impute_cols = existing_num_cols
                    
                    if impute_cols:
                        # Impute each column individually to avoid dimension mismatches
                        for col in impute_cols:
                            col_data = result[[col]]
                            try:
                                # Use a separate imputer for each column
                                col_imputer = SimpleImputer(strategy='median')
                                imputed_values = col_imputer.fit_transform(col_data)
                                result[col] = imputed_values
                            except Exception as e:
                                logger.warning(f"Error imputing column {col}: {str(e)}. Filling with zeros.")
                                result[col] = result[col].fillna(0)
                except Exception as e:
                    logger.error(f"Error during numeric imputation: {str(e)}")
                    # Fallback: fill NaN values with 0
                    logger.warning("Falling back to filling all numeric NaN values with 0")
                    for col in existing_num_cols:
                        result[col] = result[col].fillna(0)
        
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
        Predict probabilities for samples in X.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Samples to predict
            
        Returns:
        --------
        np.ndarray
            Predicted probabilities for both classes (0 and 1)
        """
        if self.model is None:
            logger.error("Model not trained. Call fit() before predict_proba().")
            raise ValueError("Model not trained. Call fit() before predict_proba().")
        
        try:
            # Process the data if not already processed
            if hasattr(self, 'processed_columns_') and not all(col in X.columns for col in self.processed_columns_):
                logger.info("Input data needs preprocessing")
                X_processed = self._preprocess_data(X)
                
                # Handle missing columns
                missing_cols = set(self.processed_columns_) - set(X_processed.columns)
                if missing_cols:
                    logger.warning(f"Missing {len(missing_cols)} columns from training data. Adding them with zeros.")
                    for col in missing_cols:
                        X_processed[col] = 0
                
                # Ensure correct column order
                X_processed = X_processed[self.processed_columns_]
            else:
                # Data is already processed or doesn't need column alignment
                X_processed = X
            
            # Get probability predictions - return full probability array (not just class 1)
            if self.model_type == "lightgbm" and hasattr(self.model, "best_iteration_"):
                probas = self.model.predict_proba(X_processed, num_iteration=self.model.best_iteration_)
            elif self.model_type == "catboost" and hasattr(self.model, "best_iteration_"):
                probas = self.model.predict_proba(X_processed, ntree_end=self.model.best_iteration_)
            elif self.model_type == "xgboost" and hasattr(self.model, "best_iteration"):
                probas = self.model.predict_proba(X_processed, iteration_range=(0, self.model.best_iteration))
            else:
                probas = self.model.predict_proba(X_processed)
            
            # For some models that might only return class 1 probability, reshape to proper format
            if len(probas.shape) == 1:
                logger.warning("Model returned 1D probability array, reshaping to 2D")
                class_1_probs = probas
                class_0_probs = 1 - class_1_probs
                probas = np.column_stack((class_0_probs, class_1_probs))
                
            return probas
        
        except Exception as e:
            logger.error(f"Error in predict_proba(): {str(e)}")
            raise
    
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
            Training features
        y : pd.Series
            Target variable
        n_folds : int
            Number of folds for cross-validation
        stratified : bool
            Whether to use stratified folds
            
        Returns:
        --------
        Dict
            Cross-validation results
        """
        # Identify categorical and numerical columns
        self.categorical_columns = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype == 'category']
        self.numeric_columns = [col for col in X.columns if col not in self.categorical_columns]
        
        logger.info(f"Identified {len(self.categorical_columns)} categorical columns and {len(self.numeric_columns)} numeric columns")
        
        # Store feature names for later use
        self.feature_names_ = X.columns.tolist()
        
        # Define cross-validation strategy
        if stratified:
            folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        else:
            folds = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            
        # Initialize result dictionaries
        cv_scores = {
            "auc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "accuracy": [],
            "train_time": [],
            "inference_time": [],
            "oof_preds": []  # Add out-of-fold predictions
        }
        
        # Preprocess the data once to avoid redundant preprocessing in each fold
        X_processed = self._preprocess_data(X)
        
        # We'll construct arrays for concatenated predictions and true values
        all_predictions = []
        all_true_values = []
        all_indices = []
        
        # Perform cross-validation
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_processed, y)):
            fold_start_time = time.time()
            logger.info(f"Training fold {n_fold + 1}/{n_folds}")
            
            # Split data
            X_train, X_valid = X_processed.iloc[train_idx], X_processed.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            # Create and train the model
            fold_model = self._create_model()
            
            # Train the model with early stopping if available
            if self.model_type in ["xgboost", "lightgbm", "catboost"]:
                # Handle categorical features for CatBoost
                if self.model_type == "catboost":
                    cat_features = [i for i, col in enumerate(X_processed.columns) if col in self.categorical_columns]
                    if cat_features:
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_valid, y_valid)],
                            cat_features=cat_features,
                            early_stopping_rounds=100,
                            verbose=False
                        )
                    else:
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_valid, y_valid)],
                            early_stopping_rounds=100,
                            verbose=False
                        )
                elif self.model_type == "xgboost":
                    # For XGBoost 2.1.x, early stopping is set in the constructor
                    fold_model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        verbose=False
                    )
                else:
                    try:
                        # Try standard approach
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_valid, y_valid)],
                            early_stopping_rounds=100,
                            verbose=False
                        )
                    except TypeError:
                        logger.warning(f"Early stopping not supported directly in fit method for {self.model_type}. Falling back to basic fit.")
                        # Fall back to basic fit without early stopping
                        fold_model.fit(X_train, y_train)
            else:
                fold_model.fit(X_train, y_train)
            
            # Get predictions for this fold
            if self.model_type == "lightgbm" and hasattr(fold_model, "best_iteration_"):
                oof_preds = fold_model.predict_proba(X_valid, num_iteration=fold_model.best_iteration_)[:, 1]
            elif self.model_type == "catboost" and hasattr(fold_model, "best_iteration_"):
                oof_preds = fold_model.predict_proba(X_valid, ntree_end=fold_model.best_iteration_)[:, 1]
            elif self.model_type == "xgboost" and hasattr(fold_model, "best_iteration"):
                oof_preds = fold_model.predict_proba(X_valid, iteration_range=(0, fold_model.best_iteration))[:, 1]
            else:
                oof_preds = fold_model.predict_proba(X_valid)[:, 1]
            
            # Save predictions and true values for later overall calculation
            all_predictions.extend(oof_preds)
            all_true_values.extend(y_valid.values)
            all_indices.extend(valid_idx)
            
            # Calculate fold metrics
            fold_auc = roc_auc_score(y_valid, oof_preds)
            
            # Collect fold metrics
            cv_scores["auc"].append(fold_auc)
            cv_scores["train_time"].append(time.time() - fold_start_time)
            cv_scores["oof_preds"].append({
                "indices": valid_idx,
                "predictions": oof_preds
            })
            
            fold_time = time.time() - fold_start_time
            logger.info(f"Fold {n_fold + 1} completed in {fold_time:.2f} seconds. AUC: {fold_auc:.4f}")
            
            # Clean up memory
            gc.collect()
        
        # Calculate overall metrics from fold results
        mean_auc = np.mean(cv_scores["auc"])
        std_auc = np.std(cv_scores["auc"])
        
        # Calculate the overall AUC directly from all predictions and true values
        try:
            logger.info(f"Calculating overall AUC from {len(all_predictions)} predictions")
            overall_auc = roc_auc_score(all_true_values, all_predictions)
            logger.info(f"Successfully calculated overall AUC: {overall_auc:.4f}")
        except Exception as e:
            logger.warning(f"Error calculating overall AUC: {str(e)}. Using mean AUC instead.")
            overall_auc = mean_auc  # Fallback to mean AUC
        
        # Create results dictionary
        cv_results = {
            "oof_predictions": cv_scores["oof_preds"],
            "fold_metrics": cv_scores["auc"],
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "overall_auc": overall_auc,
            "feature_importances": None,
            "train_time": np.mean(cv_scores["train_time"]),
            "inference_time": np.mean(cv_scores["inference_time"])
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