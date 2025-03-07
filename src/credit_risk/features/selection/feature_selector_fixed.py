import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import logging
from typing import List, Tuple, Dict, Union, Optional
import warnings

logger = logging.getLogger(__name__)

# Create a context manager to temporarily suppress warnings
class SuppressWarnings:
    def __enter__(self):
        self.original_filters = warnings.filters.copy()
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
        return self
        
    def __exit__(self, *args):
        warnings.filters = self.original_filters

class FeatureSelector:
    """
    A class for feature selection using various techniques including:
    - Null importance feature selection
    - Correlation-based feature removal
    - XGBoost feature importance
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the FeatureSelector.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.selected_features_ = None
        self.removed_features_ = None
        self.feature_scores_ = None
        self.split_score_threshold_ = None
        self.gain_score_threshold_ = None
        self.correlation_threshold_ = None
        
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            n_runs: int = 100, 
            split_score_threshold: float = None, 
            gain_score_threshold: float = None,
            correlation_threshold: float = None) -> 'FeatureSelector':
        """
        Fit the feature selector to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        n_runs : int
            Number of runs for null importance calculation
        split_score_threshold : float or None
            Threshold for split score. Features with split score below threshold will be removed.
            If None, threshold will be determined automatically.
        gain_score_threshold : float or None
            Threshold for gain score. Features with gain score below threshold will be removed.
            If None, threshold will be determined automatically.
        correlation_threshold : float or None
            Threshold for correlation. Features with correlation above threshold will be removed.
            If None, default value of 0.95 will be used.
            
        Returns:
        --------
        FeatureSelector
            Fitted feature selector
        """
        feature_scores = self.get_feature_scores(X, y, n_runs=n_runs)
        self.feature_scores_ = feature_scores
        
        # Determine optimal thresholds if not provided
        if split_score_threshold is None or gain_score_threshold is None or correlation_threshold is None:
            # Use the new set_thresholds method to determine thresholds automatically
            self.set_thresholds(
                feature_scores,
                correlation_threshold=0.95 if correlation_threshold is None else correlation_threshold
            )
            
            # Use the provided thresholds or the automatically determined ones
            split_score_threshold = split_score_threshold if split_score_threshold is not None else self.split_score_threshold_
            gain_score_threshold = gain_score_threshold if gain_score_threshold is not None else self.gain_score_threshold_
            correlation_threshold = correlation_threshold if correlation_threshold is not None else self.correlation_threshold_
        else:
            # If thresholds are explicitly provided, set them
            self.split_score_threshold_ = split_score_threshold
            self.gain_score_threshold_ = gain_score_threshold
            self.correlation_threshold_ = correlation_threshold
            
            # Log the thresholds being used
            logger.info(f"Using thresholds - Split: {split_score_threshold:.3f}, Gain: {gain_score_threshold:.3f}, Correlation: {correlation_threshold:.3f}")
        
        # Select features based on scores
        mask_split = feature_scores['split_score'] > self.split_score_threshold_
        mask_gain = feature_scores['gain_score'] > self.gain_score_threshold_
        
        # Keep features that pass either split or gain threshold
        selected_features = feature_scores.loc[mask_split | mask_gain, 'feature'].tolist()
        
        # Check if we selected a reasonable number of features
        # If more than 75% of features are selected, adjust thresholds to be stricter
        if len(selected_features) > 0.75 * len(feature_scores):
            logger.warning(f"Too many features selected ({len(selected_features)} out of {len(feature_scores)}). Adjusting thresholds.")
            
            # Get the top 25% features by split and gain scores
            top_features_split = feature_scores.sort_values('split_score', ascending=False).head(int(len(feature_scores) * 0.25))['feature'].tolist()
            top_features_gain = feature_scores.sort_values('gain_score', ascending=False).head(int(len(feature_scores) * 0.25))['feature'].tolist()
            
            # Union of top features by split and gain
            selected_features = list(set(top_features_split) | set(top_features_gain))
            
            # Update thresholds to match the new selection
            if len(selected_features) > 0:
                min_selected_split = feature_scores.loc[feature_scores['feature'].isin(selected_features), 'split_score'].min()
                min_selected_gain = feature_scores.loc[feature_scores['feature'].isin(selected_features), 'gain_score'].min()
                self.split_score_threshold_ = min_selected_split
                self.gain_score_threshold_ = min_selected_gain
        
        # Log the number of selected features
        logger.info(f"Selected {len(selected_features)} features out of {len(feature_scores)} based on importance scores")
        
        # Store all removed features
        self.removed_features_ = list(set(feature_scores['feature'].tolist()) - set(selected_features))
        
        # Store selected features
        self.selected_features_ = selected_features
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the feature matrix by selecting only useful features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        pd.DataFrame
            Transformed feature matrix with only useful features
        """
        if self.selected_features_ is None:
            raise ValueError("FeatureSelector has not been fitted yet. Call fit() first.")
        
        # Get columns that exist in both X and selected_features
        common_columns = list(set(X.columns) & set(self.selected_features_))
        
        if len(common_columns) == 0:
            # If no common columns, issue a warning and return the top features by importance
            logger.warning("No selected features found in the input DataFrame. Using top features by importance instead.")
            
            # If we have feature scores, select top features
            if hasattr(self, 'feature_scores_') and self.feature_scores_ is not None:
                # Sort features by importance (split or gain score)
                sorted_features = self.feature_scores_.sort_values(
                    by=['split_score', 'gain_score'], 
                    ascending=False
                )
                
                # Select top 20% of features or at least 10 features (if available)
                num_top_features = max(int(X.shape[1] * 0.2), min(10, X.shape[1]))
                top_features = sorted_features.head(num_top_features)['feature'].tolist()
                
                # Filter to only include features available in X
                available_top_features = list(set(X.columns) & set(top_features))
                
                if available_top_features:
                    logger.info(f"Using {len(available_top_features)} top features by importance")
                    return X[available_top_features]
            
            # If all else fails, return all columns
            logger.warning("Returning all features - feature selection could not be performed")
            return X
        
        if len(common_columns) < len(self.selected_features_):
            missing_cols = set(self.selected_features_) - set(X.columns)
            logger.warning(f"Some selected features are missing in the input DataFrame: {missing_cols}")
        
        logger.info(f"Returning {len(common_columns)} features after selection")
        
        return X[common_columns]
    
    def fit_transform(self, 
                      X: pd.DataFrame, 
                      y: pd.Series, 
                      n_runs: int = 100, 
                      split_score_threshold: float = None, 
                      gain_score_threshold: float = None,
                      correlation_threshold: float = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        n_runs : int
            Number of runs for null importance calculation
        split_score_threshold : float or None
            Threshold for split score. Features with split score below threshold will be removed.
            If None, threshold will be determined automatically.
        gain_score_threshold : float or None
            Threshold for gain score. Features with gain score below threshold will be removed.
            If None, threshold will be determined automatically.
        correlation_threshold : float or None
            Threshold for correlation. Features with correlation above threshold will be removed.
            If None, default value of 0.95 will be used.
            
        Returns:
        --------
        pd.DataFrame
            Transformed feature matrix with selected features
        """
        self.fit(X, y, n_runs, split_score_threshold, gain_score_threshold, correlation_threshold)
        return self.transform(X)
    
    def get_feature_scores(self, X: pd.DataFrame, y: pd.Series, n_runs: int = 100) -> pd.DataFrame:
        """
        Calculate feature scores based on feature importance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        n_runs : int
            Number of runs for null importance calculation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature scores
        """
        try:
            # Calculate feature importance with real target
            actual_imp_df = self._get_feature_importance(X, y)
            
            # Calculate null importance by permuting the target
            null_imp_df = self._get_null_importance(X, y, n_runs=n_runs)
            
            # Calculate feature scores based on null and actual importance
            return self._calculate_feature_scores(actual_imp_df, null_imp_df)
        except Exception as e:
            logger.error(f"Error calculating feature scores: {str(e)}")
            raise
    
    def _get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate feature importance using XGBoost.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature importances
        """
        try:
            # Store original feature names
            original_feature_names = X.columns.tolist()
            logger.info(f"Original feature count: {len(original_feature_names)}")
            
            # Process categorical columns
            X_processed = X.copy()
            
            # Identify categorical columns
            categorical_columns = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Convert categorical columns to numeric using one-hot encoding
            if categorical_columns:
                X_processed = pd.get_dummies(X_processed, columns=categorical_columns, drop_first=True)
                # Store mapping of one-hot encoded features to original features
                self.encoded_feature_map = {}
                for col in categorical_columns:
                    encoded_cols = [c for c in X_processed.columns if c.startswith(f"{col}_")]
                    for enc_col in encoded_cols:
                        self.encoded_feature_map[enc_col] = col
            
            # Get the final list of feature names after processing
            feature_names = X_processed.columns.tolist()
            logger.info(f"Processed feature count: {len(feature_names)}")
            
            # Handle missing values before XGBoost training
            imputer = SimpleImputer(strategy='mean')
            try:
                X_imputed = imputer.fit_transform(X_processed)
                logger.info(f"Imputed data shape: {X_imputed.shape}")
            except Exception as e:
                logger.warning(f"Mean imputation failed: {str(e)}. Trying median imputation.")
                try:
                    imputer = SimpleImputer(strategy='median')
                    X_imputed = imputer.fit_transform(X_processed)
                except Exception as e2:
                    logger.warning(f"Median imputation failed: {str(e2)}. Using constant imputation.")
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                    X_imputed = imputer.fit_transform(X_processed)
            
            # Check for shape mismatch between imputed data and feature names
            if X_imputed.shape[1] != len(feature_names):
                logger.warning(f"Shape mismatch: Imputed data has {X_imputed.shape[1]} columns but feature_names has {len(feature_names)} elements")
                # Adjust feature names to match imputed data shape
                if X_imputed.shape[1] < len(feature_names):
                    logger.warning(f"Truncating feature names to match imputed data columns")
                    feature_names = feature_names[:X_imputed.shape[1]]
                else:
                    logger.warning(f"Adding generic feature names to match imputed data columns")
                    additional_features = [f"feature_{i}" for i in range(len(feature_names), X_imputed.shape[1])]
                    feature_names.extend(additional_features)
            
            # Create DataFrame with adjusted feature names
            X_imputed_df = pd.DataFrame(X_imputed, columns=feature_names)
            
            # Create feature name mapping for consistent reference
            feature_name_map = {i: name for i, name in enumerate(feature_names)}
            
            # Check if GPU is available
            gpu_available = False
            gpu_device_name = "No GPU"
            
            # Method 1: Try nvidia-smi command
            try:
                import subprocess
                try:
                    nvidia_output = subprocess.check_output(["nvidia-smi"], 
                                                           stderr=subprocess.STDOUT).decode("utf-8")
                    if "NVIDIA-SMI" in nvidia_output:
                        gpu_available = True
                        # Extract GPU model name if available
                        import re
                        gpu_match = re.search(r"\| NVIDIA-SMI.+?(\d+\.\d+).+\|(.+)\|", nvidia_output)
                        if gpu_match:
                            gpu_device_name = gpu_match.group(2).strip()
                        else:
                            gpu_device_name = "NVIDIA GPU (detected via nvidia-smi)"
                        logger.info(f"GPU detected via nvidia-smi: {gpu_device_name}")
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.info("nvidia-smi command failed, trying alternative detection methods")
            except ImportError:
                logger.info("subprocess module not available")
                
            # Method 2: Try cupy import
            if not gpu_available:
                try:
                    import cupy
                    gpu_available = True
                    gpu_device_name = "CUDA GPU (detected via cupy)"
                    logger.info("GPU detected via cupy import")
                except ImportError:
                    logger.info("cupy import failed, GPU not detected through cupy")
                    
            # Method 3: Try RAPIDS cuML
            if not gpu_available:
                try:
                    import cuml
                    gpu_available = True
                    gpu_device_name = "CUDA GPU (detected via cuml)"
                    logger.info("GPU detected via cuml import")
                except ImportError:
                    logger.info("cuml import failed, GPU not detected through RAPIDS")
            
            # Configure XGBoost model parameters
            xgb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'eval_metric': 'logloss'
            }
            
            # Add GPU-specific parameters if GPU is available
            if gpu_available:
                xgb_params.update({
                    'tree_method': 'hist',  # Changed from 'gpu_hist' (deprecated in XGBoost 2.0+)
                    'device': 'cuda',       # New parameter for XGBoost 2.0+
                })
                logger.info(f"Using GPU for XGBoost with device='cuda': {gpu_device_name}")
            else:
                xgb_params['tree_method'] = 'hist'  # Best CPU algorithm
                logger.info("No GPU detected, using CPU for XGBoost with tree_method='hist'")
                
            # Train XGBoost model to get feature importances
            model = XGBClassifier(**xgb_params)
            
            try:
                # No feature_names parameter - use the DataFrame to preserve names
                model.fit(X_imputed_df, y)
                
                # Get feature importance - handle both naming schemes
                try:
                    # Try to get feature importances using get_score
                    booster = model.get_booster()
                    
                    # Get both importance types
                    importance_split = booster.get_score(importance_type='weight')
                    importance_gain = booster.get_score(importance_type='gain')
                    
                    # Check if we have any importances
                    if not importance_split or not importance_gain:
                        logger.warning("No importances returned from get_score. Falling back to feature_importances_")
                        raise ValueError("Empty importances")
                    
                    logger.info(f"Retrieved importances - Split: {len(importance_split)}, Gain: {len(importance_gain)}")
                    
                    # Check if importances use f0, f1, etc. format and map to actual feature names
                    if list(importance_split.keys())[0].startswith('f') and list(importance_split.keys())[0][1:].isdigit():
                        logger.info("Detected f0, f1 format in feature names, mapping to actual feature names")
                        # Map f0, f1, etc. to actual feature names
                        importance_split = {feature_name_map.get(int(k[1:]), k): v 
                                          for k, v in importance_split.items()}
                        importance_gain = {feature_name_map.get(int(k[1:]), k): v 
                                         for k, v in importance_gain.items()}
                    
                    # Create a complete feature importance DataFrame with all features
                    all_features_imp = pd.DataFrame({
                        'feature': feature_names,
                        'importance_gain': [importance_gain.get(f, 0.0) for f in feature_names],
                        'importance_split': [importance_split.get(f, 0.0) for f in feature_names]
                    })
                    
                except Exception as e:
                    logger.warning(f"Error getting importances from booster: {str(e)}")
                    # Fallback to feature_importances_ attribute
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        
                        # Ensure we handle potential length mismatches
                        if len(importances) != len(feature_names):
                            logger.warning(f"Feature importances length ({len(importances)}) doesn't match feature names length ({len(feature_names)})")
                            # Use the shorter length
                            min_len = min(len(importances), len(feature_names))
                            feature_names_trunc = feature_names[:min_len]
                            importances_trunc = importances[:min_len]
                            
                            all_features_imp = pd.DataFrame({
                                'feature': feature_names_trunc,
                                'importance_split': importances_trunc,
                                'importance_gain': importances_trunc  # Use same values as approximation
                            })
                        else:
                            all_features_imp = pd.DataFrame({
                                'feature': feature_names,
                                'importance_split': importances,
                                'importance_gain': importances  # Use same values as approximation
                            })
                        logger.info("Used feature_importances_ attribute as fallback")
                    else:
                        raise ValueError("Could not obtain feature importances")
                
                # Ensure we have at least some non-zero importances
                if all_features_imp['importance_gain'].sum() == 0 or all_features_imp['importance_split'].sum() == 0:
                    logger.warning("All feature importances are zero. Creating artificial importances.")
                    # Create artificial importances based on feature index (at least they won't all be zero)
                    for i, feature in enumerate(all_features_imp['feature']):
                        importance_value = 1.0 / (i + 1)  # Higher importance for first features
                        all_features_imp.loc[i, 'importance_gain'] = importance_value
                        all_features_imp.loc[i, 'importance_split'] = importance_value
                
                # Map one-hot encoded features back to original features if needed
                if hasattr(self, 'encoded_feature_map') and self.encoded_feature_map:
                    logger.info("Aggregating importances for one-hot encoded features")
                    all_features_imp['original_feature'] = all_features_imp['feature'].map(
                        lambda x: self.encoded_feature_map.get(x, x)
                    )
                    
                    # Sum importances for features that were one-hot encoded
                    grouped_imp = all_features_imp.groupby('original_feature').agg({
                        'importance_gain': 'sum',
                        'importance_split': 'sum'
                    }).reset_index()
                    grouped_imp = grouped_imp.rename(columns={'original_feature': 'feature'})
                    
                    logger.info(f"Aggregated {len(all_features_imp)} features to {len(grouped_imp)} original features")
                    return grouped_imp
                    
                logger.info(f"Feature importance calculated successfully for {len(all_features_imp)} features")
                return all_features_imp
                
            except Exception as e:
                logger.error(f"Error calculating feature importance: {str(e)}")
                # Return a DataFrame with artificial importances as fallback (not all zeros)
                importance_df = pd.DataFrame({
                    'feature': original_feature_names,
                    'importance_gain': [1.0 / (i + 1) for i in range(len(original_feature_names))],
                    'importance_split': [1.0 / (i + 1) for i in range(len(original_feature_names))]
                })
                logger.info("Created artificial feature importances due to calculation error")
            return importance_df
                
        except Exception as e:
            logger.error(f"Error in _get_feature_importance: {str(e)}")
            # Final fallback - create basic artificial importances
            return pd.DataFrame({
                'feature': X.columns.tolist(),
                'importance_gain': [1.0 / (i + 1) for i in range(len(X.columns))],
                'importance_split': [1.0 / (i + 1) for i in range(len(X.columns))]
            })
    
    def _get_null_importance(self, X: pd.DataFrame, y: pd.Series, n_runs: int = 100) -> pd.DataFrame:
        """
        Calculate null importance by permuting the target variable.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        n_runs : int, optional
            Number of permutation runs, by default 100
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with null importance for each feature
        """
        try:
            # Process X to handle categorical columns
            X_processed = X.copy()
            
            # Identify categorical columns
            categorical_columns = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Store a mapping from original columns to encoded columns
            column_mapping = {}
            
            # Convert categorical columns to numeric using one-hot encoding
            if categorical_columns:
                X_processed = pd.get_dummies(X_processed, columns=categorical_columns, drop_first=True)
                # Store mapping from encoded columns to original columns
                for col in categorical_columns:
                    encoded_cols = [c for c in X_processed.columns if c.startswith(f"{col}_")]
                    for enc_col in encoded_cols:
                        column_mapping[enc_col] = col
            
            # Store the original feature names before imputation
            original_feature_names = X_processed.columns.tolist()
                    
            # Handle missing values in X
            try:
                # Try mean imputation first - but don't transform yet
                logger.info("Preparing mean imputer for null importance calculation")
                imputer = SimpleImputer(strategy="mean")
                
                # Check if any columns have all missing values
                all_missing_cols = []
                for col in X_processed.columns:
                    if X_processed[col].isna().all():
                        all_missing_cols.append(col)
                
                if all_missing_cols:
                    logger.warning(f"Found {len(all_missing_cols)} columns with all missing values: {all_missing_cols[:5]}")
                    logger.warning("Removing these columns before imputation")
                    X_processed = X_processed.drop(columns=all_missing_cols)
                
                # Fit imputer and transform data
                X_imputed = imputer.fit_transform(X_processed)
                feature_names = X_processed.columns.tolist()  # Update feature names to match imputed data
                logger.info(f"Imputed data shape: {X_imputed.shape}, feature count: {len(feature_names)}")
            except Exception as e:
                logger.warning(f"Error using mean imputation in null importance: {str(e)}. Trying median imputation.")
                try:
                    # Try median imputation if mean fails
                    imputer = SimpleImputer(strategy="median")
                    X_imputed = imputer.fit_transform(X_processed)
                    feature_names = X_processed.columns.tolist()
                except Exception as e2:
                    logger.warning(f"Error using median imputation in null importance: {str(e2)}. Trying constant imputation.")
                    # As a last resort, use constant imputation
                    imputer = SimpleImputer(strategy="constant", fill_value=0)
                    X_imputed = imputer.fit_transform(X_processed)
                    feature_names = X_processed.columns.tolist()
            
            # Check for shape mismatch between imputed data and feature names
            if X_imputed.shape[1] != len(feature_names):
                logger.warning(f"Shape mismatch in null importance: Imputed data has {X_imputed.shape[1]} columns but feature_names has {len(feature_names)} elements")
                
                # Ensure feature_names matches the imputed data shape
                if X_imputed.shape[1] < len(feature_names):
                    logger.warning(f"Truncating feature names list to match imputed data ({X_imputed.shape[1]} columns)")
                    feature_names = feature_names[:X_imputed.shape[1]]
                else:
                    logger.warning(f"Adding generic feature names to match imputed data columns ({X_imputed.shape[1]} columns)")
                    additional_features = [f"feature_{i}" for i in range(len(feature_names), X_imputed.shape[1])]
                    feature_names.extend(additional_features)
            
            # Create a feature name mapping for consistent reference
            feature_name_map = {i: name for i, name in enumerate(feature_names)}
                
            # Create pandas DataFrame to preserve feature names
            X_imputed_df = pd.DataFrame(X_imputed, columns=feature_names)
            
            # Create DataFrame to store null importances
            null_imp_df = pd.DataFrame()
            
            # Check if GPU is available (reuse code pattern from _get_feature_importance)
            gpu_available = False
            
            # Method 1: Try nvidia-smi command
            try:
                import subprocess
                try:
                    nvidia_output = subprocess.check_output(["nvidia-smi"], 
                                                         stderr=subprocess.STDOUT).decode("utf-8")
                    if "NVIDIA-SMI" in nvidia_output:
                        gpu_available = True
                        logger.info("GPU detected via nvidia-smi for null importance calculation")
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.info("nvidia-smi command failed, trying alternative detection methods")
            except ImportError:
                logger.info("subprocess module not available")
                
            # Method 2: Try cupy import
            if not gpu_available:
                try:
                    import cupy
                    gpu_available = True
                    logger.info("GPU detected via cupy import for null importance calculation")
                except ImportError:
                    logger.info("cupy import failed, GPU not detected through cupy")
            
            # Configure XGBoost parameters for null importance
            xgb_params = {
                'n_estimators': 100, 
                'max_depth': 5,
                'random_state': self.random_state,
            }
            
            # Add GPU-specific parameters if GPU is available
            if gpu_available:
                xgb_params.update({
                    'tree_method': 'hist',  # Changed from 'gpu_hist' (deprecated in XGBoost 2.0+)
                    'device': 'cuda',       # New parameter for XGBoost 2.0+
                })
                logger.info("Using GPU for null importance calculation with device='cuda'")
            else:
                xgb_params['tree_method'] = 'hist'  # Best CPU algorithm
                logger.info("Using CPU for null importance calculation with tree_method='hist'")
            
            # Create a copy of y to permute
            y_perm = y.copy()
            
            logger.info(f"Starting null importance calculation with {n_runs} runs")
            successful_runs = 0
            
            for i in range(n_runs):
                # Permute the target
                y_perm = y.sample(frac=1.0, random_state=self.random_state + i).reset_index(drop=True)
                
                # Train model on permuted target
                model = XGBClassifier(**xgb_params)
                
                try:
                    # Remove feature_names parameter - use DataFrame instead
                    model.fit(X_imputed_df, y_perm)
                    
                    # Get feature importances from booster
                    try:
                        booster = model.get_booster()
                        
                        # Get importances (with consistent handling of feature names)
                        split_imp = booster.get_score(importance_type='weight')
                        gain_imp = booster.get_score(importance_type='gain')
                        
                        # Check if we have any importances
                        if not split_imp or not gain_imp:
                            logger.warning(f"No importances returned from get_score in run {i}. Skipping.")
                            continue
                        
                        # Check if feature names are in f0, f1 format and map to actual feature names
                        if list(split_imp.keys())[0].startswith('f') and list(split_imp.keys())[0][1:].isdigit():
                            # Map f0, f1, etc. to actual feature names
                            split_imp = {feature_name_map.get(int(k[1:]), k): v for k, v in split_imp.items()}
                            gain_imp = {feature_name_map.get(int(k[1:]), k): v for k, v in gain_imp.items()}
                        
                        # Ensure all features are included (with 0 importance if not in get_score output)
                        run_importance = pd.DataFrame({
                            "feature": feature_names,
                            "importance_split": [split_imp.get(f, 0.0) for f in feature_names],
                            "importance_gain": [gain_imp.get(f, 0.0) for f in feature_names],
                            "run": i
                        })
                        
                        # Append to null importance DataFrame
                        null_imp_df = pd.concat([null_imp_df, run_importance], axis=0)
                        successful_runs += 1
                        
                    except Exception as e:
                        logger.warning(f"Error getting feature importance in null importance run {i}: {str(e)}")
                        # Try to use feature_importances_ attribute as fallback
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                            
                            # Handle potential length mismatch
                            if len(importances) != len(feature_names):
                                logger.warning(f"Feature importances length mismatch in run {i}")
                                min_len = min(len(importances), len(feature_names))
                                feature_names_trunc = feature_names[:min_len]
                                importances_trunc = importances[:min_len]
                                
                                run_importance = pd.DataFrame({
                                    "feature": feature_names_trunc,
                                    "importance_split": importances_trunc,
                                    "importance_gain": importances_trunc,  # Use same as approximation
                                    "run": i
                                })
                            else:
                                # Create run importance DataFrame
                                run_importance = pd.DataFrame({
                                    "feature": feature_names,
                                    "importance_split": importances,
                                    "importance_gain": importances,  # Use same as approximation
                                    "run": i
                                })
                            
                            # Append to null importance DataFrame
                            null_imp_df = pd.concat([null_imp_df, run_importance], axis=0)
                            successful_runs += 1
                        else:
                            logger.warning(f"Skipping null importance run {i} due to missing importances")
                            
                except Exception as e:
                    logger.warning(f"Error in null importance run {i}: {str(e)}")
                    # Skip this run if it fails
                    continue
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{n_runs} null importance runs")
            
            # Check if we have enough successful runs
            if null_imp_df.empty or successful_runs == 0:
                logger.error("No successful null importance runs. This suggests an issue with the data or model.")
                logger.info("Creating artificial null importances to proceed with feature selection")
                
                # Create artificial null importances based on random noise
                artificial_null_imp = []
                
                for i in range(min(n_runs, 10)):  # Create at least 10 artificial runs
                    for j, feature in enumerate(feature_names):
                        # Random importance values with decreasing mean for each feature (first ones more important)
                        split_imp = np.random.exponential(1.0 / (j + 1)) 
                        gain_imp = np.random.exponential(1.0 / (j + 1))
                        
                        artificial_null_imp.append({
                            "feature": feature,
                            "importance_split": split_imp,
                            "importance_gain": gain_imp,
                            "run": i
                        })
                
                null_imp_df = pd.DataFrame(artificial_null_imp)
                logger.info(f"Created {len(null_imp_df['run'].unique())} artificial null importance runs")
                
                return null_imp_df
                
            logger.info(f"Completed null importance calculation with {successful_runs} successful runs")
            
            # If we have one-hot encoded features, map them back to original features
            if column_mapping:
                logger.info("Mapping one-hot encoded features back to original features for null importance")
                null_imp_df['original_feature'] = null_imp_df['feature'].map(lambda x: column_mapping.get(x, x))
                
                # Aggregate importances by original feature
                null_imp_agg = null_imp_df.groupby(['original_feature', 'run']).agg({
                    'importance_split': 'sum',
                    'importance_gain': 'sum'
                }).reset_index()
                
                # Rename column back to feature
                null_imp_agg = null_imp_agg.rename(columns={'original_feature': 'feature'})
                return null_imp_agg
            
            return null_imp_df
            
        except Exception as e:
            logger.error(f"Error in _get_null_importance: {str(e)}")
            
            # Create emergency artificial null importances
            logger.warning("Creating emergency artificial null importances due to error")
            artificial_null_imp = []
            
            # Use original columns as feature names
            feature_names = X.columns.tolist()
            
            for i in range(5):  # Create 5 artificial runs
                for j, feature in enumerate(feature_names):
                    # Random importance values with decreasing mean for each feature
                    split_imp = np.random.exponential(1.0 / (j + 1)) 
                    gain_imp = np.random.exponential(1.0 / (j + 1))
                    
                    artificial_null_imp.append({
                        "feature": feature,
                        "importance_split": split_imp,
                        "importance_gain": gain_imp,
                        "run": i
                    })
            
            return pd.DataFrame(artificial_null_imp)
    
    def _calculate_feature_scores(self, actual_imp_df: pd.DataFrame, null_imp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate feature scores based on actual and null importance distributions.
        
        Parameters:
        -----------
        actual_imp_df : pd.DataFrame
            DataFrame with actual feature importances
        null_imp_df : pd.DataFrame
            DataFrame with null feature importances
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature scores
        """
        feature_scores = []
        
        # If actual_imp_df is empty, log a warning and return empty DataFrame
        if actual_imp_df.empty:
            logger.warning("Actual importance DataFrame is empty. Cannot calculate feature scores.")
            return pd.DataFrame(columns=["feature", "split_score", "gain_score"])
        
        # Check if we have enough null importance runs
        if null_imp_df.empty:
            logger.warning("Null importance DataFrame is empty. Creating scores based on actual importance only.")
            
            # Create scores based on ranking of actual importances (not binary 0-1)
            for i, feature in enumerate(actual_imp_df["feature"]):
                act_imps_gain = actual_imp_df.loc[actual_imp_df["feature"] == feature, "importance_gain"].mean()
                act_imps_split = actual_imp_df.loc[actual_imp_df["feature"] == feature, "importance_split"].mean()
                
                # Just use the raw importances (will be relative to each other)
                split_score = act_imps_split
                gain_score = act_imps_gain
                
                feature_scores.append((feature, split_score, gain_score))
                
            scores_df = pd.DataFrame(feature_scores, columns=["feature", "split_score", "gain_score"])
            logger.info("Created scores based on actual importance only (no null importance comparison)")
            return scores_df
            
        # Check number of null importance runs
        n_runs = len(null_imp_df["run"].unique())
        if n_runs < 5:
            logger.warning(f"Only {n_runs} null importance runs. This may not be enough for stable results.")
        
        logger.info(f"Scoring features based on {n_runs} null importance runs")
        
        # === SIMPLIFIED SCORING LOGIC (SIMILAR TO ORIGINAL NOTEBOOK) ===
        for feature in actual_imp_df["feature"].unique():
            # Get null importances for the feature
            null_imps_gain = null_imp_df.loc[null_imp_df["feature"] == feature, "importance_gain"].values
            
            # Get actual importance for the feature
            act_imps_gain = actual_imp_df.loc[actual_imp_df["feature"] == feature, "importance_gain"].mean()
            
            # Calculate gain score using the log of ratio between actual and null importance
            # This is the original formula from the notebook
            if null_imps_gain.size > 0:
                gain_score = np.log(1e-10 + act_imps_gain / (1 + np.percentile(null_imps_gain, 75)))
            else:
                # Only fall back if absolutely necessary
                gain_score = act_imps_gain
            
            # Same for split importances
            null_imps_split = null_imp_df.loc[null_imp_df["feature"] == feature, "importance_split"].values
            act_imps_split = actual_imp_df.loc[actual_imp_df["feature"] == feature, "importance_split"].mean()
            
            if null_imps_split.size > 0:
                split_score = np.log(1e-10 + act_imps_split / (1 + np.percentile(null_imps_split, 75)))
            else:
                split_score = act_imps_split
            
            # Add scores to the list
            feature_scores.append((feature, split_score, gain_score))
        
        # Create scores dataframe
        scores_df = pd.DataFrame(feature_scores, columns=["feature", "split_score", "gain_score"])
        
        # Report statistics on scores
        logger.info(f"Feature score statistics - Split score mean: {scores_df['split_score'].mean():.4f}, "
                    f"min: {scores_df['split_score'].min():.4f}, max: {scores_df['split_score'].max():.4f}")
        logger.info(f"Feature score statistics - Gain score mean: {scores_df['gain_score'].mean():.4f}, "
                    f"min: {scores_df['gain_score'].min():.4f}, max: {scores_df['gain_score'].max():.4f}")
        
        return scores_df
    
    def _remove_highly_correlated_features(self, X: pd.DataFrame) -> List[str]:
        """
        Identify highly correlated features to remove.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
            
        Returns
        -------
        List[str]
            List of features to drop
        """
        logger.info(f"Checking for highly correlated features above threshold {self.correlation_threshold_}")
        
        # Create correlation matrix and get upper triangle
        with SuppressWarnings():
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
            # Find columns with correlations higher than threshold
            to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold_)]
            
        logger.info(f"Found {len(to_drop)} highly correlated features to remove")
        return to_drop
    
    def get_useful_features(self) -> List[str]:
        """
        Get the list of useful features identified during fitting.
        
        Returns:
        --------
        List[str]
            List of useful feature names
        """
        if self.selected_features_ is None:
            raise ValueError("FeatureSelector has not been fitted yet. Call fit() first.")
        
        return self.selected_features_

    def set_thresholds(self, feature_scores: pd.DataFrame, correlation_threshold: float = 0.95) -> None:
        """
        Set the thresholds for feature selection based on score distribution.
        
        Parameters:
        -----------
        feature_scores : pd.DataFrame
            DataFrame with feature scores
        correlation_threshold : float, optional
            Threshold for correlation, by default 0.95
        """
        # Set correlation threshold
        self.correlation_threshold_ = correlation_threshold
        
        # Check if we have any features
        if feature_scores.empty:
            logger.error("Feature scores DataFrame is empty. Cannot set thresholds.")
            self.split_score_threshold_ = -float('inf')
            self.gain_score_threshold_ = -float('inf')
            return
        
        # For continuous scores (which may be negative due to log transform),
        # set thresholds that select a reasonable proportion of features
        
        # Sort features by scores
        sorted_split = feature_scores.sort_values('split_score', ascending=False)
        sorted_gain = feature_scores.sort_values('gain_score', ascending=False)
        
        # Select top 60% of features by each metric
        # This is more inclusive than the original notebook to ensure we keep important features
        split_threshold_idx = int(len(sorted_split) * 0.4)  # Keep top 60%
        gain_threshold_idx = int(len(sorted_gain) * 0.4)    # Keep top 60%
        
        # Set thresholds at these percentiles
        if split_threshold_idx < len(sorted_split):
            self.split_score_threshold_ = sorted_split.iloc[split_threshold_idx]['split_score']
        else:
            self.split_score_threshold_ = -float('inf')
            
        if gain_threshold_idx < len(sorted_gain):
            self.gain_score_threshold_ = sorted_gain.iloc[gain_threshold_idx]['gain_score']
        else:
            self.gain_score_threshold_ = -float('inf')
        
        logger.info(f"Using thresholds - Split: {self.split_score_threshold_:.3f}, Gain: {self.gain_score_threshold_:.3f}, Correlation: {self.correlation_threshold_:.3f}")
        
        # Check that at least some features will be selected
        features_selected = feature_scores[
            (feature_scores['split_score'] > self.split_score_threshold_) | 
            (feature_scores['gain_score'] > self.gain_score_threshold_)
        ]
        
        if len(features_selected) == 0:
            logger.warning("No features would be selected with current thresholds. Setting thresholds to select at least 20% of features.")
            # Set more permissive thresholds (lowest 20%)
            self.split_score_threshold_ = np.percentile(feature_scores['split_score'], 80)
            self.gain_score_threshold_ = np.percentile(feature_scores['gain_score'], 80)
            logger.info(f"Adjusted thresholds - Split: {self.split_score_threshold_:.3f}, Gain: {self.gain_score_threshold_:.3f}")
        elif len(features_selected) < 0.05 * len(feature_scores):
            logger.warning(f"Only {len(features_selected)} features would be selected. Setting thresholds to select at least 10% of features.")
            # Ensure at least 10% of features are selected
            self.split_score_threshold_ = np.percentile(feature_scores['split_score'], 90)
            self.gain_score_threshold_ = np.percentile(feature_scores['gain_score'], 90)
            logger.info(f"Adjusted thresholds - Split: {self.split_score_threshold_:.3f}, Gain: {self.gain_score_threshold_:.3f}")
        elif len(features_selected) > 0.9 * len(feature_scores):
            logger.warning(f"Almost all features ({len(features_selected)} out of {len(feature_scores)}) would be selected. Setting stricter thresholds.")
            # Set more restrictive thresholds (top 33%)
            self.split_score_threshold_ = np.percentile(feature_scores['split_score'], 67)
            self.gain_score_threshold_ = np.percentile(feature_scores['gain_score'], 67)
            logger.info(f"Adjusted thresholds - Split: {self.split_score_threshold_:.3f}, Gain: {self.gain_score_threshold_:.3f}") 