"""
Utility functions for data preprocessing in model training.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def clean_dataset_for_optimization(X, y):
    """
    Clean a dataset by removing NaN and infinity values and encode categorical features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features DataFrame, potentially containing NaN, infinity values or categorical data
    y : pd.Series
        Target Series, potentially containing NaN values
        
    Returns:
    --------
    (pd.DataFrame, pd.Series)
        Cleaned features and target
    """
    X_clean = X.copy()
    y_clean = y.copy()
    
    # Process categorical columns
    categorical_cols = X_clean.select_dtypes(exclude=np.number).columns
    if len(categorical_cols) > 0:
        logger.info(f"Found {len(categorical_cols)} categorical columns that need encoding")
        
        # For each categorical column, apply simple label encoding or one-hot encoding
        for col in categorical_cols:
            if X_clean[col].nunique() <= 10:  # For categories with few unique values
                # One-hot encode with pandas get_dummies
                logger.info(f"One-hot encoding column {col} with {X_clean[col].nunique()} unique values")
                
                # Fill NAs with a placeholder before encoding
                X_clean[col] = X_clean[col].fillna("Unknown")
                
                # Apply one-hot encoding
                dummies = pd.get_dummies(X_clean[col], prefix=col, dummy_na=False)
                
                # Concatenate with the rest of the dataframe and drop the original column
                X_clean = pd.concat([X_clean.drop(col, axis=1), dummies], axis=1)
            else:
                # For many categories, use simple ordinal encoding
                logger.info(f"Ordinal encoding column {col} with {X_clean[col].nunique()} unique values")
                
                # Fill NAs with a placeholder
                X_clean[col] = X_clean[col].fillna("Unknown")
                
                # Convert to category codes (integers)
                X_clean[col] = X_clean[col].astype('category').cat.codes.astype(float)
    
    # Double-check if any non-numeric columns remain and convert them
    remaining_categorical = X_clean.select_dtypes(exclude=np.number).columns
    if len(remaining_categorical) > 0:
        logger.warning(f"Found {len(remaining_categorical)} remaining categorical columns after initial processing")
        for col in remaining_categorical:
            logger.info(f"Forcing column {col} to numeric using ordinal encoding")
            # Force categorical to numeric with simple ordinal encoding
            X_clean[col] = pd.Categorical(X_clean[col]).codes.astype(float)
    
    # Check for NaN or infinity values in X
    has_nans = X_clean.isna().any().any()
    has_inf = np.isinf(X_clean).any().any()
    
    if has_nans or has_inf:
        logger.info(f"Data cleaning needed: NaN values: {has_nans}, Infinity values: {has_inf}")
        
        # Fill each column separately to preserve column distributions
        for col in X_clean.columns:
            # Handle infinite values by replacing with NaN
            if np.isinf(X_clean[col]).any():
                logger.info(f"Replacing infinity values in column {col}")
                X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
                
            # Fill NaNs with column median
            if X_clean[col].isna().any():
                median_value = X_clean[col].median()
                if pd.isna(median_value):  # If median is also NaN (all values are NaN)
                    logger.warning(f"Column {col} has all NaN values, filling with 0")
                    median_value = 0
                
                logger.info(f"Filling NaN values in column {col} with median {median_value}")
                X_clean[col] = X_clean[col].fillna(median_value)
    
    # Handle NaN values in target
    if y_clean.isna().any():
        logger.info("Filling NaN values in target")
        # For binary classification, use mode (most common value)
        mode_value = y_clean.mode()[0]  # Most common value
        y_clean = y_clean.fillna(mode_value)
    
    # Final verification that all columns are numeric
    final_check = X_clean.select_dtypes(exclude=np.number).columns
    if len(final_check) > 0:
        logger.error(f"Failed to convert these columns to numeric: {list(final_check)}")
        # As a last resort, drop columns that couldn't be converted
        X_clean = X_clean.select_dtypes(include=np.number)
        logger.warning(f"Dropped {len(final_check)} columns that couldn't be converted to numeric")
        
    return X_clean, y_clean


def check_feature_quality(X):
    """
    Analyze a dataset and report quality issues such as missing values, 
    low variance features, and outliers.
    
    Parameters:
    -----------
    X : pd.DataFrame
        The feature dataset to check
        
    Returns:
    --------
    dict
        A report containing statistics and issues found in the data
    """
    report = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "issues": []
    }
    
    # Check for missing values
    missing_counts = X.isna().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    if not features_with_missing.empty:
        report["issues"].append({
            "type": "missing_values",
            "description": f"Found {len(features_with_missing)} features with missing values",
            "details": features_with_missing.to_dict()
        })
    
    # Separate numerical and categorical columns
    numeric_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(exclude=np.number).columns
    
    # Report categorical columns
    if len(categorical_cols) > 0:
        report["issues"].append({
            "type": "categorical_features",
            "description": f"Found {len(categorical_cols)} categorical features that need encoding",
            "details": list(categorical_cols)
        })
    
    # Check for constants or near-constants (only on numeric columns)
    if len(numeric_cols) > 0:
        variance = X[numeric_cols].var()
        low_variance_features = variance[variance < 0.01]
        if not low_variance_features.empty:
            report["issues"].append({
                "type": "low_variance",
                "description": f"Found {len(low_variance_features)} features with very low variance",
                "details": low_variance_features.to_dict()
            })
    
        # Check for infinity values (only on numeric columns)
        inf_counts = np.isinf(X[numeric_cols]).sum()
        features_with_inf = inf_counts[inf_counts > 0]
        if not features_with_inf.empty:
            report["issues"].append({
                "type": "infinity_values",
                "description": f"Found {len(features_with_inf)} features with infinity values",
                "details": features_with_inf.to_dict()
            })
    
        # Check for extreme outliers using IQR (only on numeric columns)
        outlier_features = {}
        for col in numeric_cols:
            try:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                
                if iqr > 0:  # Avoid division by zero
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    
                    n_outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
                    if n_outliers > 0:
                        outlier_features[col] = n_outliers
            except Exception:
                # Skip columns that cause issues with quantile calculation
                pass
        
        if outlier_features:
            report["issues"].append({
                "type": "outliers",
                "description": f"Found {len(outlier_features)} features with significant outliers",
                "details": outlier_features
            })
    
    return report 