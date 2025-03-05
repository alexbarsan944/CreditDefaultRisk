"""
Sampling utilities for the Credit Default Risk application.

This module provides functions for consistent sampling across the application.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_sampling_profile(
    datasets: Dict[str, pd.DataFrame],
    profile_name: str,
    custom_sample_size: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Apply a predefined sampling profile to datasets.
    
    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of dataset names and DataFrames
    profile_name : str
        Name of the sampling profile to apply
    custom_sample_size : int, optional
        Override the profile's sample size
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of sampled datasets
    """
    # Define sampling profiles
    profiles = {
        "Quick Exploration": {
            "description": "Small samples for quick exploration and debugging",
            "sample_size": 5000,
            "sampling_method": "Random",
            "target_variable": "TARGET"
        },
        "Balanced Development": {
            "description": "Balanced samples for model development",
            "sample_size": 20000,
            "sampling_method": "Stratified",
            "target_variable": "TARGET"
        },
        "Production Preparation": {
            "description": "Larger samples for final model testing",
            "sample_size": 50000,
            "sampling_method": "Stratified",
            "target_variable": "TARGET"
        },
        "Full Dataset": {
            "description": "Use the entire dataset (warning: may be slow)",
            "sample_size": None,
            "sampling_method": None,
            "target_variable": "TARGET"
        }
    }
    
    # Get the requested profile
    if profile_name not in profiles:
        logger.warning(f"Unknown sampling profile: {profile_name}. Using 'Balanced Development' instead.")
        profile = profiles["Balanced Development"]
    else:
        profile = profiles[profile_name]
    
    # Override sample size if provided
    if custom_sample_size is not None:
        profile["sample_size"] = custom_sample_size
    
    # If "Full Dataset" profile, return original datasets
    if profile_name == "Full Dataset" or profile["sample_size"] is None:
        logger.info("Using full datasets without sampling")
        return datasets
    
    # Apply sampling to each dataset
    sampled_datasets = {}
    for name, df in datasets.items():
        # Check if we should use stratified sampling (only for main dataset with target)
        use_stratified = (
            profile["sampling_method"] == "Stratified" and
            name == "application_train" and 
            profile["target_variable"] in df.columns
        )
        
        # Determine sample size for this dataset
        sample_size = min(profile["sample_size"], df.shape[0])
        
        if use_stratified:
            logger.info(f"Applying stratified sampling to {name} with size {sample_size}")
            # Just take the first part from stratified split to maintain class distribution
            sampled_df, _ = train_test_split(
                df, 
                train_size=sample_size,
                stratify=df[profile["target_variable"]],
                random_state=42
            )
            sampled_datasets[name] = sampled_df
        else:
            logger.info(f"Applying random sampling to {name} with size {sample_size}")
            sampled_datasets[name] = df.sample(sample_size, random_state=42)
    
    # Log sampling results
    total_rows = sum(df.shape[0] for df in sampled_datasets.values())
    logger.info(f"Sampling complete. Total rows across all datasets: {total_rows}")
    
    return sampled_datasets


def create_train_test_split(
    data: pd.DataFrame,
    test_size: float = 0.2, 
    target_column: str = "TARGET",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a train-test split with consistent parameters.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to split
    test_size : float
        Proportion of data to include in test split
    target_column : str
        Name of the target column
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    if target_column not in data.columns:
        logger.warning(f"Target column '{target_column}' not found in data")
        # If no target column, just split the data randomly
        train_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state
        )
        return train_data, test_data, None, None
    
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train-test split created: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test


def create_stratified_folds(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified cross-validation folds with consistent parameters.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame
    y : pd.Series
        Target Series
    n_splits : int
        Number of folds
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples
    """
    from sklearn.model_selection import StratifiedKFold
    
    # Create stratified K-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Generate the folds
    folds = list(skf.split(X, y))
    
    logger.info(f"Created {n_splits} stratified folds")
    
    return folds 