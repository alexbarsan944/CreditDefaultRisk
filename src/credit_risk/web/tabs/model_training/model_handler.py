"""
Model handler functions for training, evaluating and saving models.
"""

import logging
import time
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_auc_score, classification_report

from credit_risk.models.estimator import CreditRiskModel
from credit_risk.data.feature_selection import FeatureSelector
from credit_risk.utils.metadata import ModelMetadata
from credit_risk.utils.streamlit_utils import (
    save_step_data,
    load_step_data,
    get_available_step_data
)

logger = logging.getLogger(__name__)


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    model_params: Dict,
    cv_folds: int = 5,
    random_state: int = 42,
    progress_bar = None
) -> Tuple[CreditRiskModel, Dict, Dict, List[plt.Figure]]:
    """
    Train and evaluate a model for credit risk prediction.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_type : str
        Type of model to use (xgboost, lightgbm, catboost)
    model_params : Dict
        Model parameters
    cv_folds : int, optional
        Number of cross-validation folds, by default 5
    random_state : int, optional
        Random seed for reproducibility, by default 42
    progress_bar : streamlit.ProgressBar, optional
        Progress bar for tracking, by default None
    
    Returns:
    --------
    Tuple[CreditRiskModel, Dict, Dict, List[plt.Figure]]
        Trained model, metrics, cross-validation results, and figures
    """
    start_time = time.time()
    figures = []
    
    # Check data quality before proceeding
    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Check for NaN values
    train_nan_counts = X_train.isna().sum()
    nan_cols = train_nan_counts[train_nan_counts > 0]
    if not nan_cols.empty:
        logger.warning(f"Found {len(nan_cols)} columns with NaN values in training data")
        logger.warning(f"Top 5 columns with most NaNs: {nan_cols.sort_values(ascending=False).head()}")
    
    # Identify numeric and categorical columns
    numeric_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]
    logger.info(f"Detected {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
    
    # Check for columns with all NaN values
    all_nan_cols = [col for col in X_train.columns if X_train[col].isna().all()]
    if all_nan_cols:
        logger.warning(f"Found {len(all_nan_cols)} columns with all NaN values: {all_nan_cols}")
    
    # Create model
    try:
        logger.info(f"Creating {model_type} model with parameters: {model_params}")
        model = CreditRiskModel(
            model_type=model_type,
            model_params=model_params,
            random_state=random_state
        )
        
        # Update progress
        if progress_bar:
            progress_bar.progress(0.2, "Running cross-validation...")
        
        # Run cross-validation
        try:
            cv_results = model.cross_validate(X_train, y_train, n_folds=cv_folds, stratified=True)
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}", exc_info=True)
            # If we are here, there was an error in cross-validation
            # Lets try to provide more context about the data
            sample_row = X_train.iloc[0] if not X_train.empty else None
            if sample_row is not None:
                logger.info(f"Sample row data types: {sample_row.apply(type)}")
            raise
        
        # Update progress
        if progress_bar:
            progress_bar.progress(0.6, "Training final model...")
        
        # Train model on full dataset
        logger.info(f"Fitting {model_type} model on full training dataset...")
        model.fit(X_train, y_train)
        
        # Update progress
        if progress_bar:
            progress_bar.progress(0.8, "Evaluating model...")
        
        # Make predictions on test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
    except Exception as e:
        logger.error(f"Error during model creation or training: {str(e)}", exc_info=True)
        raise ValueError(f"Model training failed: {str(e)}")
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Create performance metrics dictionary
    metrics = {
        "auc": auc,
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "train_time": time.time() - start_time
    }
    
    # Generate figures
    figures = []
    
    # Feature importance
    try:
        if progress_bar:
            progress_bar.progress(0.9, "Generating feature importance plot...")
        fig = model.plot_feature_importance(top_n=20)
        figures.append(fig)
    except Exception as e:
        logger.warning(f"Error generating feature importance plot: {str(e)}")
    
    # Confusion matrix
    try:
        if progress_bar:
            progress_bar.progress(0.95, "Generating confusion matrix...")
        fig = model.plot_confusion_matrix(X_test, y_test)
        figures.append(fig)
    except Exception as e:
        logger.warning(f"Error generating confusion matrix: {str(e)}")
    
    # Complete progress
    if progress_bar:
        progress_bar.progress(1.0, "Model training and evaluation complete!")
    
    return model, metrics, cv_results, figures


def save_model_and_selector(
    model: CreditRiskModel,
    feature_selector: FeatureSelector = None,
    model_type: str = None
) -> str:
    """
    Save trained model and feature selector to disk
    
    Parameters:
    -----------
    model : CreditRiskModel
        Trained model to save
    feature_selector : FeatureSelector, optional
        Feature selector to save
    model_type : str, optional
        Type of model
        
    Returns:
    --------
    str
        Path to saved model
    """
    try:
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Generate model ID
        model_id = str(uuid.uuid4())[:8]
        
        # Add timestamp to model name for better identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # If we have feature selection timestamp, include it in the model name
        feature_selection_timestamp = st.session_state.get("feature_selection_timestamp", "unknown")
    
        # Create model metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type or model.model_type,
            created_at=datetime.now().isoformat(),
            model_params=model.model_params,
            feature_names=list(feature_selector.selected_features) if feature_selector else None,
            feature_selection_timestamp=feature_selection_timestamp
        )
        
        # Create model directory
        model_dir = models_dir / f"{model_type}_{timestamp}_{model_id}"
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.joblib"
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save feature selector if provided
        if feature_selector:
            selector_path = model_dir / "feature_selector.joblib"
            feature_selector.save(str(selector_path))
            logger.info(f"Feature selector saved to {selector_path}")
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
            logger.info(f"Model metadata saved to {metadata_path}")
            
        # Also save as a step in our step_data directory
        model_data = {
            "model": model,
            "feature_selector": feature_selector,
            "metadata": metadata.to_dict(),
            "model_dir": str(model_dir),
            "timestamp": timestamp
        }
        
        # Save to model_training step
        filepath = save_step_data("model_training", model_data, 
                                 f"{model_type} model with {len(metadata.feature_names or [])} features")
        if filepath:
            logger.info(f"Model also saved as step data to {filepath}")
    
        return str(model_dir)

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}", exc_info=True)
        return None 