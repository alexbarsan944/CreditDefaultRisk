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
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
        Testing features
    y_test : pd.Series
        Testing target
    model_type : str
        Type of model to train ('xgboost', 'lightgbm', 'catboost', 'logistic_regression', 'random_forest')
    model_params : Dict
        Model hyperparameters
    cv_folds : int
        Number of folds for cross-validation
    random_state : int
        Random seed for reproducibility
    progress_bar : st.progress
        Streamlit progress bar
        
    Returns:
    --------
    Tuple[CreditRiskModel, Dict, Dict, List[plt.Figure]]
        Trained model, evaluation metrics, cross-validation results, and figures
    """
    figures = []
    
    # Check if enough variation in target variable
    if len(y_train.unique()) < 2:
        logger.error("Training data has only one class, cannot train model")
        metrics = {
            "auc": 0.5,
            "accuracy": y_train.value_counts().max() / len(y_train),
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "train_time": 0,
            "inference_time": 0
        }
        cv_results = {
            "overall_auc": 0.5,
            "mean_auc": 0.5,
            "std_auc": 0,
            "fold_metrics": [],
            "feature_importances": None,
            "train_time": 0,
            "inference_time": 0
        }
        return None, metrics, cv_results, figures
    
    # Check if y_test is not None before checking its unique values
    if y_test is not None and len(y_test.unique()) < 2:
        logger.warning("Test data has only one class, evaluation metrics may not be reliable")
    
    # Create and train model
    start_time = time.time()
    
    try:
        # Initialize model
        logger.info(f"Initializing {model_type} model")
        model = CreditRiskModel(model_type=model_type, model_params=model_params, random_state=random_state)
        
        # Update progress
        if progress_bar:
            progress_bar.progress(0.1, text="Model initialized")
        
        # Cross-validate model if requested
        if cv_folds > 0:
            logger.info(f"Performing {cv_folds}-fold cross-validation")
            cv_results = model.cross_validate(X_train, y_train, n_folds=cv_folds, stratified=True)
            
            # Plot ROC curve for cross-validation
            if len(cv_results.get("fold_metrics", [])) > 0:
                fig = plot_cv_roc_curve(model, X_train, y_train, cv_folds)
                if fig:
                    figures.append(fig)
            
            # Update progress
            if progress_bar:
                progress_bar.progress(0.3, text="Cross-validation completed")
        else:
            cv_results = {
                "overall_auc": 0.0,
                "mean_auc": 0.0,
                "std_auc": 0.0,
                "fold_metrics": []
            }
        
        # Train final model on full training data
        logger.info("Training final model on full training data")
        model.fit(X_train, y_train)
        
        # Update progress
        if progress_bar:
            progress_bar.progress(0.6, text="Model trained")
        
        # Get model metrics
        train_time = time.time() - start_time
        inference_start = time.time()
        
        # Get predictions
        try:
            # Check that test set has enough samples
            if len(y_test) == 0:
                logger.warning("Test set is empty, cannot evaluate model")
                y_pred = []
                y_prob = []
            else:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            inference_time = time.time() - inference_start
            
            # Calculate metrics if possible
            if len(y_test) > 0 and len(y_test.unique()) > 1 and len(y_prob) > 0:
                logger.info("Calculating evaluation metrics")
                try:
                    auc = roc_auc_score(y_test, y_prob)
                except Exception as e:
                    logger.error(f"Error calculating AUC: {str(e)}")
                    auc = 0.5  # Default value
                
                try:
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                except Exception as e:
                    logger.error(f"Error calculating metrics: {str(e)}")
                    accuracy = precision = recall = f1 = 0
            else:
                logger.warning("Cannot calculate metrics: insufficient test data or predictions")
                auc = accuracy = precision = recall = f1 = 0
            
            # Prepare metrics dictionary
            metrics = {
                "auc": auc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "train_time": train_time,
                "inference_time": inference_time
            }
            
            # Plot confusion matrix
            logger.info("Generating confusion matrix")
            if len(y_test) > 0 and len(y_pred) > 0:
                cm_fig = plot_confusion_matrix(y_test, y_pred)
                figures.append(cm_fig)
            
            # Update progress
            if progress_bar:
                progress_bar.progress(1.0, text="Evaluation completed")
        
        except Exception as e:
            logger.error(f"Error during prediction or evaluation: {str(e)}")
            metrics = {
                "auc": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "train_time": train_time,
                "inference_time": 0.0
            }
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        model = None
        metrics = {
            "auc": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "train_time": 0.0,
            "inference_time": 0.0
        }
        cv_results = {
            "overall_auc": 0.0,
            "mean_auc": 0.0,
            "std_auc": 0.0,
            "fold_metrics": []
        }
    
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

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Plot confusion matrix for binary classification.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with confusion matrix
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Show all ticks and label them
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Negative (0)', 'Positive (1)'],
               yticklabels=['Negative (0)', 'Positive (1)'],
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Add accuracy, precision, recall as text
        accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
        if np.sum(cm[:, 1]) > 0:  # Avoid div by zero
            precision = cm[1, 1] / np.sum(cm[:, 1])
        else:
            precision = 0
            
        if np.sum(cm[1, :]) > 0:  # Avoid div by zero
            recall = cm[1, 1] / np.sum(cm[1, :])
        else:
            recall = 0
            
        if precision + recall > 0:  # Avoid div by zero
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}", 
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        fig.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        # Return an empty figure if we couldn't create the confusion matrix
        return plt.figure()

def plot_cv_roc_curve(model: CreditRiskModel, X: pd.DataFrame, y: pd.Series, cv_folds: int) -> Optional[plt.Figure]:
    """
    Plot ROC curves for each fold of cross-validation.
    
    Parameters:
    -----------
    model : CreditRiskModel
        Model used for predictions
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    Optional[plt.Figure]
        Matplotlib figure with ROC curves or None if plotting fails
    """
    try:
        from sklearn.model_selection import StratifiedKFold
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # We'll track mean TPR at different FPR values
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        # Create cross-validation folds
        cv = StratifiedKFold(n_splits=cv_folds)
        
        # Process for each fold
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Split data
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            # Check if we have enough class variation
            if len(y_train_fold.unique()) < 2 or len(y_test_fold.unique()) < 2:
                logger.warning(f"Skipping fold {i+1} due to lack of class variation")
                continue
                
            # Create and train a model
            fold_model = CreditRiskModel(
                model_type=model.model_type,
                model_params=model.model_params,
                random_state=model.random_state
            )
            
            try:
                # Train model
                fold_model.fit(X_train_fold, y_train_fold)
                
                # Get predictions
                y_score = fold_model.predict_proba(X_test_fold)[:, 1]
                
                # Compute ROC curve and area
                fpr, tpr, _ = roc_curve(y_test_fold, y_score)
                roc_auc = auc(fpr, tpr)
                
                # Interpolate TPR at the fixed FPR values
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)
                
                # Plot ROC curve for this fold
                ax.plot(fpr, tpr, lw=1, alpha=0.3,
                        label=f'ROC fold {i+1} (AUC = {roc_auc:.2f})')
            except Exception as e:
                logger.error(f"Error processing fold {i+1}: {str(e)}")
                continue
        
        # Check if we have any successful folds
        if not tprs:
            logger.warning("No successful CV folds to plot ROC curve")
            return None
            
        # Plot the chance level
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
        
        # Plot mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
                lw=2, alpha=.8)
        
        # Plot standard deviation around mean ROC
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=f'± 1 std. dev.')
        
        # Final plot settings
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title=f'Cross-Validation ROC Curves ({cv_folds} folds)',
               xlabel='False Positive Rate',
               ylabel='True Positive Rate')
        ax.legend(loc="lower right")
        fig.tight_layout()
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting CV ROC curves: {str(e)}")
        return None 