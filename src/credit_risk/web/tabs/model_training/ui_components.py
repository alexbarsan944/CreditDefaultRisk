"""
UI components for the model training tab.
"""

import logging
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import roc_auc_score, classification_report

from credit_risk.models.estimator import CreditRiskModel
from credit_risk.utils.streamlit_utils import (
    create_download_link,
    save_step_data,
    load_step_data,
    get_available_step_data,
    DATA_DIR
)

from .utils import check_gpu_availability

logger = logging.getLogger(__name__)

# Model types
MODEL_TYPES = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost"
}


def display_model_params(model_type, use_gpu=False):
    """
    Display model parameters in the UI and return the parameters dictionary.
    
    Parameters:
    -----------
    model_type : str
        Type of model (xgboost, lightgbm, catboost)
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default False
    
    Returns:
    --------
    dict
        Dictionary of model parameters selected by the user
    """
    st.subheader("Model Parameters")
    
    # Common parameters
    base_params = {
        "random_state": 42
    }
    
    # Check GPU availability for this model type
    gpu_available = check_gpu_availability(model_type) if use_gpu else False
    
    # GPU acceleration
    if use_gpu:
        if not gpu_available:
            st.warning(f"GPU acceleration is not available for {MODEL_TYPES[model_type]}. Using CPU instead.")
        
        if model_type == "xgboost" and gpu_available:
            # Fixed XGBoost GPU configuration to avoid deprecation warning
            base_params["tree_method"] = "hist"
            base_params["device"] = "cuda"
            st.info("Using XGBoost with CUDA acceleration (tree_method='hist', device='cuda')")
        elif model_type == "lightgbm" and gpu_available:
            base_params["device_type"] = "gpu"
            st.info("Using LightGBM with GPU acceleration")
        elif model_type == "catboost" and gpu_available:
            base_params["task_type"] = "GPU"
            st.info("Using CatBoost with GPU acceleration")
    
    # Model-specific parameters
    if model_type == "xgboost":
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.number_input(
                "Number of trees", 
                min_value=50,
                max_value=1000,
                value=100,
                step=50,
                key="xgb_n_estimators"
            )
            
            max_depth = st.number_input(
                "Max depth", 
                min_value=3,
                max_value=15,
                value=6,
                key="xgb_max_depth"
            )
            
            gamma = st.number_input(
                "Gamma (regularization)", 
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                format="%.2f",
                key="xgb_gamma"
            )
        
        with col2:
            learning_rate = st.number_input(
                "Learning rate", 
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                format="%.3f",
                key="xgb_learning_rate"
            )
            
            subsample = st.number_input(
                "Subsample ratio", 
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                format="%.2f",
                key="xgb_subsample"
            )
            
            colsample_bytree = st.number_input(
                "Column sample by tree", 
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                format="%.2f",
                key="xgb_colsample_bytree"
            )
        
        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "early_stopping_rounds": 100,  # Early stopping
            "verbosity": 0,  # Quiet mode
        }
        
    elif model_type == "lightgbm":
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.number_input(
                "Number of trees", 
                min_value=50,
                max_value=1000,
                value=100,
                step=50,
                key="lgb_n_estimators"
            )
            
            max_depth = st.number_input(
                "Max depth", 
                min_value=3,
                max_value=15,
                value=6,
                key="lgb_max_depth"
            )
            
            num_leaves = st.number_input(
                "Number of leaves", 
                min_value=10,
                max_value=255,
                value=31,
                key="lgb_num_leaves"
            )
            
        with col2:
            learning_rate = st.number_input(
                "Learning rate", 
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                format="%.3f",
                key="lgb_learning_rate"
            )
            
            subsample = st.number_input(
                "Subsample ratio", 
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                format="%.2f",
                key="lgb_subsample"
            )
            
            colsample_bytree = st.number_input(
                "Feature fraction", 
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                format="%.2f",
                key="lgb_feature_fraction"
            )
        
        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "early_stopping_rounds": 100,  # Early stopping
            "verbose": -1,  # Quiet mode
            "silent": True,  # Silent mode
            "n_jobs": -1  # Use all CPU cores
        }
        
    elif model_type == "catboost":
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.number_input(
                "Number of trees", 
                min_value=50,
                max_value=1000,
                value=100,
                step=50,
                key="cat_n_estimators"
            )
            
            max_depth = st.number_input(
                "Max depth", 
                min_value=1,
                max_value=16,
                value=6,
                key="cat_max_depth"
            )
            
            l2_leaf_reg = st.number_input(
                "L2 regularization", 
                min_value=1.0,
                max_value=100.0,
                value=3.0,
                format="%.1f",
                key="cat_l2_leaf_reg"
            )
            
        with col2:
            learning_rate = st.number_input(
                "Learning rate", 
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                format="%.3f",
                key="cat_learning_rate"
            )
            
            rsm = st.number_input(
                "RSM (feature sample ratio)", 
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                format="%.2f",
                key="cat_rsm"
            )
            
            use_best_model = st.checkbox(
                "Use best model",
                value=True,
                help="Use best model found during training",
                key="cat_use_best_model"
            )
        
        model_params = {
            "iterations": n_estimators,
            "depth": max_depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "rsm": rsm,
            "use_best_model": use_best_model,
            "verbose": False,  # Quiet mode
        }
    
    # Combine base params and model-specific params
    params = {**base_params, **model_params}
    
    return params


def render_ui_controls():
    """
    Render the UI controls for the model training tab.
    
    Returns
    -------
    dict
        Dictionary of UI control values
    """
    st.sidebar.header("Model Training Options")
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    
    # Check if we have feature selection results already
    has_feature_selection = "feature_selection_results" in st.session_state
    
    use_feature_selection = st.sidebar.checkbox(
        "Use Feature Selection Results", 
        value=has_feature_selection,
        disabled=not has_feature_selection,
        help="Use the results from the Feature Selection tab"
    )
    
    # Check if we have access to previous models
    available_data = get_available_step_data()
    has_models = "model_training" in available_data and len(available_data["model_training"]) > 0
    prev_models_available = has_models
    
    load_prev_model = st.sidebar.checkbox(
        "Load Previous Model", 
        value=False,
        disabled=not prev_models_available,
        help="Load a previously trained model"
    )
    
    # Model selection
    if load_prev_model and prev_models_available:
        # Get available model options
        model_options = []
        for idx, metadata in enumerate(available_data["model_training"]):
            model_type = metadata.get("model_type", "Unknown")
            timestamp = metadata.get("created_at", "Unknown")
            model_options.append(f"{model_type} ({timestamp})")
        
        # Allow selecting a model
        selected_model = st.sidebar.selectbox(
            "Select Model",
            options=model_options,
            index=0
        )
        
        # Get the index of the selected model
        selected_model_idx = model_options.index(selected_model)
    else:
        # Only show model selection if we're not loading a previous model
        st.sidebar.subheader("Model Selection")
        
        model_type = st.sidebar.selectbox(
            "Model Type",
            options=list(MODEL_TYPES.keys()),
            format_func=lambda x: MODEL_TYPES[x],
            index=0,
            help="Select which model type to use"
        )
        
        # GPU acceleration
        use_gpu = st.sidebar.checkbox(
            "Use GPU Acceleration", 
            value=True,
            help="Use GPU for faster training if available"
        )
        
        # Cross-validation
        cv_folds = st.sidebar.slider(
            "Cross-Validation Folds",
            min_value=2,
            max_value=10,
            value=5,
            help="Number of folds for cross-validation"
        )
        
        selected_model_idx = 0  # Placeholder
    
    # Return all control values
    return {
        "use_feature_selection": use_feature_selection,
        "load_prev_model": load_prev_model,
        "prev_models_available": prev_models_available,
        "selected_model_idx": selected_model_idx,
        "model_type": model_type if not load_prev_model else None,
        "use_gpu": use_gpu if not load_prev_model else False,
        "cv_folds": cv_folds if not load_prev_model else 5,
    }


def plot_model_results(model, metrics, cv_results, X_train, y_test, y_pred_proba, figures=None):
    """
    Plot the results of model training and evaluation.
    
    Parameters:
    -----------
    model : CreditRiskModel
        Trained model
    metrics : dict
        Performance metrics dictionary
    cv_results : dict
        Cross-validation results dictionary
    X_train : pd.DataFrame
        Training features used for feature importance
    y_test : pd.Series
        True test labels
    y_pred_proba : numpy.ndarray
        Predicted probabilities for test set
    figures : list, optional
        Additional figures to display
    """
    # Display metrics
    st.subheader("Model Performance")
    
    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AUC", f"{metrics['auc']:.4f}")
    
    with col2:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    
    with col3:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    
    with col4:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    
    # CV results
    if cv_results:
        st.subheader("Cross-Validation Results")
        
        # Format CV results
        cv_metrics = {
            "AUC": cv_results.get("test_auc", []),
            "Accuracy": cv_results.get("test_accuracy", []),
            "Precision": cv_results.get("test_precision", []),
            "Recall": cv_results.get("test_recall", [])
        }
        
        # Create a box plot
        fig = go.Figure()
        
        for metric_name, values in cv_metrics.items():
            if values:  # Only add if we have values
                fig.add_trace(go.Box(
                    y=values,
                    name=metric_name,
                    boxmean=True  # adds mean as dashed line
                ))
        
        fig.update_layout(
            template="plotly_dark",
            title="Cross-Validation Metrics Distribution",
            yaxis_title="Score",
            boxmode="group"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # CV summary table
        cv_summary = pd.DataFrame({
            "Metric": list(cv_metrics.keys()),
            "Mean": [np.mean(values) if values else np.nan for values in cv_metrics.values()],
            "Std": [np.std(values) if values else np.nan for values in cv_metrics.values()],
            "Min": [np.min(values) if values else np.nan for values in cv_metrics.values()],
            "Max": [np.max(values) if values else np.nan for values in cv_metrics.values()]
        })
        
        st.dataframe(cv_summary, use_container_width=True)
    
    # Display figures
    if figures:
        st.subheader("Model Evaluation Plots")
        
        for i, fig in enumerate(figures):
            try:
                # Check if figure is a Plotly figure (has 'update_layout' attribute)
                if hasattr(fig, 'update_layout'):
                    # It's a Plotly figure, use plotly_chart with dark theme
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                else:
                    # It's a matplotlib figure, use pyplot
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Error displaying figure {i+1}: {str(e)}")
                logger.warning(f"Error displaying figure {i+1}: {str(e)}")
    
    # Additional Feature Importance Visualization
    # This is a backup in case the model's plot_feature_importance fails
    if model and hasattr(model, 'model') and hasattr(model.model, 'feature_importances_') and X_train is not None:
        try:
            # Only show this if it wasn't already shown in figures
            feature_importance_shown = any(
                hasattr(fig, 'layout') and 
                hasattr(fig.layout, 'title') and 
                fig.layout.title.text and 
                'feature importance' in fig.layout.title.text.lower() 
                for fig in figures if hasattr(fig, 'layout')
            )
            
            if not feature_importance_shown:
                st.subheader("Additional Feature Importance")
                
                # Get feature names and importances
                features = X_train.columns.tolist()
                importances = model.model.feature_importances_
                
                # Check for length mismatch and fix if needed
                if len(importances) != len(features):
                    logger.warning(f"Feature importance length mismatch: {len(importances)} importances vs {len(features)} features")
                    # Use the minimum length of both arrays
                    min_length = min(len(importances), len(features))
                    importances = importances[:min_length]
                    features = features[:min_length]
                
                # Create DataFrame with matching lengths
                feature_importance = pd.DataFrame({
                    'feature': features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Display top 20 features
                feature_importance = feature_importance.head(20)
                
                # Create Plotly bar chart with dark theme
                fig = go.Figure(go.Bar(
                    x=feature_importance['importance'][::-1],
                    y=feature_importance['feature'][::-1],
                    orientation='h',
                    marker=dict(
                        color=feature_importance['importance'][::-1],
                        colorscale='Blues',
                        showscale=True
                    )
                ))
                
                # Update layout with dark theme
                fig.update_layout(
                    template="plotly_dark",
                    title="Feature Importance",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.warning(f"Error generating additional feature importance plot: {str(e)}")
            # Don't show warning to user as this is a backup visualization 