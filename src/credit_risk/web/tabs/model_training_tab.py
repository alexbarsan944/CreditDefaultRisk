"""
Model Training Tab for the Credit Risk Web Application
"""

import logging
import time
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from sklearn.metrics import roc_auc_score

from credit_risk.utils.streamlit_utils import (
    create_download_link,
    save_step_data,
    load_step_data,
    get_available_step_data,
    DATA_DIR
)

# Import from modular components
from .model_training import (
    render_ui_controls,
    display_model_params,
    plot_model_results,
    train_and_evaluate_model,
    save_model_and_selector,
    run_hyperparameter_optimization,
    validate_feature_data
)

logger = logging.getLogger(__name__)


def render_model_training_tab(features_df=None):
    """
    Render the model training tab.
    
    Parameters
    ----------
    features_df : pd.DataFrame, optional
        DataFrame with feature-engineered data, if available
    """
    st.header("Model Training and Evaluation")
    
    # Add intro text
    st.write("""
    In this tab, you can train and evaluate machine learning models for credit risk prediction.
    
    You can select model type, tune hyperparameters, and evaluate performance metrics.
    """)
    
    # Get UI controls from sidebar
    controls = render_ui_controls()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        # Create two tabs for classic training and hyperparameter optimization
        classic_tab, hyperopt_tab = st.tabs(["Classic Training", "Hyperparameter Optimization"])
        
        # Process data based on source (feature selection or feature engineering)
        if "feature_selection_results" in st.session_state and controls.get("use_feature_selection", False):
            # Get data from feature selection
            fs_results = st.session_state.feature_selection_results
            X_train = fs_results["X_train"]
            X_test = fs_results["X_test"]
            y_train = fs_results["y_train"]
            y_test = fs_results["y_test"]
            feature_selector = fs_results.get("feature_selector")
            
            # Show feature selection info
            st.info(f"Using selected features from Feature Selection: {X_train.shape[1]} features")
        elif features_df is not None and "TARGET" in features_df.columns:
            # Use feature-engineered data
            st.info("Using feature-engineered data for modeling")
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            X = features_df.drop("TARGET", axis=1)
            y = features_df["TARGET"]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create dummy feature selector
            from credit_risk.features.selection.feature_selector_fixed import FeatureSelector
            feature_selector = FeatureSelector(random_state=42)
            # Add the useful_features_ property with the list of columns
            if not hasattr(feature_selector, 'useful_features_'):
                feature_selector.useful_features_ = X_train.columns.tolist()
        else:
            st.warning("No feature selection results or engineered features available.")
            st.info("Please run feature selection first or upload preprocessed data.")
            return
            
        # Validate feature data
        X_train, X_test, selected_features = validate_feature_data(
            X_train, X_test, X_train.columns.tolist()
        )
        
        # Show data dimensions
        st.write(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        st.write(f"Testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Display model parameters
        model_type = controls["model_type"]
        use_gpu = controls["use_gpu"]
        
        # Define a function for training and displaying results to avoid code duplication
        def train_and_display_results(model_params, progress_bar=None):
            try:
                with st.spinner("Training model..."):
                    # Train model
                    model, metrics, cv_results, figures = train_and_evaluate_model(
                        X_train, 
                        y_train,
                        X_test,
                        y_test,
                        model_type,
                        model_params,
                        cv_folds=controls.get("cv_folds", 5),
                        progress_bar=progress_bar
                    )
                    
                    # Save model
                    model_dir = save_model_and_selector(model, feature_selector, model_type)
                    if model_dir:
                        st.success(f"Model saved to {model_dir}")
                    
                    # Get predictions for visualization
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Display model results
                    plot_model_results(
                        model, 
                        metrics, 
                        cv_results, 
                        X_train, 
                        y_test, 
                        y_pred_proba, 
                        figures
                    )
                    
                    # Store model results in session state
                    st.session_state.model_results = {
                        "model": model,
                        "metrics": metrics,
                        "model_type": model_type,
                        "model_params": model_params,
                        "feature_names": X_train.columns.tolist(),
                        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    return True
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                logger.error(f"Model training failed: {str(e)}", exc_info=True)
                return False
                
        # Classic Training Tab
        with classic_tab:
            st.subheader("Classic Model Training")
            st.write("Configure model parameters and train with fixed settings.")
            
            # Display standard model parameters
            model_params = display_model_params(model_type, use_gpu)
            
            # Train button for classic training
            if st.button("Train Model", key="classic_train_button"):
                # Run with progress bar
                progress_bar = st.progress(0.0, "Preparing for model training...")
                train_and_display_results(model_params, progress_bar)
        
        # Hyperparameter Optimization Tab
        with hyperopt_tab:
            st.subheader("Hyperparameter Optimization")
            st.write("Automatically search for the best hyperparameters using Bayesian optimization.")
            
            # Hyperparameter optimization settings
            col1, col2 = st.columns(2)
            with col1:
                n_calls = st.number_input(
                    "Number of trials", 
                    min_value=10, 
                    max_value=100,
                    value=30,
                    step=5,
                    help="Number of hyperparameter combinations to try"
                )
            
            with col2:
                cv_folds = st.number_input(
                    "Cross-validation folds", 
                    min_value=2, 
                    max_value=10,
                    value=5,
                    step=1,
                    help="Number of cross-validation folds"
                )
            
            # Show which hyperparameters will be optimized based on model type
            st.subheader("Hyperparameters to Optimize")
            st.write(f"Model type: {model_type.upper()}")
            
            hyperparams_by_model = {
                "xgboost": ["n_estimators (50-500)", "max_depth (3-10)", "learning_rate (0.01-0.3)", 
                           "subsample (0.5-1.0)", "colsample_bytree (0.5-1.0)", "gamma (0-10)", 
                           "min_child_weight (1-10)"],
                "lightgbm": ["n_estimators (50-500)", "max_depth (3-15)", "learning_rate (0.01-0.3)", 
                            "num_leaves (10-255)", "subsample (0.5-1.0)", "colsample_bytree (0.5-1.0)", 
                            "min_child_weight (1-100)"],
                "catboost": ["n_estimators (50-500)", "max_depth (3-10)", "learning_rate (0.01-0.3)", 
                            "subsample (0.5-1.0)", "l2_leaf_reg (1-100)"]
            }
            
            for param in hyperparams_by_model.get(model_type, []):
                st.write(f"• {param}")
            
            # Train button for hyperparameter optimization
            if st.button("Optimize Hyperparameters", key="hyperopt_train_button"):
                # Run with progress bar
                progress_bar = st.progress(0.0, "Preparing for hyperparameter optimization...")
                
                try:
                    st.info("Running hyperparameter optimization...")
                    
                    # Split train set into train and validation
                    from sklearn.model_selection import train_test_split
                    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                    )
                    
                    # Run optimization
                    best_params, opt_result, callback = run_hyperparameter_optimization(
                        X_opt_train, 
                        y_opt_train,
                        X_opt_val,
                        y_opt_val,
                        model_type,
                        use_gpu=use_gpu,
                        n_calls=n_calls,
                        cv_folds=cv_folds,
                        progress_bar=progress_bar
                    )
                    
                    # Display optimization results
                    st.subheader("Hyperparameter Optimization Results")
                    
                    # Calculate and display the correct AUC value
                    best_auc = 1.0 - opt_result.fun if opt_result.fun <= 1.0 else 0.0
                    st.write(f"Best AUC: {best_auc:.4f}")
                    
                    # Format parameters for better readability
                    formatted_params = {}
                    for key, value in best_params.items():
                        # Convert numeric parameters to appropriate format
                        if isinstance(value, (int, float)):
                            if isinstance(value, int) or value.is_integer():
                                formatted_params[key] = int(value)
                            else:
                                formatted_params[key] = round(value, 4)
                        else:
                            formatted_params[key] = value
                    
                    # Display parameters in a more structured way
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Model Parameters")
                        model_params = ["n_estimators", "max_depth", "learning_rate", "subsample", 
                                      "colsample_bytree", "gamma", "min_child_weight", "num_leaves", 
                                      "l2_leaf_reg"]
                        for param in model_params:
                            if param in formatted_params:
                                st.write(f"• **{param}:** {formatted_params[param]}")
                    
                    with col2:
                        st.subheader("System Parameters")
                        sys_params = ["tree_method", "device", "eval_metric", "random_state", 
                                    "n_jobs", "verbosity"]
                        for param in sys_params:
                            if param in formatted_params:
                                st.write(f"• **{param}:** {formatted_params[param]}")
                    
                    # Show raw parameters in JSON format (expandable)
                    with st.expander("View all parameters as JSON"):
                        st.json(formatted_params)
                    
                    # Plot optimization progress
                    from .model_training import plot_optimization_progress
                    progress_fig = plot_optimization_progress(callback.iterations)
                    if progress_fig:
                        st.plotly_chart(progress_fig, use_container_width=True)
                        
                    # Ask if user wants to train the model with optimized parameters
                    if st.button("Train Model with Optimized Parameters"):
                        train_and_display_results(best_params, progress_bar)
                
                except Exception as e:
                    st.error(f"Error during hyperparameter optimization: {str(e)}")
                    logger.error(f"Hyperparameter optimization failed: {str(e)}", exc_info=True)

    with col2:
        # Sidebar information
        st.subheader("Usage Instructions")
        st.write("""
        1. Select model type and parameters in the sidebar
        2. Choose cross-validation settings
        3. Click 'Train Model' to start training
        4. View model performance metrics and visualizations
        5. The model will be saved automatically
        """)
        
        # Show model types
        st.subheader("Available Model Types")
        for key, name in MODEL_TYPES.items():
            st.write(f"• **{name}** (`{key}`)")
        
        # Show feature importance if available
        if "model_results" in st.session_state:
            model_results = st.session_state.model_results
            st.subheader("Current Model")
            st.write(f"Type: {model_results.get('model_type', 'Unknown')}")
            st.write(f"Training/Loading Date: {model_results.get('training_date', model_results.get('loading_date', 'Unknown'))}")
            
            # Show metrics if available
            if "metrics" in model_results:
                metrics = model_results["metrics"]
                st.write("Performance:")
                st.write(f"• AUC: {metrics.get('auc', 'N/A'):.4f}")
                st.write(f"• Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")


# Constants for model types
MODEL_TYPES = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost"
} 