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
from datetime import datetime
import os
import json
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost

from sklearn.metrics import roc_auc_score

from credit_risk.utils.streamlit_utils import (
    create_download_link,
    save_step_data,
    load_step_data,
    get_available_step_data,
    cleanup_corrupted_files,
    load_most_recent_feature_selection,
    DATA_DIR,
    prepare_dataframe_for_streamlit,
    get_available_data_files
)
from credit_risk.web.ui_components import (
    section_header,
    display_metrics_dashboard,
    model_parameter_section,
    plot_roc_curve,
    plot_confusion_matrix,
    educational_tip
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
from .model_training.data_utils import clean_dataset_for_optimization, check_feature_quality

# Import data profiler for automatic checkpointing
from credit_risk.utils.data_profiling import DataProfiler

logger = logging.getLogger(__name__)

# Initialize the data profiler as a session state object (persistent across reruns)
def get_data_profiler():
    """Initialize or retrieve the DataProfiler from session state."""
    if 'data_profiler' not in st.session_state:
        # Create a checkpoint directory within the data directory
        checkpoint_dir = os.path.join(os.getcwd(), 'data', 'profiling_checkpoints')
        st.session_state.data_profiler = DataProfiler(checkpoint_dir=checkpoint_dir)
    return st.session_state.data_profiler

def render_model_training_tab(features_df=None):
    """
    Render the model training tab.
    
    Parameters
    ----------
    features_df : pd.DataFrame, optional
        Features DataFrame. If None, will try to load from session state.
    """
    st.header("Model Training")
    
    # Add a "Maintenance" expander with cleanup option
    with st.expander("Data Maintenance", expanded=False):
        st.write("Use this section to manage saved model and feature data.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Cleanup Corrupted Model Files", key="model_training_cleanup"):
                checked, removed = cleanup_corrupted_files("model_training")
                if removed > 0:
                    st.success(f"✅ Cleaned up {removed} corrupted model files out of {checked} checked.")
                else:
                    st.info(f"✓ No corrupted model files found. Checked {checked} files.")
        
        with col2:
            if st.button("Cleanup Feature Selection Files", key="feature_selection_cleanup_mt"):
                checked, removed = cleanup_corrupted_files("feature_selection")
                if removed > 0:
                    st.success(f"✅ Cleaned up {removed} corrupted feature selection files out of {checked} checked.")
                else:
                    st.info(f"✓ No corrupted feature selection files found. Checked {checked} files.")
                    
        with col3:
            if st.button("Cleanup All Files", key="all_cleanup"):
                checked, removed = cleanup_corrupted_files()
                if removed > 0:
                    st.success(f"✅ Cleaned up {removed} corrupted files out of {checked} checked.")
                else:
                    st.info(f"✓ No corrupted files found. Checked {checked} files.")
    
    # Create model training workflow tabs
    model_workflow_tabs = st.tabs([
        "Data Selection", 
        "Model Configuration", 
        "Training/Optimization",
        "Load Previous Models"
    ])
    
    # Initialize model results in session state if not already present
    if "model_results" not in st.session_state:
        st.session_state["model_results"] = None
    
    # Data Selection tab
    with model_workflow_tabs[0]:
        st.subheader("Select Training Data")
        
        # Data source options
        data_options = []
        data_source = "none"
        
        # Check for feature selection results
        if "feature_selection_results" in st.session_state:
            data_options.append("feature_selection")
            data_source = "feature_selection"
        
        # Check for feature engineering results
        if "feature_engineering_results" in st.session_state or features_df is not None:
            data_options.append("feature_engineering")
            if data_source == "none":
                data_source = "feature_engineering"
        
        # Select data source
        if data_options:
            data_source = st.radio(
                "Select data source for model training:",
                options=data_options,
                index=data_options.index(data_source) if data_source in data_options else 0,
                help="Choose the data source to use for model training."
            )
            
            # Display data source info
            if data_source == "feature_selection":
                fs_results = st.session_state.get("feature_selection_results")
                if fs_results is None:
                    st.warning("No feature selection results available. Please run feature selection first.")
                else:
                    # Get the list of selected features
                    selected_features = fs_results.get("selected_features", [])
                    
                    # Use X_train and X_test with only the selected features
                    if "X_train" in fs_results:
                        X_train = fs_results["X_train"][selected_features].copy() if selected_features else fs_results["X_train"].copy()
                        st.info(f"Using {len(selected_features)} selected features from feature selection")
                    else:
                        X_train = fs_results.get("X_train")
                    
                    if "X_test" in fs_results:
                        X_test = fs_results["X_test"][selected_features].copy() if selected_features and selected_features[0] in fs_results["X_test"].columns else fs_results["X_test"].copy()
                    else:
                        X_test = fs_results.get("X_test")
                    
                    y_train = fs_results.get("y_train")
                    y_test = fs_results.get("y_test")
            
            elif data_source == "feature_engineering":
                if "feature_engineering_results" in st.session_state:
                    fe_results = st.session_state["feature_engineering_results"]
                    features_shape = fe_results.get("dataset_shape", (0, 0))
                    st.info(f"Using feature engineering results with {features_shape[1]} features")
                elif features_df is not None:
                    st.info(f"Using provided feature dataset with {features_df.shape[1]} features")
                else:
                    st.warning("No feature engineering results found")
            
            # Save data source choice in session state
            st.session_state["model_data_source"] = data_source
            
            # Button to confirm data selection
            if st.button("Confirm Data Selection", use_container_width=True):
                st.success(f"Data source confirmed: {data_source}")
                st.rerun()
        else:
            st.error("No valid data sources found. Please run feature engineering or feature selection first.")
    
    # Model Configuration tab
    with model_workflow_tabs[1]:
        st.subheader("Configure Model Parameters")
        
        if "model_data_source" not in st.session_state:
            st.warning("Please select a data source in the Data Selection tab first.")
        else:
            # Create tabs for different model types
            model_tabs = st.tabs(["LightGBM", "XGBoost", "CatBoost"])
            
            # Collect model parameters
            model_params = {}
            
            # LightGBM tab
            with model_tabs[0]:
                st.markdown("### LightGBM Parameters")
                
                educational_tip(
                    "LightGBM is often the best choice for credit risk models due to its "
                    "efficiency with large datasets and ability to handle categorical features naturally."
                )
                
                # Get LightGBM parameters using the reusable component
                model_params["lightgbm"] = model_parameter_section("lightgbm")
                model_params["lightgbm"]["model_type"] = "lightgbm"
            
            # XGBoost tab
            with model_tabs[1]:
                st.markdown("### XGBoost Parameters")
                
                educational_tip(
                    "XGBoost often provides excellent performance but may require more "
                    "feature preprocessing than LightGBM."
                )
                
                # Get XGBoost parameters
                model_params["xgboost"] = model_parameter_section("xgboost")
                model_params["xgboost"]["model_type"] = "xgboost"
            
            # CatBoost tab
            with model_tabs[2]:
                st.markdown("### CatBoost Parameters")
                
                educational_tip(
                    "CatBoost handles categorical variables automatically and is "
                    "less prone to overfitting, but may be slower to train."
                )
                
                # Get CatBoost parameters
                model_params["catboost"] = model_parameter_section("catboost")
                model_params["catboost"]["model_type"] = "catboost"
            
            # Training settings
            st.subheader("General Training Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.number_input(
                    "Test Size",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Proportion of data to use for testing",
                    key="test_size_input"
                )
                
                random_seed = st.number_input(
                    "Random Seed",
                    min_value=1,
                    max_value=9999,
                    value=42,
                    help="Random seed for reproducibility",
                    key="random_seed_input"
                )
            
            with col2:
                use_gpu = st.checkbox(
                    "Use GPU",
                    value=st.session_state.get("gpu_available", False),
                    disabled=not st.session_state.get("gpu_available", False),
                    help="Use GPU for model training if available",
                    key="use_gpu_checkbox"
                )
                
                # Description for saving
                model_description = st.text_input(
                    "Model Description",
                    value="Credit risk model",
                    help="Description to identify this model run later",
                    key="model_description_input"
                )
            
            # Save model params to session state
            st.session_state["model_params"] = {
                "model_params": model_params,
                "training_params": {
                    "test_size": test_size,
                    "random_seed": random_seed,
                    "use_gpu": use_gpu,
                    "description": model_description
                }
            }
            
            # Button to confirm configuration
            if st.button("Confirm Configuration", use_container_width=True):
                st.success("Model configuration saved.")
                st.rerun()
    
    # Training/Optimization tab
    with model_workflow_tabs[2]:
        st.subheader("Train and Optimize Models")
        
        if "model_params" not in st.session_state or "model_data_source" not in st.session_state:
            st.warning("Please configure model parameters in the previous tabs first.")
        else:
            # Create tabs for Classic Training vs Hyperparameter Optimization
            train_tabs = st.tabs(["Classic Training", "Hyperparameter Optimization"])
            
            # Get data based on selected source
            data_source = st.session_state["model_data_source"]
            X_train, X_test, y_train, y_test = None, None, None, None
            
            if data_source == "feature_selection":
                fs_results = st.session_state.get("feature_selection_results")
                if fs_results is None:
                    st.warning("No feature selection results available. Please run feature selection first.")
                else:
                    # Get the list of selected features
                    selected_features = fs_results.get("selected_features", [])
                    
                    # Use X_train and X_test with only the selected features
                    if "X_train" in fs_results:
                        X_train = fs_results["X_train"][selected_features].copy() if selected_features else fs_results["X_train"].copy()
                        st.info(f"Using {len(selected_features)} selected features from feature selection")
                    else:
                        X_train = fs_results.get("X_train")
                    
                    if "X_test" in fs_results:
                        X_test = fs_results["X_test"][selected_features].copy() if selected_features and selected_features[0] in fs_results["X_test"].columns else fs_results["X_test"].copy()
                    else:
                        X_test = fs_results.get("X_test")
                    
                    y_train = fs_results.get("y_train")
                    y_test = fs_results.get("y_test")
            
            elif data_source == "feature_engineering":
                if "feature_engineering_results" in st.session_state:
                    features_df = st.session_state["feature_engineering_results"].get("features_df")
                
                if features_df is not None and "TARGET" in features_df.columns:
                    # Split features and target
                    y = features_df["TARGET"]
                    X = features_df.drop(columns=["TARGET"])
                    
                    # Train test split
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=st.session_state["model_params"]["training_params"]["test_size"],
                        random_state=st.session_state["model_params"]["training_params"]["random_seed"],
                        stratify=y
                    )
            
            # Check if we have valid data
            if X_train is None or y_train is None:
                st.error("Could not prepare data for training. Please check the selected data source.")
            else:
                # Automatically clean NaN values in target variables
                if y_train.isna().any():
                    nan_count = y_train.isna().sum()
                    logger.warning(f"Found {nan_count} NaN values in training target variable. Automatically removing them.")
                    
                    # Get indices of non-NaN values
                    valid_indices = y_train[~y_train.isna()].index
                    
                    # Filter X_train, y_train to keep only rows with valid target values
                    X_train = X_train.loc[valid_indices]
                    y_train = y_train.loc[valid_indices]
                    
                    st.warning(f"✓ Automatically removed {nan_count} rows with NaN values in target. Using {len(y_train)} valid rows for training.")
                
                # Also clean test data if needed
                if y_test is not None and y_test.isna().any():
                    test_nan_count = y_test.isna().sum()
                    logger.warning(f"Found {test_nan_count} NaN values in test target variable. Automatically removing them.")
                    
                    # Get indices of non-NaN values
                    test_valid_indices = y_test[~y_test.isna()].index
                    
                    # Filter X_test, y_test
                    if X_test is not None:
                        X_test = X_test.loc[test_valid_indices]
                    y_test = y_test.loc[test_valid_indices]
                    
                    st.warning(f"✓ Automatically removed {test_nan_count} rows with NaN values in test target. Using {len(y_test)} valid rows for testing.")
                
                # Display dataset info
                st.write(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
                
                # Check if X_test is None before trying to access its shape
                if X_test is not None:
                    st.write(f"Testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
                else:
                    st.warning("No test data available. Consider using a different data source or checking your data.")
                
                # Classic Training tab
                with train_tabs[0]:
                    st.subheader("Train Classic Models")
                    
                    # Select model type
                    model_type = st.selectbox(
                        "Select model type",
                        options=["lightgbm", "xgboost", "catboost"],
                        index=0,
                        help="Choose the model type to train",
                        key="classic_model_type"
                    )
                    
                    # Get parameters for selected model
                    params = st.session_state["model_params"]["model_params"][model_type]
                    training_params = st.session_state["model_params"]["training_params"]
                    
                    # Display selected parameters
                    with st.expander("Model Parameters", expanded=False):
                        # Create a DataFrame from the parameters
                        param_df = pd.DataFrame([
                            {"Parameter": k, "Value": v} 
                            for k, v in params.items()
                        ])
                        # Prepare dataframe for display to avoid Arrow serialization issues
                        param_df = prepare_dataframe_for_streamlit(param_df)
                        st.table(param_df)
                    
                    # Train model button
                    if st.button("Train Model", use_container_width=True, key="train_model_button"):
                        with st.spinner(f"Training {model_type} model..."):
                            # Record start time
                            start_time = time.time()
                            
                            try:
                                # Train and evaluate model
                                model, metrics, cv_results, figures = train_and_evaluate_model(
                                    X_train, y_train, X_test, y_test, 
                                    model_type=model_type,
                                    model_params=params,
                                    cv_folds=5,
                                    random_state=training_params["random_seed"]
                                )
                                
                                # Record end time and duration
                                end_time = time.time()
                                duration = end_time - start_time
                                
                                # Create model results dictionary
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                model_results = {
                                    "model": model,
                                    "model_type": model_type,
                                    "params": params,
                                    "training_params": training_params,
                                    "data_source": data_source,
                                    "metrics": metrics,
                                    "cv_results": cv_results, 
                                    "figures": figures,
                                    "timestamp": timestamp,
                                    "description": training_params["description"],
                                    "duration": duration,
                                    "optimization_type": "classic",
                                    # Store feature information
                                    "feature_names": X_train.columns.tolist(),
                                    "n_features": X_train.shape[1]
                                }
                                
                                # Save to session state
                                st.session_state["model_results"] = model_results
                                
                                # Save model to file
                                save_step_data("model_training", model_results, training_params["description"])
                                
                                st.success(f"Model trained successfully in {duration:.1f} seconds! AUC: {metrics.get('auc', 0):.4f}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error training model: {str(e)}")
                                logger.error(f"Error training model: {str(e)}", exc_info=True)
                
                # Hyperparameter Optimization tab
                with train_tabs[1]:
                    st.subheader("Hyperparameter Optimization")
                    
                    # Select model type for optimization
                    opt_model_type = st.selectbox(
                        "Select model type",
                        options=["lightgbm", "xgboost", "catboost"],
                        index=0,
                        help="Choose the model type to optimize",
                        key="opt_model_type"
                    )
                    
                    # Optimization settings
                    opt_col1, opt_col2 = st.columns(2)
                    
                    with opt_col1:
                        n_trials = st.slider(
                            "Number of Trials",
                            min_value=10,
                            max_value=100,
                            value=20,
                            step=5,
                            help="Number of different hyperparameter combinations to try",
                            key="opt_n_trials"
                        )
                        
                        timeout = st.number_input(
                            "Timeout (seconds)",
                            min_value=60,
                            max_value=3600,
                            value=600,
                            step=60,
                            help="Maximum time for optimization in seconds",
                            key="opt_timeout"
                        )
                    
                    with opt_col2:
                        optimization_metric = st.selectbox(
                            "Optimization Metric",
                            options=["auc", "accuracy", "f1"],
                            index=0,
                            help="Metric to optimize for",
                            key="opt_metric"
                        )
                        
                        opt_description = st.text_input(
                            "Optimization Description",
                            value=f"Hyperparameter optimization for {opt_model_type}",
                            help="Description to identify this optimization run",
                            key="opt_description"
                        )
                    
                    # Run optimization button
                    if st.button("Run Hyperparameter Optimization", use_container_width=True, key="run_opt_button"):
                        with st.spinner(f"Optimizing {opt_model_type} hyperparameters..."):
                            # Record start time
                            start_time = time.time()
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Progress callback
                            def update_progress(trial_num, total_trials, best_value):
                                progress = min(1.0, trial_num / total_trials)
                                progress_bar.progress(progress)
                                status_text.text(f"Trial {trial_num}/{total_trials} - Best {optimization_metric}: {best_value:.4f}")
                            
                            try:
                                # Check data quality before optimization
                                quality_report = check_feature_quality(X_train)
                                if len(quality_report["issues"]) > 0:
                                    logger.warning(f"Data quality issues detected: {quality_report['issues']}")
                                    st.warning("Data quality issues detected. Cleaning data before optimization...")
                                    
                                    # Clean the data automatically
                                    X_train_clean, y_train_clean = clean_dataset_for_optimization(X_train, y_train)
                                    
                                    # Check if test data is available
                                    if X_test is not None and y_test is not None:
                                        X_test_clean, y_test_clean = clean_dataset_for_optimization(X_test, y_test)
                                    else:
                                        # Use training data for both if test data not available
                                        logger.warning("Test data not available. Using training data for evaluation.")
                                        X_test_clean, y_test_clean = X_train_clean.copy(), y_train_clean.copy()
                                        st.warning("No test data available. Using training data for evaluation (this may lead to overly optimistic results).")
                                    
                                    # Show summary of cleaning
                                    st.info(f"Cleaned data: Filled {X_train.isna().sum().sum()} missing values and handled infinity values")
                                else:
                                    # No issues found, use original data
                                    X_train_clean, y_train_clean = X_train, y_train
                                    
                                    # Check if test data is available
                                    if X_test is not None and y_test is not None:
                                        X_test_clean, y_test_clean = X_test, y_test
                                    else:
                                        # Use training data for both if test data not available
                                        logger.warning("Test data not available. Using training data for evaluation.")
                                        X_test_clean, y_test_clean = X_train_clean.copy(), y_train_clean.copy()
                                        st.warning("No test data available. Using training data for evaluation (this may lead to overly optimistic results).")
                                    
                                    st.info("Data quality check passed. No cleaning needed.")
                                
                                # Run hyperparameter optimization with clean data
                                best_params, best_model, callback = run_hyperparameter_optimization(
                                    X_train_clean, y_train_clean, X_test_clean, y_test_clean,
                                    model_type=opt_model_type,
                                    use_gpu=training_params["use_gpu"],
                                    n_calls=n_trials,
                                    cv_folds=3,
                                    random_state=training_params.get("random_seed", 42),
                                    progress_callback=update_progress
                                )
                                
                                # Get optimization history from callback
                                optimization_history = {
                                    "scores": callback.scores,
                                    "times": callback.timestamps,
                                    "params": callback.params
                                }
                                
                                # Evaluate best model
                                try:
                                    y_pred = best_model.predict_proba(X_test_clean)[:, 1]
                                    auc = roc_auc_score(y_test_clean, y_pred)
                                    
                                    st.success(f"Optimization complete! Best AUC: {auc:.4f}")
                                except Exception as e:
                                    logger.error(f"Error evaluating best model: {str(e)}", exc_info=True)
                                    st.error(f"Error evaluating best model: {str(e)}")
                                    auc = 0.0
                                
                                # Record end time and duration
                                end_time = time.time()
                                duration = end_time - start_time
                                
                                # Create model results dictionary
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                model_results = {
                                    "model": best_model,
                                    "model_type": opt_model_type,
                                    "params": best_params,
                                    "training_params": {
                                        **training_params,
                                        "n_calls": n_trials,
                                        "timeout": timeout,
                                        "optimization_metric": optimization_metric
                                    },
                                    "optimization_history": optimization_history,
                                    "data_source": data_source,
                                    "metrics": {"auc": auc},
                                    "timestamp": timestamp,
                                    "description": opt_description,
                                    "duration": duration,
                                    "optimization_type": "hyperopt",
                                    # Store feature information
                                    "feature_names": X_train_clean.columns.tolist(),
                                    "n_features": X_train_clean.shape[1]
                                }
                                
                                # Save to session state
                                st.session_state["model_results"] = model_results
                                
                                # Save model to file
                                save_step_data("model_training", model_results, opt_description)
                                
                                # Add automatic profiling for trained model data
                                try:
                                    profiler = get_data_profiler()
                                    
                                    # Create checkpoint with model predictions
                                    X_test_with_preds = X_test_clean.copy()
                                    X_test_with_preds['predicted_probability'] = y_pred
                                    X_test_with_preds['actual_target'] = y_test_clean.values
                                    
                                    profiler.create_checkpoint(
                                        stage="model_training",
                                        X=X_test_with_preds,
                                        description=f"Model: {opt_model_type}, AUC: {auc:.4f}"
                                    )
                                    st.info("Model training profile created. Compare with previous stages in the Data Profiling tab.")
                                except Exception as e:
                                    logger.error(f"Error creating model training profile: {str(e)}")
                                
                                st.success(f"Optimization completed in {duration:.1f} seconds! Best AUC: {auc:.4f}")
                                st.rerun()
                            except ValueError as e:
                                error_message = str(e)
                                logger.error(f"Error in hyperparameter optimization: {error_message}", exc_info=True)
                                
                                # Handle specific errors with user-friendly messages
                                if "NaN values" in error_message:
                                    st.error(f"Error: {error_message}\n\nPlease check your data for missing values and ensure proper preprocessing.")
                                elif "GPU Tree Learner was not enabled" in error_message:
                                    st.error("GPU acceleration is not available for LightGBM. Please disable GPU in the training parameters or use a different model.")
                                    st.info("To fix this issue: Go to the training parameters in the previous tab and set 'Use GPU' to 'No', or select a different model type.")
                                else:
                                    st.error(f"Error in hyperparameter optimization: {error_message}")
                                
                                # Provide guidance for common errors
                                if "CUDA" in error_message or "GPU" in error_message:
                                    st.info("GPU error detected. Try disabling GPU in the training parameters.")
                                elif "memory" in error_message.lower():
                                    st.info("Memory error detected. Try reducing the dataset size or using simpler model parameters.")
                                elif "timeout" in error_message.lower():
                                    st.info("The operation timed out. Try reducing the number of trials or simplifying your model.")
                            except Exception as e:
                                st.error(f"Error in hyperparameter optimization: {str(e)}")
                                logger.error(f"Error in hyperparameter optimization: {str(e)}", exc_info=True)
    
    # Load Previous Models tab
    with model_workflow_tabs[3]:
        st.subheader("Load Previously Trained Models")
        
        # Check for existing saved models
        available_models = get_available_step_data("model_training")
        
        if available_models:
            st.info(f"Found {len(available_models)} previously trained models")
            
            # Create selectbox for available models
            model_options = []
            for timestamp, description in available_models.items():
                model_options.append(f"{timestamp} - {description}")
            
            selected_model = st.selectbox(
                "Select a model to load",
                options=model_options,
                help="Choose a previously trained model to load"
            )
            
            # Load model button
            if st.button("Load Selected Model", use_container_width=True):
                with st.spinner("Loading model..."):
                    # Extract timestamp from selection
                    timestamp = selected_model.split(" - ")[0]
                    model_results = load_step_data("model_training", timestamp)
                    
                    if model_results and "model" in model_results:
                        # Load into session state
                        st.session_state["model_results"] = model_results
                        
                        # Show success message
                        model_type = model_results.get("model_type", "unknown")
                        auc = model_results.get("metrics", {}).get("auc", 0)
                        st.success(f"Loaded {model_type} model with AUC: {auc:.4f}")
                        st.rerun()
                    else:
                        st.error("Error loading model: Invalid or corrupted model file")
        else:
            st.warning("No saved models found. Train a model first.")
    
    # Display model results if available
    if st.session_state.get("model_results"):
        model_results = st.session_state["model_results"]
        
        st.markdown("---")
        st.header("Model Results")
        
        # Basic info
        model_type = model_results.get("model_type", "Unknown")
        timestamp = model_results.get("timestamp", "Unknown")
        description = model_results.get("description", "Unknown")
        optimization_type = model_results.get("optimization_type", "classic")
        
        # Create metrics for display
        metrics = model_results.get("metrics", {})
        auc = metrics.get("auc", 0)
        
        # Display model information
        st.write(f"### {description}")
        st.write(f"**Model Type**: {model_type.upper()}")
        st.write(f"**Training Type**: {'Hyperparameter Optimization' if optimization_type == 'hyperopt' else 'Classic Training'}")
        st.write(f"**Trained On**: {timestamp.replace('_', ' ')}")
        
        # Display metrics dashboard
        display_metrics_dashboard(metrics)
        
        # Create tabs for different aspects of the model
        result_tabs = st.tabs(["Model Performance", "Parameters", "Feature Importance"])
        
        # Model Performance tab
        with result_tabs[0]:
            st.subheader("Model Performance")
            
            # ROC curve
            if "roc_curve" in metrics:
                fpr = metrics["roc_curve"]["fpr"]
                tpr = metrics["roc_curve"]["tpr"]
                plot_roc_curve(fpr, tpr, auc)
            
            # Confusion matrix
            if "confusion_matrix" in metrics:
                plot_confusion_matrix(metrics["confusion_matrix"])
            
            # Add option to export predictions if available
            if "predictions" in model_results:
                predictions = model_results["predictions"]
                
                # Create download button for predictions
                csv = pd.DataFrame({
                    "true": predictions.get("y_true", []),
                    "pred": predictions.get("y_pred", [])
                }).to_csv(index=False)
                
                st.download_button(
                    "Download Predictions",
                    csv,
                    file_name=f"predictions_{timestamp}.csv",
                    mime="text/csv"
                )
        
        # Parameters tab
        with result_tabs[1]:
            st.subheader("Model Parameters")
            
            # Model parameters
            st.write("#### Model Parameters")
            params_df = pd.DataFrame([
                {"Parameter": k, "Value": v} 
                for k, v in model_results.get("params", {}).items()
            ])
            params_df = prepare_dataframe_for_streamlit(params_df)
            st.table(params_df)
            
            # Training parameters
            st.write("#### Training Parameters")
            training_params_df = pd.DataFrame([
                {"Parameter": k, "Value": v} 
                for k, v in model_results.get("training_params", {}).items()
                if k not in ["description"]
            ])
            training_params_df = prepare_dataframe_for_streamlit(training_params_df)
            st.table(training_params_df)
            
            # Optimization history (if available)
            if "optimization_history" in model_results:
                st.write("#### Optimization History")
                
                # Create DataFrame from optimization history
                history = model_results["optimization_history"]
                if history:
                    history_df = pd.DataFrame(history)
                    history_df = prepare_dataframe_for_streamlit(history_df)
                    
                    # Plot optimization history
                    try:
                        import plotly.express as px
                        
                        # Check what columns we actually have in the dataframe
                        available_columns = history_df.columns.tolist()
                        logger.info(f"Available columns in optimization history: {available_columns}")
                        
                        # If we have 'scores' column, use that
                        if 'scores' in available_columns:
                            fig = px.line(
                                history_df, 
                                x=history_df.index, 
                                y="scores",
                                title="Optimization History",
                                labels={"index": "Trial Number", "scores": "AUC Score"}
                            )
                        # Otherwise try to adapt to available columns
                        elif len(available_columns) > 0:
                            # Use the first column that might be a y value
                            y_column = next((col for col in ['scores', 'value', 'target', 'fun', 'objective_value'] 
                                           if col in available_columns), available_columns[0])
                            
                            fig = px.line(
                                history_df, 
                                x=history_df.index, 
                                y=y_column,
                                title="Optimization History",
                                labels={"index": "Trial Number", y_column: "Score"}
                            )
                        else:
                            st.warning("No data available for optimization history plot")
                            fig = None
                            
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting optimization history: {str(e)}")
                        logger.error(f"Error plotting optimization history: {str(e)}")
                    
                    # Show history table
                    st.dataframe(history_df)
        
        # Feature Importance tab
        with result_tabs[2]:
            st.subheader("Feature Importance")
            
            # Get model and feature names
            model = model_results.get("model")
            feature_names = model_results.get("feature_names", [])
            
            if model is not None and hasattr(model, "feature_importances_") and feature_names:
                # Extract feature importances
                feature_importances = model.feature_importances_
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": feature_importances
                }).sort_values("Importance", ascending=False)
                
                # Plot feature importances
                try:
                    import plotly.express as px
                    
                    # Only show top 20 features
                    top_n = min(20, len(importance_df))
                    top_df = importance_df.head(top_n)
                    
                    fig = px.bar(
                        top_df,
                        y="Feature",
                        x="Importance",
                        orientation="h",
                        title=f"Top {top_n} Feature Importances",
                        labels={"Feature": "", "Importance": "Importance"},
                        color="Importance",
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting feature importances: {str(e)}")
                
                # Show table of all importances
                st.dataframe(importance_df)
            else:
                st.warning("Feature importances not available for this model")
        
        # Option to save model for deployment
        st.markdown("---")
        st.subheader("Export Model")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Save Model for Deployment", use_container_width=True):
                try:
                    import pickle
                    
                    # Create a simplified model dict for deployment
                    deployment_model = {
                        "model": model_results["model"],
                        "model_type": model_results["model_type"],
                        "feature_names": model_results.get("feature_names", []),
                        "timestamp": model_results["timestamp"],
                        "description": model_results["description"],
                        "metrics": model_results.get("metrics", {})
                    }
                    
                    # Save to pickle
                    model_path = f"models/credit_risk_model_{timestamp}.pkl"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    
                    with open(model_path, "wb") as f:
                        pickle.dump(deployment_model, f)
                    
                    st.success(f"Model saved to {model_path}")
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
        
        with export_col2:
            if st.button("Download Model as Pickle", use_container_width=True):
                try:
                    import pickle
                    import io
                    
                    # Pickle the model to a bytes buffer
                    buffer = io.BytesIO()
                    pickle.dump(model_results["model"], buffer)
                    buffer.seek(0)
                    
                    # Create download button
                    st.download_button(
                        "Download Pickle File",
                        buffer,
                        file_name=f"model_{timestamp}.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error creating download: {str(e)}")
    
    # Button to reset model results
    if st.session_state.get("model_results"):
        if st.button("Start New Model", use_container_width=True):
            if "model_results" in st.session_state:
                del st.session_state["model_results"]
            st.rerun()


# Constants for model types
MODEL_TYPES = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost"
} 