import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
import sys
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import the necessary modules
from credit_risk.web.model_utils import (
    train_and_evaluate_model,
    create_cv_plot,
    create_confusion_matrix_plot,
    save_model_and_selector,
    MODEL_TYPES
)
from credit_risk.web.app import (
    load_data,
    is_gpu_available,
    set_page_config,
    setup_sidebar
)

# Set page config
set_page_config("Model Training")

def display_model_params(model_type):
    """Display and let user configure model parameters."""
    st.subheader(f"{MODEL_TYPES.get(model_type, model_type.upper())} Parameters")
    
    # Get default parameters based on model type
    if model_type == "lightgbm":
        default_params = {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 7,
            "num_leaves": 31,
            "colsample_bytree": 0.8
        }
    elif model_type == "xgboost":
        default_params = {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 6,
            "colsample_bytree": 0.8,
            "subsample": 0.9
        }
    elif model_type == "catboost":
        default_params = {
            "learning_rate": 0.05,
            "iterations": 200,
            "depth": 6,
            "l2_leaf_reg": 3,
            "bootstrap_type": "Bayesian"
        }
    else:
        default_params = {}
    
    # Let user adjust parameters
    adjusted_params = {}
    
    # Common parameters for all models
    col1, col2 = st.columns(2)
    
    if model_type in ["lightgbm", "xgboost"]:
        with col1:
            adjusted_params["learning_rate"] = st.number_input(
                "Learning rate",
                min_value=0.001,
                max_value=0.5,
                value=default_params.get("learning_rate", 0.05),
                step=0.01,
                format="%.3f"
            )
            
            adjusted_params["n_estimators"] = st.number_input(
                "Number of estimators",
                min_value=50,
                max_value=1000,
                value=default_params.get("n_estimators", 200),
                step=50
            )
        
        with col2:
            adjusted_params["max_depth"] = st.number_input(
                "Max depth",
                min_value=3,
                max_value=15,
                value=default_params.get("max_depth", 6),
                step=1
            )
            
            adjusted_params["colsample_bytree"] = st.slider(
                "Column sample by tree",
                min_value=0.5,
                max_value=1.0,
                value=default_params.get("colsample_bytree", 0.8),
                step=0.1
            )
    
    elif model_type == "catboost":
        with col1:
            adjusted_params["learning_rate"] = st.number_input(
                "Learning rate",
                min_value=0.001,
                max_value=0.5,
                value=default_params.get("learning_rate", 0.05),
                step=0.01,
                format="%.3f"
            )
            
            adjusted_params["iterations"] = st.number_input(
                "Iterations",
                min_value=50,
                max_value=1000,
                value=default_params.get("iterations", 200),
                step=50
            )
        
        with col2:
            adjusted_params["depth"] = st.number_input(
                "Depth",
                min_value=3,
                max_value=15,
                value=default_params.get("depth", 6),
                step=1
            )
            
            adjusted_params["l2_leaf_reg"] = st.number_input(
                "L2 leaf regularization",
                min_value=1,
                max_value=10,
                value=default_params.get("l2_leaf_reg", 3),
                step=1
            )
            
            bootstrap_types = ["Bayesian", "Bernoulli", "MVS"]
            adjusted_params["bootstrap_type"] = st.selectbox(
                "Bootstrap type",
                options=bootstrap_types,
                index=bootstrap_types.index(default_params.get("bootstrap_type", "Bayesian"))
            )
    
    # Model-specific additional parameters
    if model_type == "lightgbm":
        adjusted_params["num_leaves"] = st.number_input(
            "Number of leaves",
            min_value=10,
            max_value=255,
            value=default_params.get("num_leaves", 31),
            step=10
        )
    
    elif model_type == "xgboost":
        adjusted_params["subsample"] = st.slider(
            "Subsample",
            min_value=0.5,
            max_value=1.0,
            value=default_params.get("subsample", 0.9),
            step=0.1
        )
    
    return adjusted_params

def main():
    """Main function for the Model Training page."""
    
    # Set up sidebar
    backend, raw_data_path, current_step = setup_sidebar(current_page="Model Training")
    
    # Display page header
    st.title("Credit Default Risk - Model Training")
    st.write(
        """
        This page allows you to train machine learning models to predict credit default risk.
        You can select from different model types and configure their parameters.
        """
    )
    
    # Check if feature selection results exist
    if "feature_selection_results" not in st.session_state:
        st.warning(
            """
            Feature selection results not found in session state. Please run feature selection first
            or use the options below to load data and train a model without feature selection.
            """
        )
        
        # Option to proceed without feature selection
        st.subheader("Proceed without Feature Selection")
        
        use_existing_data = st.checkbox(
            "Use data loaded in previous step",
            value=True,
            help="If unchecked, you'll need to load data again"
        )
        
        if use_existing_data:
            if "data" not in st.session_state:
                try:
                    with st.spinner("Loading data..."):
                        st.session_state.data = load_data(raw_data_path, backend=backend)
                    st.write(f"Data loaded with {st.session_state.data.shape[0]} rows and {st.session_state.data.shape[1]} columns.")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    logger.error(f"Data loading error: {str(e)}", exc_info=True)
                    return
            
            data = st.session_state.data
            
            # Check if TARGET column exists
            if "TARGET" not in data.columns:
                st.error("TARGET column not found in the data. Please make sure your data contains a TARGET column.")
                return
            
            # Data splitting options
            test_size = st.slider(
                "Test set size (%)",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="Percentage of data to use for testing"
            ) / 100
            
            random_state = st.number_input(
                "Random seed",
                min_value=1,
                max_value=100000,
                value=42,
                help="Random seed for reproducibility"
            )
            
            if st.button("Prepare Data", type="primary"):
                # Split features and target
                X = data.drop(columns=["TARGET"])
                y = data["TARGET"]
                
                # Split into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                st.session_state.feature_selection_results = {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "feature_selector": None,
                    "original_features": X.shape[1],
                    "selected_features": X.shape[1]
                }
                
                st.success("Data prepared for model training!")
                st.rerun()
    
    # If feature selection results exist, proceed with model training
    if "feature_selection_results" in st.session_state:
        results = st.session_state.feature_selection_results
        X_train = results["X_train"]
        X_test = results["X_test"]
        y_train = results["y_train"]
        y_test = results["y_test"]
        feature_selector = results.get("feature_selector")
        
        # Display data info
        st.info(
            f"""
            Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features
            Testing data: {X_test.shape[0]} samples, {X_test.shape[1]} features
            """
        )
        
        # Model selection and configuration
        st.header("Model Selection")
        
        model_type = st.selectbox(
            "Select model type",
            options=list(MODEL_TYPES.keys()),
            format_func=lambda x: MODEL_TYPES[x],
            index=1  # Default to LightGBM
        )
        
        # Model parameters
        with st.expander("Model Parameters", expanded=True):
            model_params = display_model_params(model_type)
        
        # Cross-validation settings
        n_folds = st.slider(
            "Number of cross-validation folds",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="More folds give more stable results but take longer"
        )
        
        # Model name for saving
        model_name = st.text_input(
            "Model name",
            value=f"{model_type}_model",
            help="Name used for saving the trained model"
        )
        
        # Train button
        if st.button("Train Model", type="primary"):
            # Create progress bar
            progress_bar = st.progress(0, "Starting model training...")
            
            try:
                start_time = time.time()
                
                # Train and evaluate model
                model, training_results = train_and_evaluate_model(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    model_type=model_type,
                    n_folds=n_folds,
                    model_params=model_params,
                    random_state=42,
                    progress_bar=progress_bar
                )
                
                elapsed_time = time.time() - start_time
                
                st.success(f"Model training completed in {elapsed_time:.2f} seconds!")
                
                # Store model in session state
                st.session_state.model_results = {
                    "model": model,
                    "model_type": model_type,
                    "training_results": training_results,
                    "feature_selector": feature_selector
                }
                
                # Display results
                st.header("Model Evaluation Results")
                
                # Metrics
                test_results = training_results["test_results"]
                cv_results = training_results["cv_results"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test AUC", f"{test_results['auc']:.4f}")
                
                with col2:
                    st.metric("Test Accuracy", f"{test_results['accuracy']:.4f}")
                
                with col3:
                    st.metric("CV AUC (mean)", f"{cv_results['mean_auc']:.4f}")
                
                # Cross-validation plot
                st.subheader("Cross-Validation Results")
                cv_plot = create_cv_plot(cv_results, model_type)
                st.plotly_chart(cv_plot, use_container_width=True)
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm_plot = create_confusion_matrix_plot(model, X_test, y_test)
                st.plotly_chart(cm_plot)
                
                # Feature importance
                st.subheader("Feature Importance")
                feature_imp_fig = model.plot_feature_importance(top_n=20)
                st.pyplot(feature_imp_fig)
                
                # Save model
                st.header("Save Model")
                st.write("Save the trained model and feature selector for future use:")
                
                save_dir = st.text_input(
                    "Save directory",
                    value="models",
                    help="Directory where the model will be saved"
                )
                
                if st.button("Save Model"):
                    try:
                        # Create directory if it doesn't exist
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Save model and feature selector
                        if feature_selector is not None:
                            save_paths = save_model_and_selector(
                                model,
                                feature_selector,
                                model_name,
                                output_dir=save_dir
                            )
                            
                            # Display save paths
                            st.success(
                                f"""
                                Model saved successfully!
                                
                                Model path: {save_paths['model_path']}
                                Feature selector path: {save_paths['selector_path']}
                                Selected features list: {save_paths['features_path']}
                                """
                            )
                        else:
                            # Save just the model if no feature selector
                            model_path = os.path.join(save_dir, f"{model_name}.joblib")
                            model.save(model_path)
                            
                            st.success(
                                f"""
                                Model saved successfully!
                                
                                Model path: {model_path}
                                """
                            )
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
                        logger.error(f"Model saving error: {str(e)}", exc_info=True)
                
                # Next steps guidance
                st.info("🔍 **Next Steps**: Go to the Prediction page to make predictions with your trained model.")
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                logger.error(f"Model training error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 