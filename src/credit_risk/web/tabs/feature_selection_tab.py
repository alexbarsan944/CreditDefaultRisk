"""
Feature Selection Tab for the Credit Risk Web App.

This module provides functions for the feature selection tab in the main app.
"""
import logging
import time
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.express as px
from typing import Dict, Any, Tuple, List, Optional, Callable
from datetime import datetime
from pathlib import Path

from credit_risk.web.model_utils import (
    perform_feature_selection
)
from credit_risk.utils.streamlit_utils import (
    prepare_dataframe_for_streamlit,
    save_step_data,
    load_step_data,
    get_available_step_data,
    cleanup_corrupted_files
)

# Import data profiler for automatic checkpointing
from credit_risk.utils.data_profiling import DataProfiler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the data profiler as a session state object (persistent across reruns)
def get_data_profiler():
    """Initialize or retrieve the DataProfiler from session state."""
    if 'data_profiler' not in st.session_state:
        # Create a checkpoint directory within the data directory
        checkpoint_dir = Path.cwd() / 'data' / 'profiling_checkpoints'
        st.session_state.data_profiler = DataProfiler(checkpoint_dir=checkpoint_dir)
    return st.session_state.data_profiler

def render_feature_selection_tab(data: pd.DataFrame):
    """
    Render the feature selection tab.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to perform feature selection on
    """
    st.header("Feature Selection")
    
    # Check if we have data to work with
    if data is None:
        st.warning("No feature data available. Please run Feature Engineering first.")
        return
    
    # Educational content in an expander
    with st.expander("About Feature Selection", expanded=False):
        st.markdown(
            """
            ### Why Feature Selection Matters
            
            Feature selection helps improve model performance by:
            - **Reducing Overfitting**: Fewer features can lead to more generalizable models
            - **Improving Performance**: Removing noise can increase prediction accuracy
            - **Faster Training**: Fewer features means faster model training
            - **Better Interpretability**: Models with fewer features are easier to understand
            
            ### Methods Used
            
            This tab implements the **Null Importance Method**:
            
            1. Train a model on the real data and record feature importances
            2. Shuffle the target variable to break its relationship with features
            3. Train models on this "null" data multiple times
            4. Compare real importances vs. null importance distribution
            5. Select features where real importance significantly exceeds null importances
            
            This method is particularly effective for identifying features that have a genuine relationship with the target.
            """
        )
        
        # Simple diagram explaining null importance
        try:
            st.image("https://i.imgur.com/XXvAYKA.png", caption="Null Importance Method", use_container_width=True)
        except:
            st.write("Null importance compares feature importance between real data and randomized target data.")
    
    # Show existing results or run new feature selection
    feature_selection_results = st.session_state.get("feature_selection_results")
    
    if feature_selection_results is None:
        # Create tabs for new feature selection
        feature_tabs = st.tabs(["Basic Settings", "Advanced Settings", "Run Feature Selection"])
        
        # Basic Settings tab
        with feature_tabs[0]:
            st.subheader("Basic Feature Selection Settings")
            
            # Basic parameters with helpful explanations
            base_col1, base_col2 = st.columns(2)
            
            with base_col1:
                selection_method = st.selectbox(
                    "Feature Selection Method",
                    options=["null_importance", "feature_importance"],
                    index=0,
                    help="Null importance is more robust but takes longer to run"
                )
                
                test_size = st.slider(
                    "Test Size",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Percentage of data to use for validation"
                )
            
            with base_col2:
                correlation_threshold = st.slider(
                    "Correlation Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                    help="Remove highly correlated features above this threshold"
                )
                
                random_seed = st.number_input(
                    "Random Seed",
                    min_value=1,
                    max_value=9999,
                    value=42,
                    help="Random seed for reproducibility"
                )
        
        # Advanced Settings tab
        with feature_tabs[1]:
            st.subheader("Advanced Feature Selection Settings")
            
            # Only show null importance settings if that method is selected
            if selection_method == "null_importance":
                adv_col1, adv_col2 = st.columns(2)
                
                with adv_col1:
                    n_runs = st.slider(
                        "Number of Null Runs",
                        min_value=5,
                        max_value=100,
                        value=20,
                        step=5,
                        help="More runs give more reliable results but take longer"
                    )
                    
                    threshold_method = st.selectbox(
                        "Threshold Method",
                        options=["percentile", "standard_deviation", "threshold"],
                        index=0,
                        help="Method to determine the importance threshold"
                    )
                
                with adv_col2:
                    if threshold_method == "percentile":
                        threshold_value = st.slider(
                            "Percentile Threshold",
                            min_value=80,
                            max_value=99,
                            value=95,
                            step=1,
                            help="Percentile of null importance distribution to use as threshold"
                        )
                    elif threshold_method == "standard_deviation":
                        threshold_value = st.slider(
                            "Standard Deviation Multiplier",
                            min_value=1.0,
                            max_value=5.0,
                            value=2.0,
                            step=0.1,
                            help="Number of standard deviations above mean null importance"
                        )
                    else:
                        # Direct threshold setting
                        importance_threshold = st.slider(
                            "Importance Threshold",
                            min_value=0.0,
                            max_value=0.1,
                            value=0.005,
                            step=0.001,
                            help="Minimum importance score to keep a feature"
                        )
                        n_runs = 1
                        threshold_method = "threshold"
                        threshold_value = importance_threshold
            else:
                # Feature importance method settings
                importance_threshold = st.slider(
                    "Importance Threshold",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.005,
                    step=0.001,
                    help="Minimum importance score to keep a feature"
                )
                n_runs = 1
                threshold_method = "threshold"
                threshold_value = importance_threshold
            
            # Feature prefiltering
            st.subheader("Feature Prefiltering")
            prefilter_col1, prefilter_col2 = st.columns(2)
            
            with prefilter_col1:
                exclude_features = st.text_area(
                    "Exclude Features (one per line)",
                    value="",
                    height=100,
                    help="Features to exclude from selection (e.g., ID columns)"
                )
            
            with prefilter_col2:
                include_features = st.text_area(
                    "Include Features (one per line)",
                    value="",
                    height=100,
                    help="Features to always include regardless of importance"
                )
        
        # Run Feature Selection tab
        with feature_tabs[2]:
            st.subheader("Run Feature Selection")
            
            # Summary of settings
            st.write("### Summary of Feature Selection Settings")
            
            # Create a summary DataFrame
            settings_summary = pd.DataFrame([
                {"Setting": "Selection Method", "Value": selection_method},
                {"Setting": "Test Size", "Value": test_size},
                {"Setting": "Correlation Threshold", "Value": correlation_threshold}
            ])
            
            if selection_method == "null_importance":
                settings_summary = pd.concat([
                    settings_summary,
                    pd.DataFrame([
                        {"Setting": "Number of Null Runs", "Value": n_runs},
                        {"Setting": "Threshold Method", "Value": threshold_method},
                        {"Setting": "Threshold Value", "Value": threshold_value}
                    ])
                ])
            else:
                settings_summary = pd.concat([
                    settings_summary,
                    pd.DataFrame([
                        {"Setting": "Importance Threshold", "Value": importance_threshold}
                    ])
                ])
            
            st.table(settings_summary)
            
            # Process exclude and include features
            exclude_list = [f.strip() for f in exclude_features.strip().split("\n") if f.strip()]
            include_list = [f.strip() for f in include_features.strip().split("\n") if f.strip()]
            
            st.write(f"Excluding {len(exclude_list)} features and always including {len(include_list)} features")
            
            # Button to run feature selection
            description = st.text_input(
                "Description",
                value=f"Feature selection with {selection_method} method",
                help="A brief description for your reference"
            )
            
            start_button = st.button(
                "Start Feature Selection",
                help="This may take some time depending on the settings",
                use_container_width=True
            )
            
            if start_button:
                with st.spinner("Running feature selection..."):
                    # Prepare parameters for feature selection
                    params = {
                        "selection_method": selection_method,
                        "test_size": test_size,
                        "correlation_threshold": correlation_threshold,
                        "random_state": random_seed,
                        "n_runs": n_runs,
                        "threshold_method": threshold_method,
                        "threshold_value": threshold_value,
                        "exclude_features": exclude_list,
                        "include_features": include_list
                    }
                    
                    # Add a progress bar for long-running operations
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress, status):
                        progress_bar.progress(progress)
                        status_text.text(status)
                    
                    # Create a progress bar adapter class that has the expected interface
                    class ProgressBarAdapter:
                        def __init__(self, update_fn):
                            self.update_fn = update_fn
                        
                        def progress(self, value, text=None):
                            self.update_fn(value, text if text else "")
                    
                    # Create an adapter instance
                    progress_adapter = ProgressBarAdapter(update_progress)
                    
                    # Set callback function in session state
                    st.session_state["update_progress"] = update_progress
                    
                    # Record start time
                    start_time = time.time()
                    
                    # Run feature selection with update_progress callback
                    try:
                        # Extract target and features
                        if "TARGET" in data.columns:
                            y = data["TARGET"]
                            X = data.drop(columns=["TARGET"])
                        else:
                            st.error("No TARGET column found in the data")
                            return
                            
                        # Call the perform_feature_selection function with individual parameters
                        feature_selector = perform_feature_selection(
                            X=X, 
                            y=y,
                            n_runs=params["n_runs"],
                            correlation_threshold=params["correlation_threshold"],
                            random_state=params["random_state"],
                            progress_bar=progress_adapter,
                            use_gpu=True
                        )
                        
                        # Process the results
                        selected_features = feature_selector[0].columns.tolist()
                        
                        # Create X_train and X_test with proper types
                        X_train = feature_selector[0].copy().reset_index(drop=True)
                        X_test = pd.DataFrame(X[feature_selector[0].columns], index=X.index).copy().reset_index(drop=True)
                        
                        # Convert COUNT columns to numeric to avoid issues with LightGBM
                        count_columns = [col for col in selected_features if col.startswith('COUNT(')]
                        if count_columns:
                            logger.info(f"Converting {len(count_columns)} COUNT columns to numeric types")
                            for col in count_columns:
                                try:
                                    # Convert to numeric, coerce errors to NaN
                                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                                    X_train[col] = X_train[col].fillna(0)
                                    X_train[col] = X_train[col].astype('int64')
                                    
                                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                                    X_test[col] = X_test[col].fillna(0)
                                    X_test[col] = X_test[col].astype('int64')
                                except Exception as e:
                                    logger.warning(f"Could not convert {col} to numeric: {str(e)}")
                        
                        # Create results dictionary
                        feature_selection_results = {
                            "selected_features": selected_features,
                            "feature_scores": feature_selector[2].get("feature_scores"),
                            "params": params,
                            # Add the split data to feature_selection_results for model_training_tab.py
                            "X_train": X_train,  # Selected features DataFrame
                            "y_train": y.copy().reset_index(drop=True),
                            "X_test": X_test,
                            "y_test": y.copy().reset_index(drop=True)
                        }
                        
                        # Record end time and calculate duration
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Add timestamp and description to results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        feature_selection_results["timestamp"] = timestamp
                        feature_selection_results["description"] = description
                        feature_selection_results["duration"] = duration
                        
                        # Prepare results dictionary
                        fs_results = {
                            "selected_features": selected_features,
                            "feature_scores": feature_selector[2].get("feature_scores"),
                            "params": params,
                            "X_train": X_train,
                            "y_train": y.copy().reset_index(drop=True),
                            "X_test": X_test,  # Include X_test
                            "y_test": y.copy().reset_index(drop=True),  # Include y_test
                            "X_selected": X_train[selected_features],
                            "timestamp": timestamp,
                            "duration": duration,
                            "description": description
                        }
                        
                        # Save to session state and file
                        st.session_state["feature_selection_results"] = fs_results
                        save_step_data("feature_selection", fs_results, description)
                        
                        # Profile the selected features dataset
                        try:
                            profiler = get_data_profiler()
                            profiler.create_checkpoint(
                                stage="feature_selection",
                                X=X_train[selected_features],
                                y=y.copy().reset_index(drop=True),
                                description=f"Selected features: {len(selected_features)} features"
                            )
                            st.info("Feature selection profile created. Compare with previous stages in the Data Profiling tab.")
                        except Exception as e:
                            logger.error(f"Error creating feature selection profile: {str(e)}")
                        
                        # Show success message
                        st.success(f"Feature selection completed in {duration:.1f} seconds!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error running feature selection: {str(e)}")
    
    else:
        # Display feature selection results
        display_feature_selection_results(feature_selection_results, data)
        
        # Add single button to clear results and run new selection
        if st.button("Clear Results and Run New Selection", key="clear_results_button", use_container_width=True):
            # Completely remove feature selection results from session state
            if "feature_selection_results" in st.session_state:
                del st.session_state["feature_selection_results"]
            st.rerun()

def display_feature_selection_results(results: Dict[str, Any], original_data: pd.DataFrame):
    """
    Display the feature selection results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Feature selection results
    original_data : pd.DataFrame
        Original data used for feature selection
    """
    # Extract key information from results
    selected_features = results.get("selected_features", [])
    feature_scores = results.get("feature_scores")
    method = results.get("params", {}).get("selection_method", "unknown")
    description = results.get("description", "Feature selection results")
    timestamp = results.get("timestamp", "unknown")
    duration = results.get("duration", 0)
    
    # Set up header with key metrics
    st.write("## Feature Selection Results")
    
    # Summary metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Selected Features", len(selected_features))
    
    with col2:
        # Calculate feature reduction percentage
        if original_data is not None and len(original_data.columns) > 0:
            total_features = len(original_data.columns)
            reduction = 100 * (1 - len(selected_features) / total_features)
            st.metric("Feature Reduction", f"{reduction:.1f}%")
        else:
            st.metric("Feature Reduction", "N/A")
    
    with col3:
        st.metric("Selection Method", method.replace("_", " ").title())
    
    # Additional information
    st.info(f"**📊 Details:** {description}\n\n**🕒 Generated:** {timestamp.replace('_', ' ')} (took {duration:.1f} seconds)")
    
    # Create tabs for different views of the results
    results_tabs = st.tabs(["Selected Features", "Feature Importance", "Data Preview"])
    
    with results_tabs[0]:
        st.subheader("Selected Features List")
        
        # Create a dataframe with more information about selected features
        if selected_features and feature_scores is not None:
            # Create a dataframe with selected features and their scores
            feature_df = pd.DataFrame({
                'Feature': selected_features
            })
            
            # Try to add importance scores if available
            if isinstance(feature_scores, dict):
                # Try different possible keys for the scores
                for key in ['importance_gain', 'importance_split', 'final_score', 'score']:
                    if key in feature_scores and len(feature_scores[key]) > 0:
                        score_dict = {feat: score for feat, score in zip(feature_scores.get('feature', selected_features), feature_scores[key])}
                        feature_df['Score'] = feature_df['Feature'].map(score_dict).fillna(0)
                        break
                
                # Sort by score if available
                if 'Score' in feature_df.columns:
                    feature_df = feature_df.sort_values('Score', ascending=False)
            
            # Display the dataframe
            st.dataframe(feature_df, use_container_width=True)
        else:
            st.warning("No selected features data available. Results may be in an unexpected format.")
    
    with results_tabs[1]:
        st.subheader("Feature Importance")
        
        # Check if we have the necessary data for plots
        if feature_scores is not None:
            # Try to create feature importance plots
            try:
                # Display importance plot
                if isinstance(feature_scores, dict) and 'final_score' in feature_scores:
                    # Prepare data for Plotly
                    importance_df = pd.DataFrame({
                        'Feature': feature_scores.get('feature', []),
                        'Importance': feature_scores.get('final_score', [])
                    }).sort_values('Importance', ascending=False).head(30)
                    
                    if len(importance_df) > 0:
                        # Create bar chart
                        import plotly.express as px
                        fig = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 30 Features by Importance',
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance plot could not be created with the available data format.")
            except Exception as e:
                st.error(f"Error creating feature importance plot: {str(e)}")
        else:
            st.warning("No feature importance data available.")
    
    with results_tabs[2]:
        st.subheader("Data with Selected Features")
        
        # Show a preview of the data with only selected features
        if original_data is not None and len(selected_features) > 0:
            # Filter columns to only show selected features (plus TARGET if it exists)
            available_columns = set(original_data.columns)
            columns_to_show = [col for col in selected_features if col in available_columns]
            
            if 'TARGET' in available_columns and 'TARGET' not in columns_to_show:
                columns_to_show.append('TARGET')
            
            # Create filtered dataframe
            if columns_to_show:
                filtered_df = original_data[columns_to_show].copy()
                st.dataframe(prepare_dataframe_for_streamlit(filtered_df.head(10)), use_container_width=True)
                st.caption(f"Showing first 10 rows with {len(columns_to_show)} selected features")
            else:
                st.warning("None of the selected features are available in the current dataset.")
        else:
            st.warning("Data preview is not available.") 