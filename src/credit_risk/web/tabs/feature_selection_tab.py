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
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import os

from credit_risk.web.model_utils import (
    perform_feature_selection,
    create_feature_score_plots
)
from credit_risk.utils.streamlit_utils import (
    prepare_dataframe_for_streamlit,
    save_step_data,
    load_step_data,
    get_available_step_data,
    DATA_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_feature_selection_tab(data: pd.DataFrame):
    """
    Render the feature selection tab.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to perform feature selection on
    """
    st.header("Feature Selection")
    
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
            st.image("https://i.imgur.com/XXvAYKA.png", caption="Null Importance Method", use_column_width=True)
        except:
            st.write("Null importance compares feature importance between real data and randomized target data.")
        
    # Check if we have data to work with
    if data is None:
        st.warning("No feature data available. Please run Feature Engineering first.")
        return
    
    # Show existing results or run new feature selection
    feature_selection_results = st.session_state.get("feature_selection_results")
    
    if feature_selection_results is None:
        # Check for existing saved feature selection results
        available_data = get_available_step_data("feature_selection")
        
        if available_data:
            st.info("Found existing feature selection results. You can load them or run a new selection.")
            
            # Create selectbox for available feature selection results
            result_options = []
            for timestamp, description in available_data.items():
                result_options.append(f"{timestamp} - {description}")
            
            selected_result = st.selectbox(
                "Load existing feature selection",
                options=result_options,
                help="Select previously saved feature selection results to load"
            )
            
            if st.button("Load Selected Results"):
                with st.spinner("Loading feature selection results..."):
                    # Extract timestamp from selection
                    timestamp = selected_result.split(" - ")[0]
                    feature_selection_results = load_step_data("feature_selection", timestamp)
                    st.session_state["feature_selection_results"] = feature_selection_results
                    st.success("Feature selection results loaded successfully!")
                    st.rerun()
        
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
                    else:  # threshold
                        threshold_value = st.slider(
                            "Absolute Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.1,
                            step=0.01,
                            help="Absolute importance score threshold"
                        )
            else:
                # Simple feature importance settings
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
                "Description for this feature selection run",
                value=f"Feature selection with {selection_method} method",
                help="A brief description to identify these results later"
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
                        
                        # Create results dictionary
                        feature_selection_results = {
                            "selected_features": selected_features,
                            "feature_scores": feature_selector[2].get("feature_scores"),
                            "params": params,
                            "feature_score_plots": create_feature_score_plots(feature_selector[2].get("feature_scores")),
                            # Add the split data to feature_selection_results for model_training_tab.py
                            "X_train": feature_selector[0],  # Selected features DataFrame
                            "y_train": y,
                            "X_test": pd.DataFrame(X[feature_selector[0].columns], index=X.index),
                            "y_test": y
                        }
                        
                        # Record end time and calculate duration
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Save feature selection results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        feature_selection_results["timestamp"] = timestamp
                        feature_selection_results["description"] = description
                        feature_selection_results["duration"] = duration
                        
                        # Save to session state and to file
                        st.session_state["feature_selection_results"] = feature_selection_results
                        save_step_data("feature_selection", feature_selection_results, description)
                        
                        # Show success message
                        st.success(f"Feature selection completed in {duration:.1f} seconds!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error running feature selection: {str(e)}")
    
    else:
        # Display feature selection results
        display_feature_selection_results(feature_selection_results, data)
        
        # Add button to clear results and run new selection
        if st.button("Clear Results and Run New Selection"):
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
    
    # Create a dashboard layout
    st.write(f"### {description}")
    st.write(f"Run on: {timestamp.replace('_', ' ')} (took {duration:.1f} seconds)")
    
    # Display metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Selected Features", len(selected_features))
    
    with metrics_col2:
        st.metric("Feature Reduction", f"{(1 - len(selected_features) / len(original_data.columns)) * 100:.1f}%")
    
    with metrics_col3:
        if method == "null_importance":
            n_runs = results.get("params", {}).get("n_runs", 0)
            st.metric("Null Importance Runs", n_runs)
    
    # Create tabs for different views of the results
    results_tabs = st.tabs(["Feature Scores", "Selected Features", "Feature Distributions"])
    
    # Feature Scores tab
    with results_tabs[0]:
        st.subheader("Feature Importance Scores")
        
        if method == "null_importance" and "feature_score_plots" in results:
            # Display the feature score plots created during feature selection
            for fig_title, fig in results["feature_score_plots"].items():
                st.write(f"#### {fig_title}")
                try:
                    # Check if this is a Plotly figure
                    if hasattr(fig, 'update_layout'):
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Assume it's a matplotlib figure if not a Plotly figure
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error displaying plot: {str(e)}")
                    st.write("Plot data available but could not be displayed.")
        
        # Show feature scores table
        if feature_scores is not None:
            # Prepare table of feature scores
            if isinstance(feature_scores, pd.DataFrame):
                score_df = feature_scores
            else:
                # Convert to DataFrame if it's a dict or other format
                score_df = pd.DataFrame({
                    "Feature": list(feature_scores.keys()),
                    "Score": list(feature_scores.values())
                }).sort_values("Score", ascending=False)
            
            st.write("#### Feature Scores Table")
            st.dataframe(
                prepare_dataframe_for_streamlit(score_df),
                use_container_width=True
            )
    
    # Selected Features tab
    with results_tabs[1]:
        st.subheader("Selected Features")
        
        # Group features by type or source
        if any("(" in f for f in selected_features):
            # Features likely have source information in them
            feature_types = {}
            for feature in selected_features:
                if "(" in feature:
                    feature_type = feature.split("(")[1].split(")")[0].split(".")[0]
                    if feature_type not in feature_types:
                        feature_types[feature_type] = []
                    feature_types[feature_type].append(feature)
                else:
                    if "Other" not in feature_types:
                        feature_types["Other"] = []
                    feature_types["Other"].append(feature)
            
            # Display features by type
            for feature_type, features in sorted(feature_types.items()):
                with st.expander(f"{feature_type} ({len(features)} features)", expanded=False):
                    st.write(", ".join(sorted(features)))
        else:
            # Just show the list of selected features
            st.write(", ".join(sorted(selected_features)))
        
        # Download button for selected features
        features_csv = "\n".join(selected_features)
        st.download_button(
            "Download Selected Features",
            features_csv,
            file_name=f"selected_features_{timestamp}.csv",
            mime="text/csv"
        )
    
    # Feature Distributions tab
    with results_tabs[2]:
        st.subheader("Feature Distributions")
        
        # Allow user to select features to visualize
        selected_viz_features = st.multiselect(
            "Select features to visualize",
            options=selected_features,
            default=selected_features[:min(5, len(selected_features))],
            help="Select features to view their distributions"
        )
        
        if selected_viz_features:
            # Check if we have target column for stratification
            has_target = "TARGET" in original_data.columns
            
            for feature in selected_viz_features:
                if feature in original_data.columns:
                    st.write(f"#### {feature}")
                    
                    try:
                        # Create a distribution plot
                        if has_target:
                            fig = px.histogram(
                                original_data, 
                                x=feature,
                                color="TARGET",
                                barmode="overlay",
                                opacity=0.7,
                                marginal="box"
                            )
                        else:
                            fig = px.histogram(
                                original_data, 
                                x=feature,
                                opacity=0.7,
                                marginal="box"
                            )
                        
                        # Update layout
                        fig.update_layout(
                            height=400,
                            width=700
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not plot {feature}: {str(e)}")
                else:
                    st.warning(f"Feature {feature} not found in the original data") 