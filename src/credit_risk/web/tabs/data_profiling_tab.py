"""
Data Profiling tab for visualizing data at different pipeline stages.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from typing import Dict, List, Optional, Union, Tuple
import logging
import json

from credit_risk.utils.data_profiling import DataProfiler
from credit_risk.utils.streamlit_utils import load_step_data, get_available_data_files, prepare_dataframe_for_streamlit

logger = logging.getLogger(__name__)

# Initialize the data profiler as a session state object (persistent across reruns)
def get_data_profiler():
    """Initialize or retrieve the DataProfiler from session state."""
    if 'data_profiler' not in st.session_state:
        # Create a checkpoint directory within the data directory
        checkpoint_dir = os.path.join(os.getcwd(), 'data', 'profiling_checkpoints')
        st.session_state.data_profiler = DataProfiler(checkpoint_dir=checkpoint_dir)
    return st.session_state.data_profiler

def render_data_profiling_tab():
    """Render the data profiling tab in the Streamlit app."""
    st.title("Data Profiling")
    
    # Initialize profiler
    profiler = get_data_profiler()
    
    # Tabs for different profiling views
    tab_data_selection, tab_checkpoint_viewer, tab_comparison = st.tabs([
        "Create Checkpoint", "View Checkpoint", "Compare Stages"
    ])
    
    with tab_data_selection:
        st.header("Create Data Checkpoint")
        st.markdown("""
        Create a checkpoint to track data at any pipeline stage. 
        This helps you monitor how data changes throughout the pipeline.
        """)
        
        # Data source selection
        st.subheader("Select Data Source")
        data_source = st.radio(
            "Choose data source",
            options=["Current Session Data", "Saved Data Files"],
            horizontal=True
        )
        
        df = None
        y = None
        
        if data_source == "Current Session Data":
            # First check if feature selection results are available
            if 'feature_selection_results' in st.session_state and st.session_state.feature_selection_results is not None:
                # Use the reduced feature set from feature selection
                fs_results = st.session_state.feature_selection_results
                if "X_train" in fs_results and "selected_features" in fs_results:
                    selected_features = fs_results.get("selected_features", [])
                    df = fs_results["X_train"][selected_features].copy() if selected_features else fs_results["X_train"].copy()
                    if "y_train" in fs_results:
                        y = fs_results["y_train"].copy()
                    st.success(f"Using feature selection results with {len(selected_features)} selected features: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # If no feature selection results or feature selection couldn't be used, try regular features_df
            elif 'features_df' in st.session_state and st.session_state.features_df is not None:
                df = st.session_state.features_df
                st.success(f"Found data in current session: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Check if target exists
                if 'target' in st.session_state and st.session_state.target is not None:
                    y = st.session_state.target
                    st.info(f"Found target variable: {y.name}")
            else:
                st.warning("No data found in current session. Please load data in the main app first or select 'Saved Data Files'.")
        
        elif data_source == "Saved Data Files":
            # Get available data files
            available_files = get_available_data_files()
            
            if available_files:
                # Display data files for selection
                file_options = [f"{meta['timestamp']} - {meta['step']} ({meta['description']})" 
                               for meta in available_files]
                
                selected_file_idx = st.selectbox(
                    "Select a saved data file",
                    options=range(len(file_options)),
                    format_func=lambda i: file_options[i]
                )
                
                if st.button("Load Selected Data"):
                    try:
                        selected_file = available_files[selected_file_idx]
                        data = load_step_data(selected_file["path"])
                        
                        if isinstance(data, tuple) and len(data) == 2:
                            df, y = data
                            st.success(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns with target variable")
                        elif isinstance(data, pd.DataFrame):
                            df = data
                            st.success(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
                        elif isinstance(data, dict) and "X_train" in data:
                            # Handle feature selection results dictionary
                            if "selected_features" in data:
                                selected_features = data.get("selected_features", [])
                                df = data["X_train"][selected_features].copy() if selected_features else data["X_train"].copy()
                            else:
                                df = data["X_train"].copy()
                                
                            if "y_train" in data:
                                y = data["y_train"].copy()
                                
                            st.success(f"Loaded data from saved results: {df.shape[0]} rows × {df.shape[1]} columns")
                        else:
                            st.error("Unknown data format. Expected DataFrame, dictionary with 'X_train', or (DataFrame, Series) tuple.")
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
            else:
                st.warning("No saved data files found. Please save data in one of the main tabs first.")
        
        if df is not None:
            # Checkpoint creation form
            st.subheader("Create Checkpoint")
            
            with st.form("checkpoint_form"):
                # Stage name
                stage_name = st.text_input(
                    "Stage Name",
                    value=f"stage_{len(profiler.get_all_checkpoints()) + 1}",
                    help="Name to identify this pipeline stage (e.g., raw_data, feature_engineering, etc.)"
                )
                
                # Description
                description = st.text_area(
                    "Description",
                    value="",
                    help="Add notes about this checkpoint or what processing was done"
                )
                
                # Preview data
                st.subheader("Data Preview")
                st.dataframe(prepare_dataframe_for_streamlit(df.head(5)))
                
                submitted = st.form_submit_button("Create Checkpoint")
                
                if submitted:
                    try:
                        # Create checkpoint
                        stats = profiler.create_checkpoint(
                            stage=stage_name,
                            X=df,
                            y=y,
                            description=description
                        )
                        
                        st.success(f"Checkpoint '{stage_name}' created successfully!")
                        
                        # Display quick stats
                        st.subheader("Quick Stats")
                        st.markdown(f"""
                        - **Rows:** {stats['n_samples']}
                        - **Columns:** {stats['n_features']}
                        - **Memory Usage:** {stats['memory_usage_mb']:.2f} MB
                        - **Missing Values:** {stats['missing_values']['total']} ({stats['missing_values']['percentage']:.2f}%)
                        - **Numeric Features:** {len(stats['features']['numeric'])}
                        - **Categorical Features:** {len(stats['features']['categorical'])}
                        """)
                    except Exception as e:
                        st.error(f"Error creating checkpoint: {str(e)}")
    
    with tab_checkpoint_viewer:
        st.header("Checkpoint Viewer")
        
        # Get all checkpoints
        checkpoints = profiler.get_all_checkpoints()
        
        if not checkpoints:
            st.warning("No checkpoints available. Create a checkpoint in the 'Create Checkpoint' tab.")
        else:
            # Select checkpoint to view
            checkpoint_names = list(checkpoints.keys())
            selected_checkpoint = st.selectbox(
                "Select checkpoint to view",
                options=checkpoint_names
            )
            
            if selected_checkpoint:
                checkpoint = checkpoints[selected_checkpoint]
                
                # Display checkpoint info
                st.subheader(f"Checkpoint: {selected_checkpoint}")
                st.markdown(f"**Description:** {checkpoint['description']}")
                
                # Overview visualizations
                st.subheader("Data Overview")
                overview_fig = profiler.plot_data_overview(selected_checkpoint)
                st.plotly_chart(overview_fig, use_container_width=True)
                
                # Missing values visualization
                st.subheader("Missing Values")
                missing_fig = profiler.plot_missing_values(selected_checkpoint)
                st.plotly_chart(missing_fig, use_container_width=True)
                
                # Column exploration
                st.subheader("Explore Columns")
                
                # Group columns by type
                numeric_cols = checkpoint["features"]["numeric"]
                categorical_cols = checkpoint["features"]["categorical"]
                
                # Select category of columns
                col_type = st.radio(
                    "Column type",
                    options=["Numeric", "Categorical"],
                    horizontal=True
                )
                
                if col_type == "Numeric" and numeric_cols:
                    # Top stats table for numeric columns
                    st.subheader("Numeric Column Statistics")
                    
                    # Create a DataFrame of stats for display
                    stats_df = pd.DataFrame({
                        col: {
                            "Mean": checkpoint["column_stats"][col]["mean"],
                            "Median": checkpoint["column_stats"][col]["median"],
                            "Std": checkpoint["column_stats"][col]["std"],
                            "Min": checkpoint["column_stats"][col]["min"],
                            "Max": checkpoint["column_stats"][col]["max"],
                            "Missing %": checkpoint["column_stats"][col]["missing_pct"]
                        } for col in numeric_cols[:20]  # Limit to 20 columns for display
                    }).T  # Transpose for better display
                    
                    st.dataframe(stats_df)
                    
                    if len(numeric_cols) > 20:
                        st.info(f"Showing stats for 20 of {len(numeric_cols)} numeric columns.")
                    
                    # Feature selector for detailed view
                    selected_feature = st.selectbox(
                        "Select a numeric feature to explore",
                        options=numeric_cols
                    )
                    
                    if selected_feature:
                        # Show detailed view of this feature
                        st.subheader(f"Exploring: {selected_feature}")
                        feature_fig = profiler.plot_feature_distribution(selected_checkpoint, selected_feature)
                        st.plotly_chart(feature_fig, use_container_width=True)
                
                elif col_type == "Categorical" and categorical_cols:
                    # Top stats table for categorical columns
                    st.subheader("Categorical Column Statistics")
                    
                    # Create a DataFrame of stats for display
                    stats_df = pd.DataFrame({
                        col: {
                            "Unique Values": checkpoint["column_stats"][col]["unique_values"],
                            "Missing %": checkpoint["column_stats"][col]["missing_pct"],
                            "Most Common": str(checkpoint["column_stats"][col]["most_common"]) if checkpoint["column_stats"][col]["most_common"] else "N/A"
                        } for col in categorical_cols[:20]  # Limit to 20 columns for display
                    }).T  # Transpose for better display
                    
                    st.dataframe(stats_df)
                    
                    if len(categorical_cols) > 20:
                        st.info(f"Showing stats for 20 of {len(categorical_cols)} categorical columns.")
                    
                    # Feature selector for detailed view
                    selected_feature = st.selectbox(
                        "Select a categorical feature to explore",
                        options=categorical_cols
                    )
                    
                    if selected_feature:
                        # Show detailed view of this feature
                        st.subheader(f"Exploring: {selected_feature}")
                        
                        # Display top categories
                        st.write("Most common values:")
                        if checkpoint["column_stats"][selected_feature]["most_common"]:
                            st.write(checkpoint["column_stats"][selected_feature]["most_common"])
                else:
                    st.info(f"No {col_type.lower()} columns found in this checkpoint.")
    
    with tab_comparison:
        st.header("Pipeline Stage Comparison")
        
        # Get all checkpoints
        checkpoints = profiler.get_all_checkpoints()
        
        if len(checkpoints) < 2:
            st.warning("Need at least 2 checkpoints to compare. Create more checkpoints in the 'Create Checkpoint' tab.")
        else:
            # Select checkpoints to compare
            checkpoint_names = list(checkpoints.keys())
            
            col1, col2 = st.columns(2)
            with col1:
                checkpoint1 = st.selectbox(
                    "Stage 1",
                    options=checkpoint_names,
                    key="stage1"
                )
            
            with col2:
                # Exclude the first selection from options for the second selection
                remaining_checkpoints = [cp for cp in checkpoint_names if cp != checkpoint1]
                checkpoint2 = st.selectbox(
                    "Stage 2",
                    options=remaining_checkpoints,
                    key="stage2"
                )
            
            if checkpoint1 and checkpoint2:
                # Get comparison
                comparison = profiler.compare_checkpoints(checkpoint1, checkpoint2)
                
                # High level changes
                st.subheader("Change Summary")
                
                # Display as metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples", checkpoints[checkpoint2]["n_samples"], 
                             delta=int(comparison["sample_count_change"]))
                with col2:
                    st.metric("Features", checkpoints[checkpoint2]["n_features"], 
                             delta=int(comparison["feature_count_change"]))
                with col3:
                    # For missing values, negative delta is better
                    delta_missing = int(comparison["missing_values_change"])
                    delta_color = "normal" if delta_missing <= 0 else "inverse"
                    st.metric("Missing Values", checkpoints[checkpoint2]["missing_values"]["total"], 
                             delta=delta_missing, delta_color=delta_color)
                
                # Show added/removed columns
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Added Columns")
                    if comparison["columns_added"]:
                        st.write(", ".join(comparison["columns_added"]))
                    else:
                        st.info("No columns added")
                
                with col2:
                    st.subheader("Removed Columns")
                    if comparison["columns_removed"]:
                        st.write(", ".join(comparison["columns_removed"]))
                    else:
                        st.info("No columns removed")
                
                # Pipeline visualization
                st.subheader("Pipeline Metrics")
                pipeline_fig = profiler.plot_pipeline_comparison()
                st.plotly_chart(pipeline_fig, use_container_width=True)
                
                # Column changes 
                st.subheader("Column Changes")
                
                # Only show columns with notable changes
                notable_changes = {}
                for col, changes in comparison["column_stats_changes"].items():
                    # Check if any change is significant
                    if (changes.get("mean_change") is not None and abs(changes.get("mean_change")) > 0.1) or \
                       (changes.get("std_change") is not None and abs(changes.get("std_change")) > 0.1) or \
                       (changes.get("missing_change") != 0) or \
                       (changes.get("unique_values_change", 0) != 0):
                        notable_changes[col] = changes
                
                if notable_changes:
                    # Create a table of notable changes
                    changes_data = []
                    for col, changes in notable_changes.items():
                        row = {"Column": col}
                        
                        # Check which type of changes are present
                        if "mean_change" in changes:
                            if changes['mean_change'] is not None:
                                row["Mean Δ"] = f"{float(changes['mean_change']):.2f}"
                            else:
                                row["Mean Δ"] = "N/A"
                                
                            if changes['std_change'] is not None:
                                row["Std Δ"] = f"{float(changes['std_change']):.2f}"
                            else:
                                row["Std Δ"] = "N/A"
                        elif "unique_values_change" in changes:
                            row["Unique Values Δ"] = int(changes["unique_values_change"])
                        
                        # Missing values change applies to both types
                        row["Missing Values Δ"] = int(changes.get("missing_change", 0))
                        
                        changes_data.append(row)
                    
                    # Convert to DataFrame for display
                    changes_df = pd.DataFrame(changes_data)
                    st.dataframe(changes_df)
                else:
                    st.info("No significant changes detected in common columns.") 