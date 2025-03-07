"""
Main application for the Credit Default Risk Prediction web interface.

This application provides a Streamlit web interface for:
- Data exploration and management
- Feature engineering
- Feature selection
- Model training 
- Prediction
"""
import sys
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
# Import plotly here to ensure it's available throughout the file
import plotly.express as px

# Add parent directory to path for package imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import utility modules
from credit_risk.web.utils import (
    set_page_config, 
    get_config, 
    is_gpu_available,
    apply_sampling_profile
)
from credit_risk.utils.streamlit_utils import (
    prepare_dataframe_for_streamlit,
    save_step_data,
    load_step_data,
    get_available_step_data,
    cleanup_corrupted_files
)
from credit_risk.utils.logging_utils import get_logger
from credit_risk.config import RunMode, ProcessingBackend
from credit_risk.web.ui_components import (
    display_info_box,
    educational_tip,
    display_dataframe_with_metrics,
    plot_missing_values,
    section_header,
    workflow_progress
)

# Import tab modules
from credit_risk.web.tabs.feature_selection_tab import render_feature_selection_tab
from credit_risk.web.tabs.model_training_tab import render_model_training_tab
from credit_risk.web.tabs.prediction_tab import render_prediction_tab
from credit_risk.web.tabs.data_profiling_tab import render_data_profiling_tab
from credit_risk.web.exploratory_analysis_tab import render_exploratory_analysis_tab, SuppressWarnings

# Import data functions from original app
from credit_risk.web.app import load_data, run_feature_engineering

# Import data profiling module
from credit_risk.utils.data_profiling import DataProfiler

# Configure logging
logger = get_logger("credit_risk_app")

# Initialize the data profiler function here for use in main file
def get_data_profiler():
    """Initialize or retrieve the DataProfiler from session state."""
    if 'data_profiler' not in st.session_state:
        # Create a checkpoint directory within the data directory
        checkpoint_dir = os.path.join(os.getcwd(), 'data', 'profiling_checkpoints')
        st.session_state.data_profiler = DataProfiler(checkpoint_dir=checkpoint_dir)
    return st.session_state.data_profiler

def main():
    """Main function for the Streamlit app."""
    # Set up page configuration
    set_page_config("Credit Risk Prediction")
    
    # Display app title and description
    st.title("Credit Default Risk Prediction")
    
    with st.expander("About this application", expanded=False):
        st.markdown(
            """
            This application demonstrates credit risk prediction using:
            - Advanced feature engineering
            - Feature selection with null importance
            - Multiple model types (LightGBM, XGBoost, CatBoost)
            - MLflow experiment tracking
            
            Use this app to explore, preprocess data, build features, train models, and evaluate performance.
            """
        )
    
    # Check if GPU is available and print message
    gpu_available = is_gpu_available()
    st.session_state["gpu_available"] = gpu_available
    
    if not gpu_available:
        st.sidebar.warning("No GPU detected. Using CPU for all operations.")
    else:
        st.sidebar.success("GPU detected and available for accelerated processing.")
    
    # Create a simple sidebar for session management
    st.sidebar.title("Session Management")
    
    # Option to reset session
    if st.sidebar.button("Reset Session"):
        for key in list(st.session_state.keys()):
            if key not in ["gpu_available"]:
                del st.session_state[key]
        st.sidebar.success("Session reset successfully!")
        st.rerun()
    
    # Get configuration based on user selections
    config = get_config()
    
    # Initialize session state variables if they don't exist
    if "datasets" not in st.session_state:
        st.session_state["datasets"] = {}
    
    if "features_df" not in st.session_state:
        st.session_state["features_df"] = None
    
    if "feature_selection_results" not in st.session_state:
        st.session_state["feature_selection_results"] = None
    
    if "model_results" not in st.session_state:
        st.session_state["model_results"] = None
    
    # Add state for active tabs
    if "active_main_tab" not in st.session_state:
        st.session_state["active_main_tab"] = 0  # Default to first tab
    
    if "active_fe_tab" not in st.session_state:
        st.session_state["active_fe_tab"] = 0  # Default to Feature Generation tab
    
    # Add state for whether feature preview should be shown
    if "show_feature_preview" not in st.session_state:
        st.session_state["show_feature_preview"] = False
    
    # Track workflow progress
    workflow_stage = 0
    if st.session_state["datasets"]:
        workflow_stage = 1
    if st.session_state.get("features_df") is not None:
        workflow_stage = 2
    if "feature_selection_results" in st.session_state:
        workflow_stage = 3
    if "model_results" in st.session_state:
        workflow_stage = 4
    
    # Display current workflow stage in sidebar
    st.sidebar.title("Workflow Progress")
    progress_bar = st.sidebar.progress(workflow_stage / 4)
    st.sidebar.caption(f"Stage {workflow_stage + 1} of 5")
    
    # Create tabs for main workflow
    tab_titles = ["Data Management", "Feature Engineering", "Feature Selection", 
                 "Model Training", "Prediction", "Data Profiling", "MLflow Dashboard"]
    selected_tab = st.session_state.get("active_main_tab", 0)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_titles)
    
    # Create subtabs for the feature engineering workflow right away so we can set their state
    with tab2:
        fe_tab_titles = ["Feature Generation", "Load Previous Features", "Feature Preview"]
        selected_fe_tab = st.session_state.get("active_fe_tab", 0)
        fe_tabs = st.tabs(fe_tab_titles)
    
    # Store tabs in session state for reference
    st.session_state["tabs"] = [tab1, tab2, tab3, tab4, tab5, tab6, tab7]
    
    # Data Management tab
    with tab1:
        section_header(
            "Data Management",
            "Load, explore, and prepare your data for modeling",
            """
            **Why Data Management Matters:**
            
            Good data preparation is essential for successful modeling:
            - Clean data leads to better model performance
            - Appropriate sampling ensures efficient development
            - Understanding your data helps with feature engineering
            
            In this tab, you can load your datasets, apply sampling strategies,
            and explore the data before proceeding to feature engineering.
            """
        )
        
        # Create tabs for data management
        data_tabs = st.tabs(["Data Loading", "Data Sampling", "Data Preview"])
        
        # Data Loading tab
        with data_tabs[0]:
            st.subheader("Data Sources")
            
            # Add a helpful tip about data loading
            educational_tip(
                "Loading smaller amounts of data during development allows for faster iterations. "
                "You can always train on the full dataset when you've finalized your approach."
            )
            
            # Data path setting
            data_path_col1, data_path_col2 = st.columns([3, 1])
            with data_path_col1:
                data_path = st.text_input(
                    "Data Directory",
                    value="data/raw",
                    help="Path to raw data files."
                )
            
            # Update config with data path
            # Convert the data path string to Path objects as expected by the loader
            config.data.raw_data_path = Path(data_path)
            config.data.processed_data_path = Path("data/processed")
            config.data.features_path = Path("data/features")
            
            # GPU settings for data loading
            use_gpu_for_loading = st.checkbox(
                "Use GPU for Data Loading", 
                value=gpu_available,
                disabled=not gpu_available,
                help="Use GPU acceleration for faster data loading (if available)."
            )
            
            # Update config with GPU setting
            if use_gpu_for_loading and gpu_available:
                config.gpu.processing_backend = ProcessingBackend.GPU
            else:
                config.gpu.processing_backend = ProcessingBackend.CPU
            
            # Button to load data
            if st.button("Load Data", key="load_data_button", use_container_width=True):
                with st.spinner("Loading datasets..."):
                    try:
                        st.session_state["datasets"] = load_data(config, force_sample=False)
                        
                        # Show the total number of rows loaded
                        if st.session_state["datasets"]:
                            total_rows = sum(df.shape[0] for df in st.session_state["datasets"].values())
                            st.success(f"Successfully loaded {len(st.session_state['datasets'])} datasets with {total_rows:,} total rows")
                            
                            # Set workflow stage to data loaded
                            workflow_stage = 1
                            
                            # Log success
                            st.success(f"Data loaded successfully: {total_rows} samples from {len(st.session_state['datasets'])} tables")
                            
                            # Add automatic profiling at data loading stage
                            try:
                                profiler = get_data_profiler()
                                profiler.create_checkpoint(
                                    stage="raw_data",
                                    X=st.session_state["datasets"]["application"],
                                    y=st.session_state["datasets"]["application"]["TARGET"] if "TARGET" in st.session_state["datasets"]["application"].columns else None,
                                    description=f"Raw data: {', '.join(st.session_state['datasets'].keys())}"
                                )
                                st.info("Data profile created. View it in the Data Profiling tab.")
                            except Exception as e:
                                logger.error(f"Error creating data profile: {str(e)}")
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
                        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        
        # Data Sampling tab
        with data_tabs[1]:
            st.subheader("Data Sampling Settings")
            
            if not st.session_state["datasets"]:
                st.warning("No data loaded yet. Please load data first.")
            else:
                # Get the original total size for reference
                original_total_rows = sum(df.shape[0] for df in st.session_state["datasets"].values())
                
                display_info_box(
                    "About Sampling Profiles", 
                    """
                    **Sampling profiles** provide preset configurations for different stages of model development:
                    
                    - **Quick Exploration**: Small samples for rapid iteration and testing
                    - **Balanced Development**: Medium-sized samples balanced for model training
                    - **Production Preparation**: Larger samples for final model testing
                    - **Full Dataset**: No sampling, use all available data
                    
                    Choose a profile based on your current needs, or customize the sample size.
                    """,
                    expanded=False
                )
                
                # Create two columns: one for profile selection and one for options
                profile_col1, profile_col2 = st.columns([2, 1])
                
                with profile_col1:
                    # Sampling profile selection
                    profile_options = [
                        "Quick Exploration", 
                        "Balanced Development", 
                        "Production Preparation", 
                        "Full Dataset"
                    ]
                    
                    selected_profile = st.radio(
                        "Sampling Profile",
                        options=profile_options,
                        index=1,  # Default to Balanced Development
                        horizontal=True
                    )
                
                with profile_col2:
                    # Custom sample size option
                    use_custom_size = st.checkbox("Custom Sample Size", value=False)
                    
                    if use_custom_size:
                        custom_sample_size = st.number_input(
                            "Sample Size per Dataset",
                            min_value=1000,
                            max_value=500000,
                            value=10000,
                            step=1000
                        )
                    else:
                        custom_sample_size = None
                
                # Show profile description
                if selected_profile == "Quick Exploration":
                    st.info("🔍 **Quick Exploration**: Small samples (5,000 rows) for rapid testing and debugging.")
                elif selected_profile == "Balanced Development":
                    st.info("⚖️ **Balanced Development**: Medium-sized samples (20,000 rows) with stratified sampling for model development.")
                elif selected_profile == "Production Preparation":
                    st.info("🚀 **Production Preparation**: Larger samples (50,000 rows) with stratified sampling for final model testing.")
                else:  # Full Dataset
                    st.warning("⚠️ **Full Dataset**: Using all data. This may be slow, especially for feature engineering and model training.")
                
                # Apply sampling button
                if st.button("Apply Sampling Profile", use_container_width=True):
                    with st.spinner(f"Applying {selected_profile} sampling profile..."):
                        # Use our new sampling utility
                        sampled_datasets = apply_sampling_profile(
                            st.session_state["datasets"],
                            selected_profile,
                            custom_sample_size
                        )
                        
                        # Update datasets in session state
                        st.session_state["datasets"] = sampled_datasets
                        
                        # Show success message with comparison to original size
                        new_total_rows = sum(df.shape[0] for df in sampled_datasets.values())
                        reduction = 100 * (1 - new_total_rows / original_total_rows) if original_total_rows > 0 else 0
                        
                        if reduction > 0:
                            st.success(f"Applied {selected_profile} sampling. New total: {new_total_rows:,} rows ({reduction:.1f}% reduction)")
                        else:
                            st.success(f"Using full dataset with {new_total_rows:,} rows")
                
                # Display current dataset sizes
                st.subheader("Current Dataset Sizes")
                
                # Create a more informative table with original and current sizes
                if "original_sizes" not in st.session_state and st.session_state["datasets"]:
                    st.session_state["original_sizes"] = {
                        name: df.shape[0] for name, df in st.session_state["datasets"].items()
                    }
                
                if "original_sizes" in st.session_state:
                    sizes_data = []
                    for name, df in st.session_state["datasets"].items():
                        original_size = st.session_state["original_sizes"].get(name, df.shape[0])
                        current_size = df.shape[0]
                        percent = 100 * (current_size / original_size) if original_size > 0 else 100
                        
                        sizes_data.append({
                            "Dataset": name,
                            "Original Rows": f"{original_size:,}",
                            "Current Rows": f"{current_size:,}",
                            "% of Original": f"{percent:.1f}%",
                            "Columns": df.shape[1]
                        })
                    
                    sizes_df = pd.DataFrame(sizes_data)
                    st.table(sizes_df)
        
        # Data Preview tab
        with data_tabs[2]:
            st.subheader("Data Preview")
            
            if not st.session_state["datasets"]:
                st.warning("No data loaded yet. Please load data first.")
            else:
                st.write("Preview and analyze the loaded datasets before feature engineering.")
                
                # Dataset selection with dataset info
                dataset_options = [f"{name} ({df.shape[0]:,} rows)" for name, df in st.session_state["datasets"].items()]
                selected_option = st.selectbox(
                    "Select Dataset to Preview",
                    options=dataset_options
                )
                
                # Extract the dataset name from the selected option
                selected_dataset = selected_option.split(" (")[0] if selected_option else None
                
                # Show selected dataset
                if selected_dataset:
                    df = st.session_state["datasets"][selected_dataset]
                    
                    # Display dataset metrics and sample
                    display_dataframe_with_metrics(df, selected_dataset)
                    
                    # Create tabs for different views
                    preview_tabs = st.tabs(["Column Statistics", "Missing Values", "Target Distribution"])
                    
                    with preview_tabs[0]:
                        # Create statistics DataFrame
                        stats_df = pd.DataFrame({
                            "Type": df.dtypes.astype(str),
                            "Non-Null Count": df.count(),
                            "Null Count": df.isnull().sum(),
                            "Null %": round(100 * df.isnull().sum() / len(df), 2),
                            "Unique Values": df.nunique()
                        })
                        st.dataframe(stats_df)
                        
                    with preview_tabs[1]:
                        # Use our reusable component for missing values
                        plot_missing_values(df)
                        
                    with preview_tabs[2]:
                        # Show target distribution if this is the main dataset
                        if "TARGET" in df.columns:
                            target_col1, target_col2 = st.columns([1, 1])
                            
                            with target_col1:
                                # Show target value counts
                                target_counts = df["TARGET"].value_counts().reset_index()
                                target_counts.columns = ["Target Value", "Count"]
                                target_counts["Percentage"] = 100 * target_counts["Count"] / target_counts["Count"].sum()
                                st.write("### Target Distribution")
                                st.table(target_counts)
                            
                            with target_col2:
                                # Create a pie chart of target distribution
                                try:
                                    import plotly.express as px
                                    fig = px.pie(
                                        target_counts, 
                                        values="Count", 
                                        names="Target Value",
                                        title="Target Distribution",
                                        color_discrete_sequence=["#0068c9", "#ef553b"]
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not create target distribution chart: {str(e)}")
                        else:
                            st.info("No TARGET column found in this dataset.")
    
    # Feature Engineering tab
    with tab2:
        section_header(
            "Feature Engineering",
            "Create and manage features for predictive modeling",
            """
            **Why Feature Engineering Matters:**
            
            Feature engineering can dramatically improve model performance by:
            - Creating meaningful representations of raw data
            - Capturing domain knowledge as numerical features
            - Revealing relationships between entities
            - Transforming data into forms that models can better understand
            
            In this tab, you can generate features or load previously created feature sets.
            """
        )
        
        # Feature Generation tab - for creating new features
        with fe_tabs[0]:
            if not st.session_state["datasets"]:
                st.warning("No data loaded yet. Please go to the Data Management tab first.")
            else:
                # Feature engineering settings
                st.subheader("Feature Engineering Settings")
                
                # Create columns for settings
                fe_col1, fe_col2 = st.columns(2)
                
                with fe_col1:
                    max_depth = st.slider(
                        "Maximum feature depth",
                        min_value=1,
                        max_value=3,
                        value=config.features.max_depth,
                        step=1,
                        help="Higher values create more complex features but take longer to compute.\n\n" + 
                             "1: Simple aggregations\n" +
                             "2: Aggregations of aggregations\n" +
                             "3: Complex nested features (slowest)"
                    )
                    
                    # Replace GPU checkbox with info message
                    st.info("⚡ **CPU Processing:** Using CPU for reliable feature generation.")
                
                with fe_col2:
                    agg_primitives = st.multiselect(
                        "Feature Types",
                        options=["mean", "sum", "std", "max", "min", "count", "percent_true", "num_unique"],
                        default=["mean", "sum", "max", "min", "count"],
                        help="Types of features to generate.\n\n" +
                             "Common choices:\n" +
                             "- sum: Total across related records\n" +
                             "- mean: Average of related records\n" +
                             "- count: Number of related records\n" +
                             "- min/max: Extreme values in related records"
                    )
                    
                    # Display an info message about parallel processing
                    st.info("🔒 **Single-Process Mode:** Feature generation runs in single-process mode for stability.")
                
                # Update config with FE settings
                config.features.max_depth = max_depth
                if agg_primitives:
                    config.features.default_agg_primitives = agg_primitives
                # Always set n_jobs to 1 for stability
                config.features.n_jobs = 1
                
                # Processing backend based on GPU setting - always use CPU for stability
                config.gpu.processing_backend = ProcessingBackend.CPU
                
                # Additional sampling for feature engineering
                with st.expander("Advanced Feature Engineering Settings"):
                    chunk_size = st.number_input(
                        "Chunk Size",
                        min_value=1000,
                        max_value=50000,
                        value=config.features.chunk_size or 10000,
                        step=1000,
                        help="Number of rows to process at once. Smaller chunks use less memory but may be slower."
                    )
                    config.features.chunk_size = chunk_size
                    
                    # Description for saving
                    fe_description = st.text_input(
                        "Description (for saving)",
                        value=f"Features with depth {max_depth} and {len(agg_primitives)} primitives",
                        help="A description to identify these features later"
                    )
                
                # Button to run feature engineering
                if st.button("Generate Features", key="run_fe_button", use_container_width=True):
                    if not st.session_state["datasets"]:
                        st.error("No data loaded. Please load data first.")
                    else:
                        with st.spinner("Generating features... (this may take a few minutes)"):
                            try:
                                # Show progress info
                                progress_container = st.empty()
                                progress_container.info("Preprocessing data and preparing feature engineering...")
                                
                                # Run feature engineering
                                features_df = run_feature_engineering(
                                    config,
                                    st.session_state["datasets"]
                                )
                                
                                if features_df is not None:
                                    # Update progress
                                    progress_container.success("Feature generation complete!")
                                    
                                    # Save to session state
                                    st.session_state["features_df"] = features_df
                                    
                                    # Create feature engineering results dictionary
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    fe_results = {
                                        "features_df": features_df,
                                        "config": {
                                            "max_depth": max_depth,
                                            "agg_primitives": agg_primitives,
                                            "n_jobs": 1,
                                            "chunk_size": chunk_size,
                                            "use_gpu": False
                                        },
                                        "timestamp": timestamp,
                                        "description": fe_description,
                                        "dataset_shape": features_df.shape,
                                        "dataset_original": list(st.session_state["datasets"].keys())
                                    }
                                    
                                    # Save to session state and file
                                    st.session_state["feature_engineering_results"] = fe_results
                                    save_step_data("feature_engineering", fe_results, fe_description)
                                    
                                    # Set active tabs and flag to show feature preview
                                    st.session_state["active_main_tab"] = 1  # Feature Engineering tab
                                    st.session_state["active_fe_tab"] = 2    # Feature Preview tab
                                    st.session_state["show_feature_preview"] = True
                                    
                                    # Add automatic profiling for feature engineering
                                    try:
                                        profiler = get_data_profiler()
                                        profiler.create_checkpoint(
                                            stage="feature_engineering",
                                            X=features_df,
                                            y=features_df["TARGET"] if "TARGET" in features_df.columns else None,
                                            description=f"Features generated: {features_df.shape[1]} features"
                                        )
                                        st.info("Feature profile created. Compare with raw data in the Data Profiling tab.")
                                    except Exception as e:
                                        logger.error(f"Error creating feature profile: {str(e)}")
                                    
                                    st.success(f"Features generated successfully: {features_df.shape[1]} features")
                                    st.rerun()
                                else:
                                    st.error("Error generating features. Please check logs for details.")
                            except Exception as e:
                                st.error(f"Error generating features: {str(e)}")
                                logger.error(f"Error generating features: {str(e)}", exc_info=True)
        
        # Load Previous Features tab
        with fe_tabs[1]:
            st.subheader("Load Previously Generated Features")
            
            # Check for existing feature engineering results
            available_fe_data = get_available_step_data("feature_engineering")
            
            if available_fe_data:
                st.info(f"Found {len(available_fe_data)} previous feature engineering results")
                
                # Create selectbox for available feature engineering results
                result_options = []
                for timestamp, description in available_fe_data.items():
                    result_options.append(f"{timestamp} - {description}")
                
                selected_result = st.selectbox(
                    "Select saved features to load",
                    options=result_options,
                    help="Choose previously saved feature sets to load"
                )
                
                # Create two columns for load button and cleanup button
                load_col, cleanup_col = st.columns([3, 1])
                
                with load_col:
                    load_button = st.button("Load Selected Features", use_container_width=True)
                
                with cleanup_col:
                    cleanup_button = st.button("Cleanup Data", help="Check for and repair corrupted data files")
                
                if cleanup_button:
                    with st.spinner("Checking for corrupted files..."):
                        checked, removed = cleanup_corrupted_files("feature_engineering")
                        if removed > 0:
                            st.success(f"Cleaned up {removed} corrupted files (checked {checked} files)")
                        else:
                            st.success(f"No corrupted files found (checked {checked} files)")
                        
                        # Refresh the page to update file listings
                        st.rerun()
                
                if load_button:
                    with st.spinner("Loading features..."):
                        # Extract timestamp from selection
                        timestamp = selected_result.split(" - ")[0].strip()
                        logger.info(f"Attempting to load features with timestamp: {timestamp}")
                        fe_results = load_step_data("feature_engineering", timestamp)
                        
                        if fe_results and "features_df" in fe_results:
                            # Load features into session state
                            st.session_state["features_df"] = fe_results["features_df"]
                            st.session_state["feature_engineering_results"] = fe_results
                            
                            # Set active tabs and flag to show feature preview
                            st.session_state["active_main_tab"] = 1  # Feature Engineering tab
                            st.session_state["active_fe_tab"] = 2    # Feature Preview tab
                            st.session_state["show_feature_preview"] = True
                            
                            # Show success message
                            features_df = fe_results["features_df"]
                            st.success(f"Loaded features: {features_df.shape[1]} features, {features_df.shape[0]} samples")
                            st.rerun()
                        else:
                            st.error("Error loading features: Invalid or corrupted data file")
                            logger.error(f"Failed to load features with timestamp: {timestamp}")
            else:
                st.warning("No saved feature engineering results found. Generate features first.")
                
            # Option to upload a saved feature set
            st.subheader("Or Upload Features File")
            uploaded_file = st.file_uploader("Upload a saved features file (.pkl or .csv)", type=["pkl", "csv"])
            
            if uploaded_file is not None:
                try:
                    # Load the uploaded file
                    with st.spinner("Loading uploaded features..."):
                        if uploaded_file.name.endswith('.csv'):
                            features_df = pd.read_csv(uploaded_file)
                        else:
                            import pickle
                            features_df = pickle.load(uploaded_file)
                        
                        # Create feature engineering results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fe_results = {
                            "features_df": features_df,
                            "config": {"uploaded_file": True},
                            "timestamp": timestamp,
                            "description": f"Uploaded features: {uploaded_file.name}",
                            "dataset_shape": features_df.shape
                        }
                        
                        # Save to session state
                        st.session_state["features_df"] = features_df
                        st.session_state["feature_engineering_results"] = fe_results
                        
                        # Set active tabs and flag to show feature preview
                        st.session_state["active_main_tab"] = 1  # Feature Engineering tab
                        st.session_state["active_fe_tab"] = 2    # Feature Preview tab
                        st.session_state["show_feature_preview"] = True
                        
                        st.success(f"Loaded features from upload: {features_df.shape[1]} features")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        # Feature Preview tab
        with fe_tabs[2]:
            # Show features if already available
            features_df = st.session_state.get("features_df")
            
            if features_df is None:
                st.warning("No features available yet. Please generate or load features first.")
            else:
                # Flag that we're showing the feature preview
                st.session_state["show_feature_preview"] = True
                
                # Display feature information
                st.subheader("Feature Overview")
                st.write(f"Dataset shape: {features_df.shape[0]} samples × {features_df.shape[1]} features")
                
                # Feature type distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    # Count by feature type
                    numeric_cols = len(features_df.select_dtypes(include=['number']).columns)
                    categorical_cols = len(features_df.select_dtypes(include=['category', 'object']).columns)
                    st.metric("Numeric Features", numeric_cols)
                    st.metric("Categorical Features", categorical_cols)
                
                with col2:
                    # Count original vs engineered
                    original_cols = [col for col in features_df.columns if '(' not in col and ')' not in col]
                    engineered_cols = [col for col in features_df.columns if '(' in col or ')' in col]
                    st.metric("Original Features", len(original_cols))
                    st.metric("Engineered Features", len(engineered_cols))
                
                # Feature data preview
                st.subheader("Feature Sample")
                
                # Create tabs for different views
                preview_tabs = st.tabs(["Data Preview", "Feature Statistics", "Correlations"])
                
                with preview_tabs[0]:
                    # Get a mix of features to display
                    sample_original = original_cols[:5]
                    sample_engineered = engineered_cols[:5]
                    sample_cols = sample_original + sample_engineered
                    
                    if 'TARGET' in features_df.columns and 'TARGET' not in sample_cols:
                        sample_cols = ['TARGET'] + sample_cols
                    
                    st.dataframe(prepare_dataframe_for_streamlit(features_df[sample_cols].head(10)))
                    st.caption(f"Showing {len(sample_cols)} of {features_df.shape[1]} features")
                    
                    # Option to export features
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        if st.button("Export to CSV", use_container_width=True):
                            csv = features_df.to_csv(index=False)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            st.download_button(
                                "Download CSV",
                                csv,
                                file_name=f"features_{timestamp}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    with export_col2:
                        # Only show full dataset if requested
                        if st.button("Show All Features", use_container_width=True):
                            st.dataframe(prepare_dataframe_for_streamlit(features_df.head(10)))
                            st.caption(f"Showing all {features_df.shape[1]} features")
                
                with preview_tabs[1]:
                    # Feature Statistics & Correlations Tab
                    if features_df is not None:
                        st.subheader("Feature Statistics")
                        
                        # Filter to only numeric columns for statistics
                        numeric_df = features_df.select_dtypes(include=np.number)
                        
                        # Display basic statistics
                        with st.expander("Basic Statistics", expanded=True):
                            with SuppressWarnings():
                                stats_df = numeric_df.describe().T
                                stats_df = prepare_dataframe_for_streamlit(stats_df)
                                st.dataframe(stats_df)
                    else:
                        st.info("No feature data available. Please generate or load features first.")
                
                with preview_tabs[2]:
                    # Correlations Tab
                    if features_df is not None and 'TARGET' in features_df.columns:
                        # Show correlations with target
                        st.subheader("Feature Correlations")
                        
                        # Filter to only numeric columns for correlations
                        numeric_df = features_df.select_dtypes(include=np.number).drop(columns=['TARGET'], errors='ignore')
                        
                        # Calculate correlations with TARGET
                        with SuppressWarnings():
                            corrs = numeric_df.corrwith(features_df['TARGET']).sort_values(ascending=False)
                        
                        # Display top positive and negative correlations
                        corr_col1, corr_col2 = st.columns(2)
                        
                        with corr_col1:
                            st.write("Top Positive Correlations")
                            st.dataframe(corrs.head(10).to_frame("Correlation with TARGET"))
                        
                        with corr_col2:
                            st.write("Top Negative Correlations")
                            st.dataframe(corrs.tail(10).to_frame("Correlation with TARGET"))
                        
                        # Plot correlation heatmap for top features
                        st.subheader("Correlation Heatmap")
                        
                        # Get top correlated features
                        with SuppressWarnings():
                            top_positive = corrs.head(10).index
                            top_negative = corrs.tail(10).index
                            target_idx = pd.Index(['TARGET'])
                            top_features = top_positive.union(top_negative).union(target_idx)
                            
                            # Calculate correlation matrix
                            corr_matrix = features_df[top_features].corr()
                        
                        # Create heatmap
                        import plotly.express as px  # Import px locally to ensure it's available
                        fig = px.imshow(
                            corr_matrix,
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1,
                            title="Feature Correlation Heatmap",
                            labels=dict(x="Feature", y="Feature", color="Correlation")
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Selection tab
    with tab3:
        # Pass the features dataframe to the feature selection tab
        features_df = st.session_state.get("features_df")
        render_feature_selection_tab(features_df)
    
    # Model Training tab
    with tab4:
        # Pass the features dataframe to the model training tab
        features_df = st.session_state.get("features_df")
        render_model_training_tab(features_df)
    
    # Prediction tab
    with tab5:
        render_prediction_tab()
    
    # Data Profiling tab
    with tab6:
        render_data_profiling_tab()
    
    # MLflow Dashboard tab
    with tab7:
        st.header("MLflow Dashboard")
        st.markdown("""
        The MLflow dashboard provides a way to track and compare experiments.
        
        You can view:
        - All tracked runs
        - Hyperparameters used
        - Model performance metrics
        - Artifacts generated
        """)
        
        # Check if MLflow UI is available
        if st.button("Launch MLflow UI (New Tab)"):
            # This won't directly open a new tab in Streamlit, but users can click the link
            st.markdown("[Open MLflow UI](http://localhost:5000)")
            
            # Attempt to start MLflow UI if it's not already running
            try:
                import subprocess
                subprocess.Popen(["mlflow", "ui"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                start_new_session=True)
                st.success("MLflow UI started. Click the link above to open it.")
            except Exception as e:
                st.error(f"Error starting MLflow UI: {str(e)}")
                st.info("You may need to start MLflow UI manually with 'mlflow ui' in your terminal.")

if __name__ == "__main__":
    main() 