"""
Web application for credit risk prediction using Streamlit.

This module provides a web interface for exploring data, running feature
engineering experiments, training models, and visualizing results.
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path for package imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from credit_risk.config import PipelineConfig, RunMode, ProcessingBackend
from credit_risk.data.gpu_loader import GPUDataLoader
from credit_risk.data.preprocessor import DataPreprocessor
from credit_risk.features.entity_builder import EntitySetBuilder
from credit_risk.features.engineer import FeatureEngineer
from credit_risk.features.feature_store import FeatureStore
from credit_risk.utils.mlflow_tracking import ExperimentTracker
from credit_risk.utils.streamlit_utils import prepare_dataframe_for_streamlit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("credit_risk_app")


def get_config():
    """Get configuration based on user selections in the sidebar."""
    # Create a default config
    config = PipelineConfig()
    
    # Update with user selections
    config.run_mode = st.session_state.get("run_mode", RunMode.DEVELOPMENT)
    config.data.sample_size = st.session_state.get("sample_size", 10000)
    
    # GPU settings
    processing_backend = st.session_state.get("processing_backend", ProcessingBackend.AUTO)
    use_gpu_for_feature_engineering = st.session_state.get("use_gpu_for_features", True)
    use_gpu_for_training = st.session_state.get("use_gpu_for_training", True)
    
    # Update GPU config
    config.gpu.processing_backend = processing_backend
    config.gpu.use_gpu_for_feature_engineering = use_gpu_for_feature_engineering
    config.gpu.use_gpu_for_training = use_gpu_for_training
    
    # MLflow settings
    config.mlflow.enabled = st.session_state.get("mlflow_enabled", True)
    config.mlflow.experiment_name = st.session_state.get("experiment_name", "credit_risk_prediction")
    
    return config


def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.title("Configuration")
    
    # Run mode
    run_mode = st.sidebar.selectbox(
        "Run Mode",
        options=[mode.value for mode in RunMode],
        index=0,  # Default to development mode
        help="Development mode uses a sample of the data for faster iteration."
    )
    st.session_state["run_mode"] = RunMode(run_mode)
    
    # Sample size (for development mode)
    sample_size = st.sidebar.number_input(
        "Sample Size",
        min_value=5000,
        max_value=300000,
        value=10000,
        step=5000,
        help="Number of samples to use in development mode."
    )
    st.session_state["sample_size"] = sample_size
    
    # Data settings
    st.sidebar.header("Data Settings")
    
    # Raw data path
    raw_data_path = st.sidebar.text_input(
        "Raw Data Path",
        value="data/raw",
        help="Path to raw data files"
    )
    st.session_state["raw_data_path"] = raw_data_path
    
    # GPU settings
    st.sidebar.header("GPU Settings")
    
    # Check if GPU is available
    gpu_available = is_gpu_available()
    
    # If GPU is not available, show a message
    if not gpu_available:
        st.sidebar.warning("No GPU detected. Using CPU for all operations.")
    
    # Processing backend options
    backend_options = [backend.value for backend in ProcessingBackend]
    # Default to GPU (index 1) when GPU is available, otherwise default to CPU (index 0)
    default_index = 1 if gpu_available else 0
    
    processing_backend = st.sidebar.selectbox(
        "Processing Backend",
        options=backend_options,
        index=default_index,
        disabled=not gpu_available,  # Disable selection if no GPU
        help="Processing backend to use for data loading and processing."
    )
    
    # If no GPU available, force CPU
    if not gpu_available and processing_backend != ProcessingBackend.CPU.value:
        st.sidebar.info("Forcing CPU backend since no GPU is available.")
        processing_backend = ProcessingBackend.CPU.value
    
    st.session_state["processing_backend"] = ProcessingBackend(processing_backend)
    
    use_gpu_for_features = st.sidebar.checkbox(
        "Use GPU for Feature Engineering",
        value=gpu_available,
        disabled=not gpu_available,
        help="Whether to use GPU for feature engineering."
    )
    st.session_state["use_gpu_for_features"] = use_gpu_for_features and gpu_available
    
    use_gpu_for_training = st.sidebar.checkbox(
        "Use GPU for Model Training",
        value=gpu_available,
        disabled=not gpu_available,
        help="Whether to use GPU for model training."
    )
    st.session_state["use_gpu_for_training"] = use_gpu_for_training and gpu_available
    
    # MLflow settings
    st.sidebar.header("MLflow Settings")
    
    mlflow_enabled = st.sidebar.checkbox(
        "Enable MLflow Tracking",
        value=True,
        help="Whether to enable MLflow experiment tracking."
    )
    st.session_state["mlflow_enabled"] = mlflow_enabled
    
    if mlflow_enabled:
        experiment_name = st.sidebar.text_input(
            "Experiment Name",
            value="credit_risk_prediction",
            help="Name of the MLflow experiment."
        )
        st.session_state["experiment_name"] = experiment_name
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info(
        "This app demonstrates credit risk prediction using GPU-accelerated "
        "feature engineering and MLflow experiment tracking."
    )


def is_gpu_available():
    """Check if GPU is available."""
    try:
        # Try to create a GPUDataLoader with GPU backend
        loader = GPUDataLoader(backend=ProcessingBackend.GPU)
        return loader.is_gpu_available()
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {str(e)}")
        return False


def load_data(config, force_sample=None):
    """Load data based on the configuration.
    
    Args:
        config: The configuration object with data settings
        force_sample: If provided, overrides config.is_development_mode for sampling
    
    Returns:
        Dictionary of datasets
    """
    st.header("Data Loading")
    
    try:
        # Create a data loader with the specified backend
        loader = GPUDataLoader(config=config, backend=config.gpu.processing_backend)
        
        # Check if GPU is being used
        using_gpu = loader.is_gpu_available()
        if using_gpu:
            st.success(f"Using GPU for data loading with {loader.get_backend_name()} backend")
        else:
            st.warning(f"Using CPU for data loading with {loader.get_backend_name()} backend")
    except Exception as e:
        # If GPU initialization fails, force CPU backend
        logger.warning(f"GPU initialization failed: {str(e)}, falling back to CPU")
        st.warning(f"GPU initialization failed: {str(e)}, falling back to CPU")
        config.gpu.processing_backend = ProcessingBackend.CPU
        loader = GPUDataLoader(config=config, backend=ProcessingBackend.CPU)
    
    # Show loading progress
    progress_bar = st.progress(0)
    
    with st.spinner("Loading datasets..."):
        # Get list of datasets
        dataset_names = list(config.data.dataset_files.keys())
        
        # Determine if we should sample (use force_sample if provided, otherwise use config)
        should_sample = force_sample if force_sample is not None else config.is_development_mode
        
        # Log sampling information
        if should_sample:
            logger.info(f"Sampling enabled with {config.data.sample_size} rows per dataset")
            st.info(f"Loading with sampling: {config.data.sample_size} rows per dataset")
        else:
            logger.info("Loading full datasets (sampling disabled)")
            st.info("Loading full datasets (sampling disabled)")
        
        # Load each dataset with progress tracking
        datasets = {}
        total_rows = 0
        
        for i, dataset_name in enumerate(dataset_names):
            try:
                st.text(f"Loading {dataset_name}...")
                # Load the dataset, ensuring we use Streamlit-compatible optimization
                df = loader.load_dataset(
                    dataset_name,
                    to_pandas=True,  # Always convert to pandas for display
                    optimize_memory=True,
                    sample=should_sample,  # Use the determined sampling flag
                )
                
                # Store the dataset
                datasets[dataset_name] = df
                total_rows += len(df)
                
                # Update progress
                progress_bar.progress((i + 1) / len(dataset_names))
                st.text(f"Loaded {dataset_name}: {len(df):,} rows, {df.shape[1]} columns")
            except (FileNotFoundError, ValueError) as e:
                st.error(f"Error loading {dataset_name}: {str(e)}")
            except Exception as e:
                # If GPU processing fails, try again with CPU
                if config.gpu.processing_backend != ProcessingBackend.CPU:
                    st.warning(f"Error processing {dataset_name} with GPU: {str(e)}")
                    st.info("Falling back to CPU processing")
                    config.gpu.processing_backend = ProcessingBackend.CPU
                    loader.set_backend(ProcessingBackend.CPU)
                    try:
                        df = loader.load_dataset(
                            dataset_name,
                            to_pandas=True,
                            sample=should_sample,  # Use the determined sampling flag
                        )
                        datasets[dataset_name] = df
                        total_rows += len(df)
                        progress_bar.progress((i + 1) / len(dataset_names))
                        st.text(f"Loaded {dataset_name}: {len(df):,} rows, {df.shape[1]} columns")
                    except Exception as inner_e:
                        st.error(f"Error loading {dataset_name} with CPU fallback: {str(inner_e)}")
                else:
                    st.error(f"Error loading {dataset_name}: {str(e)}")
    
    progress_bar.progress(100)
    
    # Show dataset summary
    if datasets:
        st.success(f"Loaded {len(datasets)} datasets with {total_rows:,} total rows")
        
        # Create an expandable section for each dataset
        for name, df in datasets.items():
            with st.expander(f"{name} ({df.shape[0]} rows, {df.shape[1]} columns)"):
                st.dataframe(prepare_dataframe_for_streamlit(df.head()))
                
                # Show column types and missing values
                col_info = pd.DataFrame({
                    "Type": df.dtypes,
                    "Missing": df.isnull().sum(),
                    "Missing (%)": 100 * df.isnull().sum() / len(df),
                    "Unique Values": df.nunique(),
                })
                st.dataframe(prepare_dataframe_for_streamlit(col_info))
    
    return datasets


def run_feature_engineering(config, datasets):
    """Run feature engineering pipeline.
    
    Args:
        config: The configuration object
        datasets: Dictionary of datasets to use for feature engineering
        
    Returns:
        DataFrame with generated features or None if an error occurred
    """
    # Check if datasets are loaded
    if not datasets:
        logger.error("No datasets loaded for feature engineering")
        return None
    
    # Log dataset sizes
    total_rows = sum(len(df) for df in datasets.values())
    logger.info(f"Running feature engineering on {len(datasets)} datasets with {total_rows:,} total rows")
    
    for name, df in datasets.items():
        logger.info(f"Feature engineering dataset: {name} with {len(df):,} rows, {df.shape[1]} columns")
    
    # Preprocessor for data cleaning
    preprocessor = DataPreprocessor()
    
    # Get application data
    app_df = datasets.get("application_train")
    if app_df is None:
        logger.error("Application train dataset not found")
        return None
    
    # Preprocess data
    logger.info("Preprocessing data...")
    app_df = preprocessor.drop_low_importance_columns(app_df)
    app_df = preprocessor.handle_missing_values(app_df)
        
    # Update the dataset
    datasets["application_train"] = app_df
    
    # Create feature engineer with forced single process configuration
    try:
        # Force config to use single process
        config.features.n_jobs = 1
        
        # Disable GPU to prevent issues
        config.gpu.processing_backend = ProcessingBackend.CPU
        
        # Disable Dask distributed
        import os  # Import here to ensure it's available
        os.environ["FEATURETOOLS_NO_DASK"] = "1"
        
        engineer = FeatureEngineer(config=config)
    except Exception as e:
        logger.warning(f"Error initializing FeatureEngineer: {str(e)}, falling back to basic config")
        config.gpu.processing_backend = ProcessingBackend.CPU
        engineer = FeatureEngineer()
    
    # Generate manual features
    logger.info("Creating manual features...")
    
    # Try to get categorical columns for one-hot encoding
    cat_cols = app_df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Generate manual features
    try:
        manual_features = engineer.create_manual_features(app_df)
        logger.info(f"Created {manual_features.shape[1]} manual features")
    except Exception as e:
        logger.error(f"Error creating manual features: {str(e)}")
        manual_features = app_df.copy()
    
    # Generate automatic features using featuretools with single process mode
    logger.info(f"Generating features with max_depth={config.features.max_depth}, primitives={config.features.default_agg_primitives}")
    
    try:
        # Try with simplified parameters
        features = engineer.generate_features(
            datasets,
            # Force single process parameters
            n_jobs=1,
            chunk_size=None
        )
        
        # Check the shape of returned features
        if isinstance(features, tuple) and len(features) >= 1 and isinstance(features[0], pd.DataFrame):
            logger.info("Features were returned as a tuple, extracting DataFrame")
            features_df = features[0]
        elif isinstance(features, pd.DataFrame):
            features_df = features
        else:
            logger.error(f"Unexpected type returned from feature generation: {type(features)}")
            return None
        
        logger.info(f"Generated {features_df.shape[1]} features with featuretools")
        
        # Combine manual and automatic features
        final_features = pd.concat([manual_features, features_df], axis=1)
        
        # Remove duplicate columns
        final_features = final_features.loc[:, ~final_features.columns.duplicated()]
        logger.info(f"Final feature set: {final_features.shape[1]} features")
        
        # Convert COUNT columns to numeric types to avoid LightGBM errors
        count_columns = [col for col in final_features.columns if col.startswith('COUNT(')]
        if count_columns:
            logger.info(f"Converting {len(count_columns)} COUNT columns to numeric types")
            for col in count_columns:
                try:
                    # Convert to numeric, coerce errors to NaN
                    final_features[col] = pd.to_numeric(final_features[col], errors='coerce')
                    # Fill NaN values with 0
                    final_features[col] = final_features[col].fillna(0)
                    # Convert to int64 for cleaner representation
                    final_features[col] = final_features[col].astype('int64')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {str(e)}")
        
        # Ensure the TARGET column is preserved without NaNs
        if 'TARGET' in app_df.columns:
            # Check if TARGET is already in the final features
            if 'TARGET' in final_features.columns:
                # Check for NaN values
                nan_count = final_features['TARGET'].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in TARGET column after feature engineering")
                    
                    # Option 1: Drop rows with NaN in TARGET
                    # final_features = final_features.dropna(subset=['TARGET'])
                    
                    # Option 2: Use TARGET from original dataset to fill NaNs
                    # This ensures we keep all rows but have valid TARGET values
                    logger.info("Using original TARGET values from application_train")
                    # Ensure indices match
                    matching_indices = final_features.index.intersection(app_df.index)
                    final_features = final_features.loc[matching_indices]
                    final_features['TARGET'] = app_df.loc[matching_indices, 'TARGET']
            else:
                # If TARGET is not in final_features, add it from app_df
                logger.info("Adding TARGET column from application_train")
                # Get common indices
                matching_indices = final_features.index.intersection(app_df.index)
                final_features = final_features.loc[matching_indices]
                final_features['TARGET'] = app_df.loc[matching_indices, 'TARGET']
        
        logger.info(f"Final features shape: {final_features.shape}")
        return final_features
        
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}", exc_info=True)
        # Return manual features as a fallback
        logger.info("Returning only manual features due to error in feature generation")
        return manual_features


def show_mlflow_ui():
    """Show MLflow UI integration."""
    st.header("MLflow Experiment Tracking")
    
    try:
        # Get MLflow client
        client = MlflowClient()
        
        # Get experiment by name
        experiment_name = st.session_state.get("experiment_name", "credit_risk_prediction")
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            st.warning(f"No experiment found with name '{experiment_name}'")
            return
        
        # Get runs for the experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if not runs:
            st.warning("No runs found for this experiment")
            return
        
        # Show experiment info
        st.subheader(f"Experiment: {experiment_name}")
        st.text(f"Experiment ID: {experiment.experiment_id}")
        st.text(f"Artifact Location: {experiment.artifact_location}")
        st.text(f"Number of Runs: {len(runs)}")
        
        # Show runs in a table
        run_data = []
        for run in runs:
            run_data.append({
                "Run ID": run.info.run_id,
                "Status": run.info.status,
                "Start Time": pd.to_datetime(run.info.start_time, unit="ms"),
                "End Time": pd.to_datetime(run.info.end_time, unit="ms") if run.info.end_time else None,
                "Run Name": run.data.tags.get("mlflow.runName", ""),
                "Run Mode": run.data.tags.get("run_mode", ""),
                "Num Features": run.data.metrics.get("num_features", 0),
                "Execution Time (ms)": run.data.metrics.get("execution_time_ms", 0),
            })
        
        run_df = pd.DataFrame(run_data)
        st.dataframe(prepare_dataframe_for_streamlit(run_df))
        
        # Show details for a selected run
        if runs:
            st.subheader("Run Details")
            run_ids = [run.info.run_id for run in runs]
            run_names = [
                f"{run.data.tags.get('mlflow.runName', '')} ({run.info.run_id[:8]}...)"
                for run in runs
            ]
            
            selected_run_index = st.selectbox(
                "Select a run to view details",
                options=range(len(run_ids)),
                format_func=lambda i: run_names[i]
            )
            
            selected_run = runs[selected_run_index]
            
            # Show run info
            st.text(f"Run ID: {selected_run.info.run_id}")
            st.text(f"Status: {selected_run.info.status}")
            st.text(f"Start Time: {pd.to_datetime(selected_run.info.start_time, unit='ms')}")
            if selected_run.info.end_time:
                st.text(f"End Time: {pd.to_datetime(selected_run.info.end_time, unit='ms')}")
            
            # Show run parameters
            st.subheader("Parameters")
            param_df = pd.DataFrame({
                "Parameter": list(selected_run.data.params.keys()),
                "Value": list(selected_run.data.params.values())
            })
            st.dataframe(prepare_dataframe_for_streamlit(param_df))
            
            # Show run metrics
            st.subheader("Metrics")
            metric_df = pd.DataFrame({
                "Metric": list(selected_run.data.metrics.keys()),
                "Value": list(selected_run.data.metrics.values())
            })
            st.dataframe(prepare_dataframe_for_streamlit(metric_df))
            
            # Show link to MLflow UI
            tracking_uri = mlflow.get_tracking_uri()
            if tracking_uri.startswith("http"):
                mlflow_url = f"{tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{selected_run.info.run_id}"
                st.markdown(f"[View in MLflow UI]({mlflow_url})")
    
    except Exception as e:
        st.error(f"Error accessing MLflow: {str(e)}")


def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="Credit Risk Prediction",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("Credit Default Risk Prediction")
    st.markdown(
        "This application demonstrates credit risk prediction using GPU-accelerated "
        "feature engineering and MLflow experiment tracking."
    )
    
    # Check if GPU is available and print message
    gpu_available = is_gpu_available()
    if not gpu_available:
        st.warning("No GPU detected. Using CPU for all operations.")
        # Set default processing backend to CPU
        if "processing_backend" not in st.session_state:
            st.session_state["processing_backend"] = ProcessingBackend.CPU
    
    # Setup sidebar with configuration options
    setup_sidebar()
    
    # Get configuration based on user selections
    config = get_config()
    
    # If no GPU available, force CPU mode
    if not gpu_available:
        config.gpu.processing_backend = ProcessingBackend.CPU
        config.gpu.use_gpu_for_feature_engineering = False
        config.gpu.use_gpu_for_training = False
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Data Explorer", "Feature Engineering", "MLflow Dashboard"])
    
    # Store data in session state
    if "datasets" not in st.session_state:
        st.session_state["datasets"] = {}
    
    if "features_df" not in st.session_state:
        st.session_state["features_df"] = None
    
    # Data Explorer tab
    with tab1:
        # Button to load data
        if st.button("Load Data", key="load_data_button"):
            st.session_state["datasets"] = load_data(config)
        
        # Show data if already loaded
        if st.session_state["datasets"]:
            st.success(f"Data loaded: {len(st.session_state['datasets'])} datasets")
            
            # Show dataset summary
            for name, df in st.session_state["datasets"].items():
                with st.expander(f"{name} ({df.shape[0]} rows, {df.shape[1]} columns)"):
                    st.dataframe(prepare_dataframe_for_streamlit(df.head()))
    
    # Feature Engineering tab
    with tab2:
        # Button to run feature engineering
        if st.button("Run Feature Engineering", key="run_fe_button"):
            st.session_state["features_df"] = run_feature_engineering(
                config,
                st.session_state["datasets"]
            )
        
        # Show features if already generated
        if st.session_state["features_df"] is not None:
            features_df = st.session_state["features_df"]
            st.success(f"Features generated: {features_df.shape[1]} features")
            
            # Sample a subset of features for display
            if features_df.shape[1] > 20:
                sample_cols = list(features_df.columns[:20])
                st.dataframe(prepare_dataframe_for_streamlit(features_df[sample_cols].head()))
                st.info(f"Showing {len(sample_cols)} of {features_df.shape[1]} features")
            else:
                st.dataframe(prepare_dataframe_for_streamlit(features_df.head()))
    
    # MLflow Dashboard tab
    with tab3:
        show_mlflow_ui()


if __name__ == "__main__":
    main() 