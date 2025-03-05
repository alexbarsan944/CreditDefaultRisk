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
    st.header("Feature Engineering")
    
    # Check if datasets are loaded
    if not datasets:
        st.error("No datasets loaded. Please load data first.")
        return None
    
    # Show dataset sizes being used
    total_rows = sum(len(df) for df in datasets.values())
    st.info(f"Running feature engineering on {len(datasets)} datasets with {total_rows:,} total rows")
    
    # Log dataset sizes
    for name, df in datasets.items():
        logger.info(f"Feature engineering dataset: {name} with {len(df):,} rows, {df.shape[1]} columns")
    
    # Preprocessor for data cleaning
    preprocessor = DataPreprocessor()
    
    # Get application data
    app_df = datasets.get("application_train")
    if app_df is None:
        st.error("Application train dataset not found")
        return None
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        st.text("Dropping low importance columns...")
        app_df = preprocessor.drop_low_importance_columns(app_df)
        
        st.text("Handling missing values...")
        app_df = preprocessor.handle_missing_values(app_df)
        
        # Update the dataset
        datasets["application_train"] = app_df
    
    # Create feature engineer
    try:
        engineer = FeatureEngineer(config=config)
    except Exception as e:
        logger.warning(f"Error initializing FeatureEngineer: {str(e)}, falling back to CPU")
        st.warning(f"Error initializing FeatureEngineer: {str(e)}, falling back to CPU")
        config.gpu.processing_backend = ProcessingBackend.CPU
        engineer = FeatureEngineer(config=config)
    
    # Add manual features
    with st.spinner("Creating manual features..."):
        st.text("Adding manual features...")
        app_df = engineer.create_manual_features(
            app_df=app_df,
            bureau_df=datasets.get("bureau"),
            previous_df=datasets.get("previous_application"),
            installments_df=datasets.get("installments_payments")
        )
        
        # Update the dataset
        datasets["application_train"] = app_df
        
        st.success(f"Created manual features. Total features: {app_df.shape[1]}")
    
    # Generate automated features
    with st.spinner("Generating automated features..."):
        st.text(f"Generating features with max_depth={config.features.max_depth}...")
        
        # Track with MLflow if enabled
        tracker = ExperimentTracker(config=config)
        
        with tracker.start_run(run_name="feature_engineering"):
            # Log run parameters
            tracker.log_params({
                "run_mode": config.run_mode.value,
                "sample_size": config.data.sample_size,
                "max_depth": config.features.max_depth,
                "backend": str(config.gpu.processing_backend),
            })
            
            # Generate features
            try:
                # Use configuration settings for performance
                chunk_size = config.features.chunk_size
                if chunk_size is None and config.data.sample_size > 10000:
                    chunk_size = 10000  # Default chunk size for large datasets
                
                features_result = engineer.generate_features(
                    datasets=datasets,
                    max_depth=config.features.max_depth,
                    verbose=True,
                    chunk_size=chunk_size,
                    n_jobs=config.features.n_jobs
                )
                
                # Handle the case where featuretools returns a tuple
                if isinstance(features_result, tuple):
                    logger.info("Features were returned as a tuple, extracting DataFrame")
                    features_df = features_result[0]  # Extract DataFrame from tuple
                else:
                    features_df = features_result
                
                # Log feature stats
                tracker.log_metrics({
                    "num_features": features_df.shape[1],
                    "num_samples": features_df.shape[0],
                })
                
                # Save features to feature store
                store = FeatureStore(config=config)
                feature_set_name = f"features_depth{config.features.max_depth}"
                store.save_feature_set(
                    features=features_df,
                    name=feature_set_name,
                    description=f"Features with max_depth={config.features.max_depth}",
                    tags=["web_app", config.run_mode.value],
                    metadata={
                        "max_depth": config.features.max_depth,
                        "run_mode": config.run_mode.value,
                    }
                )
                
                st.success(f"Generated {features_df.shape[1]} features")
                
                # Display detailed statistics
                with st.expander("Feature Engineering Statistics", expanded=True):
                    # Create two columns for stats
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Dataset Information")
                        st.write(f"Number of rows: {features_df.shape[0]}")
                        st.write(f"Total features: {features_df.shape[1]}")
                        
                        # Original vs new features - correctly identify app_train columns as original
                        app_train_df = datasets.get("application_train", pd.DataFrame())
                        
                        # To get the true original column count, we need to check the raw file
                        # as some columns might have been dropped during preprocessing
                        try:
                            raw_app_df = pd.read_csv(f"{config.data.raw_data_path}/application_train.csv")
                            original_cols = raw_app_df.shape[1]
                        except Exception as e:
                            # Fallback to the current dataframe if raw file can't be loaded
                            logger.warning(f"Could not load raw application_train.csv: {str(e)}")
                            original_cols = len(app_train_df.columns) if not app_train_df.empty else 0
                        
                        # Output the actual count
                        st.write(f"Original features: {original_cols}")
                        st.write(f"New features: {max(0, features_df.shape[1] - original_cols)}")
                    
                    with col2:
                        st.subheader("Feature Types")
                        # Count feature types
                        numeric_cols = len(features_df.select_dtypes(include=['number']).columns)
                        categorical_cols = len(features_df.select_dtypes(include=['category', 'object']).columns)
                        bool_cols = len(features_df.select_dtypes(include=['bool']).columns)
                        date_cols = len(features_df.select_dtypes(include=['datetime']).columns)
                        other_cols = features_df.shape[1] - numeric_cols - categorical_cols - bool_cols - date_cols
                        
                        st.write(f"Numeric features: {numeric_cols}")
                        st.write(f"Categorical features: {categorical_cols}")
                        st.write(f"Boolean features: {bool_cols}")
                        st.write(f"Date/time features: {date_cols}")
                        if other_cols > 0:
                            st.write(f"Other types: {other_cols}")
                    
                    # Memory usage information
                    st.subheader("Memory Usage")
                    memory_usage = features_df.memory_usage(deep=True).sum() / (1024 * 1024)
                    st.write(f"Feature matrix memory usage: {memory_usage:.2f} MB")
                
                # Sample a subset of features for display
                st.subheader("Feature Preview")
                if features_df.shape[1] > 20:
                    sample_cols = list(features_df.columns[:20])
                    st.dataframe(prepare_dataframe_for_streamlit(features_df[sample_cols].head()))
                    st.info(f"Showing {len(sample_cols)} of {features_df.shape[1]} features")
                else:
                    st.dataframe(prepare_dataframe_for_streamlit(features_df.head()))
                
                return features_df
                
            except Exception as e:
                logger.error(f"Error generating features: {str(e)}")
                st.error(f"Error generating features: {str(e)}")
                
                # Try again with CPU if we're not already using it
                if config.gpu.processing_backend != ProcessingBackend.CPU:
                    st.warning("Trying again with CPU backend...")
                    try:
                        config.gpu.processing_backend = ProcessingBackend.CPU
                        engineer = FeatureEngineer(config=config)
                        
                        # For CPU fallback, use conservative settings to avoid memory issues
                        reduced_n_jobs = 1  # Use single process to minimize memory usage
                        
                        # Calculate a reasonable chunk size based on dataset size
                        app_df = datasets.get("application_train")
                        if app_df is not None:
                            auto_chunk_size = min(5000, app_df.shape[0] // 2)
                        else:
                            auto_chunk_size = 5000
                            
                        logger.info(f"CPU fallback: Using n_jobs={reduced_n_jobs} and chunk_size={auto_chunk_size}")
                        
                        features_result = engineer.generate_features(
                            datasets=datasets,
                            max_depth=config.features.max_depth,
                            verbose=True,
                            chunk_size=auto_chunk_size,
                            n_jobs=reduced_n_jobs
                        )
                        
                        # Handle the case where featuretools returns a tuple
                        if isinstance(features_result, tuple):
                            logger.info("Features were returned as a tuple, extracting DataFrame")
                            features_df = features_result[0]  # Extract DataFrame from tuple
                        else:
                            features_df = features_result
                        
                        # Log feature stats
                        tracker.log_metrics({
                            "num_features": features_df.shape[1],
                            "num_samples": features_df.shape[0],
                        })
                        
                        st.success(f"Generated {features_df.shape[1]} features with CPU backend")
                        
                        # Display detailed statistics
                        with st.expander("Feature Engineering Statistics", expanded=True):
                            # Create two columns for stats
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Dataset Information")
                                st.write(f"Number of rows: {features_df.shape[0]}")
                                st.write(f"Total features: {features_df.shape[1]}")
                                
                                # Original vs new features - correctly identify app_train columns as original
                                app_train_df = datasets.get("application_train", pd.DataFrame())
                                
                                # To get the true original column count, we need to check the raw file
                                # as some columns might have been dropped during preprocessing
                                try:
                                    raw_app_df = pd.read_csv(f"{config.data.raw_data_path}/application_train.csv")
                                    original_cols = raw_app_df.shape[1]
                                except Exception as e:
                                    # Fallback to the current dataframe if raw file can't be loaded
                                    logger.warning(f"Could not load raw application_train.csv: {str(e)}")
                                    original_cols = len(app_train_df.columns) if not app_train_df.empty else 0
                                
                                # Output the actual count
                                st.write(f"Original features: {original_cols}")
                                st.write(f"New features: {max(0, features_df.shape[1] - original_cols)}")
                            
                            with col2:
                                st.subheader("Feature Types")
                                # Count feature types
                                numeric_cols = len(features_df.select_dtypes(include=['number']).columns)
                                categorical_cols = len(features_df.select_dtypes(include=['category', 'object']).columns)
                                bool_cols = len(features_df.select_dtypes(include=['bool']).columns)
                                date_cols = len(features_df.select_dtypes(include=['datetime']).columns)
                                other_cols = features_df.shape[1] - numeric_cols - categorical_cols - bool_cols - date_cols
                                
                                st.write(f"Numeric features: {numeric_cols}")
                                st.write(f"Categorical features: {categorical_cols}")
                                st.write(f"Boolean features: {bool_cols}")
                                st.write(f"Date/time features: {date_cols}")
                                if other_cols > 0:
                                    st.write(f"Other types: {other_cols}")
                            
                            # Memory usage information
                            st.subheader("Memory Usage")
                            memory_usage = features_df.memory_usage(deep=True).sum() / (1024 * 1024)
                            st.write(f"Feature matrix memory usage: {memory_usage:.2f} MB")
                        
                        # Sample a subset of features for display
                        st.subheader("Feature Preview")
                        if features_df.shape[1] > 20:
                            sample_cols = list(features_df.columns[:20])
                            st.dataframe(prepare_dataframe_for_streamlit(features_df[sample_cols].head()))
                            st.info(f"Showing {len(sample_cols)} of {features_df.shape[1]} features")
                        else:
                            st.dataframe(prepare_dataframe_for_streamlit(features_df.head()))
                        
                        return features_df
                    except Exception as inner_e:
                        logger.error(f"Error generating features with CPU fallback: {str(inner_e)}")
                        st.error(f"Error generating features with CPU fallback: {str(inner_e)}")
                        return None
                return None


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