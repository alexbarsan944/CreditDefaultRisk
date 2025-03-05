"""
Utility functions for the Credit Risk Prediction web application.

This module contains shared functions used across different parts of the web app.
"""
import os
import logging
from typing import Tuple, Any, Optional

import streamlit as st

from credit_risk.config import PipelineConfig, RunMode, ProcessingBackend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("credit_risk_utils")


def set_page_config(page_title: str = "Credit Risk Prediction"):
    """
    Configure the Streamlit page.
    
    Parameters
    ----------
    page_title : str
        Title of the page
    """
    st.set_page_config(
        page_title=f"Credit Risk - {page_title}",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def setup_sidebar(current_page: str = None) -> Tuple[ProcessingBackend, str]:
    """
    Setup the sidebar with configuration options.
    
    Parameters
    ----------
    current_page : str, optional
        Name of the current page
        
    Returns
    -------
    Tuple[ProcessingBackend, str]
        Tuple containing (backend, raw_data_path)
    """
    st.sidebar.title("Configuration")
    
    # Show current page if provided
    if current_page:
        st.sidebar.markdown(f"**Current Page:** {current_page}")
    
    # Run mode
    run_mode = st.sidebar.selectbox(
        "Run Mode",
        options=[mode.value for mode in RunMode],
        index=0,  # Default to development mode
        help="Development mode uses a sample of the data for faster iteration."
    )
    st.session_state["run_mode"] = RunMode(run_mode)
    
    # Sample size has been moved to the Data Explorer tab
    # We'll keep the run_mode setting in the sidebar for global configuration
    
    # Data path
    raw_data_path = st.sidebar.text_input(
        "Data Directory",
        value="data/raw",
        help="Path to raw data files."
    )
    
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
    
    # Show the processing backend selection with appropriate default
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
    
    # Return backend and data path only
    return ProcessingBackend(processing_backend), raw_data_path


def is_gpu_available() -> bool:
    """
    Check if GPU is available.
    
    Returns
    -------
    bool
        True if GPU is available, False otherwise
    """
    try:
        # Try to create a GPUDataLoader with GPU backend
        from credit_risk.data.gpu_loader import GPUDataLoader
        loader = GPUDataLoader(backend=ProcessingBackend.GPU)
        return loader.is_gpu_available()
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {str(e)}")
        return False


def get_config() -> PipelineConfig:
    """
    Get configuration based on user selections in the sidebar.
    
    Returns
    -------
    PipelineConfig
        Configuration object
    """
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


def downcast_dtypes(df):
    """
    Downcast data types to reduce memory usage.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize
        
    Returns
    -------
    pd.DataFrame
        Optimized DataFrame
    """
    import numpy as np
    import pandas as pd
    
    # Vectorized approach for downcasting data types to reduce memory usage
    # Handle float columns
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # Handle integer columns
    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        # Ensure no integer overflow during downcast
        c_min = df[col].min()
        c_max = df[col].max()
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

    return df 