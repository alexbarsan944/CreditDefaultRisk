import streamlit as st
import logging
from pathlib import Path
from credit_risk.config import RunMode
from credit_risk.utils.processing_strategy import ProcessingBackend

logger = logging.getLogger("credit_risk_utils")

def setup_sidebar(current_page=None):
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
    
    # Sample size (for development mode)
    sample_size = st.sidebar.number_input(
        "Sample Size",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="Number of rows to sample in development mode."
    )
    st.session_state["sample_size"] = sample_size
    
    # Processing backend
    backend_options = [backend.value for backend in ProcessingBackend]
    # Check if GPU is available
    gpu_available = True  # Simplified check, we'll assume GPU is available here
    # Default to GPU (index 1) when GPU is available, otherwise default to CPU (index 0)
    default_index = 1 if gpu_available else 0
    
    backend = st.sidebar.selectbox(
        "Processing Backend",
        options=backend_options,
        index=default_index,  # Default to GPU if available
        help="Backend to use for data processing."
    )
    backend = ProcessingBackend(backend)
    st.session_state["backend"] = backend
    
    # Data path
    raw_data_path = st.sidebar.text_input(
        "Raw Data Path",
        value="data/raw",
        help="Path to raw data files."
    )
    st.session_state["raw_data_path"] = raw_data_path
    
    # Return only backend and raw_data_path
    return backend, raw_data_path 