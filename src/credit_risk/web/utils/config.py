import streamlit as st
from credit_risk.config import PipelineConfig

def get_config():
    """
    Get the application configuration.
    
    Returns
    -------
    PipelineConfig
        Application configuration
    """
    # Create a default config
    config = PipelineConfig()
    
    # Return the config
    return config 