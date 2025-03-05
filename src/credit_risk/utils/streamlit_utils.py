"""
Utility functions for Streamlit integration in Credit Risk Prediction.
"""
import logging
import pandas as pd
import numpy as np
import base64
import json
import pickle
import os
from typing import Union, Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)

# Create a directory for storing step data using absolute path
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
DATA_DIR = PROJECT_ROOT / "data" / "step_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Using data directory: {DATA_DIR.absolute()}")

def prepare_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for display in Streamlit
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to prepare
    
    Returns:
    --------
    pd.DataFrame
        DataFrame prepared for Streamlit display
    """
    # Make a copy to avoid modifying the original
    df_streamlit = df.copy()
    
    # For each column, handle special data types for better display
    for col in df_streamlit.columns:
        # Handle pandas extension dtypes that can cause issues with Arrow
        if hasattr(df_streamlit[col].dtype, 'name'):
            dtype_name = df_streamlit[col].dtype.name
            
            # Convert pandas extension dtypes to standard types
            if 'Int' in dtype_name or 'UInt' in dtype_name:
                df_streamlit[col] = df_streamlit[col].astype('float64')
            elif 'string' in dtype_name:
                df_streamlit[col] = df_streamlit[col].astype('object')
                
        # If column contains objects, check if they need conversion
        if df_streamlit[col].dtype == 'object':
            # Sample first non-null value to check type
            non_null_vals = df_streamlit[col].dropna()
            sample = non_null_vals.iloc[0] if not non_null_vals.empty else None
            
            # Convert complex Python objects to strings for display
            if isinstance(sample, (list, dict, tuple, set, np.ndarray)):
                df_streamlit[col] = df_streamlit[col].apply(lambda x: str(x) if x is not None else x)
    
    # For columns that are dtype objects but appear to be all strings/numbers,
    # try to convert to appropriate types
    for col in df_streamlit.select_dtypes(include=['object']).columns:
        try:
            # Try to convert to numeric if appropriate
            pd.to_numeric(df_streamlit[col], errors='raise')
            df_streamlit[col] = pd.to_numeric(df_streamlit[col], errors='coerce')
        except (ValueError, TypeError):
            # If conversion to numeric fails, just leave as object
            pass
    
    # Reset index for better display
    if not df_streamlit.index.equals(pd.RangeIndex(len(df_streamlit))):
        df_streamlit = df_streamlit.reset_index()
    
    return df_streamlit 

def create_download_link(object_to_download, download_filename, download_link_text):
    """
    Generate a link to download the object
    
    Parameters:
    -----------
    object_to_download : Any
        Object to be downloaded
    download_filename : str
        Filename for the download
    download_link_text : str
        Text for the download link
    
    Returns:
    --------
    str
        HTML link to download the object
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    elif isinstance(object_to_download, dict):
        object_to_download = json.dumps(object_to_download, indent=4)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    else:
        b64 = base64.b64encode(pickle.dumps(object_to_download)).decode()
        href = f'<a href="data:file/pkl;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    
    return href

def save_step_data(step_name: str, data: Union[pd.DataFrame, Dict, Any], description: str = None) -> str:
    """
    Save data from a processing step to disk for persistence across tabs
    
    Parameters:
    -----------
    step_name : str
        Name of the processing step (e.g., 'feature_engineering', 'feature_selection')
    data : Union[pd.DataFrame, Dict, Any]
        Data to be saved
    description : str, optional
        Description of the saved data
        
    Returns:
    --------
    str
        Path where the data was saved
    """
    try:
        # Make sure the DATA_DIR exists
        if not os.path.exists(DATA_DIR):
            logger.info(f"Creating data directory: {DATA_DIR}")
            os.makedirs(DATA_DIR, exist_ok=True)
        else:
            logger.info(f"Data directory already exists: {DATA_DIR}")
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step_name}_{timestamp}.pkl"
        filepath = DATA_DIR / filename
        
        # Extra logging for feature_selection
        if step_name == "feature_selection":
            logger.info(f"Saving feature_selection data with timestamp {timestamp}")
            logger.info(f"Save path: {filepath.absolute()}")
            
            # Add more detailed logging for debugging
            if isinstance(data, dict):
                logger.info(f"Feature selection data keys: {list(data.keys())}")
                if "feature_selector" in data:
                    logger.info(f"Feature selector has {len(data['feature_selector'].get_useful_features())} useful features")
                if "X_train" in data:
                    logger.info(f"X_train shape: {data['X_train'].shape}")
        
        # Create metadata
        metadata = {
            "step_name": step_name,
            "description": description or f"Data from {step_name} step",
            "timestamp": timestamp,
            "filename": filename,
            "filepath": str(filepath.absolute())
        }
        
        # Save data
        logger.info(f"Attempting to save data to {filepath.absolute()}")
        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Pickle dump completed for {step_name} data")
        except Exception as e:
            logger.error(f"Error during pickle dump: {str(e)}", exc_info=True)
            return None
        
        # Verify the file was saved
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            logger.info(f"Verified file was created: {filepath.absolute()} (Size: {file_size} bytes)")
            if file_size == 0:
                logger.error(f"File was created but is empty: {filepath.absolute()}")
                return None
        else:
            logger.error(f"File was not created: {filepath.absolute()}")
            return None
        
        # Save metadata
        metadata_file = DATA_DIR / f"{filename}.meta"
        logger.info(f"Saving metadata to {metadata_file.absolute()}")
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata JSON dump completed")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}", exc_info=True)
            return None
        
        # Verify metadata file was saved
        if os.path.exists(metadata_file):
            logger.info(f"Verified metadata file was created: {metadata_file.absolute()}")
        else:
            logger.error(f"Metadata file was not created: {metadata_file.absolute()}")
            return None
        
        # Store the most recent file for each step type
        if "step_data_files" not in st.session_state:
            st.session_state.step_data_files = {}
        
        st.session_state.step_data_files[step_name] = str(filepath.absolute())
        
        logger.info(f"Successfully saved {step_name} data to {filepath.absolute()}")
        return str(filepath.absolute())
    
    except Exception as e:
        logger.error(f"Error saving {step_name} data: {str(e)}", exc_info=True)
        return None

def load_step_data(step_name: str, specific_file: str = None) -> Any:
    """
    Load data from a previous processing step
    
    Parameters:
    -----------
    step_name : str
        Name of the processing step to load data from
    specific_file : str, optional
        Path to a specific file to load, if None, load the most recent file for the step
        
    Returns:
    --------
    Any
        The loaded data
    """
    try:
        if specific_file:
            filepath = specific_file
            logger.info(f"Loading specific file: {filepath}")
        elif "step_data_files" in st.session_state and step_name in st.session_state.step_data_files:
            filepath = st.session_state.step_data_files[step_name]
            logger.info(f"Loading from session state: {filepath}")
        else:
            # Find the most recent file for this step
            files = list(DATA_DIR.glob(f"{step_name}_*.pkl"))
            if not files:
                logger.warning(f"No saved data found for step: {step_name}")
                return None
            
            # Sort by modification time (most recent first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            filepath = files[0]
            logger.info(f"Loading most recent file: {filepath.absolute()}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        # Load the data
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        logger.info(f"Successfully loaded {step_name} data from {filepath}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading {step_name} data: {str(e)}", exc_info=True)
        st.error(f"Error loading data: {str(e)}")
        return None

def get_available_step_data(step_name: str = None) -> Dict:
    """
    Get a list of all available saved step data, including metadata about the files
    
    Parameters:
    -----------
    step_name : str, optional
        If provided, only return data for this specific step
    
    Returns:
    --------
    Dict
        Dictionary with step names as keys and lists of metadata as values
    """
    result = {}
    try:
        # Check if DATA_DIR exists
        if not os.path.exists(DATA_DIR):
            logger.warning(f"Data directory does not exist: {DATA_DIR}")
            return result
            
        # List all metadata files
        logger.info(f"Searching for metadata files in {DATA_DIR}")
        
        # Get all meta files using os.listdir for more reliable file listing
        all_files = os.listdir(DATA_DIR)
        meta_files = [f for f in all_files if f.endswith('.meta')]
        logger.info(f"Found {len(meta_files)} metadata files")
        
        # If step_name is provided, filter files by step name
        if step_name:
            meta_files = [f for f in meta_files if f.startswith(f'{step_name}_')]
            logger.info(f"Found {len(meta_files)} {step_name} metadata files")
        
        # Specifically check for feature_selection files (for logging/debugging)
        feature_selection_files = [f for f in meta_files if f.startswith('feature_selection_')]
        logger.info(f"Found {len(feature_selection_files)} feature_selection metadata files")
        
        if feature_selection_files:
            # List the files for debugging
            for file in feature_selection_files:
                logger.info(f"Feature selection metadata file: {file}")
                
        # Process all metadata files
        for meta_filename in meta_files:
            try:
                meta_file = os.path.join(DATA_DIR, meta_filename)
                # Read metadata file
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
                    
                current_step_name = metadata.get("step_name")
                logger.info(f"Processing metadata for step: {current_step_name}, file: {meta_filename}")
                
                # Skip if we're filtering by step_name and this doesn't match
                if step_name and current_step_name != step_name:
                    continue
                
                # Create entry for current_step_name if it doesn't exist
                if current_step_name not in result:
                    result[current_step_name] = []
                    
                # Check if the data file exists
                data_file_path = metadata.get("filepath")
                file_exists = os.path.exists(data_file_path) if data_file_path else False
                
                # Add file_exists to metadata
                metadata["file_exists"] = file_exists
                
                # Log more detailed info about the file
                logger.info(f"Metadata file: {meta_filename}, Data file exists: {file_exists}, Step: {current_step_name}")
                
                if current_step_name == "feature_selection":
                    if file_exists:
                        file_size = os.path.getsize(data_file_path)
                        logger.info(f"Feature selection file exists at {data_file_path} with size {file_size} bytes")
                    else:
                        logger.warning(f"Feature selection data file does NOT exist at {data_file_path}")
                
                # Append metadata to result
                result[current_step_name].append(metadata)
            
            except json.JSONDecodeError:
                logger.error(f"Error decoding metadata file {meta_filename}")
            except Exception as e:
                logger.error(f"Error processing metadata file {meta_filename}: {str(e)}", exc_info=True)
        
        # Sort metadata by timestamp (most recent first)
        for step in result:
            result[step] = sorted(result[step], key=lambda x: x.get("timestamp", ""), reverse=True)
            
        # Check for feature_selection results
        if "feature_selection" in result and result["feature_selection"]:
            logger.info(f"Found {len(result['feature_selection'])} feature_selection results")
            # Log details of each result
            for i, meta in enumerate(result["feature_selection"]):
                logger.info(f"Feature selection result {i+1}: {meta.get('timestamp')} - {meta.get('description')}")
        elif step_name == "feature_selection":
            logger.warning(f"No feature_selection results found")
            
        # If step_name is provided, convert the result to a more convenient format
        if step_name and step_name in result:
            # Convert to a dictionary mapping timestamp to description
            simplified_result = {}
            for meta in result[step_name]:
                timestamp = meta.get("timestamp", "")
                description = meta.get("description", f"{step_name} data")
                if timestamp:
                    simplified_result[timestamp] = description
            return simplified_result
            
        return result
    except Exception as e:
        logger.error(f"Error in get_available_step_data: {str(e)}", exc_info=True)
        return {} 