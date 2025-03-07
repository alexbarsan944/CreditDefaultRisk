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
from typing import Union, Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)

# Create a directory for storing step data using absolute path
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
DATA_DIR = PROJECT_ROOT / "data" / "step_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Using data directory: {DATA_DIR.absolute()}")

def prepare_dataframe_for_streamlit(df):
    """
    Prepare a DataFrame for display in Streamlit, handling issues with PyArrow serialization.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to prepare
    
    Returns
    -------
    pd.DataFrame
        DataFrame ready for display in Streamlit
    """
    if df is None:
        return None
    
    try:
        # Try to make a copy to avoid modifying the original
        display_df = df.copy()
        
        # Handle mixed type columns that could cause Arrow serialization issues
        for col in display_df.columns:
            # For columns with mixed types (typically 'object' dtype)
            if display_df[col].dtype == 'object':
                # Check the first few non-null values to see if they're consistent
                non_null_values = display_df[col].dropna().head(5).tolist()
                if len(non_null_values) > 0:
                    # If first value is a number but the column has strings too, convert all to strings
                    if (isinstance(non_null_values[0], (int, float)) and 
                        any(isinstance(x, str) for x in non_null_values)):
                        display_df[col] = display_df[col].astype(str)
                    # Convert complex Python objects to strings
                    elif any(isinstance(x, (dict, list, tuple, set)) for x in non_null_values):
                        display_df[col] = display_df[col].apply(lambda x: str(x) if x is not None else x)
                # If all checks pass, still convert to string to be safe
                display_df[col] = display_df[col].astype(str)
        
        return display_df
    except Exception as e:
        logger.warning(f"Error preparing DataFrame for display: {str(e)}. Falling back to basic conversion.")
        try:
            # Fallback: convert everything to strings
            return df.astype(str)
        except:
            # Ultimate fallback: convert to dictionary and create new dataframe
            logger.warning("Final fallback: converting to dict and rebuilding dataframe")
            data = []
            for _, row in df.iterrows():
                data.append({col: str(val) for col, val in row.items()})
            return pd.DataFrame(data)

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
        If this is a timestamp string, it will find the file with that timestamp
        
    Returns:
    --------
    Any
        The loaded data
    """
    try:
        # Check if specific_file is a timestamp rather than a full path
        if specific_file and not os.path.exists(specific_file) and not specific_file.endswith('.pkl'):
            # Try to find file with matching timestamp in the filename
            logger.info(f"Looking for file with timestamp: {specific_file}")
            matching_files = list(DATA_DIR.glob(f"{step_name}_{specific_file}.pkl"))
            if matching_files:
                filepath = matching_files[0]
                logger.info(f"Found matching file: {filepath}")
            else:
                logger.error(f"File not found with timestamp: {specific_file}")
                return None
        elif specific_file:
            filepath = specific_file
            logger.info(f"Loading specific file: {filepath}")
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None
        elif "step_data_files" in st.session_state and step_name in st.session_state.step_data_files:
            filepath = st.session_state.step_data_files[step_name]
            logger.info(f"Loading from session state: {filepath}")
            if not os.path.exists(filepath):
                logger.error(f"File referenced in session state not found: {filepath}")
                # Remove invalid reference from session state
                del st.session_state.step_data_files[step_name]
                return None
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
            
            # Verify file is valid and not empty
            if os.path.getsize(filepath) == 0:
                logger.error(f"File is empty: {filepath.absolute()}")
                return None
        
        # Load the data with proper error handling
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            
            # Validate data structure based on step_name
            if step_name == "feature_selection":
                required_keys = ["selected_features", "X_train", "y_train", "X_test", "y_test"]
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    logger.error(f"Invalid feature_selection data: Missing keys {missing_keys}")
                    return None
                
                # Validate data types
                if not isinstance(data["X_train"], pd.DataFrame):
                    logger.error(f"Invalid X_train data type: {type(data['X_train'])}")
                    return None
                
                # Log data dimensions
                logger.info(f"Loaded X_train shape: {data['X_train'].shape}")
                logger.info(f"Loaded y_train shape: {len(data['y_train'])}")
            
            logger.info(f"Successfully loaded {step_name} data from {filepath}")
            return data
            
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"File is corrupted or invalid: {filepath}")
            logger.error(f"Pickle error: {str(e)}")
            return None
    
    except Exception as e:
        logger.error(f"Error loading {step_name} data: {str(e)}", exc_info=True)
        st.error(f"Error loading {step_name} data: {str(e)}")
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
        
        # Get all files in the data directory
        all_files = []
        try:
            all_files = os.listdir(DATA_DIR)
            logger.info(f"Total files in data directory: {len(all_files)}")
        except Exception as e:
            logger.error(f"Error listing directory {DATA_DIR}: {str(e)}")
            return result
        
        # First, look for .meta files - use consistent approach
        meta_files = sorted([f for f in all_files if f.endswith('.meta')])
        logger.info(f"Found {len(meta_files)} metadata files")
        
        # Process each metadata file to build result structure first
        # This way we can correctly count files per step_name
        step_counts = {}
        for meta_filename in meta_files:
            try:
                meta_file = os.path.join(DATA_DIR, meta_filename)
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
                    
                current_step_name = metadata.get("step_name")
                if current_step_name not in step_counts:
                    step_counts[current_step_name] = 0
                step_counts[current_step_name] += 1
                
                # Create entry for current_step_name if it doesn't exist
                if current_step_name not in result:
                    result[current_step_name] = []
                
                # Check if the data file exists
                data_file_path = metadata.get("filepath")
                file_exists = os.path.exists(data_file_path) if data_file_path else False
                
                # Add file_exists to metadata
                metadata["file_exists"] = file_exists
                
                # Log info about the data file
                data_file_size = os.path.getsize(data_file_path) if file_exists else 0
                logger.info(f"Metadata file: {meta_filename}, Data file exists: {file_exists}, File size: {data_file_size}, Step: {current_step_name}")
                
                # Append metadata to result
                result[current_step_name].append(metadata)
            except Exception as e:
                logger.error(f"Error processing metadata file {meta_filename}: {str(e)}")
        
        # Log counts for each step type
        for step, count in step_counts.items():
            logger.info(f"Found {count} {step} metadata files")
        
        # If step_name is provided, filter results
        if step_name and step_name in result:
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

def cleanup_corrupted_files(step_name: str = None) -> Tuple[int, int]:
    """
    Clean up corrupted data files by checking for invalid or incomplete files
    
    Parameters:
    -----------
    step_name : str, optional
        If provided, only clean files for this specific step
    
    Returns:
    --------
    Tuple[int, int]
        Number of files checked, number of files removed
    """
    checked = 0
    removed = 0
    
    try:
        # Get available data
        available_data = get_available_step_data()
        
        # Iterate through steps
        for current_step, metadata_list in available_data.items():
            # Skip if step_name is provided and doesn't match
            if step_name and current_step != step_name:
                continue
            
            # Check each file in the step
            for metadata in metadata_list:
                checked += 1
                
                # Check if the data file exists
                filepath = metadata.get("filepath")
                if not filepath or not os.path.exists(filepath):
                    logger.warning(f"File not found: {filepath}")
                    continue
                
                try:
                    # Try to open and load the file
                    with open(filepath, "rb") as f:
                        try:
                            # Try to load just a small part of the file without loading the whole thing
                            pickle.load(f)
                        except (pickle.UnpicklingError, EOFError) as e:
                            logger.error(f"Corrupted file detected: {filepath}")
                            logger.error(f"Pickle error: {str(e)}")
                            
                            # Move the file to a backup location instead of deleting
                            backup_dir = DATA_DIR / "corrupted"
                            backup_dir.mkdir(exist_ok=True)
                            
                            # Create backup filename
                            backup_filename = f"{Path(filepath).name}.corrupted"
                            backup_path = backup_dir / backup_filename
                            
                            try:
                                # Move the file
                                import shutil
                                shutil.move(filepath, backup_path)
                                logger.info(f"Moved corrupted file to: {backup_path}")
                                removed += 1
                            except Exception as move_error:
                                logger.error(f"Failed to move corrupted file: {str(move_error)}")
                except Exception as e:
                    logger.error(f"Error checking file {filepath}: {str(e)}")
        
        return checked, removed
    
    except Exception as e:
        logger.error(f"Error in cleanup_corrupted_files: {str(e)}", exc_info=True)
        return checked, removed

def load_most_recent_feature_selection() -> Any:
    """
    Load the most recent feature selection data from file.
    
    Returns:
    --------
    Any
        The loaded feature selection data or None if no data is found
    """
    try:
        # Check if DATA_DIR exists
        if not os.path.exists(DATA_DIR):
            logger.warning(f"Data directory does not exist: {DATA_DIR}")
            return None
        
        # Find all feature selection pkl files
        fs_files = [f for f in os.listdir(DATA_DIR) if f.startswith('feature_selection_') and f.endswith('.pkl')]
        if not fs_files:
            logger.warning("No feature selection files found")
            return None
        
        # Sort by modification time (most recent first)
        fs_files.sort(key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)), reverse=True)
        most_recent = fs_files[0]
        
        logger.info(f"Loading most recent feature selection file: {most_recent}")
        
        # Load the data
        filepath = os.path.join(DATA_DIR, most_recent)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        # Validate data format
        required_keys = ["selected_features", "X_train", "y_train", "X_test", "y_test"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            logger.error(f"Invalid feature selection data: Missing keys {missing_keys}")
            return None
        
        logger.info(f"Successfully loaded feature selection data: {len(data.get('selected_features', []))} features")
        return data
    
    except Exception as e:
        logger.error(f"Error loading feature selection data: {str(e)}", exc_info=True)
        return None

def get_available_data_files() -> List[Dict]:
    """
    Get all available data files across all steps.
    
    Returns:
    --------
    List[Dict]
        List of data file metadata sorted by timestamp (newest first)
    """
    all_files = []
    
    # Make sure the data directory exists
    if not DATA_DIR.exists():
        logger.warning(f"Data directory does not exist: {DATA_DIR}")
        return all_files
    
    # Find all metadata files
    logger.info(f"Searching for metadata files in {DATA_DIR}")
    all_metadata_files = list(DATA_DIR.glob("*.meta"))
    logger.info(f"Found {len(all_metadata_files)} metadata files")
    
    # Process each metadata file
    for meta_file in all_metadata_files:
        try:
            # Get corresponding data file path
            data_file = meta_file.with_suffix("")
            
            # Check if data file exists and get its size
            if data_file.exists():
                file_size = data_file.stat().st_size
                
                # Load metadata
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
                
                # Add file information
                all_files.append({
                    "path": str(data_file),
                    "size": file_size,
                    "timestamp": metadata.get("timestamp", "unknown"),
                    "step": metadata.get("step", "unknown"),
                    "description": metadata.get("description", "No description available")
                })
        except Exception as e:
            logger.error(f"Error processing metadata file {meta_file}: {str(e)}")
    
    # Sort by timestamp (newest first)
    all_files.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return all_files 