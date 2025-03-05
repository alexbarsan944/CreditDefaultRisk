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
    st.write(
        """
        This tab allows you to perform feature selection using the null importance method.
        The null importance method identifies important features by comparing their importance in the original data
        with their importance when the target variable is randomly permuted.
        """
    )
    
    # Check if data is loaded
    if data is None or data.empty:
        st.warning("No data loaded. Please load data first.")
        return
    
    # Save button for original data
    st.sidebar.subheader("Save/Load Data")
    save_col1, save_col2 = st.sidebar.columns(2)
    
    with save_col1:
        if st.button("💾 Save Input Data", key="save_input_data"):
            with st.spinner("Saving input data..."):
                try:
                    description = f"Input data with {data.shape[1]} features for feature selection"
                    filepath = save_step_data("feature_engineering", data, description)
                    if filepath:
                        st.success(f"Input data saved successfully! Path: {filepath}")
                        logger.info(f"Input data saved to {filepath}")
                    else:
                        st.error("Failed to save input data. Check logs for details.")
                        logger.error("Failed to save input data")
                except Exception as e:
                    st.error(f"Error saving input data: {str(e)}")
                    logger.error(f"Error saving input data: {str(e)}", exc_info=True)
    
    # Load data from previous steps
    try:
        available_data = get_available_step_data()
        if "feature_engineering" in available_data and available_data["feature_engineering"]:
            st.sidebar.subheader("Load Previous Data")
            data_options = [f"{item['timestamp']} - {item['description']}" for item in available_data["feature_engineering"]]
            
            if data_options:
                selected_data_idx = st.sidebar.selectbox(
                    "Select data to load:", 
                    range(len(data_options)), 
                    format_func=lambda i: data_options[i]
                )
                
                if st.sidebar.button("📂 Load Selected Data"):
                    with st.spinner("Loading data..."):
                        try:
                            selected_metadata = available_data["feature_engineering"][selected_data_idx]
                            if "filepath" in selected_metadata:
                                filepath = selected_metadata["filepath"]
                            else:
                                filepath = str(DATA_DIR / selected_metadata["filename"])
                                
                            logger.info(f"Attempting to load data from {filepath}")
                            loaded_data = load_step_data("feature_engineering", specific_file=filepath)
                            
                            if loaded_data is not None:
                                data = loaded_data
                                st.sidebar.success(f"Data loaded successfully! {data.shape[0]} rows, {data.shape[1]} columns")
                                st.rerun()
                            else:
                                st.sidebar.error("Failed to load data. Check logs for details.")
                        except Exception as e:
                            st.sidebar.error(f"Error loading data: {str(e)}")
                            logger.error(f"Error loading data: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Error getting available data: {str(e)}", exc_info=True)
        st.sidebar.error(f"Error retrieving saved data: {str(e)}")
    
    # Check if TARGET column exists
    if "TARGET" not in data.columns:
        st.error("TARGET column not found in the data. Please make sure your data contains a TARGET column.")
        return
    
    st.write(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    
    # Settings for feature selection
    st.subheader("Feature Selection Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test set size (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Percentage of data to use for testing",
            key="feature_selection_test_size"
        ) / 100
        
        n_runs = st.slider(
            "Number of null importance runs",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="More runs give more stable results but take longer",
            key="feature_selection_n_runs"
        )
        
        # Add GPU options
        use_gpu = st.checkbox(
            "Use GPU for XGBoost (if available)",
            value=True,
            help="Enable GPU acceleration for faster feature selection",
            key="feature_selection_use_gpu"
        )
    
    with col2:
        use_auto_thresholds = st.checkbox(
            "Use automatic threshold detection",
            value=True,
            help="Automatically determine optimal thresholds for feature selection",
            key="feature_selection_auto_thresholds"
        )
        
        if not use_auto_thresholds:
            split_score_threshold = st.slider(
                "Split score threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.05,
                help="Higher values keep fewer features (more strict)",
                key="feature_selection_split_score_threshold"
            )
            
            gain_score_threshold = st.slider(
                "Gain score threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.05,
                help="Higher values keep fewer features (more strict)",
                key="feature_selection_gain_score_threshold"
            )
            
            correlation_threshold = st.slider(
                "Correlation threshold",
                min_value=0.70,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Features with correlation above this value will be removed",
                key="feature_selection_correlation_threshold"
            )
        else:
            st.info("Thresholds will be determined automatically based on the feature importance distributions.")
            # Set thresholds to None to trigger automatic detection
            split_score_threshold = None
            gain_score_threshold = None
            correlation_threshold = None
    
    random_state = st.number_input(
        "Random seed",
        min_value=1,
        max_value=100000,
        value=42,
        help="Random seed for reproducibility",
        key="feature_selection_random_state"
    )
    
    # Run button
    if st.button("Run Feature Selection", type="primary", key="run_feature_selection"):
        # Split features and target
        X = data.drop(columns=["TARGET"])
        y = data["TARGET"]
        
        # Log data information for debugging
        st.write(f"Data shape: {X.shape}")
        st.write(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        st.write(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Create progress bar
        progress_bar = st.progress(0, "Starting feature selection...")
        
        try:
            start_time = time.time()
            
            # Log that we're running feature selection with GPU if enabled
            if use_gpu:
                st.info("Running feature selection with GPU support (if available)")
                logger.info("Running feature selection with GPU support")
            else:
                st.info("Running feature selection without GPU support")
                logger.info("Running feature selection without GPU support")
            
            # Perform feature selection
            X_train_selected, feature_selector, results = perform_feature_selection(
                X_train, 
                y_train,
                n_runs=n_runs,
                split_score_threshold=split_score_threshold,
                gain_score_threshold=gain_score_threshold,
                correlation_threshold=correlation_threshold,
                random_state=random_state,
                progress_bar=progress_bar,
                use_gpu=use_gpu
            )
            
            X_test_selected = feature_selector.transform(X_test)
            
            elapsed_time = time.time() - start_time
            
            st.success(f"Feature selection completed in {elapsed_time:.2f} seconds!")
            
            # Show thresholds used (automatically determined or manually set)
            if use_auto_thresholds:
                st.info(f"""
                Automatically determined thresholds:
                - Split score threshold: {feature_selector.split_score_threshold_:.3f}
                - Gain score threshold: {feature_selector.gain_score_threshold_:.3f}
                - Correlation threshold: {feature_selector.correlation_threshold_:.3f}
                """)
            
            # Store results in session state for use in model training
            st.session_state.feature_selection_results = {
                "X_train": X_train_selected,
                "X_test": X_test_selected,
                "y_train": y_train,
                "y_test": y_test,
                "feature_selector": feature_selector,
                "feature_scores": results["feature_scores"],
                "original_features": results["original_features"],
                "selected_features": results["selected_features"]
            }
            
            # Display results
            st.subheader("Feature Selection Results")
            
            # Show metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Original Features", 
                    f"{results['original_features']}",
                    help="Number of features before selection"
                )
            
            with col2:
                st.metric(
                    "Selected Features", 
                    f"{results['selected_features']}",
                    f"-{results.get('reduction_percentage', 0.0):.1f}%",
                    help="Number of features after selection"
                )
            
            with col3:
                percent_kept = 100 - results.get('reduction_percentage', 0.0)
                st.metric(
                    "Features Kept", 
                    f"{percent_kept:.1f}%",
                    help="Percentage of original features kept"
                )
            
            # Create feature score plots
            plots = create_feature_score_plots(results["feature_scores"])
            
            # Display plots in tabs
            plot_tabs = st.tabs(["Top Features", "Bottom Features", "Score Distributions"])
            
            with plot_tabs[0]:
                st.subheader("Top Features by Score")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plots["top_split"], use_container_width=True)
                
                with col2:
                    st.plotly_chart(plots["top_gain"], use_container_width=True)
            
            with plot_tabs[1]:
                st.subheader("Bottom Features by Score")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plots["bottom_split"], use_container_width=True)
                
                with col2:
                    st.plotly_chart(plots["bottom_gain"], use_container_width=True)
            
            with plot_tabs[2]:
                st.subheader("Score Distributions")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plots["split_distribution"], use_container_width=True)
                
                with col2:
                    st.plotly_chart(plots["gain_distribution"], use_container_width=True)
            
            # Display feature selection results
            with st.expander("Selected Features List", expanded=False):
                st.write(f"Selected {len(feature_selector.get_useful_features())} features out of {X_train.shape[1]} original features")
                
                # Create a more informative selected features dataframe
                selected_features = feature_selector.get_useful_features()
                selected_features_df = results["feature_scores"][results["feature_scores"]["feature"].isin(selected_features)]
                selected_features_df = selected_features_df.sort_values("split_score", ascending=False)
                
                # Add an index column to display rank
                selected_features_df = selected_features_df.reset_index(drop=True)
                selected_features_df.index = selected_features_df.index + 1
                selected_features_df = selected_features_df.rename_axis("Rank")
                
                st.dataframe(prepare_dataframe_for_streamlit(selected_features_df))
            
            # Display feature scores
            with st.expander("All Feature Scores", expanded=False):
                all_scores_df = results["feature_scores"].sort_values("split_score", ascending=False).reset_index(drop=True)
                all_scores_df.index = all_scores_df.index + 1
                all_scores_df = all_scores_df.rename_axis("Rank")
                st.dataframe(prepare_dataframe_for_streamlit(all_scores_df))
            
            # Save feature selection results
            col1, col2 = st.columns(2)
            
            with col1:
                save_button = st.button("💾 Save Selected Features", 
                              key="save_feature_selection",
                              help="Save the selected features for use in model training")
                
                if save_button:
                    with st.spinner("Saving selected features..."):
                        try:
                            # Save the entire dataset with only selected features and target
                            selected_features = feature_selector.get_useful_features()
                            selected_data = data[selected_features + ["TARGET"]]
                            
                            # Save data with explicit step name
                            description = f"Selected {len(selected_features)} features out of {X_train.shape[1]} original features"
                            
                            # Explicitly print what we're trying to save
                            logger.info(f"Attempting to save feature selection data with {len(selected_features)} features")
                            
                            # Prepare the data to save
                            save_data = {
                                "data": selected_data,
                                "feature_selector": feature_selector,
                                "results": results,
                                "X_train": X_train[selected_features],
                                "X_test": X_test[selected_features],
                                "y_train": y_train,
                                "y_test": y_test,
                                "selected_features": selected_features,
                                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                            }
                            
                            # Log the keys in the save_data for debugging
                            logger.info(f"Save data contains keys: {list(save_data.keys())}")
                            
                            # Check if the data directory exists
                            if not os.path.exists(DATA_DIR):
                                logger.error(f"Data directory does not exist: {DATA_DIR}")
                                os.makedirs(DATA_DIR, exist_ok=True)
                                logger.info(f"Created data directory: {DATA_DIR}")
                            else:
                                logger.info(f"Data directory exists: {DATA_DIR}")
                            
                            # Debug the filepath before saving
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            expected_filename = f"feature_selection_{timestamp}.pkl"
                            expected_filepath = os.path.join(DATA_DIR, expected_filename)
                            logger.info(f"Expected filepath for saving: {expected_filepath}")
                            
                            # Save to feature_selection step (NOT feature_engineering)
                            logger.info(f"About to save feature_selection data with description: '{description}'")
                            filepath = save_step_data("feature_selection", save_data, description)
                            logger.info(f"Returned filepath after save_step_data: {filepath}")
                            
                            if filepath:
                                # Verify the file actually exists
                                if os.path.exists(filepath):
                                    file_size = os.path.getsize(filepath)
                                    logger.info(f"Verified file exists: {filepath} (Size: {file_size} bytes)")
                                    
                                    # Also verify the metadata file
                                    metadata_path = f"{filepath}.meta"
                                    if os.path.exists(metadata_path):
                                        metadata_size = os.path.getsize(metadata_path)
                                        logger.info(f"Verified metadata file exists: {metadata_path} (Size: {metadata_size} bytes)")
                                        
                                        # Read the metadata to confirm it's correct
                                        try:
                                            import json
                                            with open(metadata_path, 'r') as f:
                                                metadata = json.load(f)
                                            logger.info(f"Successfully read metadata: {metadata}")
                                            
                                            # Check available data after saving
                                            available_data_after_save = get_available_step_data()
                                            if "feature_selection" in available_data_after_save:
                                                logger.info(f"Feature selection data found in available_data after saving!")
                                                logger.info(f"Found {len(available_data_after_save['feature_selection'])} feature selection entries")
                                            else:
                                                logger.warning("Feature selection not found in available_data even after saving!")
                                                
                                        except Exception as e:
                                            logger.error(f"Error reading metadata: {str(e)}", exc_info=True)
                                    else:
                                        logger.error(f"Metadata file does NOT exist: {metadata_path}")
                                    
                                    st.success(f"Selected features saved successfully as feature_selection data!")
                                    st.info(f"You can now go to the Model Training tab to train a model using these features.")
                                    logger.info(f"Feature selection results saved to {filepath}")
                                else:
                                    logger.error(f"File does NOT exist after save: {filepath}")
                                    st.error("Failed to save feature selection results. File not created.")
                            else:
                                logger.error("Failed to save feature selection results. No filepath returned.")
                                st.error("Failed to save feature selection results.")
                        except Exception as e:
                            st.error(f"Error saving selected features: {str(e)}")
                            logger.error(f"Error saving feature selection results: {str(e)}", exc_info=True)
                            
            with col2:
                st.markdown("""
                ⚠️ **Important**: After saving the selected features, go to the 
                **Model Training** tab to train a model using these features.
                """)
            
            # Next steps guidance
            st.info("🔍 **Next Steps**: Proceed to the Model Training tab to train a model using these selected features. Your feature selection results have been saved.")
            
        except Exception as e:
            st.error(f"Error during feature selection: {str(e)}")
            logger.error(f"Feature selection error: {str(e)}", exc_info=True)
    
    # If we already have feature selection results, show a summary
    elif "feature_selection_results" in st.session_state:
        results = st.session_state.feature_selection_results
        
        st.success(f"Feature selection completed previously.")
        
        # Display results summary
        st.subheader("Feature Selection Results")
        
        # Show metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Original Features", 
                f"{results['original_features']}",
                help="Number of features before selection"
            )
        
        with col2:
            st.metric(
                "Selected Features", 
                f"{results['selected_features']}",
                f"-{results.get('reduction_percentage', 0.0):.1f}%",
                help="Number of features after selection"
            )
        
        with col3:
            percent_kept = 100 - results.get('reduction_percentage', 0.0)
            st.metric(
                "Features Kept", 
                f"{percent_kept:.1f}%",
                help="Percentage of original features kept"
            )
        
        # Show button to view detailed results
        if st.button("View Detailed Results", key="view_fs_details"):
            # Create feature score plots
            plots = create_feature_score_plots(results["feature_scores"])
            
            # Display plots in tabs
            plot_tabs = st.tabs(["Top Features", "Bottom Features", "Score Distributions"])
            
            with plot_tabs[0]:
                st.subheader("Top Features by Score")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plots["top_split"], use_container_width=True)
                
                with col2:
                    st.plotly_chart(plots["top_gain"], use_container_width=True)
            
            with plot_tabs[1]:
                st.subheader("Bottom Features by Score")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plots["bottom_split"], use_container_width=True)
                
                with col2:
                    st.plotly_chart(plots["bottom_gain"], use_container_width=True)
            
            with plot_tabs[2]:
                st.subheader("Score Distributions")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plots["split_distribution"], use_container_width=True)
                
                with col2:
                    st.plotly_chart(plots["gain_distribution"], use_container_width=True)
            
            # Show list of selected features
            with st.expander("Selected Features"):
                st.dataframe(pd.DataFrame({"feature": results["feature_selector"].get_useful_features()})) 