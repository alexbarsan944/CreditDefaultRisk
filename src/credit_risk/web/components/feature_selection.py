import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import the necessary modules
from credit_risk.web.model_utils import (
    perform_feature_selection,
    create_feature_score_plots
)
from credit_risk.web.app import (
    load_data,
    is_gpu_available,
    set_page_config,
    setup_sidebar
)

# Set page config
set_page_config("Feature Selection")

def main():
    """Main function for the Feature Selection page."""
    
    # Set up sidebar
    backend, raw_data_path, current_step = setup_sidebar(current_page="Feature Selection")
    
    # Display page header
    st.title("Credit Default Risk - Feature Selection")
    st.write(
        """
        This page allows you to perform feature selection using the null importance method.
        The null importance method identifies important features by comparing their importance in the original data
        with their importance when the target variable is randomly permuted.
        """
    )
    
    # Check if data is loaded
    try:
        if "data" not in st.session_state:
            with st.spinner("Loading data..."):
                st.session_state.data = load_data(raw_data_path, backend=backend)
        
        data = st.session_state.data
        st.write(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
        
        # Settings for feature selection
        st.header("Feature Selection Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider(
                "Test set size (%)",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="Percentage of data to use for testing"
            ) / 100
            
            n_runs = st.slider(
                "Number of null importance runs",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="More runs give more stable results but take longer"
            )
        
        with col2:
            split_score_threshold = st.slider(
                "Split score threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.05,
                help="Higher values keep fewer features (more strict)"
            )
            
            gain_score_threshold = st.slider(
                "Gain score threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.05,
                help="Higher values keep fewer features (more strict)"
            )
            
            correlation_threshold = st.slider(
                "Correlation threshold",
                min_value=0.70,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Features with correlation above this value will be removed"
            )
        
        random_state = st.number_input(
            "Random seed",
            min_value=1,
            max_value=100000,
            value=42,
            help="Random seed for reproducibility"
        )
        
        # Run button
        if st.button("Run Feature Selection", type="primary"):
            # Check if TARGET column exists
            if "TARGET" not in data.columns:
                st.error("TARGET column not found in the data. Please make sure your data contains a TARGET column.")
                return
            
            # Split features and target
            X = data.drop(columns=["TARGET"])
            y = data["TARGET"]
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            st.write(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            # Create progress bar
            progress_bar = st.progress(0, "Starting feature selection...")
            
            try:
                start_time = time.time()
                
                # Perform feature selection
                X_train_selected, feature_selector, results = perform_feature_selection(
                    X_train, 
                    y_train,
                    n_runs=n_runs,
                    split_score_threshold=split_score_threshold,
                    gain_score_threshold=gain_score_threshold,
                    correlation_threshold=correlation_threshold,
                    random_state=random_state,
                    progress_bar=progress_bar
                )
                
                X_test_selected = feature_selector.transform(X_test)
                
                elapsed_time = time.time() - start_time
                
                st.success(f"Feature selection completed in {elapsed_time:.2f} seconds!")
                
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
                st.header("Feature Selection Results")
                st.metric(
                    "Selected Features", 
                    f"{results['selected_features']} / {results['original_features']}",
                    f"-{results['reduction_percentage']:.1f}%"
                )
                
                # Create feature score plots
                plots = create_feature_score_plots(results["feature_scores"])
                
                # Display plots
                st.subheader("Top Features by Score")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plots["top_split"], use_container_width=True)
                
                with col2:
                    st.plotly_chart(plots["top_gain"], use_container_width=True)
                
                st.subheader("Bottom Features by Score")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plots["bottom_split"], use_container_width=True)
                
                with col2:
                    st.plotly_chart(plots["bottom_gain"], use_container_width=True)
                
                # Show list of selected features
                with st.expander("Selected Features"):
                    st.dataframe(pd.DataFrame({"feature": feature_selector.get_useful_features()}))
                
                # Show detailed scores for all features
                with st.expander("All Feature Scores"):
                    st.dataframe(results["feature_scores"])
                
                # Next steps guidance
                st.info("🔍 **Next Steps**: Go to the Model Training page to train a model using these selected features.")
                
            except Exception as e:
                st.error(f"Error during feature selection: {str(e)}")
                logger.error(f"Feature selection error: {str(e)}", exc_info=True)
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 