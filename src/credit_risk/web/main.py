"""
Main application for the Credit Default Risk Prediction web interface.

This application provides a Streamlit web interface for:
- Data exploration
- Feature engineering
- Feature selection
- Model training 
- Prediction
"""
import logging
import sys
import streamlit as st
from pathlib import Path
import pandas as pd

# Add parent directory to path for package imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import utility modules
from credit_risk.web.utils import (
    set_page_config, 
    setup_sidebar, 
    get_config, 
    is_gpu_available
)
from credit_risk.utils.streamlit_utils import prepare_dataframe_for_streamlit
from credit_risk.config import RunMode, ProcessingBackend

# Import tab modules
from credit_risk.web.tabs.feature_selection_tab import render_feature_selection_tab
from credit_risk.web.tabs.model_training_tab import render_model_training_tab
from credit_risk.web.tabs.prediction_tab import render_prediction_tab

# Import data functions from original app
from credit_risk.web.app import load_data, run_feature_engineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("credit_risk_app")

def main():
    """Main function for the Streamlit app."""
    # Set up page configuration
    set_page_config("Credit Risk Prediction")
    
    # Display app title and description
    st.title("Credit Default Risk Prediction")
    st.markdown(
        """
        This application demonstrates credit risk prediction using:
        - Advanced feature engineering
        - Feature selection with null importance
        - Multiple model types (LightGBM, XGBoost, CatBoost)
        - MLflow experiment tracking
        """
    )
    
    # Check if GPU is available and print message
    gpu_available = is_gpu_available()
    st.session_state["gpu_available"] = gpu_available
    
    if not gpu_available:
        st.warning("No GPU detected. Using CPU for all operations.")
    
    # Setup sidebar with configuration options
    backend, raw_data_path = setup_sidebar()
    
    # Get configuration based on user selections
    config = get_config()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Explorer", 
        "Feature Engineering", 
        "Feature Selection", 
        "Model Training", 
        "Prediction"
    ])
    
    # Store data in session state
    if "datasets" not in st.session_state:
        st.session_state["datasets"] = {}
    
    if "features_df" not in st.session_state:
        st.session_state["features_df"] = None
    
    # Data Explorer tab
    with tab1:
        st.header("Data Explorer")
        st.write(
            """
            This tab allows you to load and explore the raw data files.
            You can view dataset statistics and preview the data.
            """
        )
        
        # Add sampling controls directly in the Data Explorer tab
        st.subheader("Data Loading Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample size per dataset
            sample_per_dataset = st.number_input(
                "Sample Size per Dataset",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="Number of samples to load from each dataset. Set to 0 for no sampling.",
                key="sample_per_dataset"
            )
            
            # Update the configuration
            config.data.sample_size = sample_per_dataset
            
        with col2:
            use_sampling = st.checkbox(
                "Enable Sampling", 
                value=True,
                help="Toggle sampling on/off. When off, all data will be loaded.",
                key="use_sampling"
            )
            
            # Update the configuration
            config.run_mode = RunMode.DEVELOPMENT if use_sampling else RunMode.PRODUCTION
        
        # Button to load data
        if st.button("Load Data", key="load_data_button"):
            with st.spinner("Loading datasets..."):
                # Pass the sampling parameters to the load_data function
                st.session_state["datasets"] = load_data(config, force_sample=use_sampling)
                
                # Show the total number of rows loaded
                if st.session_state["datasets"]:
                    total_rows = sum(df.shape[0] for df in st.session_state["datasets"].values())
                    st.info(f"Total rows loaded across all datasets: {total_rows:,}")
        
        # Show data if already loaded
        if st.session_state["datasets"]:
            st.success(f"Data loaded: {len(st.session_state['datasets'])} datasets")
            
            # Show dataset summary
            for name, df in st.session_state["datasets"].items():
                with st.expander(f"{name} ({df.shape[0]:,} rows, {df.shape[1]} columns)"):
                    # Create tabs instead of nested expanders
                    data_tab, stats_tab = st.tabs(["Preview Data", "Column Statistics"])
                    
                    with data_tab:
                        st.dataframe(prepare_dataframe_for_streamlit(df.head()))
                    
                    with stats_tab:
                        # Create statistics dataframe with proper type handling
                        col_types = pd.DataFrame({
                            "Type": df.dtypes.astype(str),  # Convert dtypes to strings to avoid PyArrow issues
                            "Non-Null Count": df.count(),
                            "Null Count": df.isnull().sum(),
                            "Null %": round(100 * df.isnull().sum() / len(df), 2),
                        })
                        st.dataframe(prepare_dataframe_for_streamlit(col_types))
    
    # Feature Engineering tab
    with tab2:
        st.header("Feature Engineering")
        st.write(
            """
            This tab allows you to run automated feature engineering to create new features.
            These features will be used for model training.
            """
        )
        
        # Check if datasets are loaded
        if not st.session_state["datasets"]:
            st.warning("No data loaded. Please load data in the Data Explorer tab first.")
        else:
            # Feature engineering settings
            st.subheader("Feature Engineering Settings")
            
            # Create 2 rows of settings
            row1_col1, row1_col2 = st.columns(2)
            
            with row1_col1:
                max_depth = st.slider(
                    "Maximum feature depth",
                    min_value=1,
                    max_value=3,
                    value=config.features.max_depth,
                    step=1,
                    help="Higher values create more complex features but take longer",
                    key="feature_engineering_max_depth"
                )
                
            with row1_col2:
                agg_primitives = st.multiselect(
                    "Aggregation primitives",
                    options=["sum", "mean", "count", "max", "min", "std", "median", "mode"],
                    default=config.features.default_agg_primitives,
                    help="Functions to use for aggregating related entities",
                    key="feature_engineering_agg_primitives"
                )
            
            # Add sampling controls for feature engineering
            st.subheader("Feature Engineering Sampling")
            row2_col1, row2_col2 = st.columns(2)
            
            with row2_col1:
                # Additional sampling for feature engineering
                use_fe_sampling = st.checkbox(
                    "Enable Additional Sampling for Feature Engineering", 
                    value=False,
                    help="Sample from the already loaded data to speed up feature engineering.",
                    key="use_fe_sampling"
                )
            
            with row2_col2:
                # Only show if sampling is enabled
                fe_sample_size = st.number_input(
                    "Sample Size for Feature Engineering",
                    min_value=1000,
                    max_value=100000,
                    value=5000,
                    step=1000,
                    help="Number of samples to use for feature engineering.",
                    key="fe_sample_size",
                    disabled=not use_fe_sampling
                )
            
            # Update config with user selections
            config.features.max_depth = max_depth
            if agg_primitives:
                config.features.default_agg_primitives = agg_primitives
            
            # Button to run feature engineering
            if st.button("Run Feature Engineering", key="run_fe_button"):
                if not st.session_state["datasets"]:
                    st.error("No data loaded. Please load data first.")
                else:
                    # Apply additional sampling if requested
                    datasets_for_fe = st.session_state["datasets"].copy()
                    
                    if use_fe_sampling:
                        with st.spinner("Applying additional sampling for feature engineering..."):
                            st.info(f"Sampling {fe_sample_size} rows from each dataset for feature engineering")
                            
                            # Sample each dataset
                            for name, df in datasets_for_fe.items():
                                sample_size = min(fe_sample_size, df.shape[0])
                                datasets_for_fe[name] = df.sample(sample_size, random_state=config.random_seed)
                            
                            total_rows = sum(df.shape[0] for df in datasets_for_fe.values())
                            st.info(f"Total rows for feature engineering: {total_rows:,}")
                    
                    # Run feature engineering with the potentially sampled datasets
                    st.session_state["features_df"] = run_feature_engineering(
                        config,
                        datasets_for_fe
                    )
        
        # Show features if already generated
        if st.session_state["features_df"] is not None:
            features_df = st.session_state["features_df"]
            st.success(f"Features generated: {features_df.shape[1]} features")
            
            # Create a more informative feature sample display
            with st.expander("Sample of Features", expanded=True):
                # Get a mix of original and generated features
                original_cols = [col for col in features_df.columns if '(' not in col and ')' not in col][:20]
                generated_cols = [col for col in features_df.columns if '(' in col and ')' in col][:20]
                
                # Combine original and generated features
                sample_cols = original_cols + generated_cols
                
                # Add TARGET column if it exists
                if 'TARGET' in features_df.columns and 'TARGET' not in sample_cols:
                    sample_cols = ['TARGET'] + sample_cols
                
                # Display information about the sample
                st.write(f"Showing {len(original_cols)} original features and {len(generated_cols)} generated features:")
                
                # Create tabs for different views
                sample_tab1, sample_tab2 = st.tabs(["Sample Rows", "Feature Statistics"])
                
                with sample_tab1:
                    st.dataframe(prepare_dataframe_for_streamlit(features_df[sample_cols].head(10)))
                
                with sample_tab2:
                    # Show basic statistics for the selected features
                    stats_df = features_df[sample_cols].describe().T
                    stats_df['missing'] = features_df[sample_cols].isna().sum()
                    stats_df['missing_pct'] = (features_df[sample_cols].isna().sum() / len(features_df)) * 100
                    st.dataframe(prepare_dataframe_for_streamlit(stats_df))
    
    # Feature Selection tab
    with tab3:
        # Pass the features dataframe to the feature selection tab
        features_df = st.session_state.get("features_df")
        render_feature_selection_tab(features_df)
    
    # Model Training tab
    with tab4:
        # Pass the features dataframe to the model training tab if no feature selection done
        features_df = st.session_state.get("features_df")
        render_model_training_tab(features_df)
    
    # Prediction tab
    with tab5:
        render_prediction_tab()

if __name__ == "__main__":
    main() 