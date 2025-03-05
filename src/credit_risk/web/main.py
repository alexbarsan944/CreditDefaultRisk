"""
Main application for the Credit Default Risk Prediction web interface.

This application provides a Streamlit web interface for:
- Data exploration and management
- Feature engineering
- Feature selection
- Model training 
- Prediction
"""
import logging
import sys
import streamlit as st
import pandas as pd
from pathlib import Path

# Add parent directory to path for package imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import utility modules
from credit_risk.web.utils import (
    set_page_config, 
    get_config, 
    is_gpu_available,
    apply_sampling_profile
)
from credit_risk.utils.streamlit_utils import prepare_dataframe_for_streamlit
from credit_risk.config import RunMode, ProcessingBackend
from credit_risk.web.ui_components import (
    display_info_box,
    educational_tip,
    display_dataframe_with_metrics,
    plot_missing_values,
    section_header,
    workflow_progress
)

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
    
    with st.expander("About this application", expanded=False):
        st.markdown(
            """
            This application demonstrates credit risk prediction using:
            - Advanced feature engineering
            - Feature selection with null importance
            - Multiple model types (LightGBM, XGBoost, CatBoost)
            - MLflow experiment tracking
            
            Use this app to explore, preprocess data, build features, train models, and evaluate performance.
            """
        )
    
    # Check if GPU is available and print message
    gpu_available = is_gpu_available()
    st.session_state["gpu_available"] = gpu_available
    
    if not gpu_available:
        st.sidebar.warning("No GPU detected. Using CPU for all operations.")
    else:
        st.sidebar.success("GPU detected and available for accelerated processing.")
    
    # Create a simple sidebar for session management
    st.sidebar.title("Session Management")
    
    # Option to reset session
    if st.sidebar.button("Reset Session"):
        for key in list(st.session_state.keys()):
            if key not in ["gpu_available"]:
                del st.session_state[key]
        st.sidebar.success("Session reset successfully!")
        st.experimental_rerun()
    
    # Get configuration based on user selections
    config = get_config()
    
    # Create tabs for different sections with workflow guidance
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Data Management", 
        "2️⃣ Feature Engineering", 
        "3️⃣ Feature Selection", 
        "4️⃣ Model Training", 
        "5️⃣ Prediction"
    ])
    
    # Store data in session state
    if "datasets" not in st.session_state:
        st.session_state["datasets"] = {}
    
    if "features_df" not in st.session_state:
        st.session_state["features_df"] = None
    
    # Track workflow progress
    workflow_stage = 0
    if st.session_state["datasets"]:
        workflow_stage = 1
    if st.session_state.get("features_df") is not None:
        workflow_stage = 2
    if "feature_selection_results" in st.session_state:
        workflow_stage = 3
    if "model_results" in st.session_state:
        workflow_stage = 4
    
    # Display current workflow stage in sidebar
    st.sidebar.title("Workflow Progress")
    progress_bar = st.sidebar.progress(workflow_stage / 4)
    st.sidebar.caption(f"Stage {workflow_stage + 1} of 5")
    
    # Data Management tab
    with tab1:
        section_header(
            "Data Management",
            "Load, explore, and prepare your data for modeling",
            """
            **Why Data Management Matters:**
            
            Good data preparation is essential for successful modeling:
            - Clean data leads to better model performance
            - Appropriate sampling ensures efficient development
            - Understanding your data helps with feature engineering
            
            In this tab, you can load your datasets, apply sampling strategies,
            and explore the data before proceeding to feature engineering.
            """
        )
        
        # Create tabs for data management
        data_tabs = st.tabs(["Data Loading", "Data Sampling", "Data Preview"])
        
        # Data Loading tab
        with data_tabs[0]:
            st.subheader("Data Sources")
            
            # Add a helpful tip about data loading
            educational_tip(
                "Loading smaller amounts of data during development allows for faster iterations. "
                "You can always train on the full dataset when you've finalized your approach."
            )
            
            # Data path setting
            data_path_col1, data_path_col2 = st.columns([3, 1])
            with data_path_col1:
                data_path = st.text_input(
                    "Data Directory",
                    value="data/raw",
                    help="Path to raw data files."
                )
            
            # Update config with data path
            # Convert the data path string to Path objects as expected by the loader
            config.data.raw_data_path = Path(data_path)
            config.data.processed_data_path = Path("data/processed")
            config.data.features_path = Path("data/features")
            
            # GPU settings for data loading
            use_gpu_for_loading = st.checkbox(
                "Use GPU for Data Loading", 
                value=gpu_available,
                disabled=not gpu_available,
                help="Use GPU acceleration for faster data loading (if available)."
            )
            
            # Update config with GPU setting
            if use_gpu_for_loading and gpu_available:
                config.gpu.processing_backend = ProcessingBackend.GPU
            else:
                config.gpu.processing_backend = ProcessingBackend.CPU
            
            # Button to load data
            if st.button("Load Data", key="load_data_button", use_container_width=True):
                with st.spinner("Loading datasets..."):
                    try:
                        st.session_state["datasets"] = load_data(config, force_sample=False)
                        
                        # Show the total number of rows loaded
                        if st.session_state["datasets"]:
                            total_rows = sum(df.shape[0] for df in st.session_state["datasets"].values())
                            st.success(f"Successfully loaded {len(st.session_state['datasets'])} datasets with {total_rows:,} total rows")
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
                        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        
        # Data Sampling tab
        with data_tabs[1]:
            st.subheader("Data Sampling Settings")
            
            if not st.session_state["datasets"]:
                st.warning("No data loaded yet. Please load data first.")
            else:
                # Get the original total size for reference
                original_total_rows = sum(df.shape[0] for df in st.session_state["datasets"].values())
                
                display_info_box(
                    "About Sampling Profiles", 
                    """
                    **Sampling profiles** provide preset configurations for different stages of model development:
                    
                    - **Quick Exploration**: Small samples for rapid iteration and testing
                    - **Balanced Development**: Medium-sized samples balanced for model training
                    - **Production Preparation**: Larger samples for final model testing
                    - **Full Dataset**: No sampling, use all available data
                    
                    Choose a profile based on your current needs, or customize the sample size.
                    """,
                    expanded=False
                )
                
                # Create two columns: one for profile selection and one for options
                profile_col1, profile_col2 = st.columns([2, 1])
                
                with profile_col1:
                    # Sampling profile selection
                    profile_options = [
                        "Quick Exploration", 
                        "Balanced Development", 
                        "Production Preparation", 
                        "Full Dataset"
                    ]
                    
                    selected_profile = st.radio(
                        "Sampling Profile",
                        options=profile_options,
                        index=1,  # Default to Balanced Development
                        horizontal=True
                    )
                
                with profile_col2:
                    # Custom sample size option
                    use_custom_size = st.checkbox("Custom Sample Size", value=False)
                    
                    if use_custom_size:
                        custom_sample_size = st.number_input(
                            "Sample Size per Dataset",
                            min_value=1000,
                            max_value=500000,
                            value=10000,
                            step=1000
                        )
                    else:
                        custom_sample_size = None
                
                # Show profile description
                if selected_profile == "Quick Exploration":
                    st.info("🔍 **Quick Exploration**: Small samples (5,000 rows) for rapid testing and debugging.")
                elif selected_profile == "Balanced Development":
                    st.info("⚖️ **Balanced Development**: Medium-sized samples (20,000 rows) with stratified sampling for model development.")
                elif selected_profile == "Production Preparation":
                    st.info("🚀 **Production Preparation**: Larger samples (50,000 rows) with stratified sampling for final model testing.")
                else:  # Full Dataset
                    st.warning("⚠️ **Full Dataset**: Using all data. This may be slow, especially for feature engineering and model training.")
                
                # Apply sampling button
                if st.button("Apply Sampling Profile", use_container_width=True):
                    with st.spinner(f"Applying {selected_profile} sampling profile..."):
                        # Use our new sampling utility
                        sampled_datasets = apply_sampling_profile(
                            st.session_state["datasets"],
                            selected_profile,
                            custom_sample_size
                        )
                        
                        # Update datasets in session state
                        st.session_state["datasets"] = sampled_datasets
                        
                        # Show success message with comparison to original size
                        new_total_rows = sum(df.shape[0] for df in sampled_datasets.values())
                        reduction = 100 * (1 - new_total_rows / original_total_rows) if original_total_rows > 0 else 0
                        
                        if reduction > 0:
                            st.success(f"Applied {selected_profile} sampling. New total: {new_total_rows:,} rows ({reduction:.1f}% reduction)")
                        else:
                            st.success(f"Using full dataset with {new_total_rows:,} rows")
                
                # Display current dataset sizes
                st.subheader("Current Dataset Sizes")
                
                # Create a more informative table with original and current sizes
                if "original_sizes" not in st.session_state and st.session_state["datasets"]:
                    st.session_state["original_sizes"] = {
                        name: df.shape[0] for name, df in st.session_state["datasets"].items()
                    }
                
                if "original_sizes" in st.session_state:
                    sizes_data = []
                    for name, df in st.session_state["datasets"].items():
                        original_size = st.session_state["original_sizes"].get(name, df.shape[0])
                        current_size = df.shape[0]
                        percent = 100 * (current_size / original_size) if original_size > 0 else 100
                        
                        sizes_data.append({
                            "Dataset": name,
                            "Original Rows": f"{original_size:,}",
                            "Current Rows": f"{current_size:,}",
                            "% of Original": f"{percent:.1f}%",
                            "Columns": df.shape[1]
                        })
                    
                    sizes_df = pd.DataFrame(sizes_data)
                    st.table(sizes_df)
        
        # Data Preview tab
        with data_tabs[2]:
            st.subheader("Data Preview")
            
            if not st.session_state["datasets"]:
                st.warning("No data loaded yet. Please load data first.")
            else:
                st.write("Preview and analyze the loaded datasets before feature engineering.")
                
                # Dataset selection with dataset info
                dataset_options = [f"{name} ({df.shape[0]:,} rows)" for name, df in st.session_state["datasets"].items()]
                selected_option = st.selectbox(
                    "Select Dataset to Preview",
                    options=dataset_options
                )
                
                # Extract the dataset name from the selected option
                selected_dataset = selected_option.split(" (")[0] if selected_option else None
                
                # Show selected dataset
                if selected_dataset:
                    df = st.session_state["datasets"][selected_dataset]
                    
                    # Display dataset metrics and sample
                    display_dataframe_with_metrics(df, selected_dataset)
                    
                    # Create tabs for different views
                    preview_tabs = st.tabs(["Column Statistics", "Missing Values", "Target Distribution"])
                    
                    with preview_tabs[0]:
                        # Create statistics DataFrame
                        stats_df = pd.DataFrame({
                            "Type": df.dtypes.astype(str),
                            "Non-Null Count": df.count(),
                            "Null Count": df.isnull().sum(),
                            "Null %": round(100 * df.isnull().sum() / len(df), 2),
                            "Unique Values": df.nunique()
                        })
                        st.dataframe(stats_df)
                        
                    with preview_tabs[1]:
                        # Use our reusable component for missing values
                        plot_missing_values(df)
                        
                    with preview_tabs[2]:
                        # Show target distribution if this is the main dataset
                        if "TARGET" in df.columns:
                            target_col1, target_col2 = st.columns([1, 1])
                            
                            with target_col1:
                                # Show target value counts
                                target_counts = df["TARGET"].value_counts().reset_index()
                                target_counts.columns = ["Target Value", "Count"]
                                target_counts["Percentage"] = 100 * target_counts["Count"] / target_counts["Count"].sum()
                                st.write("### Target Distribution")
                                st.table(target_counts)
                            
                            with target_col2:
                                # Create a pie chart of target distribution
                                try:
                                    import plotly.express as px
                                    fig = px.pie(
                                        target_counts, 
                                        values="Count", 
                                        names="Target Value",
                                        title="Target Distribution",
                                        color_discrete_sequence=["#0068c9", "#ef553b"]
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not create target distribution chart: {str(e)}")
                        else:
                            st.info("No TARGET column found in this dataset.")
    
    # Feature Engineering tab
    with tab2:
        st.header("Feature Engineering")
        st.write("""
        This tab allows you to create new features from the loaded data.
        These features will be used for model training.
        """)
        
        if not st.session_state["datasets"]:
            st.warning("No data loaded yet. Please go to the Data Management tab first.")
        else:
            # Feature engineering settings
            st.subheader("Feature Engineering Settings")
            
            # Create columns for settings
            fe_col1, fe_col2 = st.columns(2)
            
            with fe_col1:
                max_depth = st.slider(
                    "Maximum feature depth",
                    min_value=1,
                    max_value=3,
                    value=config.features.max_depth,
                    step=1,
                    help="Higher values create more complex features but take longer to compute.\n\n" + 
                         "1: Simple aggregations\n" +
                         "2: Aggregations of aggregations\n" +
                         "3: Complex nested features (slowest)"
                )
                
                # GPU for feature engineering
                use_gpu_for_fe = st.checkbox(
                    "Use GPU Acceleration", 
                    value=gpu_available,
                    disabled=not gpu_available,
                    help="Use GPU to accelerate feature generation (if available)."
                )
            
            with fe_col2:
                agg_primitives = st.multiselect(
                    "Aggregation primitives",
                    options=["sum", "mean", "count", "max", "min", "std", "median", "mode"],
                    default=config.features.default_agg_primitives,
                    help="Functions used to aggregate related entities.\n\n" + 
                         "Common choices:\n" +
                         "- sum: Total across related records\n" +
                         "- mean: Average of related records\n" +
                         "- count: Number of related records\n" +
                         "- min/max: Extreme values in related records"
                )
                
                n_jobs = st.number_input(
                    "Parallel Jobs",
                    min_value=1,
                    max_value=8,
                    value=config.features.n_jobs if config.features.n_jobs > 0 else 4,
                    help="Number of parallel processes to use for feature generation. -1 uses all available cores."
                )
            
            # Update config with FE settings
            config.features.max_depth = max_depth
            if agg_primitives:
                config.features.default_agg_primitives = agg_primitives
            config.features.n_jobs = n_jobs
            
            # Processing backend based on GPU setting
            if use_gpu_for_fe and gpu_available:
                config.gpu.processing_backend = ProcessingBackend.GPU
            else:
                config.gpu.processing_backend = ProcessingBackend.CPU
            
            # Additional sampling for feature engineering
            with st.expander("Advanced Feature Engineering Settings"):
                chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=1000,
                    max_value=50000,
                    value=config.features.chunk_size or 10000,
                    step=1000,
                    help="Number of rows to process at once. Smaller chunks use less memory but may be slower."
                )
                config.features.chunk_size = chunk_size
            
            # Button to run feature engineering
            if st.button("Generate Features", key="run_fe_button"):
                if not st.session_state["datasets"]:
                    st.error("No data loaded. Please load data first.")
                else:
                    with st.spinner("Generating features..."):
                        try:
                            st.session_state["features_df"] = run_feature_engineering(
                                config,
                                st.session_state["datasets"]
                            )
                            if st.session_state["features_df"] is not None:
                                st.success(f"Features generated successfully: {st.session_state['features_df'].shape[1]} features")
                        except Exception as e:
                            st.error(f"Error generating features: {str(e)}")
                            logger.error(f"Error generating features: {str(e)}", exc_info=True)
            
            # Show features if already generated
            if st.session_state["features_df"] is not None:
                features_df = st.session_state["features_df"]
                
                with st.expander("Feature Overview", expanded=True):
                    # Create a more informative summary
                    st.write(f"Generated {features_df.shape[1]} features from {features_df.shape[0]} samples")
                    
                    # Display feature type distribution
                    feature_types_col1, feature_types_col2 = st.columns(2)
                    
                    with feature_types_col1:
                        # Count original vs derived features
                        try:
                            raw_app_df = pd.read_csv(f"{config.data.raw_data_path}/application_train.csv")
                            original_cols = raw_app_df.shape[1]
                        except Exception as e:
                            original_cols = len(features_df.columns) if not features_df.empty else 0
                        
                        st.write(f"Original features: {original_cols}")
                        st.write(f"Derived features: {max(0, features_df.shape[1] - original_cols)}")
                    
                    with feature_types_col2:
                        # Count by data type
                        numeric_cols = len(features_df.select_dtypes(include=['number']).columns)
                        categorical_cols = len(features_df.select_dtypes(include=['category', 'object']).columns)
                        st.write(f"Numeric features: {numeric_cols}")
                        st.write(f"Categorical features: {categorical_cols}")
                
                # Sample of features
                with st.expander("Feature Preview"):
                    # Get a mix of features to display
                    original_cols = [col for col in features_df.columns if '(' not in col and ')' not in col][:10]
                    generated_cols = [col for col in features_df.columns if '(' in col and ')' in col][:10]
                    sample_cols = original_cols + generated_cols
                    
                    if 'TARGET' in features_df.columns and 'TARGET' not in sample_cols:
                        sample_cols = ['TARGET'] + sample_cols
                    
                    st.dataframe(prepare_dataframe_for_streamlit(features_df[sample_cols].head(5)))
                    st.caption(f"Showing a sample of {len(sample_cols)} columns out of {features_df.shape[1]} total features")
    
    # Feature Selection tab
    with tab3:
        # Pass the features dataframe to the feature selection tab
        features_df = st.session_state.get("features_df")
        render_feature_selection_tab(features_df)
    
    # Model Training tab
    with tab4:
        # Pass the features dataframe to the model training tab
        features_df = st.session_state.get("features_df")
        render_model_training_tab(features_df)
    
    # Prediction tab
    with tab5:
        render_prediction_tab()

if __name__ == "__main__":
    main() 