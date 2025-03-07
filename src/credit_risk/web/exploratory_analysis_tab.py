import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from typing import List, Optional, Dict, Any, Tuple

from credit_risk.utils.logging_utils import get_logger

# Configure logging
logger = get_logger(__name__)

# Create a context manager to temporarily suppress warnings
class SuppressWarnings:
    def __enter__(self):
        self.original_filters = warnings.filters.copy()
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
        return self
        
    def __exit__(self, *args):
        warnings.filters = self.original_filters

def render_exploratory_analysis_tab(data: pd.DataFrame = None):
    """
    Render the exploratory analysis tab.
    
    Parameters
    ----------
    data : pd.DataFrame, optional
        Data to analyze
    """
    st.header("Exploratory Data Analysis")
    
    if data is None:
        st.warning("No data available for analysis. Please load data first.")
        return
    
    # Data overview
    st.subheader("Data Overview")
    st.write(f"Shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Basic statistics
    with st.expander("Basic Statistics", expanded=False):
        with SuppressWarnings():
            st.dataframe(data.describe())
    
    # Missing values
    with st.expander("Missing Values", expanded=False):
        missing = data.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        
        if missing.empty:
            st.success("No missing values found in the dataset!")
        else:
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing Percentage': round(missing.values / len(data) * 100, 2)
            })
            st.dataframe(missing_df)
            
            # Plot missing values
            if not missing.empty:
                fig = px.bar(
                    missing_df.head(20), 
                    x='Column', 
                    y='Missing Percentage',
                    title='Top 20 Columns with Missing Values',
                    color='Missing Percentage',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Numeric columns distribution
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) > 0:
        with st.expander("Numeric Distributions", expanded=False):
            selected_col = st.selectbox(
                "Select column to visualize", 
                options=numeric_cols,
                key="numeric_dist_col"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    data, 
                    x=selected_col,
                    title=f"Histogram of {selected_col}",
                    color_discrete_sequence=['#3366CC']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    data, 
                    y=selected_col,
                    title=f"Box Plot of {selected_col}",
                    color_discrete_sequence=['#3366CC']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    if len(numeric_cols) > 1:
        with st.expander("Correlation Analysis", expanded=False):
            st.subheader("Correlation Heatmap")
            
            # Allow users to select how many features to include
            num_features = st.slider(
                "Number of features to include", 
                min_value=5, 
                max_value=min(30, len(numeric_cols)),
                value=15,
                key="corr_num_features"
            )
            
            # Select top variable features
            with SuppressWarnings():
                var = data[numeric_cols].var().sort_values(ascending=False)
                top_var_cols = var.head(num_features).index.tolist()
                
                # Calculate correlation matrix
                corr_matrix = data[top_var_cols].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="Feature Correlation Heatmap",
                labels=dict(x="Feature", y="Feature", color="Correlation")
            )
            st.plotly_chart(fig, use_container_width=True) 