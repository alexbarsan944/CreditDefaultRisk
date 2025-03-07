"""
UI Components for the Credit Default Risk Prediction application.

This module provides reusable UI components to maintain consistency across the application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from credit_risk.utils.logging_utils import get_logger
from credit_risk.utils.streamlit_utils import prepare_dataframe_for_streamlit

# Configure logging
logger = get_logger(__name__)

# Styling constants
PRIMARY_COLOR = "#0068c9"
SECONDARY_COLOR = "#83c9ff" 
WARNING_COLOR = "#ffaa00"
SUCCESS_COLOR = "#00cc96"
DANGER_COLOR = "#ef553b"

# ============================================================================
# Information and educational components
# ============================================================================

def display_info_box(title: str, content: str, expanded: bool = False):
    """
    Display an information box with consistent styling.
    
    Parameters
    ----------
    title : str
        Title of the information box
    content : str
        Markdown content to display
    expanded : bool
        Whether the expander should be initially expanded
    """
    with st.expander(title, expanded=expanded):
        st.markdown(content)


def parameter_help(title: str, description: str, examples: Optional[List[str]] = None):
    """
    Generate consistent help text for parameters.
    
    Parameters
    ----------
    title : str
        Brief title/summary of the parameter
    description : str
        Detailed description of what the parameter does
    examples : List[str], optional
        Examples of parameter values and their effects
        
    Returns
    -------
    str
        Formatted help text
    """
    help_text = f"**{title}**\n\n{description}"
    
    if examples:
        examples_text = "\n".join([f"- {example}" for example in examples])
        help_text += f"\n\n**Examples:**\n{examples_text}"
    
    return help_text


def educational_tip(content: str):
    """
    Display an educational tip with consistent styling.
    
    Parameters
    ----------
    content : str
        Markdown content for the tip
    """
    st.info(f"💡 **Tip:**\n\n{content}")


# ============================================================================
# Data display components
# ============================================================================

def display_dataframe_with_metrics(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Display a DataFrame with summary metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to display
    name : str
        Name of the DataFrame
    """
    if df is None:
        st.warning(f"No {name} data available to display.")
        return
        
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    
    with col2:
        st.metric("Columns", df.shape[1])
    
    with col3:
        numeric_cols = df.select_dtypes(include=np.number).columns
        st.metric("Numeric Features", len(numeric_cols))
    
    # Prepare dataframe for display to avoid Arrow serialization issues
    display_df = prepare_dataframe_for_streamlit(df)
    
    # Display dataframe
    st.dataframe(display_df)
    
    # Add download option
    if st.button(f"Download {name} as CSV"):
        csv = df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name.lower().replace(' ', '_')}_{timestamp}.csv"
        
        st.download_button(
            label="Click to download",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )


def plot_missing_values(df: pd.DataFrame, max_cols: int = 20):
    """
    Create a bar plot of missing values in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    max_cols : int
        Maximum number of columns to display
    """
    # Calculate missing value percentages
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum(),
        "Missing %": round(100 * df.isnull().sum() / len(df), 2)
    }).sort_values("Missing %", ascending=False)
    
    # Filter to show only columns with missing values
    missing_df = missing_df[missing_df["Missing Values"] > 0]
    
    if len(missing_df) > 0:
        # Display summary
        st.write(f"Found {len(missing_df)} columns with missing values")
        
        # Display data table
        st.dataframe(missing_df)
        
        # Create plot for top columns
        if len(missing_df) > max_cols:
            plot_df = missing_df.head(max_cols)
            st.write(f"Showing top {max_cols} columns with highest percentage of missing values")
        else:
            plot_df = missing_df
        
        # Create and display plot
        fig = px.bar(
            plot_df, 
            x="Column", 
            y="Missing %",
            color="Missing %",
            color_continuous_scale=["#83c9ff", "#ef553b"],
            title="Missing Values by Column"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found in this dataset!")


def plot_feature_importance(feature_scores, n_features: int = 20, 
                           height: int = 600, width: int = 800):
    """
    Create a feature importance plot.
    
    Parameters
    ----------
    feature_scores : dict or DataFrame
        Feature importance scores
    n_features : int
        Number of top features to display
    height : int
        Height of the plot
    width : int
        Width of the plot
    """
    # Convert to DataFrame if it's not already
    if not isinstance(feature_scores, pd.DataFrame):
        if isinstance(feature_scores, dict):
            df = pd.DataFrame({
                "Feature": list(feature_scores.keys()),
                "Importance": list(feature_scores.values())
            })
        else:
            st.error("feature_scores must be a dict or DataFrame")
            return
    else:
        df = feature_scores
    
    # Ensure we have the right columns
    if "Feature" not in df.columns or not any(c in df.columns for c in ["Importance", "Score", "importance", "score"]):
        st.error("DataFrame must have 'Feature' column and an importance column")
        return
    
    # Standardize column names
    if "Feature" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Feature"})
    
    importance_col = next((c for c in df.columns if c.lower() in ["importance", "score"]), df.columns[1])
    if importance_col != "Importance":
        df = df.rename(columns={importance_col: "Importance"})
    
    # Sort and limit to top n features
    df = df.sort_values("Importance", ascending=False).head(n_features)
    
    # Create horizontal bar chart
    fig = px.bar(
        df,
        y="Feature",
        x="Importance",
        orientation='h',
        color="Importance",
        color_continuous_scale=px.colors.sequential.Blues,
        title=f"Top {n_features} Feature Importance"
    )
    
    fig.update_layout(
        height=height,
        width=width,
        yaxis=dict(
            title="",
            autorange="reversed"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Process tracking components
# ============================================================================

class ProgressTracker:
    """
    A class to track progress of multi-step processes.
    """
    def __init__(self, total_steps: int, description: str = "Progress"):
        """
        Initialize a progress tracker.
        
        Parameters
        ----------
        total_steps : int
            Total number of steps in the process
        description : str
            Description of the process
        """
        self.total_steps = total_steps
        self.description = description
        self.start_time = time.time()
        self.current_step = 0
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
    def update(self, step: int = None, status: str = None):
        """
        Update the progress tracker.
        
        Parameters
        ----------
        step : int, optional
            Current step (if None, increment by 1)
        status : str, optional
            Status message to display
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Calculate progress as a fraction
        progress = min(self.current_step / self.total_steps, 1.0)
        
        # Update progress bar
        self.progress_bar.progress(progress)
        
        # Update status text if provided
        if status:
            elapsed = time.time() - self.start_time
            self.status_text.text(f"{status} ({elapsed:.1f}s elapsed)")
    
    def complete(self, message: str = "Process completed!"):
        """
        Mark the process as complete.
        
        Parameters
        ----------
        message : str
            Completion message
        """
        self.progress_bar.progress(1.0)
        elapsed = time.time() - self.start_time
        self.status_text.success(f"{message} (Completed in {elapsed:.1f}s)")


# ============================================================================
# Parameter selection components
# ============================================================================

def model_parameter_section(model_type: str):
    """
    Create a model parameter input section based on model type.
    
    Parameters
    ----------
    model_type : str
        Type of model (xgboost, lightgbm, catboost)
        
    Returns
    -------
    dict
        Dictionary of model parameters
    """
    params = {}
    
    # Common parameters across all models
    col1, col2 = st.columns(2)
    
    with col1:
        params["n_estimators"] = st.slider(
            "Number of Estimators",
            min_value=50,
            max_value=1000,
            value=100,
            step=50,
            key=f"{model_type}_n_estimators",
            help=parameter_help(
                "Number of trees in the ensemble",
                "Controls model complexity and training time. More trees generally improve performance but increase training time.",
                ["100: Good balance for most problems", 
                 "500+: May be needed for complex problems"]
            )
        )
    
    with col2:
        params["learning_rate"] = st.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            key=f"{model_type}_learning_rate",
            help=parameter_help(
                "Step size shrinkage",
                "Controls how quickly the model learns. Lower values require more trees but can lead to better generalization.",
                ["0.1: Default value, works well in many cases",
                 "0.01: For more precise learning (needs more trees)",
                 "0.3: For faster learning (but may overfit)"]
            )
        )
    
    # Model-specific parameters
    if model_type == "xgboost":
        col1, col2 = st.columns(2)
        
        with col1:
            params["max_depth"] = st.slider(
                "Maximum Tree Depth",
                min_value=3,
                max_value=10,
                value=6,
                step=1,
                key="xgboost_max_depth",
                help=parameter_help(
                    "Maximum depth of trees",
                    "Controls complexity of individual trees. Deeper trees can model more complex patterns but may overfit.",
                    ["3-4: For simpler relationships", 
                     "6: Balanced default", 
                     "8-10: For complex relationships (risk of overfitting)"]
                )
            )
            
            params["gamma"] = st.slider(
                "Gamma (Min Split Loss)",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                key="xgboost_gamma",
                help=parameter_help(
                    "Minimum loss reduction for split",
                    "Controls complexity via regularization. Higher values create more conservative trees.",
                    ["0.0: No minimum requirement", 
                     "1.0: Moderate regularization", 
                     "5.0: Strong regularization"]
                )
            )
        
        with col2:
            params["subsample"] = st.slider(
                "Subsample Ratio",
                min_value=0.5,
                max_value=1.0,
                value=1.0,
                step=0.1,
                key="xgboost_subsample",
                help=parameter_help(
                    "Fraction of samples used per tree",
                    "Controls overfitting by using random subsets of data. Values below 1.0 make trees more independent.",
                    ["0.8: Commonly used to reduce overfitting", 
                     "1.0: Use all data for each tree"]
                )
            )
            
            params["colsample_bytree"] = st.slider(
                "Column Sample by Tree",
                min_value=0.5,
                max_value=1.0,
                value=1.0,
                step=0.1,
                key="xgboost_colsample_bytree",
                help=parameter_help(
                    "Fraction of features used per tree",
                    "Like feature bagging in random forests. Lower values create more diverse trees.",
                    ["0.8: Commonly used for feature diversity", 
                     "1.0: Use all features for each tree"]
                )
            )
    
    elif model_type == "lightgbm":
        col1, col2 = st.columns(2)
        
        with col1:
            params["num_leaves"] = st.slider(
                "Number of Leaves",
                min_value=8,
                max_value=256,
                value=31,
                step=8,
                key="lightgbm_num_leaves",
                help=parameter_help(
                    "Maximum number of leaves in tree",
                    "LightGBM grows trees leaf-wise rather than depth-wise. This controls tree complexity.",
                    ["31: Default value", 
                     "< 31: Reduces overfitting",
                     "> 31: May improve accuracy but risk overfitting"]
                )
            )
            
            params["max_depth"] = st.slider(
                "Maximum Tree Depth",
                min_value=-1,
                max_value=15,
                value=-1,
                step=1,
                key="lightgbm_max_depth",
                help=parameter_help(
                    "Maximum tree depth",
                    "Limits the max depth of the tree. -1 means no limit.",
                    ["-1: No limit (recommended with num_leaves constraint)", 
                     "Positive values: Explicitly limit tree depth"]
                )
            )
        
        with col2:
            params["min_child_samples"] = st.slider(
                "Min Child Samples",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                key="lightgbm_min_child_samples",
                help=parameter_help(
                    "Minimum samples per leaf",
                    "Prevents creating leaves with too few samples. Helps control overfitting.",
                    ["20: Default value", 
                     "Higher values: More conservative model"]
                )
            )
            
            params["feature_fraction"] = st.slider(
                "Feature Fraction",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.1,
                key="lightgbm_feature_fraction",
                help=parameter_help(
                    "Fraction of features to use",
                    "Similar to colsample_bytree in XGBoost. Lower values create more diverse trees.",
                    ["0.8: Default value", 
                     "Lower values: More regularization"]
                )
            )
    
    elif model_type == "catboost":
        col1, col2 = st.columns(2)
        
        with col1:
            params["depth"] = st.slider(
                "Tree Depth",
                min_value=4,
                max_value=10,
                value=6,
                step=1,
                key="catboost_depth",
                help=parameter_help(
                    "Depth of the tree",
                    "Controls complexity of individual trees. Deeper trees can model more complex patterns.",
                    ["6: Default balanced value", 
                     "4-5: For simpler models", 
                     "8-10: For more complex data relationships"]
                )
            )
            
            params["l2_leaf_reg"] = st.slider(
                "L2 Regularization",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=1.0,
                key="catboost_l2_leaf_reg",
                help=parameter_help(
                    "L2 regularization coefficient",
                    "Controls model complexity via regularization. Higher values create a more conservative model.",
                    ["3.0: Default value", 
                     "Higher values: Stronger regularization"]
                )
            )
        
        with col2:
            params["rsm"] = st.slider(
                "Random Strength",
                min_value=0.5,
                max_value=1.0,
                value=1.0,
                step=0.1,
                key="catboost_rsm",
                help=parameter_help(
                    "Random strength for feature selection",
                    "Similar to colsample in other models. Controls randomness in feature selection.",
                    ["1.0: No randomization", 
                     "Lower values: More randomness"]
                )
            )
            
            params["border_count"] = st.slider(
                "Border Count",
                min_value=32,
                max_value=255,
                value=128,
                step=32,
                key="catboost_border_count",
                help=parameter_help(
                    "Number of splits for numerical features",
                    "Controls granularity of numerical feature splits.",
                    ["128: Default value", 
                     "Higher values: More precise splits but may overfit"]
                )
            )
    
    # Return the parameter dictionary
    return params


def sampling_profiles():
    """
    Provide predefined sampling profiles for different use cases.
    
    Returns
    -------
    dict
        Dictionary of sampling profiles with their parameters
    """
    return {
        "Quick Exploration": {
            "description": "Small samples for quick exploration and debugging",
            "sample_size": 5000,
            "sampling_method": "Random",
            "target_variable": "TARGET"
        },
        "Balanced Development": {
            "description": "Balanced samples for model development",
            "sample_size": 20000,
            "sampling_method": "Stratified",
            "target_variable": "TARGET"
        },
        "Production Preparation": {
            "description": "Larger samples for final model testing",
            "sample_size": 50000,
            "sampling_method": "Stratified",
            "target_variable": "TARGET"
        },
        "Full Dataset": {
            "description": "Use the entire dataset (warning: may be slow)",
            "sample_size": None,
            "sampling_method": None,
            "target_variable": "TARGET"
        }
    }


# ============================================================================
# Result visualization components
# ============================================================================

def plot_confusion_matrix(confusion_matrix, class_names=None):
    """
    Create an interactive confusion matrix plot.
    
    Parameters
    ----------
    confusion_matrix : array-like
        Confusion matrix to plot
    class_names : list, optional
        Names for the classes (default: ["Negative", "Positive"])
    """
    if class_names is None:
        class_names = ["Negative", "Positive"]
    
    # Create annotations for the heatmap
    annotations = []
    for i, row in enumerate(confusion_matrix):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(value),
                    font=dict(color="white" if value > confusion_matrix.max() / 2 else "black"),
                    showarrow=False
                )
            )
    
    # Create confusion matrix figure
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=class_names,
        y=class_names,
        colorscale="Blues",
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="Actual", autorange="reversed"),
        annotations=annotations,
        width=500,
        height=500
    )
    
    st.plotly_chart(fig)


def plot_roc_curve(fpr, tpr, auc_score):
    """
    Create an interactive ROC curve plot.
    
    Parameters
    ----------
    fpr : array-like
        False positive rates
    tpr : array-like
        True positive rates
    auc_score : float
        Area under the ROC curve
    """
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ROC (AUC = {auc_score:.4f})",
        line=dict(color=PRIMARY_COLOR, width=2)
    ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier (AUC = 0.5)",
        line=dict(color="gray", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title="ROC Curve",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        width=600,
        height=500
    )
    
    st.plotly_chart(fig)


def plot_precision_recall_curve(precision, recall, average_precision):
    """
    Create an interactive precision-recall curve plot.
    
    Parameters
    ----------
    precision : array-like
        Precision values
    recall : array-like
        Recall values
    average_precision : float
        Average precision score
    """
    fig = go.Figure()
    
    # Add PR curve
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=f"PR (AP = {average_precision:.4f})",
        line=dict(color=PRIMARY_COLOR, width=2)
    ))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis=dict(title="Recall"),
        yaxis=dict(title="Precision"),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)"),
        width=600,
        height=500
    )
    
    st.plotly_chart(fig)


def display_metrics_dashboard(metrics: Dict[str, float]):
    """
    Display a dashboard of model performance metrics.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics to display
    """
    # Create multiple rows of metrics
    metrics_per_row = 3
    metrics_items = list(metrics.items())
    
    for i in range(0, len(metrics_items), metrics_per_row):
        row_metrics = metrics_items[i:i+metrics_per_row]
        cols = st.columns(metrics_per_row)
        
        for j, (metric_name, metric_value) in enumerate(row_metrics):
            # Format the metric value based on type
            if isinstance(metric_value, float):
                formatted_value = f"{metric_value:.4f}"
            else:
                formatted_value = str(metric_value)
            
            # Determine display color based on metric name
            if any(term in metric_name.lower() for term in ["auc", "accuracy", "precision", "recall", "f1"]):
                delta_color = "normal"  # Use standard Streamlit coloring
            else:
                delta_color = "off"  # No coloring
            
            # Display the metric
            cols[j].metric(
                label=metric_name,
                value=formatted_value,
                delta=None,
                delta_color=delta_color
            )

# ============================================================================
# Workflow guidance components
# ============================================================================

def workflow_progress(current_step: int, total_steps: int, step_names: List[str] = None):
    """
    Display a workflow progress indicator.
    
    Parameters
    ----------
    current_step : int
        Current step in the workflow (0-indexed)
    total_steps : int
        Total number of steps in the workflow
    step_names : List[str], optional
        Names for each step
    """
    # Calculate progress as a fraction
    progress = current_step / (total_steps - 1) if total_steps > 1 else 1.0
    
    # Create a progress bar
    st.progress(progress)
    
    # If step names are provided, display them
    if step_names:
        # Create a row of step markers
        cols = st.columns(total_steps)
        
        for i, (col, name) in enumerate(zip(cols, step_names)):
            if i < current_step:
                # Completed step
                col.markdown(f"✅ **{name}**")
            elif i == current_step:
                # Current step
                col.markdown(f"🔷 **{name}**")
            else:
                # Future step
                col.markdown(f"⚪ {name}")


def next_steps_guide(completed_steps: List[str], next_steps: List[str]):
    """
    Display guidance for next steps in the workflow.
    
    Parameters
    ----------
    completed_steps : List[str]
        List of completed steps
    next_steps : List[str]
        List of recommended next steps
    """
    with st.expander("Workflow Progress", expanded=True):
        st.write("### Your Progress")
        
        if completed_steps:
            st.write("✅ **Completed:**")
            for step in completed_steps:
                st.write(f"- {step}")
        
        if next_steps:
            st.write("⏭️ **Recommended Next Steps:**")
            for step in next_steps:
                st.write(f"- {step}")
        
        if not completed_steps and not next_steps:
            st.write("No steps recorded yet. Begin by loading data.")


def section_header(title: str, description: str = None, help_text: str = None):
    """
    Create a consistently styled section header.
    
    Parameters
    ----------
    title : str
        Section title
    description : str, optional
        Brief description of the section
    help_text : str, optional
        Detailed help text to show in an expander
    """
    st.write(f"## {title}")
    
    if description:
        st.write(description)
    
    if help_text:
        with st.expander("ℹ️ Learn More", expanded=False):
            st.markdown(help_text) 