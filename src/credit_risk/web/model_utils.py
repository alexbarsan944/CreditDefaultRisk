"""
Utility functions for the web application to interact with feature selection and model modules.

This module provides functions to:
1. Load and preprocess data
2. Perform feature selection
3. Train and evaluate models
4. Make predictions
5. Visualize results
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import sys
import importlib
import json
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, 
    precision_score, recall_score, f1_score,
    confusion_matrix
)

# Attempt to import FeatureSelector with fallback
try:
    from credit_risk.features.selection import FeatureSelector
    from credit_risk.models import CreditRiskModel
except ImportError:
    # Check if we can import directly from the module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from src.credit_risk.features.selection.selector import FeatureSelector
        from src.credit_risk.models.estimator import CreditRiskModel
    except ImportError:
        # If all else fails, define a simple feature selector class
        print("Failed to import FeatureSelector, using a simplified version")
        
        class FeatureSelector:
            """A simplified version of FeatureSelector for use when imports fail."""
            
            def __init__(self, random_state=42):
                self.random_state = random_state
                self.useful_features = None
            
            def fit(self, X, y, n_runs=100, split_score_threshold=0, gain_score_threshold=0, correlation_threshold=0.95):
                """Simple fit that just keeps all features."""
                self.useful_features = X.columns.tolist()
                return self
            
            def transform(self, X):
                """Return the input DataFrame."""
                return X
            
            def fit_transform(self, X, y, n_runs=100, split_score_threshold=0, gain_score_threshold=0, correlation_threshold=0.95):
                """Fit and transform in one step."""
                self.fit(X, y, n_runs, split_score_threshold, gain_score_threshold, correlation_threshold)
                return self.transform(X)
            
            def get_feature_scores(self, X, y, n_runs=100):
                """Return a dummy feature scores DataFrame."""
                return pd.DataFrame({
                    "feature": X.columns.tolist(),
                    "split_score": [1.0] * len(X.columns),
                    "gain_score": [1.0] * len(X.columns)
                })
            
            def get_useful_features(self):
                """Get the list of useful features."""
                return self.useful_features

logger = logging.getLogger(__name__)

# Define model types and their display names
MODEL_TYPES = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost"
}

def perform_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_runs: int = 20,
    split_score_threshold: float = None,
    gain_score_threshold: float = None,
    correlation_threshold: float = None,
    random_state: int = 42,
    progress_bar=None,
    use_gpu: bool = True
) -> Tuple[pd.DataFrame, FeatureSelector, Dict[str, Any]]:
    """
    Perform feature selection using the FeatureSelector.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    n_runs : int, optional
        Number of null importance runs, by default 20
    split_score_threshold : float or None, optional
        Threshold for feature selection based on split score.
        If None, threshold will be determined automatically.
    gain_score_threshold : float or None, optional
        Threshold for feature selection based on gain score.
        If None, threshold will be determined automatically.
    correlation_threshold : float or None, optional
        Threshold for removing highly correlated features.
        If None, default value of 0.95 will be used.
    random_state : int, optional
        Random seed for reproducibility, by default 42
    progress_bar : streamlit.ProgressBar, optional
        Streamlit progress bar to update
    use_gpu : bool, optional
        Whether to use GPU for feature selection, by default True
        
    Returns
    -------
    Tuple[pd.DataFrame, FeatureSelector, Dict[str, Any]]
        Transformed feature matrix, FeatureSelector instance, and results dictionary
    """
    try:
        # Log start of feature selection
        logger.info(f"Starting feature selection with {X.shape[1]} features")
        logger.info(f"Feature selection parameters: n_runs={n_runs}, use_gpu={use_gpu}")
        
        # Initialize feature selector
        feature_selector = FeatureSelector(
            random_state=random_state
        )
        
        # Update progress
        if progress_bar:
            progress_bar.progress(0.1, "Calculating feature scores...")
        
        # Check if we have any missing values in X or y and log it
        x_missing = X.isna().sum().sum()
        y_missing = y.isna().sum()
        if x_missing > 0:
            logger.warning(f"X contains {x_missing} missing values. This might affect feature selection.")
        if y_missing > 0:
            logger.error(f"y contains {y_missing} missing values. This will likely cause feature selection to fail.")
            # Drop missing values from y and corresponding rows from X
            mask = y.notna()
            X = X[mask]
            y = y[mask]
            logger.info(f"Dropped {y_missing} rows with missing target values. New shapes: X {X.shape}, y {y.shape}")
        
        # Get feature scores
        feature_scores = feature_selector.get_feature_scores(X, y, n_runs=n_runs)
        
        # Log feature scores summary
        if not feature_scores.empty:
            logger.info(f"Feature scores calculated: {len(feature_scores)} features with scores")
            logger.info(f"Split score range: {feature_scores['split_score'].min():.4f} to {feature_scores['split_score'].max():.4f}")
            logger.info(f"Gain score range: {feature_scores['gain_score'].min():.4f} to {feature_scores['gain_score'].max():.4f}")
        else:
            logger.error("Feature scores calculation returned empty DataFrame. Something went wrong.")
            if progress_bar:
                progress_bar.progress(1.0, "Error: Feature scores calculation failed")
            raise ValueError("Feature scores calculation failed")
        
        # Update progress
        if progress_bar:
            progress_bar.progress(0.5, "Applying feature selection...")
        
        # Determine if we're using automatic thresholds
        using_auto_thresholds = (
            split_score_threshold is None or 
            gain_score_threshold is None or 
            correlation_threshold is None
        )
        
        # Apply feature selection
        X_selected = feature_selector.fit_transform(
            X, 
            y, 
            n_runs=n_runs, 
            split_score_threshold=split_score_threshold,
            gain_score_threshold=gain_score_threshold,
            correlation_threshold=correlation_threshold
        )
        
        # Check if any features were selected
        if X_selected.shape[1] == 0:
            logger.warning("No features were selected. This indicates an issue with the feature selection process.")
            # Fall back to using all features
            X_selected = X
            logger.info(f"Falling back to using all {X_selected.shape[1]} features")
        
        # Update progress
        if progress_bar:
            progress_bar.progress(1.0, "Feature selection completed!")
        
        # Prepare results
        results = {
            "feature_scores": feature_scores,
            "original_features": X.shape[1],
            "selected_features": X_selected.shape[1],
            "reduction_percentage": (1 - (X_selected.shape[1] / X.shape[1])) * 100
        }
        
        return X_selected, feature_selector, results
    
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        if progress_bar:
            progress_bar.progress(1.0, f"Error: {str(e)}")
        raise
    
def create_feature_score_plots(feature_scores: pd.DataFrame) -> Dict[str, plt.Figure]:
    """
    Create plots for feature scores.
    
    Parameters
    ----------
    feature_scores : pd.DataFrame
        DataFrame containing feature scores
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of plot figures
    """
    plots = {}
    
    # Define color scheme to match web app
    colors = {
        "positive": "#0068c9",  # Blue color for positive 
        "negative": "#ff5252",  # Red color for negative
        "darker_positive": "#003f77",
        "darker_negative": "#b30000"
    }
    
    # Clean feature names if they're too long (make visible in plot)
    def clean_feature_name(name):
        if isinstance(name, str) and len(name) > 50:
            return name[:47] + "..."
        return name
    
    # Create a copy of the dataframe with cleaned feature names
    feature_scores_clean = feature_scores.copy()
    feature_scores_clean['display_name'] = feature_scores_clean['feature'].apply(clean_feature_name)
    
    # Top features by split score
    top_split = feature_scores_clean.sort_values("split_score", ascending=False).head(20)
    fig1 = px.bar(
        top_split,
        y="display_name",
        x="split_score",
        orientation="h",
        title="Top Features by Split Score",
        color_discrete_sequence=[colors["positive"]],
        text_auto='.2f',
        labels={"display_name": "Feature", "split_score": "Split Score"}
    )
    fig1.update_layout(
        yaxis_title="",
        xaxis_title="Split Score",
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis={'categoryorder':'total ascending'},
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12),
        hovermode="y"
    )
    fig1.update_traces(
        hovertemplate="<b>%{y}</b><br>Split Score: %{x:.3f}<extra></extra>"
    )
    plots["top_split"] = fig1
    
    # Top features by gain score
    top_gain = feature_scores_clean.sort_values("gain_score", ascending=False).head(20)
    fig2 = px.bar(
        top_gain,
        y="display_name",
        x="gain_score",
        orientation="h",
        title="Top Features by Gain Score",
        color_discrete_sequence=[colors["darker_positive"]],
        text_auto='.2f',
        labels={"display_name": "Feature", "gain_score": "Gain Score"}
    )
    fig2.update_layout(
        yaxis_title="",
        xaxis_title="Gain Score",
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis={'categoryorder':'total ascending'},
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12),
        hovermode="y"
    )
    fig2.update_traces(
        hovertemplate="<b>%{y}</b><br>Gain Score: %{x:.3f}<extra></extra>"
    )
    plots["top_gain"] = fig2
    
    # Bottom features by split score
    bottom_split = feature_scores_clean.sort_values("split_score", ascending=True).head(20)
    fig3 = px.bar(
        bottom_split,
        y="display_name",
        x="split_score",
        orientation="h",
        title="Bottom Features by Split Score",
        color_discrete_sequence=[colors["negative"]],
        text_auto='.2f',
        labels={"display_name": "Feature", "split_score": "Split Score"}
    )
    fig3.update_layout(
        yaxis_title="",
        xaxis_title="Split Score",
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis={'categoryorder':'total ascending'},
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12),
        hovermode="y"
    )
    fig3.update_traces(
        hovertemplate="<b>%{y}</b><br>Split Score: %{x:.3f}<extra></extra>"
    )
    plots["bottom_split"] = fig3
    
    # Bottom features by gain score
    bottom_gain = feature_scores_clean.sort_values("gain_score", ascending=True).head(20)
    fig4 = px.bar(
        bottom_gain,
        y="display_name",
        x="gain_score",
        orientation="h",
        title="Bottom Features by Gain Score",
        color_discrete_sequence=[colors["darker_negative"]],
        text_auto='.2f',
        labels={"display_name": "Feature", "gain_score": "Gain Score"}
    )
    fig4.update_layout(
        yaxis_title="",
        xaxis_title="Gain Score",
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis={'categoryorder':'total ascending'},
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12),
        hovermode="y"
    )
    fig4.update_traces(
        hovertemplate="<b>%{y}</b><br>Gain Score: %{x:.3f}<extra></extra>"
    )
    plots["bottom_gain"] = fig4
    
    # Distribution plot for split scores
    fig5 = px.histogram(
        feature_scores_clean,
        x="split_score",
        title="Distribution of Split Scores",
        color_discrete_sequence=[colors["positive"]],
        labels={"split_score": "Split Score"}
    )
    fig5.update_layout(
        xaxis_title="Split Score",
        yaxis_title="Count",
        height=400,
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    plots["split_distribution"] = fig5
    
    # Distribution plot for gain scores
    fig6 = px.histogram(
        feature_scores_clean,
        x="gain_score",
        title="Distribution of Gain Scores",
        color_discrete_sequence=[colors["darker_positive"]],
        labels={"gain_score": "Gain Score"}
    )
    fig6.update_layout(
        xaxis_title="Gain Score",
        yaxis_title="Count",
        height=400,
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    plots["gain_distribution"] = fig6
    
    return plots

def log_model_to_mlflow(
    model, 
    model_type: str, 
    params: Dict[str, Any], 
    metrics: Dict[str, float],
    training_params: Dict[str, Any],
    feature_names: List[str],
    data_source: str,
    tags: Dict[str, str] = None
) -> str:
    """
    Log a model and its associated metadata to MLflow.
    
    Parameters
    ----------
    model : Any
        The trained model object
    model_type : str
        Type of model (lightgbm, xgboost, catboost)
    params : Dict[str, Any]
        Model parameters
    metrics : Dict[str, float]
        Model performance metrics
    training_params : Dict[str, Any]
        Training parameters
    feature_names : List[str]
        Names of features used in the model
    data_source : str
        Source of the data (feature_selection, feature_engineering)
    tags : Dict[str, str], optional
        Additional tags to log with the run
        
    Returns
    -------
    str
        MLflow run ID
    """
    # Setup MLflow - ensures the mlruns directory is created in the current directory
    if not os.path.exists("mlruns"):
        os.makedirs("mlruns", exist_ok=True)
    
    # Get experiment ID or create if it doesn't exist
    experiment_name = f"credit_risk_{model_type}"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Error getting/creating MLflow experiment: {str(e)}")
        experiment_id = mlflow.create_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        
        # Log model parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Log training parameters with prefix
        for key, value in training_params.items():
            # Convert non-serializable types to strings
            if not isinstance(value, (str, int, float, bool)):
                value = str(value)
            mlflow.log_param(f"train_{key}", value)
        
        # Log metrics
        for key, value in metrics.items():
            # Skip complex metrics (like confusion matrix or ROC curve data)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(key, value)
        
        # Log key data characteristics
        mlflow.log_param("data_source", data_source)
        mlflow.log_param("n_features", len(feature_names))
        
        # Log feature names as a separate file
        features_file = "feature_names.json"
        with open(features_file, "w") as f:
            json.dump(feature_names, f)
        mlflow.log_artifact(features_file)
        os.remove(features_file)  # Clean up
        
        # Log tags
        run_tags = {
            "model_type": model_type,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "data_source": data_source
        }
        
        # Add custom tags if provided
        if tags:
            run_tags.update(tags)
        
        # Set tags
        for key, value in run_tags.items():
            mlflow.set_tag(key, value)
        
        # Log model
        if model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, "model")
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        elif model_type == "catboost":
            # CatBoost uses sklearn flavor
            mlflow.sklearn.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"Model logged to MLflow with run ID: {run_id}")
        return run_id

def train_and_evaluate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "lightgbm",
    params: Dict[str, Any] = None,
    use_gpu: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train and evaluate a model with the given parameters.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Testing features
    y_test : pd.Series
        Testing target
    model_type : str, optional
        Type of model to train, by default "lightgbm"
    params : Dict[str, Any], optional
        Model parameters, by default None
    use_gpu : bool, optional
        Whether to use GPU for training, by default False
        
    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        Trained model and evaluation metrics
    """
    logger.info(f"Training {model_type} model with {X_train.shape[1]} features")
    
    # Default parameters for each model type
    default_params = {
        "lightgbm": {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 42,
            "n_estimators": 100,
            "learning_rate": 0.1
        },
        "xgboost": {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
            "seed": 42,
            "n_estimators": 100,
            "learning_rate": 0.1
        },
        "catboost": {
            "objective": "Logloss",
            "eval_metric": "AUC",
            "verbose": 0,
            "random_seed": 42,
            "iterations": 100,
            "learning_rate": 0.1
        }
    }
    
    # Use default parameters if none provided
    if params is None:
        params = default_params.get(model_type, {})
    else:
        # Update default parameters with provided ones
        model_defaults = default_params.get(model_type, {})
        for key, value in model_defaults.items():
            if key not in params:
                params[key] = value
    
    # Add GPU parameters if requested
    if use_gpu:
        if model_type == "lightgbm":
            params["device"] = "gpu"
        elif model_type == "xgboost":
            params["tree_method"] = "gpu_hist"
        elif model_type == "catboost":
            params["task_type"] = "GPU"
    
    # Train the model based on type
    model = None
    if model_type == "lightgbm":
        import lightgbm as lgb
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_data)
    
    elif model_type == "xgboost":
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, dtrain)
    
    elif model_type == "catboost":
        from catboost import CatBoost, Pool
        
        # Identify categorical features
        cat_features = [i for i, col in enumerate(X_train.columns) 
                        if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category']
        
        # Create pool with categorical features
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        
        # Initialize and train model
        model = CatBoost(params)
        model.fit(train_pool, verbose=False)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Make predictions
    y_pred_proba = None
    if model_type == "xgboost":
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = model.predict(dtest)
    elif model_type == "catboost":
        test_pool = Pool(X_test, cat_features=cat_features)
        y_pred_proba = model.predict_proba(test_pool)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
    
    # Convert to binary predictions
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Log to MLflow
    log_model_to_mlflow(
        model=model,
        model_type=model_type,
        params=params,
        metrics=metrics,
        training_params={"use_gpu": use_gpu},
        feature_names=X_train.columns.tolist(),
        data_source="trained_model",
        tags={"training_type": "classic"}
    )
    
    return model, metrics

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for a model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    y_pred_proba : np.ndarray
        Predicted probabilities
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Core metrics
    metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    
    return metrics

def create_cv_plot(cv_results: Dict[str, Any], model_type: str) -> go.Figure:
    """
    Create a plot for cross-validation results.
    
    Parameters
    ----------
    cv_results : Dict[str, Any]
        Cross-validation results
    model_type : str
        Type of model
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    display_name = MODEL_TYPES.get(model_type, model_type.upper())
    
    fig = go.Figure()
    
    # Add bars for each fold
    fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(cv_results["fold_metrics"]))],
        y=cv_results["fold_metrics"],
        marker_color="royalblue",
        name="Fold AUC"
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=[f"Fold {i+1}" for i in range(len(cv_results["fold_metrics"]))],
        y=[cv_results["mean_auc"]] * len(cv_results["fold_metrics"]),
        mode="lines",
        line=dict(color="red", width=2, dash="solid"),
        name=f"Mean AUC: {cv_results['mean_auc']:.4f}"
    ))
    
    # Add std dev lines
    fig.add_trace(go.Scatter(
        x=[f"Fold {i+1}" for i in range(len(cv_results["fold_metrics"]))],
        y=[cv_results["mean_auc"] + cv_results["std_auc"]] * len(cv_results["fold_metrics"]),
        mode="lines",
        line=dict(color="black", width=1, dash="dash"),
        name=f"Standard Deviation: ±{cv_results['std_auc']:.4f}"
    ))
    
    fig.add_trace(go.Scatter(
        x=[f"Fold {i+1}" for i in range(len(cv_results["fold_metrics"]))],
        y=[cv_results["mean_auc"] - cv_results["std_auc"]] * len(cv_results["fold_metrics"]),
        mode="lines",
        line=dict(color="black", width=1, dash="dash"),
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{display_name} Cross-Validation AUC Scores",
        xaxis_title="",
        yaxis_title="AUC Score",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    return fig

def create_confusion_matrix_plot(model: CreditRiskModel, X_test: pd.DataFrame, y_test: pd.Series) -> go.Figure:
    """
    Create a confusion matrix plot.
    
    Parameters
    ----------
    model : CreditRiskModel
        Trained model
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        Test target variable
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Get confusion matrix
    cm, tn, fp, fn, tp = model._get_confusion_matrix(X_test, y_test)
    
    # Create annotation text
    annotations = [
        [f"TN: {tn}<br>({tn/len(y_test):.1%})", f"FP: {fp}<br>({fp/len(y_test):.1%})"],
        [f"FN: {fn}<br>({fn/len(y_test):.1%})", f"TP: {tp}<br>({tp/len(y_test):.1%})"]
    ]
    
    # Create confusion matrix plot
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted No Default", "Predicted Default"],
        y=["Actual No Default", "Actual Default"],
        colorscale=[[0, "#4CAF50"], [1, "#F44336"]],
        showscale=False
    ))
    
    # Add annotations
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j,
                y=i,
                text=annotations[i][j],
                showarrow=False,
                font=dict(
                    color="white" if (i == 1 and j == 1) or (i == 0 and j == 0) else "black",
                    size=14
                )
            )
    
    # Add metrics
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = (
        f"Accuracy: {accuracy:.4f}<br>"
        f"Precision: {precision:.4f}<br>"
        f"Recall: {recall:.4f}<br>"
        f"F1 Score: {f1:.4f}"
    )
    
    fig.add_annotation(
        x=1.35,
        y=0.5,
        text=metrics_text,
        showarrow=False,
        font=dict(size=14),
        align="left"
    )
    
    # Update layout
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="", constrain="domain"),
        yaxis=dict(title="", autorange="reversed"),
        width=700,
        height=500,
        margin=dict(l=50, r=200, t=50, b=50)
    )
    
    return fig

def get_application_prediction(
    model: CreditRiskModel,
    application_data: pd.DataFrame,
    feature_selector: Optional[FeatureSelector] = None
) -> Dict[str, Any]:
    """
    Get prediction for a loan application.
    
    Parameters
    ----------
    model : CreditRiskModel
        Trained model
    application_data : pd.DataFrame
        Application data (single row DataFrame)
    feature_selector : Optional[FeatureSelector], optional
        Feature selector, by default None
        
    Returns
    -------
    Dict[str, Any]
        Prediction results
    """
    try:
        # Apply feature selection if provided
        if feature_selector is not None:
            application_data = feature_selector.transform(application_data)
        
        # Make prediction
        prob = model.predict_proba(application_data)[0, 1]
        prediction = 1 if prob > 0.5 else 0
        
        # Get feature importance for this prediction
        if hasattr(model.model, "feature_importances_"):
            feature_imp = pd.DataFrame({
                "feature": application_data.columns,
                "importance": model.model.feature_importances_
            }).sort_values("importance", ascending=False)
            top_features = feature_imp.head(10)
        else:
            top_features = None
        
        return {
            "probability": prob,
            "prediction": prediction,
            "top_features": top_features
        }
    
    except Exception as e:
        logger.error(f"Error in application prediction: {e}")
        raise

def save_model_and_selector(
    model: CreditRiskModel,
    feature_selector: FeatureSelector,
    model_name: str,
    output_dir: str = "models"
) -> Dict[str, str]:
    """
    Save model and feature selector to disk.
    
    Parameters
    ----------
    model : CreditRiskModel
        Trained model
    feature_selector : FeatureSelector
        Feature selector
    model_name : str
        Name for the saved model files
    output_dir : str, optional
        Directory to save models, by default "models"
        
    Returns
    -------
    Dict[str, str]
        Dictionary with file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    model.save(model_path)
    
    # Save feature selector
    selector_path = os.path.join(output_dir, f"{model_name}_feature_selector.joblib")
    feature_selector.save(selector_path)
    
    # Save selected features list
    selected_features = pd.DataFrame({"feature": feature_selector.get_useful_features()})
    features_path = os.path.join(output_dir, f"{model_name}_selected_features.csv")
    selected_features.to_csv(features_path, index=False)
    
    return {
        "model_path": model_path,
        "selector_path": selector_path,
        "features_path": features_path
    }

def load_model_and_selector(
    model_path: str,
    selector_path: str
) -> Tuple[CreditRiskModel, FeatureSelector]:
    """
    Load model and feature selector from disk.
    
    Parameters
    ----------
    model_path : str
        Path to saved model
    selector_path : str
        Path to saved feature selector
        
    Returns
    -------
    Tuple[CreditRiskModel, FeatureSelector]
        Loaded model and feature selector
    """
    try:
        # Load model
        model = CreditRiskModel.load(model_path)
        
        # Load feature selector
        feature_selector = FeatureSelector.load(selector_path)
        
        return model, feature_selector
    
    except Exception as e:
        logger.error(f"Error loading model and selector: {e}")
        raise

def run_hyperparameter_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "lightgbm",
    n_trials: int = 20,
    timeout: int = 600,
    metric: str = "auc",
    use_gpu: bool = False,
    progress_callback: Callable = None
) -> Tuple[Dict[str, Any], Any, List[Dict[str, Any]]]:
    """
    Run hyperparameter optimization for a model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Testing features
    y_test : pd.Series
        Testing target
    model_type : str, optional
        Type of model to train, by default "lightgbm"
    n_trials : int, optional
        Number of trials for optimization, by default 20
    timeout : int, optional
        Maximum time for optimization in seconds, by default 600
    metric : str, optional
        Metric to optimize for, by default "auc"
    use_gpu : bool, optional
        Whether to use GPU for training, by default False
    progress_callback : Callable, optional
        Callback function for progress updates, by default None
        
    Returns
    -------
    Tuple[Dict[str, Any], Any, List[Dict[str, Any]]]
        Best parameters, best model, and optimization history
    """
    try:
        import optuna
        from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback
    except ImportError:
        logger.error("Optuna is required for hyperparameter optimization")
        raise ImportError("Optuna is required for hyperparameter optimization")
    
    logger.info(f"Starting hyperparameter optimization for {model_type} model")
    
    # Store optimization history
    optimization_history = []
    best_value = 0.0
    
    # Define the objective function for different model types
    def objective(trial):
        nonlocal best_value
        
        # Define hyperparameter search space based on model type
        if model_type == "lightgbm":
            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "seed": 42,
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 8, 128),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
            }
            
            if use_gpu:
                params["device"] = "gpu"
            
            import lightgbm as lgb
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            # Train the model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                verbose_eval=False
            )
            
            # Make predictions
            y_pred_proba = model.predict(X_test)
        
        elif model_type == "xgboost":
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "verbosity": 0,
                "seed": 42,
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
            }
            
            if use_gpu:
                params["tree_method"] = "gpu_hist"
            
            import xgboost as xgb
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # Train the model
            model = xgb.train(
                params,
                dtrain,
                evals=[(dtest, "test")],
                verbose_eval=False
            )
            
            # Make predictions
            y_pred_proba = model.predict(dtest)
        
        elif model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100, log=True),
                "rsm": trial.suggest_float("rsm", 0.5, 1.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
                "random_seed": 42,
                "verbose": 0
            }
            
            if use_gpu:
                params["task_type"] = "GPU"
            
            from catboost import CatBoost, Pool
            
            # Identify categorical features
            cat_features = [i for i, col in enumerate(X_train.columns) 
                            if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category']
            
            # Create pools
            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            test_pool = Pool(X_test, y_test, cat_features=cat_features)
            
            # Initialize and train model
            model = CatBoost(params)
            model.fit(train_pool, eval_set=test_pool, verbose=False)
            
            # Make predictions
            y_pred_proba = model.predict_proba(test_pool)[:, 1]
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Calculate the selected metric
        if metric == "auc":
            value = roc_auc_score(y_test, y_pred_proba)
        elif metric == "accuracy":
            y_pred = (y_pred_proba > 0.5).astype(int)
            value = accuracy_score(y_test, y_pred)
        elif metric == "f1":
            y_pred = (y_pred_proba > 0.5).astype(int)
            value = f1_score(y_test, y_pred)
        else:
            # Default to AUC
            value = roc_auc_score(y_test, y_pred_proba)
        
        # Update best value and call progress callback if provided
        if value > best_value:
            best_value = value
        
        if progress_callback:
            progress_callback(trial.number + 1, n_trials, best_value)
        
        # Store trial information
        optimization_history.append({
            "trial": trial.number,
            "value": value,
            "params": trial.params.copy()
        })
        
        return value
    
    # Create a study and optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters and train final model with them
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")
    
    # Add default parameters not included in the search space
    if model_type == "lightgbm":
        best_params["objective"] = "binary"
        best_params["metric"] = "auc"
        best_params["verbosity"] = -1
        best_params["seed"] = 42
        
        if use_gpu:
            best_params["device"] = "gpu"
        
        import lightgbm as lgb
        train_data = lgb.Dataset(X_train, label=y_train)
        final_model = lgb.train(best_params, train_data)
    
    elif model_type == "xgboost":
        best_params["objective"] = "binary:logistic"
        best_params["eval_metric"] = "auc"
        best_params["verbosity"] = 0
        best_params["seed"] = 42
        
        if use_gpu:
            best_params["tree_method"] = "gpu_hist"
        
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=y_train)
        final_model = xgb.train(best_params, dtrain)
    
    elif model_type == "catboost":
        best_params["random_seed"] = 42
        best_params["verbose"] = 0
        
        if use_gpu:
            best_params["task_type"] = "GPU"
        
        from catboost import CatBoost, Pool
        
        # Identify categorical features
        cat_features = [i for i, col in enumerate(X_train.columns) 
                        if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category']
        
        # Create pools
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        
        # Initialize and train model
        final_model = CatBoost(best_params)
        final_model.fit(train_pool, verbose=False)
    
    # Log optimization results to MLflow
    log_model_to_mlflow(
        model=final_model,
        model_type=model_type,
        params=best_params,
        metrics={"best_" + metric: study.best_value},
        training_params={
            "use_gpu": use_gpu,
            "n_trials": n_trials,
            "timeout": timeout,
            "optimization_metric": metric
        },
        feature_names=X_train.columns.tolist(),
        data_source="optimized_model",
        tags={
            "training_type": "hyperopt",
            "n_trials": str(n_trials),
            "best_trial": str(study.best_trial.number)
        }
    )
    
    return best_params, final_model, optimization_history