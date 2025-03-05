"""
Hyperparameter optimization functionality for model training.
"""

import logging
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import streamlit as st

from credit_risk.models.estimator import CreditRiskModel

logger = logging.getLogger(__name__)

# Parameter ranges for hyperparameter optimization
PARAM_RANGES = {
    "xgboost": {
        "n_estimators": (50, 500),
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "gamma": (0, 10),
        "min_child_weight": (1, 10)
    },
    "lightgbm": {
        "n_estimators": (50, 500),
        "max_depth": (3, 15),
        "learning_rate": (0.01, 0.3),
        "num_leaves": (10, 255),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "min_child_weight": (1, 100)
    },
    "catboost": {
        "n_estimators": (50, 500),
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.5, 1.0),
        "colsample_bylevel": (0.5, 1.0),
        "l2_leaf_reg": (1, 100)
    }
}


class IterationTrackingCallback:
    """Callback for tracking iterations during hyperparameter optimization"""
    
    def __init__(self):
        self.iterations = []
        self.best_score = None
        self.best_params = None
    
    def __call__(self, res):
        """Called after each iteration"""
        self.iterations.append(res)
        
        # The result dict format can vary depending on the optimization library
        # It might have 'fun' instead of 'target' or other keys
        target_score = None
        
        # Try different possible keys for the target score
        if 'target' in res:
            target_score = res['target']
        elif 'fun' in res:
            target_score = res['fun']
        elif 'objective_value' in res:
            target_score = res['objective_value']
        else:
            # If we can't find a recognized score key, just append the result
            logger.warning(f"Could not find score in optimization result. Keys: {list(res.keys())}")
            return
        
        # Get the parameters
        params = res.get('params', res.get('x', None))
        
        # Update best score and params if this is better
        if self.best_score is None or target_score < self.best_score:
            self.best_score = target_score
            self.best_params = params
            
            # Log progress
            logger.info(f"New best score: {1 - self.best_score:.4f}, params: {self.best_params}")


def run_hyperparameter_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str,
    use_gpu: bool = False,
    n_calls: int = 30,
    cv_folds: int = 3,
    random_state: int = 42,
    progress_bar = None
) -> Tuple[Dict, Any, IterationTrackingCallback]:
    """
    Run hyperparameter optimization for a given model type.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    model_type : str
        Type of model (xgboost, lightgbm, catboost)
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default False
    n_calls : int, optional
        Number of iterations for optimization, by default 30
    cv_folds : int, optional
        Number of cross-validation folds, by default 3
    random_state : int, optional
        Random seed for reproducibility, by default 42
    progress_bar : streamlit.ProgressBar, optional
        Progress bar for tracking, by default None
        
    Returns:
    --------
    Tuple[Dict, Any, IterationTrackingCallback]
        Best parameters, optimization result, and tracking callback
    """
    start_time = time.time()
    
    # Set base parameters based on whether GPU is available
    base_params = {
        "eval_metric": "auc",
        "random_state": random_state,
        "n_jobs": 1,  # Use 1 for cross-validation when GPU is used
        "verbosity": 0
    }
    
    if use_gpu:
        if model_type == "xgboost":
            # Fix for the deprecated gpu_hist parameter in XGBoost 2.0+
            base_params["tree_method"] = "hist"
            base_params["device"] = "cuda"
        elif model_type == "lightgbm":
            base_params["device_type"] = "gpu"
        elif model_type == "catboost":
            base_params["task_type"] = "GPU"
    
    # Define parameter space based on model type
    if model_type == "xgboost":
        space = [
            Integer(PARAM_RANGES["xgboost"]["n_estimators"][0], 
                    PARAM_RANGES["xgboost"]["n_estimators"][1], 
                    name="n_estimators"),
            Integer(PARAM_RANGES["xgboost"]["max_depth"][0], 
                    PARAM_RANGES["xgboost"]["max_depth"][1], 
                    name="max_depth"),
            Real(PARAM_RANGES["xgboost"]["learning_rate"][0], 
                 PARAM_RANGES["xgboost"]["learning_rate"][1], 
                 name="learning_rate", prior="log-uniform"),
            Real(PARAM_RANGES["xgboost"]["subsample"][0], 
                 PARAM_RANGES["xgboost"]["subsample"][1], 
                 name="subsample", prior="uniform"),
            Real(PARAM_RANGES["xgboost"]["colsample_bytree"][0], 
                 PARAM_RANGES["xgboost"]["colsample_bytree"][1], 
                 name="colsample_bytree", prior="uniform"),
            Real(PARAM_RANGES["xgboost"]["gamma"][0], 
                 PARAM_RANGES["xgboost"]["gamma"][1], 
                 name="gamma", prior="uniform"),
            Integer(PARAM_RANGES["xgboost"]["min_child_weight"][0], 
                    PARAM_RANGES["xgboost"]["min_child_weight"][1], 
                    name="min_child_weight")
        ]
        
    elif model_type == "lightgbm":
        space = [
            Integer(PARAM_RANGES["lightgbm"]["n_estimators"][0], 
                    PARAM_RANGES["lightgbm"]["n_estimators"][1], 
                    name="n_estimators"),
            Integer(PARAM_RANGES["lightgbm"]["max_depth"][0], 
                    PARAM_RANGES["lightgbm"]["max_depth"][1], 
                    name="max_depth"),
            Real(PARAM_RANGES["lightgbm"]["learning_rate"][0], 
                 PARAM_RANGES["lightgbm"]["learning_rate"][1], 
                 name="learning_rate", prior="log-uniform"),
            Integer(PARAM_RANGES["lightgbm"]["num_leaves"][0], 
                    PARAM_RANGES["lightgbm"]["num_leaves"][1], 
                    name="num_leaves"),
            Real(PARAM_RANGES["lightgbm"]["subsample"][0], 
                 PARAM_RANGES["lightgbm"]["subsample"][1], 
                 name="subsample", prior="uniform"),
            Real(PARAM_RANGES["lightgbm"]["colsample_bytree"][0], 
                 PARAM_RANGES["lightgbm"]["colsample_bytree"][1], 
                 name="colsample_bytree", prior="uniform")
        ]
        
    elif model_type == "catboost":
        space = [
            Integer(PARAM_RANGES["catboost"]["n_estimators"][0], 
                    PARAM_RANGES["catboost"]["n_estimators"][1], 
                    name="n_estimators"),
            Integer(PARAM_RANGES["catboost"]["max_depth"][0], 
                    PARAM_RANGES["catboost"]["max_depth"][1], 
                    name="max_depth"),
            Real(PARAM_RANGES["catboost"]["learning_rate"][0], 
                 PARAM_RANGES["catboost"]["learning_rate"][1], 
                 name="learning_rate", prior="log-uniform"),
            Real(PARAM_RANGES["catboost"]["l2_leaf_reg"][0], 
                 PARAM_RANGES["catboost"]["l2_leaf_reg"][1], 
                 name="l2_leaf_reg", prior="uniform")
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create callback to track iterations
    callback = IterationTrackingCallback()
    
    # Define objective function
    @use_named_args(space)
    def objective(**params):
        # Set base parameters
        all_params = {**base_params, **params}
        
        # Create model
        model = CreditRiskModel(
            model_type=model_type,
            model_params=all_params,
            random_state=random_state
        )
        
        if cv_folds > 1:
            # Perform cross-validation
            try:
                cv_results = model.cross_validate(X_train, y_train, n_folds=cv_folds, stratified=True)
                # The correct key is 'mean_auc', not 'test_auc'
                cv_auc = cv_results.get("mean_auc", 0.0)
                logger.info(f"Parameters: {params}, CV AUC: {cv_auc:.4f}")
                # We minimize, so return 1 - AUC
                return 1.0 - cv_auc
            except Exception as e:
                logger.error(f"Error in cross-validation: {str(e)}")
                # Return a high value on error to penalize this parameter set
                return 1.0
        else:
            # Train once and evaluate on validation set
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, y_pred)
                logger.info(f"Parameters: {params}, Val AUC: {val_auc:.4f}")
                # We minimize, so return 1 - AUC
                return 1.0 - val_auc
            except Exception as e:
                logger.error(f"Error in model training: {str(e)}")
                # Return a high value on error to penalize this parameter set
                return 1.0
    
    # Run optimization
    logger.info(f"Starting hyperparameter optimization for {model_type} with {n_calls} iterations...")
    
    try:
        # Update progress
        total_time_start = time.time()
        if progress_bar:
            progress_bar.progress(0.05, "Starting hyperparameter optimization...")
        
        # Run the optimizer
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=random_state,
            n_jobs=1,  # Ensure we don't conflict with model parallelism
            verbose=True,
            callback=callback
        )
        
        # Finish up optimization process
        optimization_time = time.time() - total_time_start
        
        # Get the best parameters
        space_names = [dim.name for dim in space]
        best_params = {}
        for idx, param_name in enumerate(space_names):
            best_params[param_name] = result.x[idx]
        
        # Apply parameter conversion for any integer parameters
        integer_params = [dim.name for dim in space if isinstance(dim, Integer)]
        for param_name, param_value in best_params.items():
            if param_name in integer_params:
                best_params[param_name] = int(round(param_value))
        
        # Add fixed parameters
        best_params.update(base_params)
        
        # Log results
        best_score = 1.0 - result.fun if result.fun <= 1.0 else 0.0
        logger.info(f"Hyperparameter optimization complete. Time taken: {optimization_time:.2f} seconds")
        logger.info(f"Best score: {best_score:.4f}, best params: {best_params}")
        
        # Final progress update
        if progress_bar:
            progress_bar.progress(1.0, "Hyperparameter optimization complete")
        
        return best_params, result, callback
        
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {str(e)}", exc_info=True)
        raise ValueError(f"Hyperparameter optimization failed: {str(e)}")


def plot_optimization_progress(iterations):
    """
    Plot the progress of hyperparameter optimization.
    
    Parameters:
    -----------
    iterations : list
        List of iteration results from the IterationTrackingCallback
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure showing optimization progress
    """
    if not iterations:
        return None
    
    try:
        # Extract values
        x = list(range(1, len(iterations) + 1))
        
        # Get target values (could be 'fun', 'target', etc.)
        y_values = []
        for res in iterations:
            if 'target' in res:
                y_values.append(res['target'])
            elif 'fun' in res:
                y_values.append(res['fun'])
            elif 'objective_value' in res:
                y_values.append(res['objective_value'])
            else:
                logger.warning(f"Could not find score in iteration result. Keys: {list(res.keys())}")
                y_values.append(None)
        
        # Create cumulative minimum to show best score at each iteration
        y_cum_min = [y_values[0]]
        for i in range(1, len(y_values)):
            if y_values[i] is None:
                y_cum_min.append(y_cum_min[-1])
            else:
                y_cum_min.append(min(y_cum_min[-1], y_values[i]))
        
        # Convert to AUC (we minimize 1-AUC in the optimization)
        y_auc = [1 - val if val is not None else None for val in y_values]
        y_cum_max_auc = [1 - val for val in y_cum_min]
        
        # Create figure
        fig = go.Figure()
        
        # Add iteration points
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y_auc, 
                mode='markers', 
                name='Iteration AUC',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.6
                )
            )
        )
        
        # Add best so far line
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y_cum_max_auc, 
                mode='lines+markers', 
                name='Best AUC',
                line=dict(
                    color='green',
                    width=2
                ),
                marker=dict(
                    size=6,
                    color='green',
                    opacity=0.8
                )
            )
        )
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            title="Hyperparameter Optimization Progress",
            xaxis_title="Iteration",
            yaxis_title="AUC Score",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating optimization progress plot: {str(e)}", exc_info=True)
        return None 