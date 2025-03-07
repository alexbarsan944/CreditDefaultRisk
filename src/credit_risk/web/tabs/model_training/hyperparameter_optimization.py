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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import os
import sys
from datetime import datetime

# Add MLflow imports
import mlflow

from credit_risk.models.estimator import CreditRiskModel
from credit_risk.utils.mlflow_tracking import ExperimentTracker

logger = logging.getLogger(__name__)

# Initialize MLflow tracker
tracker = ExperimentTracker()

# Parameter ranges for hyperparameter optimization
PARAM_RANGES = {
    "xgboost": {
        "n_estimators": [50, 500],
        "max_depth": [3, 10],
        "learning_rate": [0.01, 0.3],
        "subsample": [0.5, 1.0],
        "colsample_bytree": [0.5, 1.0],
        "gamma": [0, 5],
        "min_child_weight": [1, 10]
    },
    "lightgbm": {
        "n_estimators": [50, 500],
        "max_depth": [3, 10],
        "learning_rate": [0.01, 0.3],
        "num_leaves": [20, 200],
        "subsample": [0.5, 1.0],
        "colsample_bytree": [0.5, 1.0]
    },
    "catboost": {
        "n_estimators": [50, 500],
        "max_depth": [3, 10],
        "learning_rate": [0.01, 0.3],
        "random_strength": [0.1, 10],
        "bagging_temperature": [0, 10]
    }
}


class IterationTrackingCallback:
    """Callback for tracking iterations during hyperparameter optimization"""
    
    def __init__(self):
        self.iterations = []
        self.best_score = None
        self.best_params = None
        # Add explicit tracking for scores, timestamps, and params
        self.scores = []
        self.timestamps = []
        self.params = []
        self.start_time = time.time()
    
    def __call__(self, res):
        """Called after each iteration"""
        self.iterations.append(res)
        
        # The result dict format can vary depending on the optimization library
        # It might have 'fun' instead of 'target' or other keys
        target_score = None
        
        # Try different possible keys for the target score
        if hasattr(res, 'fun'):
            target_score = res.fun
        elif isinstance(res, dict) and 'target' in res:
            target_score = res['target']
        elif isinstance(res, dict) and 'fun' in res:
            target_score = res['fun']
        elif isinstance(res, dict) and 'objective_value' in res:
            target_score = res['objective_value']
        else:
            # If we can't find a recognized score key, just append 0
            logger.warning(f"Could not find score in optimization result. Type: {type(res)}")
            target_score = 0.0
        
        # Get the parameters - handle both dict and object formats
        if hasattr(res, 'x'):
            params = res.x
        elif isinstance(res, dict) and 'params' in res:
            params = res['params']
        elif isinstance(res, dict) and 'x' in res:
            params = res['x']
        else:
            logger.warning(f"Could not find params in optimization result. Type: {type(res)}")
            params = []
        
        # Track scores, timestamps, and params
        auc_score = 1.0 - target_score if target_score <= 1.0 else 0.0
        self.scores.append(auc_score)
        self.timestamps.append(time.time() - self.start_time)
        self.params.append(params)
        
        # Update best score and params if this is better
        if self.best_score is None or target_score < self.best_score:
            self.best_score = target_score
            self.best_params = params
            
            # Log progress
            logger.info(f"New best score: {1 - self.best_score:.4f}, params: {self.best_params}")


def _objective_function(params, X_train, y_train, X_val, y_val, model_type, random_state, use_gpu):
    """Objective function for hyperparameter optimization."""
    # Convert params to a dict with proper parameter names
    if model_type == "xgboost":
        model_params = {
            "n_estimators": params[0],
            "max_depth": params[1],
            "learning_rate": params[2],
            "subsample": params[3],
            "colsample_bytree": params[4],
            "random_state": random_state,
            "eval_metric": "auc"
        }
        
        if use_gpu:
            model_params["tree_method"] = "gpu_hist"
        else:
            model_params["tree_method"] = "hist"
            
    elif model_type == "lightgbm":
        model_params = {
            "n_estimators": params[0],
            "max_depth": params[1],
            "learning_rate": params[2],
            "num_leaves": params[3],
            "subsample": params[4],
            "colsample_bytree": params[5],
            "random_state": random_state,
            "eval_metric": "auc",
            "n_jobs": -1  # Always use all cores
        }
            
        # Only add GPU config if requested and in a try/except block to handle failures
        if use_gpu:
            model_params["device"] = "gpu"
        
    elif model_type == "catboost":
        model_params = {
            "n_estimators": params[0],
            "depth": params[1],
            "learning_rate": params[2],
            "random_strength": params[3],
            "bagging_temperature": params[4],
            "random_state": random_state,
            "eval_metric": "AUC"
        }
        
        if use_gpu:
            model_params["task_type"] = "GPU"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Safely filter to keep only numeric columns that can be used by all models
    try:
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(numeric_cols) < X_train.shape[1]:
            logger.warning(f"Filtering out {X_train.shape[1] - len(numeric_cols)} non-numeric columns for model training")
            X_train = X_train[numeric_cols]
            if X_val is not None:
                # Only keep columns that exist in X_val
                common_cols = [col for col in numeric_cols if col in X_val.columns]
                X_val = X_val[common_cols]
                X_train = X_train[common_cols]
    except Exception as e:
        logger.warning(f"Error while filtering numeric columns: {str(e)}")
    
    # Define a cross-validation strategy
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    
    try:
        # Initialize the appropriate model
        if model_type == "xgboost":
            model = XGBClassifier(**model_params)
        elif model_type == "lightgbm":
            try:
                # First try with GPU if requested
                model = LGBMClassifier(**model_params)
                if use_gpu:
                    # Try a small fit to see if GPU works
                    try:
                        sample_X = X_train.iloc[:100]
                        sample_y = y_train.iloc[:100]
                        # Make sure sample data is clean
                        sample_X = sample_X.fillna(0)
                        model.fit(sample_X, sample_y)
                    except Exception as e:
                        if "GPU Tree Learner was not enabled" in str(e):
                            raise e  # Re-raise this specific error to be caught below
                        else:
                            logger.warning(f"Error during GPU test fit: {str(e)}")
            except Exception as e:
                # If GPU fails, fall back to CPU
                if "GPU Tree Learner was not enabled" in str(e) and use_gpu:
                    logger.warning(f"GPU not available for LightGBM: {str(e)}")
                    logger.info("Falling back to CPU for LightGBM")
                    # Remove GPU-specific params and retry
                    if "device" in model_params:
                        del model_params["device"]
                    model = LGBMClassifier(**model_params)
        elif model_type == "catboost":
            model = CatBoostClassifier(**model_params)
        
        # Perform cross-validation
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            try:
                cv_X_train, cv_X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                cv_y_train, cv_y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Handle potential NaN values in the data
                cv_X_train = cv_X_train.fillna(0)
                cv_X_val = cv_X_val.fillna(0)
                
                # Handle potential infinite values in numeric columns
                for col in cv_X_train.select_dtypes(include=['int64', 'float64']).columns:
                    try:
                        # Safe replacement of inf values
                        mask_train = (cv_X_train[col] == np.inf) | (cv_X_train[col] == -np.inf)
                        mask_val = (cv_X_val[col] == np.inf) | (cv_X_val[col] == -np.inf)
                        
                        if mask_train.any():
                            cv_X_train.loc[mask_train, col] = 0
                        if mask_val.any():
                            cv_X_val.loc[mask_val, col] = 0
                    except Exception:
                        # If direct check fails, try catching extremely large values
                        try:
                            too_large_train = cv_X_train[col].abs() > 1e30
                            too_large_val = cv_X_val[col].abs() > 1e30
                            
                            if too_large_train.any():
                                cv_X_train.loc[too_large_train, col] = 0
                            if too_large_val.any():
                                cv_X_val.loc[too_large_val, col] = 0
                        except Exception:
                            # If all else fails, skip this column's inf check
                            pass
                
                # Train and evaluate
                model.fit(cv_X_train, cv_y_train)
                y_pred = model.predict_proba(cv_X_val)[:, 1]
                score = roc_auc_score(cv_y_val, y_pred)
                cv_scores.append(score)
            except Exception as e:
                logger.error(f"Error in CV fold: {str(e)}")
                # Instead of returning 0.5, add a slightly lower score to avoid NaN
                cv_scores.append(0.5)
        
        # Get mean AUC score across CV folds (higher is better, so we negate for minimization)
        if len(cv_scores) > 0:
            mean_auc = np.mean(cv_scores)
        else:
            mean_auc = 0.5  # Default if no scores were computed
        
        # For our optimization, we return 1 - AUC (to minimize)
        target = 1.0 - mean_auc
        
    except Exception as e:
        logger.error(f"Exception in hyperparameter evaluation: {str(e)}")
        # Return a default value instead of NaN to avoid breaking optimization
        target = 0.5  # This represents AUC of 0.5 (1-0.5=0.5, representing random model)
        
    # Never return NaN - if something went wrong, return 0.5 (1-0.5=0.5, representing random model)
    if np.isnan(target):
        logger.warning("NaN detected in objective function output, returning 0.5 instead")
        target = 0.5
        
    return target


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
    progress_callback = None
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
    use_gpu : bool
        Whether to use GPU for model training
    n_calls : int
        Number of optimization iterations
    cv_folds : int
        Number of CV folds for evaluation
    random_state : int
        Random seed for reproducibility
    progress_callback : callable
        Callback function to update progress. Should accept three args: trial_num, total_trials, best_value
        
    Returns:
    --------
    Tuple[Dict, Any, IterationTrackingCallback]
        Best parameters, best model, and callback with optimization history
    """
    logger.info(f"Starting hyperparameter optimization for {model_type} with {n_calls} iterations")
    
    # Start MLflow run for this optimization
    run_name = f"hyperopt_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with tracker.start_run(run_name=run_name, tags={"run_type": "hyperparameter_optimization", "model_type": model_type}):
        # Log basic information about this optimization run
        tracker.log_params({
            "model_type": model_type,
            "n_calls": n_calls,
            "cv_folds": cv_folds,
            "random_state": random_state,
            "use_gpu": use_gpu,
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0],
            "n_val_samples": X_val.shape[0] if X_val is not None else 0,
        })
        
        # Handle NaN values in all data
        if X_train.isna().any().any():
            logger.warning("NaN values detected in training data. Cleaning data.")
            X_train = X_train.fillna(0)
        
        # Safely handle infinite values - use try/except to handle type errors
        try:
            # Get numeric columns using a safer approach
            numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if len(numeric_cols) > 0:
                # Replace inf values safely using a loop to handle exceptions
                for col in numeric_cols:
                    try:
                        # Check if the column contains inf values
                        if (X_train[col] == np.inf).any() or (X_train[col] == -np.inf).any():
                            logger.warning(f"Infinite values detected in column {col}. Replacing with zeros.")
                            X_train[col] = X_train[col].replace([np.inf, -np.inf], 0)
                    except (TypeError, ValueError):
                        # If we can't check for inf directly, try using the absolute value
                        try:
                            too_large = X_train[col].abs() > 1e30
                            if too_large.any():
                                logger.warning(f"Very large values detected in column {col}. Replacing with zeros.")
                                X_train.loc[too_large, col] = 0
                        except Exception as e:
                            logger.warning(f"Could not check column {col} for extremely large values: {str(e)}")
        except Exception as e:
            logger.warning(f"Error checking for infinite values: {str(e)}")
        
        # Check for non-numeric columns and warn about them
        non_numeric_cols = [col for col in X_train.columns if col not in numeric_cols]
        if len(non_numeric_cols) > 0:
            logger.warning(f"Found {len(non_numeric_cols)} non-numeric columns which will be automatically encoded or dropped during model training.")
            
        if y_train.isna().any():
            logger.warning(f"NaN values detected in training target. Removing {y_train.isna().sum()} rows.")
            valid_indices = y_train[~y_train.isna()].index
            X_train = X_train.loc[valid_indices]
            y_train = y_train.loc[valid_indices]
            
        if X_val is not None and y_val is not None:
            if X_val.isna().any().any():
                logger.warning("NaN values detected in validation data. Cleaning data.")
                X_val = X_val.fillna(0)
                
            # Safely handle infinite values in validation data
            try:
                numeric_cols_val = X_val.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if len(numeric_cols_val) > 0:
                    for col in numeric_cols_val:
                        try:
                            if (X_val[col] == np.inf).any() or (X_val[col] == -np.inf).any():
                                logger.warning(f"Infinite values detected in validation column {col}. Replacing with zeros.")
                                X_val[col] = X_val[col].replace([np.inf, -np.inf], 0)
                        except (TypeError, ValueError):
                            try:
                                too_large = X_val[col].abs() > 1e30
                                if too_large.any():
                                    logger.warning(f"Very large values detected in validation column {col}. Replacing with zeros.")
                                    X_val.loc[too_large, col] = 0
                            except Exception as e:
                                logger.warning(f"Could not check validation column {col} for extremely large values: {str(e)}")
            except Exception as e:
                logger.warning(f"Error checking validation data for infinite values: {str(e)}")
                
            if y_val.isna().any():
                logger.warning(f"NaN values detected in validation target. Removing {y_val.isna().sum()} rows.")
                valid_indices = y_val[~y_val.isna()].index
                X_val = X_val.loc[valid_indices]
                y_val = y_val.loc[valid_indices]
        
        # Define hyperparameter search spaces
        dimensions = get_parameter_space(model_type)
        
        # Create a callback to track iterations
        callback = IterationTrackingCallback()
        
        # Handle absent validation data by using None
        if X_val is None or y_val is None:
            logger.info("No validation data available. Using training data for validation.")
            X_val = X_train
            y_val = y_train
        
        # Define objective function via partial application to avoid global vars
        def obj_func(params):
            # Track trial start time for logging
            trial_start = time.time()
            
            # Get the result from the objective function
            result = _objective_function(
                params=params,
                X_train=X_train,
                y_train=y_train, 
                X_val=X_val,
                y_val=y_val,
                model_type=model_type,
                random_state=random_state,
                use_gpu=use_gpu
            )
            
            # Calculate trial duration
            trial_duration = time.time() - trial_start
            
            # Log this trial to MLflow
            with mlflow.start_run(nested=True, run_name=f"trial_{len(callback.scores) + 1}"):
                # Convert parameter array to dictionary for logging
                param_dict = get_model_params_from_space(model_type, params, random_state)
                
                # Log parameters
                mlflow.log_params(param_dict)
                
                # Log key metrics
                auc_score = 1.0 - result
                mlflow.log_metric("auc", auc_score)
                mlflow.log_metric("objective_value", result)
                mlflow.log_metric("trial_duration", trial_duration)
                
                # Log this trial as artifact for tracking results over time
                trial_data = {
                    "trial_num": len(callback.scores) + 1,
                    "params": param_dict,
                    "auc": auc_score,
                    "objective_value": result,
                    "duration": trial_duration
                }
                
                # Create temporary file to store the trial data
                import json
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                    json.dump(trial_data, f, default=str)
                    temp_path = f.name
                
                # Log the trial data as an artifact
                mlflow.log_artifact(temp_path, "trials")
                
                # Clean up the temporary file
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {str(e)}")
            
            return result
        
        # Run optimization
        try:
            result = gp_minimize(
                func=obj_func,
                dimensions=dimensions,
                n_calls=n_calls,
                random_state=random_state,
                n_random_starts=min(10, n_calls),
                callback=[callback, 
                          # Add a progress update callback
                          lambda res: _update_progress(res, n_calls, progress_callback) if progress_callback else None,
                          # Add early stopping to prevent wasting time on poor models
                          lambda res: _early_stopping(res) if len(res.func_vals) > 5 else False],
                n_jobs=1  # Ensure we're not starting parallel processes which could cause CUDA issues
            )
            
            # Get the best parameters
            best_params = get_model_params_from_space(model_type, result.x, random_state)
            
            # Log the best parameters and metrics to MLflow
            tracker.log_params({f"best_{k}": v for k, v in best_params.items()})
            tracker.log_metric("best_auc", 1.0 - result.fun)
            tracker.log_metric("total_trials", len(result.func_vals))
            tracker.log_metric("optimization_time", sum(callback.timestamps))
            
            # Log optimization convergence as a plot
            try:
                # Create a convergence plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(1, len(callback.scores) + 1), callback.scores, 'b-', label='AUC Score')
                
                # Add best score line
                best_scores = np.maximum.accumulate(callback.scores)
                ax.plot(range(1, len(best_scores) + 1), best_scores, 'g-', label='Best AUC')
                
                # Add labels and title
                ax.set_xlabel('Iteration')
                ax.set_ylabel('AUC Score')
                ax.set_title('Hyperparameter Optimization Convergence')
                ax.legend()
                ax.grid(True)
                
                # Log the figure to MLflow
                tracker.log_figure(fig, "convergence_plot.png")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating convergence plot: {str(e)}")
            
            # Train a final model with the best parameters
            try:
                best_model = CreditRiskModel(model_type=model_type, model_params=best_params, random_state=random_state)
                best_model.fit(X_train, y_train)
                
                # Evaluate on validation data
                if X_val is not None and y_val is not None:
                    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
                    final_auc = roc_auc_score(y_val, y_pred_proba)
                    tracker.log_metric("final_validation_auc", final_auc)
                
                # Log the final model to MLflow
                tracker.log_model(best_model, "best_model")
                
                # Log feature importances if available
                if hasattr(best_model, "feature_importances_"):
                    feature_importances = {
                        feature: importance 
                        for feature, importance in zip(X_train.columns, best_model.feature_importances_)
                    }
                    tracker.log_feature_importance(feature_importances)
                
            except Exception as e:
                logger.error(f"Error training final model: {str(e)}")
                best_model = None
                
            return best_params, best_model, callback
            
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}")
            
            # Instead of raising an error, return default params
            default_params = get_default_params(model_type, random_state)
            
            # Log the error to MLflow
            tracker.log_param("optimization_error", str(e))
            
            try:
                # Try to train with default params 
                default_model = CreditRiskModel(model_type=model_type, model_params=default_params, random_state=random_state)
                default_model.fit(X_train, y_train)
                
                logger.info("Successfully trained model with default parameters as fallback")
                
                # Log the fallback model
                tracker.log_model(default_model, "fallback_model")
                
                return default_params, default_model, callback
            except Exception as e2:
                logger.error(f"Error training fallback model: {str(e2)}")
                raise ValueError(f"Hyperparameter optimization failed: {str(e)}")

# Helper function to update progress with each iteration
def _update_progress(res, total_trials, progress_callback):
    """
    Helper function to update progress during optimization
    
    Parameters:
    -----------
    res : OptimizeResult
        Current optimization result
    progress_callback : callable
        Callback function for tracking progress
    total_trials : int
        Total number of trials expected
    """
    if progress_callback is None:
        return False
    
    current_trial = len(res.x_iters)
    best_value = 1.0 - res.fun if res.fun <= 1.0 else 0.0
    
    # Call the progress callback with trial number, total trials, and best value
    progress_callback(current_trial, total_trials, best_value)
    
    # Always return False to continue optimization
    return False


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

def get_parameter_space(model_type):
    """Get the parameter search space for the specified model type."""
    if model_type == "xgboost":
        return [
            Integer(PARAM_RANGES["xgboost"]["n_estimators"][0], 
                    PARAM_RANGES["xgboost"]["n_estimators"][1]),
            Integer(PARAM_RANGES["xgboost"]["max_depth"][0], 
                    PARAM_RANGES["xgboost"]["max_depth"][1]),
            Real(PARAM_RANGES["xgboost"]["learning_rate"][0], 
                 PARAM_RANGES["xgboost"]["learning_rate"][1], 
                 prior="log-uniform"),
            Real(PARAM_RANGES["xgboost"]["subsample"][0], 
                 PARAM_RANGES["xgboost"]["subsample"][1], 
                 prior="uniform"),
            Real(PARAM_RANGES["xgboost"]["colsample_bytree"][0], 
                 PARAM_RANGES["xgboost"]["colsample_bytree"][1], 
                 prior="uniform")
        ]
    elif model_type == "lightgbm":
        return [
            Integer(PARAM_RANGES["lightgbm"]["n_estimators"][0], 
                    PARAM_RANGES["lightgbm"]["n_estimators"][1]),
            Integer(PARAM_RANGES["lightgbm"]["max_depth"][0], 
                    PARAM_RANGES["lightgbm"]["max_depth"][1]),
            Real(PARAM_RANGES["lightgbm"]["learning_rate"][0], 
                 PARAM_RANGES["lightgbm"]["learning_rate"][1], 
                 prior="log-uniform"),
            Integer(PARAM_RANGES["lightgbm"]["num_leaves"][0], 
                    PARAM_RANGES["lightgbm"]["num_leaves"][1]),
            Real(PARAM_RANGES["lightgbm"]["subsample"][0], 
                 PARAM_RANGES["lightgbm"]["subsample"][1], 
                 prior="uniform"),
            Real(PARAM_RANGES["lightgbm"]["colsample_bytree"][0], 
                 PARAM_RANGES["lightgbm"]["colsample_bytree"][1], 
                 prior="uniform")
        ]
    elif model_type == "catboost":
        return [
            Integer(PARAM_RANGES["catboost"]["n_estimators"][0], 
                    PARAM_RANGES["catboost"]["n_estimators"][1]),
            Integer(PARAM_RANGES["catboost"]["max_depth"][0], 
                    PARAM_RANGES["catboost"]["max_depth"][1]),
            Real(PARAM_RANGES["catboost"]["learning_rate"][0], 
                 PARAM_RANGES["catboost"]["learning_rate"][1], 
                 prior="log-uniform"),
            Real(PARAM_RANGES["catboost"]["random_strength"][0], 
                 PARAM_RANGES["catboost"]["random_strength"][1], 
                 prior="uniform"),
            Real(PARAM_RANGES["catboost"]["bagging_temperature"][0], 
                 PARAM_RANGES["catboost"]["bagging_temperature"][1], 
                 prior="uniform")
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_params_from_space(model_type, params, random_state):
    """Convert the parameter array from search space to a model parameter dictionary."""
    if model_type == "xgboost":
        return {
            "n_estimators": int(params[0]),
            "max_depth": int(params[1]),
            "learning_rate": float(params[2]),
            "subsample": float(params[3]),
            "colsample_bytree": float(params[4]),
            "random_state": random_state,
            "eval_metric": "auc",
            "tree_method": "hist"
        }
    elif model_type == "lightgbm":
        return {
            "n_estimators": int(params[0]),
            "max_depth": int(params[1]),
            "learning_rate": float(params[2]),
            "num_leaves": int(params[3]),
            "subsample": float(params[4]),
            "colsample_bytree": float(params[5]),
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1
        }
    elif model_type == "catboost":
        return {
            "n_estimators": int(params[0]),
            "depth": int(params[1]),
            "learning_rate": float(params[2]),
            "random_strength": float(params[3]),
            "bagging_temperature": float(params[4]),
            "random_state": random_state,
            "eval_metric": "AUC"
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_default_params(model_type, random_state):
    """Get default parameters for a model type."""
    if model_type == "xgboost":
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "tree_method": "hist"
        }
    elif model_type == "lightgbm":
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1
        }
    elif model_type == "catboost":
        return {
            "n_estimators": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "random_strength": 1,
            "bagging_temperature": 1,
            "random_state": random_state
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _early_stopping(res):
    """
    Check if optimization can be stopped early.
    
    Returns True if optimization should stop, False to continue.
    """
    # If we have at least 10 iterations and the best score didn't improve in the last 5
    if len(res.func_vals) >= 10:
        # Get the best value so far (remember we're minimizing 1-AUC)
        best_value = min(res.func_vals)
        
        # Check if any of the last 5 trials improved the best score
        last_5_best = min(res.func_vals[-5:])
        
        # If we haven't improved in the last 5 iterations and we're at a reasonable score
        if last_5_best > best_value and best_value < 0.4:  # 0.4 means AUC > 0.6
            logger.info(f"Early stopping triggered. Best value: {1-best_value:.4f} AUC")
            return True
    
    # Continue optimization
    return False 