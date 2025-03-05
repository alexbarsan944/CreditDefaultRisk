"""
MLflow experiment tracking module.

This module provides utilities for tracking experiments with MLflow,
including metrics, parameters, artifacts, and system metrics.
"""
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from psutil import Process, virtual_memory

from credit_risk.config import MLflowConfig, PipelineConfig, default_config

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow experiment tracker for credit risk prediction.
    
    This class manages experiment tracking with MLflow, including
    metrics, parameters, artifacts, and system metrics.
    
    Attributes:
        config: Configuration for the experiment tracker
        experiment_id: ID of the current MLflow experiment
        active_run: Currently active MLflow run
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the experiment tracker.
        
        Args:
            config: Configuration for the experiment tracker. If None, uses default config.
        """
        self.config = config or default_config
        self.mlflow_config = self.config.mlflow
        self.experiment_id = None
        self.active_run = None
        
        if not self.mlflow_config.enabled:
            logger.info("MLflow tracking is disabled")
            return
        
        # Set up MLflow tracking
        try:
            mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.mlflow_config.tracking_uri}")
            
            # Get or create the experiment
            try:
                self.experiment_id = mlflow.create_experiment(
                    name=self.mlflow_config.experiment_name
                )
                logger.info(f"Created new MLflow experiment: {self.mlflow_config.experiment_name}")
            except MlflowException:
                self.experiment_id = mlflow.get_experiment_by_name(
                    self.mlflow_config.experiment_name
                ).experiment_id
                logger.info(f"Using existing MLflow experiment: {self.mlflow_config.experiment_name}")
            
            mlflow.set_experiment(self.mlflow_config.experiment_name)
            
        except Exception as e:
            logger.warning(f"Failed to set up MLflow tracking: {str(e)}")
            self.mlflow_config.enabled = False
    
    @contextmanager
    def start_run(
        self, 
        run_name: Optional[str] = None, 
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None
    ):
        """Start a new MLflow run.
        
        Args:
            run_name: Name of the run
            nested: Whether this is a nested run
            tags: Additional tags to add to the run
        
        Yields:
            The active MLflow run
        """
        if not self.mlflow_config.enabled:
            # Create a dummy run object to prevent NoneType errors
            class DummyRun:
                def __init__(self):
                    self.info = type('obj', (object,), {
                        'run_id': 'dummy-run',
                        'experiment_id': 'dummy-experiment',
                        'status': 'DUMMY'
                    })
                    
            dummy_run = DummyRun()
            self.active_run = dummy_run
            try:
                yield dummy_run
            finally:
                self.active_run = None
            return
        
        # Combine default tags with provided tags
        all_tags = {**self.mlflow_config.tags}
        if tags:
            all_tags.update(tags)
        
        # Add run mode tag
        all_tags["run_mode"] = self.config.run_mode.value
        
        # Start the run
        active_run = mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=all_tags
        )
        self.active_run = active_run
        
        # Log system info
        if self.mlflow_config.log_system_metrics:
            self._log_system_info()
        
        # Log key configuration parameters
        self._log_config_params()
        
        try:
            yield active_run
        finally:
            # End the run
            mlflow.end_run()
            if not nested:
                self.active_run = None
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to MLflow.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        if not self.mlflow_config.enabled:
            logger.debug(f"MLflow disabled, not logging parameter {key}={value}")
            return
            
        # Convert non-string values to strings if needed
        if not isinstance(value, (str, int, float, bool)):
            value = str(value)
            
        mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters to MLflow.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if not self.mlflow_config.enabled:
            logger.debug(f"MLflow disabled, not logging {len(params)} parameters")
            return
            
        # Convert non-string values to strings if needed
        clean_params = {}
        for k, v in params.items():
            if not isinstance(v, (str, int, float, bool)):
                clean_params[k] = str(v)
            else:
                clean_params[k] = v
                
        mlflow.log_params(clean_params)
    
    def log_metric(
        self, 
        key: str, 
        value: float, 
        step: Optional[int] = None
    ) -> None:
        """Log a metric to MLflow.
        
        Args:
            key: Metric name
            value: Metric value
            step: Step value
        """
        if not self.mlflow_config.enabled:
            logger.debug(f"MLflow disabled, not logging metric {key}={value}")
            return
            
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(
        self, 
        metrics: Dict[str, float], 
        step: Optional[int] = None
    ) -> None:
        """Log multiple metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step value
        """
        if not self.mlflow_config.enabled:
            logger.debug(f"MLflow disabled, not logging {len(metrics)} metrics")
            return
            
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """Log an artifact (file) to MLflow.
        
        Args:
            local_path: Path to the artifact file
        """
        if not self.mlflow_config.enabled:
            logger.debug(f"MLflow disabled, not logging artifact: {local_path}")
            return
            
        mlflow.log_artifact(str(local_path))
    
    def log_artifacts(self, local_dir: Union[str, Path]) -> None:
        """Log all artifacts in a directory to MLflow.
        
        Args:
            local_dir: Path to the directory containing artifacts
        """
        if not self.mlflow_config.enabled:
            logger.debug(f"MLflow disabled, not logging artifacts from: {local_dir}")
            return
            
        mlflow.log_artifacts(str(local_dir))
    
    def log_figure(self, figure, artifact_path: str) -> None:
        """Log a matplotlib figure to the current run.
        
        Args:
            figure: Matplotlib figure to log
            artifact_path: Path where the figure should be logged
        """
        if not self.mlflow_config.enabled or not self.mlflow_config.log_artifacts:
            return
        
        try:
            mlflow.log_figure(figure, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log figure: {str(e)}")
    
    def log_dataframe(
        self, 
        df: pd.DataFrame, 
        artifact_path: str, 
        format: str = "csv"
    ) -> None:
        """Log a pandas DataFrame to the current run.
        
        Args:
            df: DataFrame to log
            artifact_path: Path where the DataFrame should be logged
            format: Format to use ('csv' or 'parquet')
        """
        if not self.mlflow_config.enabled or not self.mlflow_config.log_artifacts:
            return
        
        # Create a temporary file to save the DataFrame
        from tempfile import NamedTemporaryFile
        try:
            with NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
                if format == "csv":
                    df.to_csv(tmp.name, index=False)
                elif format == "parquet":
                    df.to_parquet(tmp.name, index=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                # Log the temporary file as an artifact
                mlflow.log_artifact(tmp.name, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log DataFrame: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    
    def log_model(
        self, 
        model: Any, 
        artifact_path: str, 
        conda_env: Optional[Dict] = None,
        **kwargs
    ) -> None:
        """Log a model to MLflow.
        
        Args:
            model: Model to log
            artifact_path: Path within the artifact directory to store the model
            conda_env: Conda environment for the model
            **kwargs: Additional arguments to pass to the appropriate MLflow log_model function
        """
        if not self.mlflow_config.enabled:
            logger.debug(f"MLflow disabled, not logging model to: {artifact_path}")
            return
            
        # Determine the correct log_model function based on the model type
        if hasattr(model, "__module__"):
            module_name = model.__module__.split(".")[0].lower()
            
            if module_name == "lightgbm":
                logger.info(f"Logging LightGBM model to {artifact_path}")
                import mlflow.lightgbm
                mlflow.lightgbm.log_model(model, artifact_path, conda_env=conda_env, **kwargs)
                
            elif module_name == "xgboost":
                logger.info(f"Logging XGBoost model to {artifact_path}")
                import mlflow.xgboost
                mlflow.xgboost.log_model(model, artifact_path, conda_env=conda_env, **kwargs)
                
            elif module_name == "sklearn" or module_name == "sklearnex":
                logger.info(f"Logging scikit-learn model to {artifact_path}")
                import mlflow.sklearn
                mlflow.sklearn.log_model(model, artifact_path, conda_env=conda_env, **kwargs)
                
            elif module_name == "catboost":
                logger.info(f"Logging CatBoost model to {artifact_path}")
                import mlflow.catboost
                mlflow.catboost.log_model(model, artifact_path, conda_env=conda_env, **kwargs)
                
            else:
                logger.info(f"Logging generic model to {artifact_path}")
                import mlflow.pyfunc
                mlflow.pyfunc.log_model(artifact_path, python_model=model, conda_env=conda_env, **kwargs)
        else:
            logger.info(f"Logging unknown model type to {artifact_path}")
            import mlflow.pyfunc
            mlflow.pyfunc.log_model(artifact_path, python_model=model, conda_env=conda_env, **kwargs)
    
    def log_feature_importance(
        self, 
        importance: Dict[str, float],
        artifact_path: str = "feature_importance.csv"
    ) -> None:
        """Log feature importance to the current run.
        
        Args:
            importance: Dictionary mapping feature names to importance scores
            artifact_path: Path where the feature importance should be logged
        """
        if not self.mlflow_config.enabled or not self.mlflow_config.log_artifacts:
            return
        
        try:
            # Convert to DataFrame for better visualization
            df = pd.DataFrame({
                "feature": list(importance.keys()),
                "importance": list(importance.values())
            }).sort_values("importance", ascending=False)
            
            # Log as artifact
            self.log_dataframe(df, artifact_path)
            
            # Also log top features as parameters for easy filtering
            top_features = {
                f"top_feature_{i+1}": feature
                for i, (feature, _) in enumerate(sorted(
                    importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10])
            }
            self.log_params(top_features)
        except Exception as e:
            logger.warning(f"Failed to log feature importance: {str(e)}")
    
    def log_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        artifact_path: str = "confusion_matrix.png"
    ) -> None:
        """Log a confusion matrix to the current run.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            artifact_path: Path where the confusion matrix should be logged
        """
        if not self.mlflow_config.enabled or not self.mlflow_config.log_artifacts:
            return
        
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            
            # Create the confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot the confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            
            # Log the figure
            self.log_figure(fig, artifact_path)
            plt.close(fig)
            
            # Also log the confusion matrix as a parameter
            self.log_param("confusion_matrix", str(cm.tolist()))
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {str(e)}")
    
    def _log_system_info(self) -> None:
        """Log system information as parameters."""
        if not self.mlflow_config.enabled:
            return
        
        try:
            # Get system information
            system_info = {
                "python_version": sys.version.split()[0],
                "system": sys.platform,
                "cpu_count": os.cpu_count(),
                "ram_gb": round(virtual_memory().total / (1024**3), 2),
            }
            
            # Try to get GPU information
            try:
                # NVIDIA GPU information (using subprocess)
                nvidia_smi = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
                ).decode("utf-8").strip()
                
                if nvidia_smi:
                    gpus = nvidia_smi.split("\n")
                    for i, gpu in enumerate(gpus):
                        name, memory = gpu.split(",")
                        system_info[f"gpu_{i}_name"] = name.strip()
                        system_info[f"gpu_{i}_memory_gb"] = float(memory.strip()) / 1024
                    
                    system_info["gpu_count"] = len(gpus)
            except (subprocess.SubprocessError, FileNotFoundError):
                # No NVIDIA GPU or nvidia-smi not available
                system_info["gpu_count"] = 0
            
            # Log as parameters
            self.log_params(system_info)
        
        except Exception as e:
            logger.warning(f"Failed to log system information: {str(e)}")
    
    def _log_config_params(self) -> None:
        """Log key configuration parameters."""
        if not self.mlflow_config.enabled:
            return
        
        try:
            # Log key configuration parameters
            params = {
                "run_mode": self.config.run_mode.value,
                "random_seed": self.config.random_seed,
                "sample_size": self.config.data.sample_size,
                "feature_max_depth": self.config.features.max_depth,
            }
            
            # Log GPU configuration
            if hasattr(self.config, "gpu"):
                params.update({
                    "gpu_backend": str(self.config.gpu.processing_backend),
                    "gpu_for_features": self.config.gpu.use_gpu_for_feature_engineering,
                    "gpu_for_training": self.config.gpu.use_gpu_for_training,
                })
            
            self.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log configuration parameters: {str(e)}")


def track_with_mlflow(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
):
    """Decorator to track a function with MLflow.
    
    Args:
        run_name: Name of the run
        tags: Additional tags to add to the run
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create the experiment tracker
            config = kwargs.get("config", default_config)
            tracker = ExperimentTracker(config)
            
            # Extract function parameters
            params = {}
            for i, arg in enumerate(args):
                params[f"arg_{i}"] = str(arg)
            for key, value in kwargs.items():
                if key != "config" and not isinstance(value, (pd.DataFrame, np.ndarray)):
                    params[key] = str(value)
            
            # Start a run
            with tracker.start_run(run_name=run_name, tags=tags):
                # Log function parameters
                tracker.log_params(params)
                
                # Call the function
                start_time = int(time.time() * 1000)
                result = func(*args, **kwargs)
                end_time = int(time.time() * 1000)
                
                # Log execution time
                tracker.log_metric("execution_time_ms", end_time - start_time)
                
                return result
        
        return wrapper
    
    return decorator 