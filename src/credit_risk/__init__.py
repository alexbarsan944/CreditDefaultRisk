"""
Credit Risk Prediction package.

This package provides functionality for credit risk prediction using
automated feature engineering and machine learning.
"""

import logging

from credit_risk.config import (
    PipelineConfig, 
    RunMode, 
    GPUConfig,
    MLflowConfig,
    default_config
)
from credit_risk.data.loader import DataLoader
from credit_risk.data.gpu_loader import GPUDataLoader
from credit_risk.data.preprocessor import DataPreprocessor
from credit_risk.features.engineer import FeatureEngineer
from credit_risk.features.entity_builder import EntitySetBuilder
from credit_risk.features.feature_store import FeatureStore
from credit_risk.utils.processing_strategy import (
    ProcessingBackend,
    ProcessingContext,
    ProcessingStrategy,
    PandasStrategy,
    PolarsStrategy,
    CudfStrategy
)
from credit_risk.utils.mlflow_tracking import (
    ExperimentTracker,
    track_with_mlflow
)

__version__ = "0.1.0"

# Configure package-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Export public API
__all__ = [
    # Configuration
    "PipelineConfig",
    "RunMode",
    "GPUConfig",
    "MLflowConfig",
    "default_config",
    
    # Data Loading
    "DataLoader",
    "GPUDataLoader",
    "DataPreprocessor",
    
    # Feature Engineering
    "FeatureEngineer",
    "EntitySetBuilder",
    "FeatureStore",
    
    # GPU Processing
    "ProcessingBackend",
    "ProcessingContext",
    "ProcessingStrategy",
    "PandasStrategy",
    "PolarsStrategy",
    "CudfStrategy",
    
    # MLflow Tracking
    "ExperimentTracker",
    "track_with_mlflow",
] 