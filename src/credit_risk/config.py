"""
Configuration settings for the credit risk prediction pipeline.

This module contains all configuration parameters that can be used to
customize the behavior of the pipeline.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from credit_risk.utils.processing_strategy import ProcessingBackend


class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RunMode(str, Enum):
    """Pipeline run mode."""
    DEVELOPMENT = "development"  # Use sample data
    PRODUCTION = "production"    # Use full dataset


@dataclass
class DataConfig:
    """Data loading and processing configuration."""
    base_path: Path = Path("./data")
    raw_data_path: Path = Path("./data/raw")
    processed_data_path: Path = Path("./data/processed") 
    features_path: Path = Path("./data/features")
    sample_size: int = 10000  # Number of samples to use in development mode
    
    # Map of dataset names to file paths (relative to raw_data_path)
    dataset_files: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default dataset files if none provided."""
        if self.dataset_files is None:
            self.dataset_files = {
                "application_train": "application_train.csv",
                "application_test": "application_test.csv",
                "bureau": "bureau.csv",
                "bureau_balance": "bureau_balance.csv",
                "credit_card_balance": "credit_card_balance.csv",
                "installments_payments": "installments_payments.csv",
                "pos_cash_balance": "POS_CASH_balance.csv",
                "previous_application": "previous_application.csv"
            }
    
    def get_file_path(self, dataset_name: str) -> Path:
        """Get the full path to a dataset file.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Path to the dataset file
            
        Raises:
            ValueError: If the dataset name is not configured
        """
        if dataset_name not in self.dataset_files:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        return self.raw_data_path / self.dataset_files[dataset_name]


@dataclass
class FeaturesConfig:
    """Feature engineering configuration."""
    # Default aggregation primitives to use with featuretools
    default_agg_primitives: list = None
    
    # Maximum depth for feature engineering
    max_depth: int = 2
    
    # Whether to save generated features
    save_features: bool = True
    
    # Map of entity definitions for featuretools
    entity_definitions: Dict[str, Dict] = None
    
    # Performance settings
    # Chunk size for feature calculation (helps with memory usage)
    chunk_size: Optional[int] = None
    
    # Number of parallel jobs for feature calculation (-1 uses all cores)
    n_jobs: int = -1
    
    # Whether to use reduced feature set in development mode
    use_reduced_features_in_dev: bool = True
    
    def __post_init__(self):
        """Initialize defaults if not provided."""
        if self.default_agg_primitives is None:
            self.default_agg_primitives = ["sum", "mean", "count", "max", "min", "std"]
            
        if self.entity_definitions is None:
            # Default entity relationships will be initialized during pipeline setup
            self.entity_definitions = {}


@dataclass
class GPUConfig:
    """GPU acceleration configuration."""
    # Processing backend to use (AUTO, CPU, or GPU)
    processing_backend: ProcessingBackend = ProcessingBackend.AUTO
    
    # Whether to use GPU for feature engineering
    use_gpu_for_feature_engineering: bool = True
    
    # Whether to use GPU for model training
    use_gpu_for_training: bool = True
    
    # Memory utilization threshold (0-1) to prevent GPU memory exhaustion
    gpu_memory_utilization: float = 0.8
    
    # Whether to automatically fall back to CPU if GPU errors occur
    auto_fallback_to_cpu: bool = True
    
    # Chunk size for processing large datasets on GPU (to avoid OOM errors)
    gpu_chunk_size: Optional[int] = None


@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration."""
    # Whether to enable MLflow tracking
    enabled: bool = True
    
    # MLflow tracking URI
    tracking_uri: str = "sqlite:///mlruns.db"
    
    # Experiment name
    experiment_name: str = "credit_risk_prediction"
    
    # Additional tags to add to all runs
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Whether to log models to MLflow model registry
    log_models: bool = True
    
    # Whether to log artifacts (plots, feature importance, etc.)
    log_artifacts: bool = True
    
    # Whether to log system metrics (CPU, memory, GPU usage)
    log_system_metrics: bool = True


@dataclass
class PipelineConfig:
    """Main configuration for the credit risk prediction pipeline."""
    run_mode: RunMode = RunMode.PRODUCTION
    log_level: LogLevel = LogLevel.INFO
    random_seed: int = 42
    data: DataConfig = None
    features: FeaturesConfig = None
    gpu: GPUConfig = None
    mlflow: MLflowConfig = None
    
    def __post_init__(self):
        """Initialize nested configuration if not provided."""
        if self.data is None:
            self.data = DataConfig()
            
        if self.features is None:
            self.features = FeaturesConfig()
            
        if self.gpu is None:
            self.gpu = GPUConfig()
            
        if self.mlflow is None:
            self.mlflow = MLflowConfig()
    
    @property
    def is_development_mode(self) -> bool:
        """Check if pipeline is running in development mode."""
        return self.run_mode == RunMode.DEVELOPMENT


# Default configuration instance to be used throughout the package
default_config = PipelineConfig() 