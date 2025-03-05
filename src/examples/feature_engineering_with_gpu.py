#!/usr/bin/env python3
"""
Example script for GPU-accelerated feature engineering with MLflow tracking.

This script demonstrates how to use the enhanced credit risk prediction
pipeline with GPU acceleration and MLflow experiment tracking.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Add parent directory to path for package imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from credit_risk import (
    PipelineConfig,
    RunMode,
    ProcessingBackend,
    GPUDataLoader,
    DataPreprocessor,
    FeatureEngineer,
    FeatureStore,
    ExperimentTracker,
    track_with_mlflow
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("feature_engineering_with_gpu")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="GPU-accelerated feature engineering for credit risk prediction"
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["development", "production"],
        default="development",
        help="Run mode ('development' or 'production')"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of samples to use in development mode"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum depth for feature engineering"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Processing backend to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="credit_risk_features",
        help="Name of the feature set to save"
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="sqlite:///mlruns.db",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="credit_risk_prediction",
        help="MLflow experiment name"
    )
    return parser.parse_args()


def get_config(args: argparse.Namespace) -> PipelineConfig:
    """Create configuration from command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Pipeline configuration
    """
    # Create base configuration
    config = PipelineConfig(
        run_mode=RunMode(args.run_mode)
    )
    
    # Update with command line arguments
    config.data.sample_size = args.sample_size
    config.features.max_depth = args.max_depth
    
    # Set GPU configuration
    config.gpu.processing_backend = ProcessingBackend(args.backend)
    
    # Set MLflow configuration
    config.mlflow.tracking_uri = args.mlflow_tracking_uri
    config.mlflow.experiment_name = args.experiment_name
    
    return config


@track_with_mlflow(run_name="feature_engineering_with_gpu")
def run_feature_engineering(config: PipelineConfig) -> Optional[pd.DataFrame]:
    """Run the feature engineering pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Generated features DataFrame or None if error
    """
    try:
        # Create GPU data loader
        logger.info("Creating GPU data loader...")
        loader = GPUDataLoader(
            config=config,
            backend=config.gpu.processing_backend
        )
        
        # Log GPU availability
        backend_name = loader.get_backend_name()
        is_gpu = loader.is_gpu_available()
        logger.info(f"Using {backend_name} backend (GPU: {is_gpu})")
        
        # Load datasets
        logger.info("Loading datasets...")
        datasets = loader.load_all_datasets(
            sample=config.is_development_mode,
            optimize_memory=True,
            to_pandas=True  # Convert to pandas for compatibility
        )
        logger.info(f"Loaded {len(datasets)} datasets")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        
        app_df = datasets["application_train"]
        app_df = preprocessor.drop_low_importance_columns(app_df)
        app_df = preprocessor.handle_missing_values(app_df)
        
        # Update the dataset
        datasets["application_train"] = app_df
        
        # Create manual features
        logger.info("Creating manual features...")
        feature_engineer = FeatureEngineer(config=config)
        app_df = feature_engineer.create_manual_features(
            app_df=app_df,
            bureau_df=datasets.get("bureau"),
            previous_df=datasets.get("previous_application"),
            installments_df=datasets.get("installments_payments")
        )
        datasets["application_train"] = app_df
        
        # Generate automated features
        logger.info(f"Generating automated features with max_depth={config.features.max_depth}...")
        features_result = feature_engineer.generate_features(
            datasets=datasets,
            max_depth=config.features.max_depth,
            verbose=True
        )
        
        # Handle the case where features_result might be a tuple or a DataFrame
        if isinstance(features_result, tuple):
            features_df = features_result[0]  # In some cases, dfs returns a tuple
            logger.info(f"Generated features successfully")
        else:
            features_df = features_result
            logger.info(f"Generated {features_df.shape[1]} features")
        
        # Save features to feature store
        logger.info("Saving features to feature store...")
        store = FeatureStore(config=config)
        feature_set_name = f"{config.run_mode.value}_features_depth{config.features.max_depth}"
        store.save_feature_set(
            features=features_df,
            name=feature_set_name,
            description=f"Features with max_depth={config.features.max_depth}",
            tags=[config.run_mode.value, f"backend_{backend_name}"],
            metadata={
                "max_depth": config.features.max_depth,
                "run_mode": config.run_mode.value,
                "backend": backend_name,
                "gpu_available": is_gpu,
            }
        )
        
        return features_df
    
    except Exception as e:
        logger.exception(f"Error running feature engineering: {str(e)}")
        return None


def main() -> int:
    """Main function.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration
    config = get_config(args)
    
    logger.info(f"Running in {config.run_mode.value} mode")
    logger.info(f"Using {config.gpu.processing_backend.value} backend")
    logger.info(f"Feature engineering max depth: {config.features.max_depth}")
    
    # Run feature engineering
    features_df = run_feature_engineering(config=config)
    
    if features_df is not None:
        logger.info("Feature engineering completed successfully")
        return 0
    else:
        logger.error("Feature engineering failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 