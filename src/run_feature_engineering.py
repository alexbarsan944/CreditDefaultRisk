#!/usr/bin/env python3
"""
Feature engineering pipeline for credit risk prediction.

This script runs the end-to-end feature engineering pipeline, including:
1. Loading data
2. Preprocessing data
3. Creating manual features
4. Generating automated features with featuretools
5. Saving features to the feature store
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent directory to path for package imports
sys.path.append(str(Path(__file__).parent.parent))

from credit_risk import (
    DataLoader, 
    DataPreprocessor, 
    EntitySetBuilder, 
    FeatureEngineer, 
    FeatureStore,
    PipelineConfig, 
    RunMode, 
    default_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("feature_engineering")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Feature engineering pipeline for credit risk prediction"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode (uses sample data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="credit_risk_features",
        help="Name of the feature set to save"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum depth for feature engineering"
    )
    parser.add_argument(
        "--skip-manual-features",
        action="store_true",
        help="Skip manual feature creation"
    )
    return parser.parse_args()


def load_and_preprocess_data(
    config: PipelineConfig
) -> Dict[str, pd.DataFrame]:
    """Load and preprocess data.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dictionary mapping dataset names to preprocessed DataFrames
    """
    logger.info("Loading data...")
    loader = DataLoader(config=config)
    
    # Load all datasets
    datasets = loader.load_all_datasets()
    logger.info(f"Loaded {len(datasets)} datasets")
    
    # Preprocess application data
    logger.info("Preprocessing application data...")
    preprocessor = DataPreprocessor()
    
    app_df = datasets["application_train"]
    app_df = preprocessor.drop_low_importance_columns(app_df)
    app_df = preprocessor.handle_missing_values(app_df)
    
    # Update datasets with preprocessed application data
    datasets["application_train"] = app_df
    
    return datasets


def create_manual_features(
    datasets: Dict[str, pd.DataFrame],
    skip_manual: bool = False
) -> pd.DataFrame:
    """Create manual features from datasets.
    
    Args:
        datasets: Dictionary mapping dataset names to DataFrames
        skip_manual: Whether to skip manual feature creation
        
    Returns:
        DataFrame with manual features
    """
    app_df = datasets["application_train"]
    
    if skip_manual:
        logger.info("Skipping manual feature creation")
        return app_df
    
    logger.info("Creating manual features...")
    
    # Get optional datasets
    bureau_df = datasets.get("bureau")
    previous_df = datasets.get("previous_application")
    installments_df = datasets.get("installments_payments")
    
    # Create manual features
    feature_engineer = FeatureEngineer()
    app_df = feature_engineer.create_manual_features(
        app_df=app_df,
        bureau_df=bureau_df,
        previous_df=previous_df,
        installments_df=installments_df
    )
    
    logger.info(f"Created manual features. Total features: {app_df.shape[1]}")
    return app_df


def generate_automated_features(
    datasets: Dict[str, pd.DataFrame],
    config: PipelineConfig,
    manual_features_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate automated features using featuretools.
    
    Args:
        datasets: Dictionary mapping dataset names to DataFrames
        config: Pipeline configuration
        manual_features_df: DataFrame with manual features
        
    Returns:
        DataFrame with all features (manual + automated)
    """
    logger.info("Generating automated features using featuretools...")
    
    # Update application data with manual features
    datasets["application_train"] = manual_features_df
    
    # Generate automated features
    feature_engineer = FeatureEngineer(config=config)
    automated_features = feature_engineer.generate_features(
        datasets=datasets,
        max_depth=config.features.max_depth,
        verbose=True
    )
    
    logger.info(f"Generated automated features. Total features: {automated_features.shape[1]}")
    return automated_features


def save_features(
    features_df: pd.DataFrame,
    feature_set_name: str,
    config: PipelineConfig
) -> None:
    """Save features to the feature store.
    
    Args:
        features_df: DataFrame with features to save
        feature_set_name: Name of the feature set
        config: Pipeline configuration
    """
    logger.info(f"Saving features to feature store as '{feature_set_name}'...")
    
    # Save features to feature store
    feature_store = FeatureStore(config=config)
    feature_store.save_feature_set(
        features=features_df,
        name=feature_set_name,
        description="Credit risk prediction features",
        tags=["credit_risk", "featuretools"],
        metadata={
            "max_depth": config.features.max_depth,
            "run_mode": config.run_mode.value
        }
    )
    
    logger.info(f"Saved {features_df.shape[1]} features to feature store")


def main() -> None:
    """Run the feature engineering pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Configure pipeline
    config = PipelineConfig(
        run_mode=RunMode.DEVELOPMENT if args.dev else RunMode.PRODUCTION,
        features=PipelineConfig().features
    )
    config.features.max_depth = args.max_depth
    
    logger.info(f"Running in {config.run_mode.value} mode")
    logger.info(f"Feature engineering max depth: {config.features.max_depth}")
    
    try:
        # Load and preprocess data
        datasets = load_and_preprocess_data(config)
        
        # Create manual features
        manual_features_df = create_manual_features(
            datasets=datasets,
            skip_manual=args.skip_manual_features
        )
        
        # Generate automated features
        features_df = generate_automated_features(
            datasets=datasets,
            config=config,
            manual_features_df=manual_features_df
        )
        
        # Save features
        save_features(
            features_df=features_df,
            feature_set_name=args.output,
            config=config
        )
        
        logger.info("Feature engineering pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Error running feature engineering pipeline: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 