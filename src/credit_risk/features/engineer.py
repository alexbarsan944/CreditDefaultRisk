"""
Feature engineering module for credit risk prediction.

This module provides functionality for generating features using featuretools
and other feature engineering techniques for credit risk prediction.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import featuretools as ft
import numpy as np
import pandas as pd

from credit_risk.config import PipelineConfig, default_config
from credit_risk.features.entity_builder import EntitySetBuilder

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineer for credit risk prediction.
    
    This class handles feature engineering for credit risk prediction,
    including automated feature generation using featuretools.
    
    Attributes:
        config: Configuration for feature engineering
        entity_builder: EntitySetBuilder for creating featuretools EntitySets
    """
    
    def __init__(
        self, 
        config: Optional[PipelineConfig] = None,
        entity_builder: Optional[EntitySetBuilder] = None
    ):
        """Initialize the feature engineer.
        
        Args:
            config: Configuration for feature engineering. If None, uses default config.
            entity_builder: EntitySetBuilder instance. If None, creates a new one.
        """
        self.config = config or default_config
        self.entity_builder = entity_builder or EntitySetBuilder(config=self.config)
    
    def generate_features(
        self, 
        datasets: Dict[str, pd.DataFrame],
        agg_primitives: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        features_only: bool = False,
        verbose: bool = True,
        chunk_size: Optional[int] = None,
        n_jobs: int = -1
    ) -> Union[pd.DataFrame, List[str]]:
        """
        Generate features from datasets using Featuretools.
        
        Parameters
        ----------
        datasets : Dict[str, pd.DataFrame]
            Dictionary of datasets, with keys as dataset names
        agg_primitives : Optional[List[str]], optional
            List of aggregation primitives to use, by default None
        max_depth : Optional[int], optional
            Maximum depth for feature stacking, by default None
        features_only : bool, optional
            Whether to return only feature names without calculating values, by default False
        verbose : bool, optional
            Whether to print progress, by default True
        chunk_size : Optional[int], optional
            Chunk size for parallel processing, by default None
        n_jobs : int, optional
            Number of parallel jobs, by default -1
            
        Returns
        -------
        Union[pd.DataFrame, List[str]]
            DataFrame with calculated features, or list of feature names if features_only=True
            
        Raises
        ------
        ValueError
            If no datasets are provided
        """
        if not datasets:
            raise ValueError("No datasets provided for feature engineering")
            
        # Use config values if parameters are not provided
        if agg_primitives is None and self.config:
            agg_primitives = self.config.features.default_agg_primitives
            
        if max_depth is None and self.config:
            max_depth = self.config.features.max_depth
            
        # Force n_jobs to 1 to avoid distributed errors
        # Override any passed n_jobs value
        n_jobs = 1
        
        # Default values if still None
        agg_primitives = agg_primitives or ["mean", "min", "max", "count", "sum", "std"]
        max_depth = max_depth or 2
        
        # Log parameters
        logger.info(f"Feature engineering - max_depth: {max_depth}, n_jobs: {n_jobs}")
        logger.info(f"Aggregation primitives: {agg_primitives}")
        
        # Build entityset if we don't have one
        if self.entity_builder is None:
            self.entity_builder = EntitySetBuilder()
            
        entity_set = self.entity_builder.build_entity_set(datasets)
        
        # Try to generate features
        try:
            # Define progress callback
            def progress_callback(*args):
                if verbose:
                    if len(args) == 1:
                        logger.info(f"Progress: {args[0]}")
                    else:
                        logger.info(f"Progress update: {args}")
                
            # Disable Dask/distributed to avoid errors
            import os
            os.environ["FEATURETOOLS_NO_DASK"] = "1"
            
            features = ft.dfs(
                entityset=entity_set,
                target_dataframe_name="app",
                agg_primitives=agg_primitives,
                max_depth=max_depth,
                features_only=features_only,
                verbose=verbose,
                chunk_size=chunk_size,  # Not used when n_jobs=1
                n_jobs=1,  # Force single process to avoid distributed errors
                progress_callback=progress_callback
            )
            
            # Log feature generation results
            if not features_only:
                # Get the features DataFrame if a tuple was returned
                if isinstance(features, tuple):
                    features_df = features[0]
                    feature_defs = features[1]
                    logger.info(f"Feature engineering complete. Generated DataFrame with shape: {features_df.shape}")
                else:
                    features_df = features
                    logger.info(f"Feature engineering complete. Generated DataFrame with shape: {features_df.shape}")
                
                return features_df
            else:
                logger.info(f"Feature engineering complete. Generated {len(features)} feature definitions")
                self._save_feature_names([f.get_name() for f in features])
                return features
                
        except Exception as e:
            logger.error(f"Error during feature generation: {str(e)}", exc_info=True)
            # Try to create a simple set of features as a fallback
            logger.info("Attempting to use simplified feature generation as a fallback")
            try:
                # Get application dataframe
                app_df = datasets.get("application_train", None)
                if app_df is not None:
                    # Just do basic feature engineering on the app dataframe
                    result_df = self.create_manual_features(app_df)
                    logger.info(f"Successfully created basic features: {result_df.shape}")
                    return result_df
                else:
                    raise ValueError("No application_train dataset found")
            except Exception as fallback_error:
                logger.error(f"Fallback feature generation also failed: {str(fallback_error)}", exc_info=True)
                raise
    
    def _save_feature_names(self, feature_names: List[str]) -> None:
        """Save feature names to a file.
        
        Args:
            feature_names: List of feature names to save
        """
        # Create features directory if it doesn't exist
        self.config.data.features_path.mkdir(parents=True, exist_ok=True)
        
        # Save as text file
        features_file = self.config.data.features_path / "feature_names.txt"
        logger.info(f"Saving {len(feature_names)} feature names to {features_file}")
        
        with open(features_file, "w") as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
    
    def save_features(
        self, 
        features_df: pd.DataFrame,
        filename: str = "features.pkl"
    ) -> Path:
        """Save generated features to a file.
        
        Args:
            features_df: DataFrame with generated features
            filename: Name of the file to save features to
            
        Returns:
            Path to the saved file
        """
        # Create features directory if it doesn't exist
        self.config.data.features_path.mkdir(parents=True, exist_ok=True)
        
        # Save features
        output_path = self.config.data.features_path / filename
        logger.info(f"Saving features to {output_path}")
        
        # Save as pickle
        with open(output_path, "wb") as f:
            pickle.dump(features_df, f)
        
        return output_path
    
    def load_features(
        self, 
        filename: str = "features.pkl"
    ) -> pd.DataFrame:
        """Load features from a file.
        
        Args:
            filename: Name of the file to load features from
            
        Returns:
            DataFrame with loaded features
            
        Raises:
            FileNotFoundError: If the features file does not exist
        """
        file_path = self.config.data.features_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Features file not found: {file_path}")
        
        logger.info(f"Loading features from {file_path}")
        
        with open(file_path, "rb") as f:
            features = pickle.load(f)
        
        return features
    
    @staticmethod
    def create_manual_features(
        app_df: pd.DataFrame,
        bureau_df: Optional[pd.DataFrame] = None,
        previous_df: Optional[pd.DataFrame] = None,
        installments_df: Optional[pd.DataFrame] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """Create manual features from application and related datasets.
        
        This method creates hand-crafted features that may be more interpretable
        or domain-specific than automated features.
        
        Args:
            app_df: Application DataFrame
            bureau_df: Bureau DataFrame (optional)
            previous_df: Previous applications DataFrame (optional)
            installments_df: Installments payments DataFrame (optional)
            inplace: Whether to modify the app_df in place
            
        Returns:
            DataFrame with manual features added
        """
        # Make a copy if not inplace
        if not inplace:
            app_df = app_df.copy()
        
        # Basic application features
        app_df["credit_income_ratio"] = app_df["AMT_CREDIT"] / app_df["AMT_INCOME_TOTAL"]
        app_df["annuity_income_ratio"] = app_df["AMT_ANNUITY"] / app_df["AMT_INCOME_TOTAL"]
        app_df["credit_term"] = app_df["AMT_CREDIT"] / app_df["AMT_ANNUITY"]
        app_df["days_employed_ratio"] = app_df["DAYS_EMPLOYED"] / app_df["DAYS_BIRTH"]
        
        # Count NaN values in external sources
        ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        app_df["ext_sources_nan"] = app_df[ext_cols].isna().sum(axis=1)
        
        # Add features from bureau data if available
        if bureau_df is not None:
            # Count active loans
            active_loans = bureau_df[bureau_df["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR").size()
            app_df["active_loans_count"] = active_loans
            
            # Count closed loans
            closed_loans = bureau_df[bureau_df["CREDIT_ACTIVE"] == "Closed"].groupby("SK_ID_CURR").size()
            app_df["closed_loans_count"] = closed_loans
            
            # Calculate average credit duration
            bureau_df["credit_duration"] = bureau_df["DAYS_CREDIT"] - bureau_df["DAYS_CREDIT_ENDDATE"]
            app_df["avg_credit_duration"] = bureau_df.groupby("SK_ID_CURR")["credit_duration"].mean()
        
        # Add features from previous applications if available
        if previous_df is not None:
            # Count previous applications
            prev_count = previous_df.groupby("SK_ID_CURR").size()
            app_df["previous_application_count"] = prev_count
            
            # Calculate approval ratio
            approved = previous_df[previous_df["NAME_CONTRACT_STATUS"] == "Approved"]
            approved_count = approved.groupby("SK_ID_CURR").size()
            app_df["previous_approval_ratio"] = approved_count / prev_count
        
        # Add features from installments if available
        if installments_df is not None:
            # Calculate late payment ratio
            installments_df["is_late"] = installments_df["DAYS_ENTRY_PAYMENT"] > installments_df["DAYS_INSTALMENT"]
            late_ratio = installments_df.groupby("SK_ID_CURR")["is_late"].mean()
            app_df["late_payment_ratio"] = late_ratio
            
            # Calculate average payment to installment ratio
            installments_df["payment_ratio"] = (installments_df["AMT_PAYMENT"] / installments_df["AMT_INSTALMENT"]).replace([np.inf, -np.inf], np.nan)
            app_df["avg_payment_installment_ratio"] = installments_df.groupby("SK_ID_CURR")["payment_ratio"].mean()
        
        return app_df
    
    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB.
        
        Returns:
            Available memory in GB (float)
        """
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            logger.warning("psutil not available, using default memory estimate of 8GB")
            return 8.0 