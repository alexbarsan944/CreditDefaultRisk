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
        """Generate features using featuretools.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            agg_primitives: List of aggregation primitives to use.
                If None, uses config defaults.
            max_depth: Maximum depth for feature engineering.
                If None, uses config default.
            features_only: If True, return list of feature names instead of DataFrame
            verbose: Whether to display progress information
            chunk_size: Size of chunks to use for feature calculation (helps with memory usage)
            n_jobs: Number of parallel jobs for feature calculation (-1 uses all cores)
            
        Returns:
            If features_only is True, returns a list of feature names.
            Otherwise, returns a DataFrame with the generated features.
        """
        # Log information about input datasets
        total_rows = 0
        total_cols = 0
        dataset_info = []
        
        for name, df in datasets.items():
            rows, cols = df.shape
            total_rows += rows
            total_cols += cols
            dataset_info.append(f"{name}: {rows} rows, {cols} columns")
        
        logger.info(f"Input datasets: {len(datasets)} datasets with {total_rows} total rows and {total_cols} total columns")
        for info in dataset_info:
            logger.info(f"  - {info}")
            
        # Get default parameters from config if not provided
        if agg_primitives is None:
            agg_primitives = self.config.features.default_agg_primitives
            
        if max_depth is None:
            max_depth = self.config.features.max_depth
            
        # Consider using a reduced set of aggregation primitives for better performance
        # For example, prioritize sum, mean and count which are often most informative
        # while skipping more computationally expensive operations like std
        if len(agg_primitives) > 3 and self.config.is_development_mode:
            logger.info("Using reduced set of aggregation primitives for development mode")
            agg_primitives = agg_primitives[:3]  # Use only the first three primitives in dev mode
        
        # Fix date formats for woodwork to avoid warnings
        # Identify and preprocess date-related columns to prevent Woodwork parsing warnings
        reference_date = pd.Timestamp('2018-01-01')  # Use a reference date as a base
        
        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                date_columns = []
                
                # First pass: identify date-like columns
                for col in df.columns:
                    # Keep track of columns we've modified to avoid double processing
                    column_modified = False
                    
                    # Check if column looks like it contains date values
                    if any(date_term in col.upper() for date_term in ['DATE', 'DAY', 'DAYS', 'MONTH', 'MONTHS', 'YEAR']):
                        # For numeric days/months values (common in this dataset), convert to numeric but don't parse as dates
                        if pd.api.types.is_numeric_dtype(df[col].dtype):
                            try:
                                # Replace extreme values (often used as missing indicators)
                                if df[col].max() > 10000:
                                    outlier_value = 365243  # Common outlier value in this dataset
                                    df[col] = df[col].replace(outlier_value, np.nan)
                                
                                # Just ensure it's numeric without datetime conversion
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                logger.debug(f"Preprocessed numeric date-related column {col} in {name}")
                                column_modified = True
                            except Exception as e:
                                logger.warning(f"Could not preprocess date-related column {col} in {name}: {str(e)}")
                        
                        # For actual date columns that need parsing
                        elif pd.api.types.is_object_dtype(df[col].dtype) and not column_modified:
                            # Try to identify if this is an actual date column that needs conversion
                            sample_values = df[col].dropna().head(10).astype(str)
                            if sample_values.empty:
                                continue
                            
                            # Check if values look like dates by trying a few common formats
                            common_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
                            for date_format in common_formats:
                                try:
                                    # If this works for at least some values, it's probably a date
                                    pd.to_datetime(sample_values.iloc[0], format=date_format)
                                    # Convert with explicit format
                                    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                                    logger.debug(f"Converted date column {col} in {name} using format {date_format}")
                                    date_columns.append(col)
                                    column_modified = True
                                    break
                                except (ValueError, TypeError):
                                    continue
                
                # Handle special case DATE columns if they weren't already processed
                for col in df.columns:
                    if 'DATE' in col.upper() and col not in date_columns and not pd.api.types.is_numeric_dtype(df[col].dtype):
                        try:
                            # Try one more time with a flexible parser but with explicit errors='coerce'
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            logger.debug(f"Converted remaining date column {col} in {name}")
                        except Exception as e:
                            logger.warning(f"Could not convert date column {col} in {name}: {str(e)}")
        
        # Build entity set
        entity_set = self.entity_builder.build_entity_set(datasets)
        
        # Calculate optimal number of workers and chunk size based on dataset size
        total_memory_gb = self._get_available_memory_gb()
        logger.info(f"Available system memory: {total_memory_gb:.2f} GB")
        
        # Calculate memory per row (approximately)
        app_df = datasets.get('app', datasets.get('application_train', None))
        if app_df is not None:
            rows = app_df.shape[0]
            # Calculate memory more accurately with deep=True
            mem_per_row_kb = app_df.memory_usage(deep=True).sum() / rows / 1024
            
            # Estimate memory based on dataset size and complexity
            # The expansion factor depends on the depth and aggregation primitives
            # For complex operations, we need a higher memory estimation
            expansion_factor = 3  # Default for simple operations
            
            # Adjust expansion factor based on max_depth
            if max_depth is not None:
                expansion_factor = max(3, 2 * max_depth)
            
            # Adjust expansion factor based on number of primitives
            if agg_primitives and len(agg_primitives) > 5:
                expansion_factor = max(expansion_factor, len(agg_primitives))
            
            # Add additional factor for each related table (joins increase memory usage)
            related_tables_factor = 1 + (0.5 * (len(datasets) - 1)) if len(datasets) > 1 else 1
            
            # Final memory estimation
            estimated_memory_mb = (mem_per_row_kb * rows * expansion_factor * related_tables_factor) / 1024
            logger.info(f"Estimated memory needed per worker: {estimated_memory_mb:.2f} MB with expansion factor: {expansion_factor:.1f}x")
            
            # Calculate optimal workers based on memory constraints
            if n_jobs == -1 or n_jobs > 1:
                # Reserve more memory for system and other processes (at least 4GB)
                system_reserve_gb = min(4, total_memory_gb * 0.2)
                usable_memory_gb = max(1, total_memory_gb - system_reserve_gb)
                
                # Be more conservative with memory allocation - aim to use at most 60% of usable memory
                # This helps prevent OOM errors
                target_memory_gb = usable_memory_gb * 0.6
                
                # Estimate memory needed for feature generation
                worker_memory_gb = estimated_memory_mb / 1024
                
                if worker_memory_gb > 0:
                    # Calculate optimal number of workers
                    optimal_workers = int(target_memory_gb / worker_memory_gb)
                    
                    # For very large datasets, be even more conservative
                    if total_rows > 10000000:  # If more than 10M rows total
                        optimal_workers = max(1, optimal_workers - 1)
                        logger.info("Reducing worker count by 1 due to large dataset size")
                    
                    # Ensure at least 1 worker, at most system CPU count
                    import multiprocessing
                    cpu_count = multiprocessing.cpu_count()
                    optimal_workers = max(1, min(cpu_count - 1, optimal_workers))
                    
                    # If optimal is very low, that's a sign we might need smaller chunks
                    if optimal_workers <= 2 and chunk_size is None:
                        # Use a smaller chunk size for memory constrained environments
                        chunk_size = min(5000, rows // 10) if rows > 5000 else rows
                        logger.info(f"Memory constrained environment detected, setting smaller chunk size: {chunk_size}")
                    
                    if n_jobs == -1 or n_jobs > optimal_workers:
                        logger.info(f"Adjusting number of workers from {n_jobs} to {optimal_workers} based on memory constraints")
                        n_jobs = optimal_workers
                    
                    # For very large feature engineering tasks, lower the parallelism further
                    if max_depth is not None and max_depth >= 2 and len(datasets) >= 4:
                        safe_workers = max(1, n_jobs // 2)
                        logger.info(f"Complex feature engineering task detected. Further reducing workers from {n_jobs} to {safe_workers}")
                        n_jobs = safe_workers
        
        # Set a reasonable chunk size if not specified
        if chunk_size is None:
            if app_df is not None:
                rows = app_df.shape[0]
                # For small datasets, process all at once
                if rows <= 5000:
                    chunk_size = rows
                # For medium datasets, use approximately 20% of rows
                elif rows <= 50000:
                    chunk_size = max(1000, min(rows // 5, 10000))
                # For large datasets, use a smaller percentage
                else:
                    chunk_size = max(1000, min(rows // 10, 5000))
                logger.info(f"Setting automatic chunk size to {chunk_size}")
        
        if chunk_size is not None:
            logger.info(f"Using chunking with chunk_size={chunk_size}")
            # Handle both DataFrame and EntitySet dataframe formats
            # Sometimes 'app' is a DataFrame and sometimes it's an EntityDataFrame with df attribute
            try:
                if hasattr(entity_set['app'], 'df'):
                    chunk_size = min(chunk_size, entity_set['app'].df.shape[0])
                else:
                    # If entity_set['app'] is just a DataFrame itself
                    chunk_size = min(chunk_size, entity_set['app'].shape[0])
            except (AttributeError, KeyError) as e:
                logger.warning(f"Could not get entity shape for chunk sizing: {str(e)}")
                # Fall back to using app_df from datasets
                if app_df is not None:
                    chunk_size = min(chunk_size, app_df.shape[0])
        
        logger.info(f"Generating features with max_depth={max_depth}, n_jobs={n_jobs}")
        logger.info(f"Using aggregation primitives: {agg_primitives}")
        
        # Generate features using featuretools with performance optimizations
        try:
            # Define a proper progress callback that can handle multiple arguments
            def progress_callback(*args):
                if verbose:
                    if len(args) == 1:
                        logger.info(f"Progress: {args[0]}")
                    else:
                        logger.info(f"Progress update: {args}")
                
            features = ft.dfs(
                entityset=entity_set,
                target_dataframe_name="app",
                agg_primitives=agg_primitives,
                max_depth=max_depth,
                features_only=features_only,
                verbose=verbose,
                chunk_size=chunk_size,
                n_jobs=n_jobs,
                # Use a proper progress callback that can handle multiple arguments
                progress_callback=progress_callback
            )
            
            # Log feature generation results
            if not features_only:
                # If we have a DataFrame, log its shape and some stats
                if isinstance(features, pd.DataFrame):
                    rows, cols = features.shape
                    original_cols = len(datasets.get("app", datasets.get("application_train", pd.DataFrame())).columns)
                    new_cols = cols - original_cols  # Approximate number of new columns
                    
                    logger.info(f"Feature generation complete: {rows} rows, {cols} total columns")
                    logger.info(f"Original features: {original_cols}, New features generated: {new_cols}")
                    
                    # Additional stats about the feature matrix
                    numeric_cols = len(features.select_dtypes(include=['number']).columns)
                    logger.info(f"Numeric features: {numeric_cols}")
        
        except MemoryError as me:
            logger.error(f"Memory error during feature generation: {str(me)}")
            # Reduce workers to 1 and chunk size further for memory-constrained environments
            if n_jobs > 1:
                n_jobs = 1
                chunk_size = min(chunk_size, 5000) if chunk_size else 5000
                logger.info(f"Retrying with reduced resources: n_jobs={n_jobs}, chunk_size={chunk_size}")
                return self.generate_features(
                    datasets=datasets,
                    agg_primitives=agg_primitives,
                    max_depth=max_depth,
                    features_only=features_only,
                    verbose=verbose,
                    chunk_size=chunk_size,
                    n_jobs=n_jobs
                )
            else:
                # If already at minimum resources, raise the error
                raise
        except Exception as e:
            logger.error(f"Error during feature generation: {str(e)}")
            raise
        
        return features
    
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