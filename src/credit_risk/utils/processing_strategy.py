"""
Processing strategy module for CPU and GPU acceleration.

This module provides a strategy pattern implementation for switching between
CPU and GPU data processing based on hardware availability and user preference.
"""
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProcessingBackend(str, Enum):
    """Enumeration of available processing backends."""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"  # Automatically select the best available backend


class ProcessingStrategy(ABC):
    """Abstract base class for processing strategies.
    
    This abstract class defines the interface for all processing strategies,
    regardless of whether they use CPU or GPU acceleration.
    """
    
    @abstractmethod
    def read_csv(self, file_path: str, **kwargs) -> Any:
        """Read a CSV file into a dataframe.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to the underlying implementation
            
        Returns:
            DataFrame object (implementation-specific)
        """
        pass
    
    @abstractmethod
    def to_pandas(self, df: Any) -> pd.DataFrame:
        """Convert a dataframe to pandas DataFrame.
        
        Args:
            df: DataFrame object from the current strategy
            
        Returns:
            pandas DataFrame
        """
        pass
    
    @abstractmethod
    def from_pandas(self, df: pd.DataFrame) -> Any:
        """Convert a pandas DataFrame to the strategy's dataframe type.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            DataFrame object for the current strategy
        """
        pass
    
    @abstractmethod
    def optimize_memory(self, df: Any, streamlit_compatible: bool = True) -> Tuple[Any, float]:
        """Optimize memory usage of a dataframe.
        
        Args:
            df: DataFrame to optimize
            streamlit_compatible: If True, use only types compatible with Streamlit/PyArrow
            
        Returns:
            Tuple containing:
                - Optimized DataFrame
                - Percentage of memory reduction
        """
        pass
    
    @abstractmethod
    def groupby_agg(
        self, 
        df: Any, 
        group_cols: List[str],
        agg_dict: Dict[str, str]
    ) -> Any:
        """Perform a groupby aggregation.
        
        Args:
            df: DataFrame to aggregate
            group_cols: Columns to group by
            agg_dict: Dictionary mapping column names to aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        pass
    
    @abstractmethod
    def merge(
        self,
        left: Any,
        right: Any,
        how: str = "inner",
        on: Optional[List[str]] = None,
        left_on: Optional[List[str]] = None,
        right_on: Optional[List[str]] = None
    ) -> Any:
        """Merge two dataframes.
        
        Args:
            left: Left DataFrame
            right: Right DataFrame
            how: Type of merge ('inner', 'outer', 'left', 'right')
            on: Column names to join on
            left_on: Column names from the left DataFrame to join on
            right_on: Column names from the right DataFrame to join on
            
        Returns:
            Merged DataFrame
        """
        pass
    
    @abstractmethod
    def filter(self, df: Any, condition: Any) -> Any:
        """Filter a dataframe based on a condition.
        
        Args:
            df: DataFrame to filter
            condition: Condition to filter on (implementation-specific)
            
        Returns:
            Filtered DataFrame
        """
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Get the name of the current backend.
        
        Returns:
            Name of the backend
        """
        pass
    
    @abstractmethod
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for this backend.
        
        Returns:
            True if GPU is available, False otherwise
        """
        pass


class PandasStrategy(ProcessingStrategy):
    """CPU-based processing strategy using pandas.
    
    This strategy uses pandas for data processing, which runs on the CPU.
    """
    
    def read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read a CSV file into a pandas DataFrame.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            pandas DataFrame
        """
        return pd.read_csv(file_path, **kwargs)
    
    def to_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the pandas DataFrame as is.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Same pandas DataFrame
        """
        return df
    
    def from_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the pandas DataFrame as is.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Same pandas DataFrame
        """
        return df
    
    def optimize_memory(self, df: pd.DataFrame, streamlit_compatible: bool = True) -> Tuple[pd.DataFrame, float]:
        """Optimize memory usage of a pandas DataFrame.
        
        Args:
            df: DataFrame to optimize
            streamlit_compatible: If True, use only types compatible with Streamlit/PyArrow
            
        Returns:
            Tuple containing:
                - Optimized DataFrame
                - Percentage of memory reduction
        """
        from credit_risk.data.optimizers import reduce_mem_usage
        return reduce_mem_usage(df, verbose=False, streamlit_compatible=streamlit_compatible)
    
    def groupby_agg(
        self, 
        df: pd.DataFrame, 
        group_cols: List[str],
        agg_dict: Dict[str, str]
    ) -> pd.DataFrame:
        """Perform a groupby aggregation using pandas.
        
        Args:
            df: DataFrame to aggregate
            group_cols: Columns to group by
            agg_dict: Dictionary mapping column names to aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        return df.groupby(group_cols).agg(agg_dict).reset_index()
    
    def merge(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        how: str = "inner",
        on: Optional[List[str]] = None,
        left_on: Optional[List[str]] = None,
        right_on: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Merge two pandas DataFrames.
        
        Args:
            left: Left DataFrame
            right: Right DataFrame
            how: Type of merge ('inner', 'outer', 'left', 'right')
            on: Column names to join on
            left_on: Column names from the left DataFrame to join on
            right_on: Column names from the right DataFrame to join on
            
        Returns:
            Merged DataFrame
        """
        return pd.merge(
            left=left,
            right=right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on
        )
    
    def filter(self, df: pd.DataFrame, condition: pd.Series) -> pd.DataFrame:
        """Filter a pandas DataFrame based on a condition.
        
        Args:
            df: DataFrame to filter
            condition: Boolean Series to filter on
            
        Returns:
            Filtered DataFrame
        """
        return df[condition]
    
    def get_backend_name(self) -> str:
        """Get the name of the current backend.
        
        Returns:
            Name of the backend
        """
        return "pandas"
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for this backend.
        
        Returns:
            Always False for pandas
        """
        return False


class PolarsStrategy(ProcessingStrategy):
    """CPU/GPU processing strategy using Polars.
    
    This strategy uses Polars for data processing, which can run on either
    CPU or GPU depending on available hardware and configuration.
    """
    
    def __init__(self, use_gpu: bool = False):
        """Initialize the Polars strategy.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        try:
            import polars as pl
            self.pl = pl
            self._use_gpu = use_gpu and self._check_gpu_support()
            
            if self._use_gpu:
                logger.info("Using Polars with GPU acceleration")
            else:
                logger.info("Using Polars with CPU processing")
                
        except ImportError:
            logger.warning("Polars not available. Install with 'pip install polars'")
            raise ImportError("Polars is required for PolarsStrategy")
    
    def _check_gpu_support(self) -> bool:
        """Check if GPU support is available in Polars.
        
        Returns:
            True if GPU support is available, False otherwise
        """
        try:
            import polars as pl
            # Check if polars has been compiled with CUDA support
            return hasattr(pl, "using_cuda") and pl.using_cuda()
        except (ImportError, AttributeError):
            return False
    
    def read_csv(self, file_path: str, **kwargs) -> "pl.DataFrame":
        """Read a CSV file into a Polars DataFrame.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pl.read_csv
            
        Returns:
            Polars DataFrame
        """
        return self.pl.read_csv(file_path, **kwargs)
    
    def to_pandas(self, df: "pl.DataFrame") -> pd.DataFrame:
        """Convert a Polars DataFrame to pandas DataFrame.
        
        Args:
            df: Polars DataFrame
            
        Returns:
            pandas DataFrame
        """
        return df.to_pandas()
    
    def from_pandas(self, df: pd.DataFrame) -> "pl.DataFrame":
        """Convert a pandas DataFrame to Polars DataFrame.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Polars DataFrame
        """
        return self.pl.from_pandas(df)
    
    def optimize_memory(self, df: "pl.DataFrame", streamlit_compatible: bool = True) -> Tuple["pl.DataFrame", float]:
        """Optimize memory usage of a Polars DataFrame.
        
        Args:
            df: DataFrame to optimize
            streamlit_compatible: If True, use only types compatible with Streamlit/PyArrow
            
        Returns:
            Tuple containing:
                - Optimized DataFrame
                - Percentage of memory reduction
        """
        # If streamlit_compatible is True, convert to pandas and optimize there
        if streamlit_compatible:
            # Get current memory usage
            current_mem = df.estimated_size()
            
            # Convert to pandas for streamlit-compatible optimization
            pandas_df = self.to_pandas(df)
            
            # Optimize the pandas DataFrame
            from credit_risk.data.optimizers import reduce_mem_usage
            pandas_df, _ = reduce_mem_usage(pandas_df, verbose=False, streamlit_compatible=True)
            
            # Convert back to polars
            optimized_df = self.from_pandas(pandas_df)
            
            # Calculate memory reduction
            new_mem = optimized_df.estimated_size()
            reduction_percentage = 100 * (current_mem - new_mem) / current_mem
            
            return optimized_df, reduction_percentage
        
        # Otherwise, use polars' native optimization
        # Get current memory usage
        current_mem = df.estimated_size()
        
        # Create schema with optimized types
        schema = {}
        for col_name in df.columns:
            col = df[col_name]
            dtype = col.dtype
            
            # Integer optimization
            if dtype in (self.pl.Int64, self.pl.UInt64):
                # Check the value range
                min_val = col.min()
                max_val = col.max()
                
                # Choose the smallest integer type that can hold the data
                if min_val is not None and max_val is not None:
                    if min_val >= 0:  # Unsigned
                        if max_val <= 255:
                            schema[col_name] = self.pl.UInt8
                        elif max_val <= 65535:
                            schema[col_name] = self.pl.UInt16
                        elif max_val <= 4294967295:
                            schema[col_name] = self.pl.UInt32
                    else:  # Signed
                        if min_val >= -128 and max_val <= 127:
                            schema[col_name] = self.pl.Int8
                        elif min_val >= -32768 and max_val <= 32767:
                            schema[col_name] = self.pl.Int16
                        elif min_val >= -2147483648 and max_val <= 2147483647:
                            schema[col_name] = self.pl.Int32
            
            # Float optimization
            elif dtype == self.pl.Float64:
                # Check the value range
                min_val = col.min()
                max_val = col.max()
                
                # Check if Float32 can represent the data without loss
                if min_val is not None and max_val is not None:
                    if abs(min_val) <= 3.4e38 and abs(max_val) <= 3.4e38:
                        schema[col_name] = self.pl.Float32
        
        # Apply the optimized schema
        if schema:
            optimized_df = df.cast(schema)
        else:
            optimized_df = df
        
        # Calculate memory reduction
        new_mem = optimized_df.estimated_size()
        reduction_percentage = 100 * (current_mem - new_mem) / current_mem
        
        return optimized_df, reduction_percentage
    
    def groupby_agg(
        self, 
        df: "pl.DataFrame", 
        group_cols: List[str],
        agg_dict: Dict[str, str]
    ) -> "pl.DataFrame":
        """Perform a groupby aggregation using Polars.
        
        Args:
            df: DataFrame to aggregate
            group_cols: Columns to group by
            agg_dict: Dictionary mapping column names to aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        # Convert agg_dict to Polars format
        agg_exprs = []
        for col, agg_func in agg_dict.items():
            if agg_func == "sum":
                agg_exprs.append(self.pl.col(col).sum().alias(f"{col}_sum"))
            elif agg_func == "mean":
                agg_exprs.append(self.pl.col(col).mean().alias(f"{col}_mean"))
            elif agg_func == "count":
                agg_exprs.append(self.pl.col(col).count().alias(f"{col}_count"))
            elif agg_func == "min":
                agg_exprs.append(self.pl.col(col).min().alias(f"{col}_min"))
            elif agg_func == "max":
                agg_exprs.append(self.pl.col(col).max().alias(f"{col}_max"))
            elif agg_func == "std":
                agg_exprs.append(self.pl.col(col).std().alias(f"{col}_std"))
        
        return df.group_by(group_cols).agg(agg_exprs)
    
    def merge(
        self,
        left: "pl.DataFrame",
        right: "pl.DataFrame",
        how: str = "inner",
        on: Optional[List[str]] = None,
        left_on: Optional[List[str]] = None,
        right_on: Optional[List[str]] = None
    ) -> "pl.DataFrame":
        """Merge two Polars DataFrames.
        
        Args:
            left: Left DataFrame
            right: Right DataFrame
            how: Type of merge ('inner', 'outer', 'left', 'right')
            on: Column names to join on
            left_on: Column names from the left DataFrame to join on
            right_on: Column names from the right DataFrame to join on
            
        Returns:
            Merged DataFrame
        """
        # Handle different join types
        if how == "inner":
            join_type = "inner"
        elif how == "outer":
            join_type = "outer"
        elif how == "left":
            join_type = "left"
        elif how == "right":
            join_type = "right"
        else:
            raise ValueError(f"Unsupported join type: {how}")
        
        # Handle different join columns
        if on is not None:
            return left.join(right, on=on, how=join_type)
        elif left_on is not None and right_on is not None:
            return left.join(right, left_on=left_on, right_on=right_on, how=join_type)
        else:
            raise ValueError("Either 'on' or both 'left_on' and 'right_on' must be specified")
    
    def filter(self, df: "pl.DataFrame", condition: "pl.Expr") -> "pl.DataFrame":
        """Filter a Polars DataFrame based on a condition.
        
        Args:
            df: DataFrame to filter
            condition: Polars expression to filter on
            
        Returns:
            Filtered DataFrame
        """
        return df.filter(condition)
    
    def get_backend_name(self) -> str:
        """Get the name of the current backend.
        
        Returns:
            Name of the backend
        """
        if self._use_gpu:
            return "polars-gpu"
        else:
            return "polars-cpu"
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for this backend.
        
        Returns:
            True if GPU is available and configured, False otherwise
        """
        return self._use_gpu


class CudfStrategy(ProcessingStrategy):
    """GPU processing strategy using cuDF.
    
    This strategy uses cuDF for data processing, which runs on NVIDIA GPUs.
    """
    
    def __init__(self):
        """Initialize the cuDF strategy.
        
        Raises:
            ImportError: If cuDF is not available
        """
        try:
            import cudf
            import cupy
            self.cudf = cudf
            self.cupy = cupy
            logger.info("Using cuDF for GPU-accelerated data processing")
        except ImportError:
            logger.warning(
                "cuDF not available. Install RAPIDS with "
                "'conda install -c rapidsai -c conda-forge cudf=23.04 python=3.10 cudatoolkit=11.8'"
            )
            raise ImportError("cuDF is required for CudfStrategy")
    
    def read_csv(self, file_path: str, **kwargs) -> "cudf.DataFrame":
        """Read a CSV file into a cuDF DataFrame.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to cudf.read_csv
            
        Returns:
            cuDF DataFrame
        """
        return self.cudf.read_csv(file_path, **kwargs)
    
    def to_pandas(self, df: "cudf.DataFrame") -> pd.DataFrame:
        """Convert a cuDF DataFrame to pandas DataFrame.
        
        Args:
            df: cuDF DataFrame
            
        Returns:
            pandas DataFrame
        """
        return df.to_pandas()
    
    def from_pandas(self, df: pd.DataFrame) -> "cudf.DataFrame":
        """Convert a pandas DataFrame to cuDF DataFrame.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            cuDF DataFrame
        """
        return self.cudf.DataFrame.from_pandas(df)
    
    def optimize_memory(self, df: "cudf.DataFrame", streamlit_compatible: bool = True) -> Tuple["cudf.DataFrame", float]:
        """Optimize memory usage of a cuDF DataFrame.
        
        Args:
            df: DataFrame to optimize
            streamlit_compatible: If True, use only types compatible with Streamlit/PyArrow
            
        Returns:
            Tuple containing:
                - Optimized DataFrame
                - Percentage of memory reduction
        """
        # If streamlit_compatible is True, convert to pandas and optimize there
        if streamlit_compatible:
            # Get current memory usage
            current_mem = df.memory_usage().sum() / 1024**2
            
            # Convert to pandas for streamlit-compatible optimization
            pandas_df = self.to_pandas(df)
            
            # Optimize the pandas DataFrame
            from credit_risk.data.optimizers import reduce_mem_usage
            pandas_df, _ = reduce_mem_usage(pandas_df, verbose=False, streamlit_compatible=True)
            
            # Convert back to cudf
            optimized_df = self.from_pandas(pandas_df)
            
            # Calculate memory reduction
            new_mem = optimized_df.memory_usage().sum() / 1024**2
            reduction_percentage = 100 * (current_mem - new_mem) / current_mem
            
            return optimized_df, reduction_percentage
        
        # Otherwise, use cudf's native optimization (simplified version)
        # Get current memory usage
        current_mem = df.memory_usage().sum() / 1024**2
        
        # Process each column to find optimal dtype
        for col in df.columns:
            col_type = df[col].dtype
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(col_type):
                continue
                
            # Integer columns
            if pd.api.types.is_integer_dtype(col_type):
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Find smallest integer type
                if min_val >= 0:  # unsigned
                    if max_val <= 255:
                        df[col] = df[col].astype('uint8')
                    elif max_val <= 65535:
                        df[col] = df[col].astype('uint16')
                    elif max_val <= 4294967295:
                        df[col] = df[col].astype('uint32')
                else:  # signed
                    if min_val >= -128 and max_val <= 127:
                        df[col] = df[col].astype('int8')
                    elif min_val >= -32768 and max_val <= 32767:
                        df[col] = df[col].astype('int16')
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        df[col] = df[col].astype('int32')
            
            # Float columns - downcast when possible
            elif pd.api.types.is_float_dtype(col_type):
                min_val = df[col].min()
                max_val = df[col].max()
                
                if min_val is not None and max_val is not None:
                    if abs(min_val) <= 3.4e38 and abs(max_val) <= 3.4e38:
                        df[col] = df[col].astype('float32')
        
        # Calculate memory reduction
        new_mem = df.memory_usage().sum() / 1024**2
        reduction_percentage = 100 * (current_mem - new_mem) / current_mem
        
        return df, reduction_percentage
    
    def groupby_agg(
        self, 
        df: "cudf.DataFrame", 
        group_cols: List[str],
        agg_dict: Dict[str, str]
    ) -> "cudf.DataFrame":
        """Perform a groupby aggregation using cuDF.
        
        Args:
            df: DataFrame to aggregate
            group_cols: Columns to group by
            agg_dict: Dictionary mapping column names to aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        return df.groupby(group_cols).agg(agg_dict).reset_index()
    
    def merge(
        self,
        left: "cudf.DataFrame",
        right: "cudf.DataFrame",
        how: str = "inner",
        on: Optional[List[str]] = None,
        left_on: Optional[List[str]] = None,
        right_on: Optional[List[str]] = None
    ) -> "cudf.DataFrame":
        """Merge two cuDF DataFrames.
        
        Args:
            left: Left DataFrame
            right: Right DataFrame
            how: Type of merge ('inner', 'outer', 'left', 'right')
            on: Column names to join on
            left_on: Column names from the left DataFrame to join on
            right_on: Column names from the right DataFrame to join on
            
        Returns:
            Merged DataFrame
        """
        return self.cudf.merge(
            left=left,
            right=right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on
        )
    
    def filter(self, df: "cudf.DataFrame", condition: "cudf.Series") -> "cudf.DataFrame":
        """Filter a cuDF DataFrame based on a condition.
        
        Args:
            df: DataFrame to filter
            condition: Boolean Series to filter on
            
        Returns:
            Filtered DataFrame
        """
        return df[condition]
    
    def get_backend_name(self) -> str:
        """Get the name of the current backend.
        
        Returns:
            Name of the backend
        """
        return "cudf"
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for this backend.
        
        Returns:
            Always True for cuDF (since it requires GPU)
        """
        return True


class ProcessingContext:
    """Context for using different processing strategies.
    
    This class manages the processing strategy and provides a unified interface
    for data processing operations.
    """
    
    def __init__(self, backend: ProcessingBackend = ProcessingBackend.AUTO):
        """Initialize the processing context.
        
        Args:
            backend: Processing backend to use
        """
        self._strategy = self._get_strategy(backend)
        logger.info(f"Using {self._strategy.get_backend_name()} backend for data processing")
    
    def _get_strategy(self, backend: ProcessingBackend) -> ProcessingStrategy:
        """Get a processing strategy based on the specified backend.
        
        Args:
            backend: Processing backend to use
            
        Returns:
            Appropriate processing strategy
        """
        if backend == ProcessingBackend.CPU:
            try:
                # Try Polars CPU first for better performance
                return PolarsStrategy(use_gpu=False)
            except ImportError:
                # Fall back to pandas
                return PandasStrategy()
        
        elif backend == ProcessingBackend.GPU:
            # Try GPU-accelerated libraries in order of preference
            try:
                # Try cuDF first
                return CudfStrategy()
            except ImportError:
                try:
                    # Try Polars with GPU acceleration
                    strategy = PolarsStrategy(use_gpu=True)
                    if strategy.is_gpu_available():
                        return strategy
                    else:
                        logger.warning("Polars GPU acceleration not available, falling back to CPU")
                        return PolarsStrategy(use_gpu=False)
                except ImportError:
                    # Fall back to pandas
                    logger.warning("No GPU-accelerated libraries available, falling back to pandas")
                    return PandasStrategy()
        
        elif backend == ProcessingBackend.AUTO:
            # Try all backends in order of preference
            try:
                # Try cuDF first
                return CudfStrategy()
            except ImportError:
                try:
                    # Try Polars with GPU acceleration
                    strategy = PolarsStrategy(use_gpu=True)
                    if strategy.is_gpu_available():
                        return strategy
                    else:
                        # Fall back to Polars CPU
                        return PolarsStrategy(use_gpu=False)
                except ImportError:
                    # Fall back to pandas
                    return PandasStrategy()
        
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    @property
    def strategy(self) -> ProcessingStrategy:
        """Get the current processing strategy.
        
        Returns:
            Current processing strategy
        """
        return self._strategy
    
    def set_strategy(self, backend: ProcessingBackend) -> None:
        """Set a new processing strategy.
        
        Args:
            backend: Processing backend to use
        """
        self._strategy = self._get_strategy(backend)
        logger.info(f"Switched to {self._strategy.get_backend_name()} backend for data processing")
    
    def get_backend_name(self) -> str:
        """Get the name of the current backend.
        
        Returns:
            Name of the backend
        """
        return self._strategy.get_backend_name()
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for the current backend.
        
        Returns:
            True if GPU is available, False otherwise
        """
        return self._strategy.is_gpu_available()
    
    def read_csv(self, file_path: str, **kwargs) -> Any:
        """Read a CSV file into a dataframe.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to the underlying implementation
            
        Returns:
            DataFrame object (implementation-specific)
        """
        return self._strategy.read_csv(file_path, **kwargs)
    
    def to_pandas(self, df: Any) -> pd.DataFrame:
        """Convert a dataframe to pandas DataFrame.
        
        Args:
            df: DataFrame object from the current strategy
            
        Returns:
            pandas DataFrame
        """
        return self._strategy.to_pandas(df)
    
    def from_pandas(self, df: pd.DataFrame) -> Any:
        """Convert a pandas DataFrame to the strategy's dataframe type.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            DataFrame object for the current strategy
        """
        return self._strategy.from_pandas(df)
    
    def optimize_memory(self, df: Any, streamlit_compatible: bool = True) -> Tuple[Any, float]:
        """Optimize memory usage of a dataframe.
        
        Args:
            df: DataFrame to optimize
            streamlit_compatible: If True, use only types compatible with Streamlit/PyArrow
            
        Returns:
            Tuple containing:
                - Optimized DataFrame
                - Percentage of memory reduction
        """
        return self._strategy.optimize_memory(df, streamlit_compatible=streamlit_compatible)
    
    def groupby_agg(
        self, 
        df: Any, 
        group_cols: List[str],
        agg_dict: Dict[str, str]
    ) -> Any:
        """Perform a groupby aggregation.
        
        Args:
            df: DataFrame to aggregate
            group_cols: Columns to group by
            agg_dict: Dictionary mapping column names to aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        return self._strategy.groupby_agg(df, group_cols, agg_dict)
    
    def merge(
        self,
        left: Any,
        right: Any,
        how: str = "inner",
        on: Optional[List[str]] = None,
        left_on: Optional[List[str]] = None,
        right_on: Optional[List[str]] = None
    ) -> Any:
        """Merge two dataframes.
        
        Args:
            left: Left DataFrame
            right: Right DataFrame
            how: Type of merge ('inner', 'outer', 'left', 'right')
            on: Column names to join on
            left_on: Column names from the left DataFrame to join on
            right_on: Column names from the right DataFrame to join on
            
        Returns:
            Merged DataFrame
        """
        return self._strategy.merge(left, right, how, on, left_on, right_on)
    
    def filter(self, df: Any, condition: Any) -> Any:
        """Filter a dataframe based on a condition.
        
        Args:
            df: DataFrame to filter
            condition: Condition to filter on (implementation-specific)
            
        Returns:
            Filtered DataFrame
        """
        return self._strategy.filter(df, condition) 