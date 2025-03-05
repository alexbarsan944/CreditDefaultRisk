"""
Tests for data optimizers functionality.
"""
import numpy as np
import pandas as pd
import pytest

from credit_risk.data.optimizers import chunk_dataframe, reduce_mem_usage


class TestReduceMemUsage:
    """Tests for the reduce_mem_usage function."""
    
    def test_reduces_memory_for_numeric_columns(self):
        """Test that memory usage is reduced for numeric columns."""
        # Create a test DataFrame with numeric columns
        df = pd.DataFrame({
            "int_col": np.random.randint(0, 100, size=1000),  # int64
            "float_col": np.random.random(1000) * 100,  # float64
            "small_int_col": np.random.randint(0, 10, size=1000),  # Should become int8
            "binary_col": np.random.randint(0, 2, size=1000),  # Should become int8
        })
        
        # Get initial memory usage
        initial_mem = df.memory_usage().sum()
        
        # Optimize memory usage
        optimized_df, reduction = reduce_mem_usage(df, verbose=False)
        
        # Get final memory usage
        final_mem = optimized_df.memory_usage().sum()
        
        # Check that memory usage is reduced
        assert final_mem < initial_mem
        assert reduction > 0
        
        # Check data types are as expected
        assert optimized_df["binary_col"].dtype == np.int8
        assert optimized_df["small_int_col"].dtype == np.int8
    
    def test_preserves_values_after_optimization(self):
        """Test that data values are preserved after memory optimization."""
        # Create a test DataFrame
        df = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
        })
        
        # Optimize memory usage
        optimized_df, _ = reduce_mem_usage(df, verbose=False)
        
        # Check that values are preserved
        pd.testing.assert_series_equal(
            df["int_col"], 
            optimized_df["int_col"], 
            check_dtype=False
        )
        pd.testing.assert_series_equal(
            df["float_col"], 
            optimized_df["float_col"], 
            check_dtype=False
        )
    
    def test_converts_object_columns_to_category(self):
        """Test that object columns are converted to category dtype."""
        # Create a test DataFrame with an object column
        df = pd.DataFrame({
            "category_col": ["A", "B", "A", "C", "B", "A"] * 100,
        })
        
        # Optimize memory usage
        optimized_df, _ = reduce_mem_usage(df, verbose=False)
        
        # Check that object column is converted to category
        assert optimized_df["category_col"].dtype == "category"
        
        # Check that values are preserved
        assert list(optimized_df["category_col"].value_counts().index) == ["A", "B", "C"]


class TestChunkDataframe:
    """Tests for the chunk_dataframe function."""
    
    def test_chunks_dataframe_correctly(self):
        """Test that dataframe is chunked correctly."""
        # Create a test DataFrame
        df = pd.DataFrame({"A": range(100)})
        
        # Split into chunks of size 10
        chunks = chunk_dataframe(df, chunk_size=10)
        
        # Check the number of chunks
        assert len(chunks) == 10
        
        # Check the size of each chunk
        for chunk in chunks:
            assert len(chunk) == 10
        
        # Check that all data is preserved
        reconstructed = pd.concat(chunks)
        pd.testing.assert_frame_equal(reconstructed.reset_index(drop=True), df)
    
    def test_handles_non_divisible_chunk_size(self):
        """Test that non-divisible chunk sizes are handled correctly."""
        # Create a test DataFrame
        df = pd.DataFrame({"A": range(105)})
        
        # Split into chunks of size 20
        chunks = chunk_dataframe(df, chunk_size=20)
        
        # Check the number of chunks
        assert len(chunks) == 6
        
        # Check the size of each chunk
        for i in range(5):
            assert len(chunks[i]) == 20
        assert len(chunks[5]) == 5
        
        # Check that all data is preserved
        reconstructed = pd.concat(chunks)
        pd.testing.assert_frame_equal(reconstructed.reset_index(drop=True), df)
    
    def test_handles_empty_dataframe(self):
        """Test that empty dataframes are handled correctly."""
        # Create an empty DataFrame
        df = pd.DataFrame({"A": []})
        
        # Split into chunks
        chunks = chunk_dataframe(df, chunk_size=10)
        
        # Check that no chunks are created
        assert len(chunks) == 0 