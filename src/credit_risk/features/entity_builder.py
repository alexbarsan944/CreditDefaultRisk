"""
Entity set builder for featuretools feature engineering.

This module provides functionality for creating featuretools EntitySets
from credit risk datasets, defining the relationships between entities.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import featuretools as ft
import pandas as pd

from credit_risk.config import PipelineConfig, default_config

logger = logging.getLogger(__name__)


class EntitySetBuilder:
    """Builder for featuretools EntitySets.
    
    This class handles the creation of featuretools EntitySets from
    credit risk prediction datasets, including defining the proper
    relationships between entities.
    
    Attributes:
        config: Configuration for the entity set builder
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the entity set builder.
        
        Args:
            config: Configuration for the entity set builder. If None, uses default config.
        """
        self.config = config or default_config
    
    def build_entity_set(
        self, 
        datasets: Dict[str, pd.DataFrame],
        entity_set_id: str = "clients"
    ) -> ft.EntitySet:
        """Build a featuretools EntitySet from the provided datasets.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            entity_set_id: ID for the created EntitySet
            
        Returns:
            The built EntitySet with defined relationships
            
        Raises:
            ValueError: If required datasets are missing
        """
        # Check for required datasets
        required_datasets = ["application_train"]
        for ds in required_datasets:
            if ds not in datasets:
                raise ValueError(f"Required dataset '{ds}' is missing")
        
        # Initialize EntitySet
        es = ft.EntitySet(id=entity_set_id)
        
        # Add application DataFrame (main entity)
        app_df = datasets["application_train"]
        es = es.add_dataframe(
            dataframe_name="app",
            dataframe=app_df,
            index="SK_ID_CURR"
        )
        
        # Add bureau DataFrame if available
        if "bureau" in datasets:
            bureau_df = datasets["bureau"]
            es = es.add_dataframe(
                dataframe_name="bureau",
                dataframe=bureau_df,
                index="SK_ID_BUREAU"
            )
            # Add relationship between app and bureau
            es = es.add_relationship(
                parent_dataframe_name="app",
                parent_column_name="SK_ID_CURR",
                child_dataframe_name="bureau",
                child_column_name="SK_ID_CURR"
            )
        
        # Add previous applications DataFrame if available
        if "previous_application" in datasets:
            previous_df = datasets["previous_application"]
            es = es.add_dataframe(
                dataframe_name="previous",
                dataframe=previous_df,
                index="SK_ID_PREV"
            )
            # Add relationship between app and previous
            es = es.add_relationship(
                parent_dataframe_name="app",
                parent_column_name="SK_ID_CURR",
                child_dataframe_name="previous",
                child_column_name="SK_ID_CURR"
            )
        
        # Add bureau balance DataFrame if available
        if "bureau_balance" in datasets and "bureau" in datasets:
            bureau_balance_df = datasets["bureau_balance"]
            # Add index column if not present
            if "bureaubalance_index" not in bureau_balance_df.columns:
                bureau_balance_df["bureaubalance_index"] = range(1, len(bureau_balance_df) + 1)
            
            es = es.add_dataframe(
                dataframe_name="bureau_balance",
                dataframe=bureau_balance_df,
                index="bureaubalance_index"
            )
            # Add relationship between bureau and bureau_balance
            es = es.add_relationship(
                parent_dataframe_name="bureau",
                parent_column_name="SK_ID_BUREAU",
                child_dataframe_name="bureau_balance",
                child_column_name="SK_ID_BUREAU"
            )
        
        # Add POS cash balance DataFrame if available
        if "pos_cash_balance" in datasets and "previous_application" in datasets:
            cash_df = datasets["pos_cash_balance"]
            # Add index column if not present
            if "cash_index" not in cash_df.columns:
                cash_df["cash_index"] = range(1, len(cash_df) + 1)
            
            es = es.add_dataframe(
                dataframe_name="cash",
                dataframe=cash_df,
                index="cash_index"
            )
            # Add relationship between previous and cash
            es = es.add_relationship(
                parent_dataframe_name="previous",
                parent_column_name="SK_ID_PREV",
                child_dataframe_name="cash",
                child_column_name="SK_ID_PREV"
            )
        
        # Add installments payments DataFrame if available
        if "installments_payments" in datasets and "previous_application" in datasets:
            installments_df = datasets["installments_payments"]
            # Add index column if not present
            if "installments_index" not in installments_df.columns:
                installments_df["installments_index"] = range(1, len(installments_df) + 1)
            
            es = es.add_dataframe(
                dataframe_name="installments",
                dataframe=installments_df,
                index="installments_index"
            )
            # Add relationship between previous and installments
            es = es.add_relationship(
                parent_dataframe_name="previous",
                parent_column_name="SK_ID_PREV",
                child_dataframe_name="installments",
                child_column_name="SK_ID_PREV"
            )
        
        # Add credit card balance DataFrame if available
        if "credit_card_balance" in datasets and "previous_application" in datasets:
            credit_df = datasets["credit_card_balance"]
            # Add index column if not present
            if "credit_index" not in credit_df.columns:
                credit_df["credit_index"] = range(1, len(credit_df) + 1)
            
            es = es.add_dataframe(
                dataframe_name="credit",
                dataframe=credit_df,
                index="credit_index"
            )
            # Add relationship between previous and credit
            es = es.add_relationship(
                parent_dataframe_name="previous",
                parent_column_name="SK_ID_PREV",
                child_dataframe_name="credit",
                child_column_name="SK_ID_PREV"
            )
        
        logger.info(f"Built EntitySet with {len(es.dataframes)} dataframes")
        logger.info(f"EntitySet relationships: {len(es.relationships)}")
        
        return es
    
    def add_custom_index_columns(
        self, 
        datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Add custom index columns to datasets without a natural primary key.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            
        Returns:
            Dictionary with updated DataFrames
        """
        updated_datasets = {}
        
        # Define datasets that need custom index columns
        needs_index = {
            "bureau_balance": "bureaubalance_index",
            "pos_cash_balance": "cash_index",
            "installments_payments": "installments_index",
            "credit_card_balance": "credit_index"
        }
        
        for name, df in datasets.items():
            if name in needs_index and needs_index[name] not in df.columns:
                updated_df = df.copy()
                updated_df[needs_index[name]] = range(1, len(df) + 1)
                updated_datasets[name] = updated_df
            else:
                updated_datasets[name] = df
        
        return updated_datasets 