"""
Features package for Credit Risk prediction.

This package includes modules for:
- Feature engineering and generation
- Feature selection and filtering
- Entity building and feature store functionality
"""

# Import key classes that might be used elsewhere
from .feature_store import FeatureStore
from .entity_builder import EntitySetBuilder
from .engineer import FeatureEngineer

__all__ = ["FeatureStore", "EntitySetBuilder", "FeatureEngineer"]
