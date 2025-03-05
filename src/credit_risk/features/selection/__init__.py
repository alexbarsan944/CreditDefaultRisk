"""
Feature Selection module for Credit Risk prediction.

This module provides tools for selecting important features using null importance
methods and removing highly correlated features.
"""

# Use the fixed feature selector file
from .feature_selector_fixed import FeatureSelector

__all__ = ["FeatureSelector"] 