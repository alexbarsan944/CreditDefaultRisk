"""
Feature store for credit risk prediction.

This module provides functionality for managing, versioning, and accessing
features for credit risk prediction models.
"""
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from credit_risk.config import PipelineConfig, default_config

logger = logging.getLogger(__name__)


class FeatureStore:
    """Feature store for credit risk prediction.
    
    This class provides functionality for storing, retrieving, and managing
    features for credit risk prediction models.
    
    Attributes:
        config: Configuration for the feature store
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the feature store.
        
        Args:
            config: Configuration for the feature store. If None, uses default config.
        """
        self.config = config or default_config
        self._ensure_feature_store_exists()
    
    def _ensure_feature_store_exists(self) -> None:
        """Ensure that the feature store directory exists."""
        # Create features directory if it doesn't exist
        self.config.data.features_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file if it doesn't exist
        metadata_path = self.config.data.features_path / "metadata.json"
        if not metadata_path.exists():
            with open(metadata_path, "w") as f:
                json.dump({"feature_sets": {}}, f)
    
    def save_feature_set(
        self, 
        features: pd.DataFrame,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Path:
        """Save a feature set to the feature store.
        
        Args:
            features: DataFrame containing the features
            name: Name of the feature set
            description: Description of the feature set
            tags: List of tags for the feature set
            metadata: Additional metadata for the feature set
            
        Returns:
            Path to the saved feature set
        """
        # Generate version identifier based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{name}_{timestamp}"
        
        # Create directory for this feature set if it doesn't exist
        feature_set_dir = self.config.data.features_path / name
        feature_set_dir.mkdir(exist_ok=True)
        
        # Save features as parquet
        features_path = feature_set_dir / f"{version}.parquet"
        features.to_parquet(features_path)
        
        # Update metadata
        metadata_path = self.config.data.features_path / "metadata.json"
        with open(metadata_path, "r") as f:
            store_metadata = json.load(f)
        
        # Initialize feature set in metadata if it doesn't exist
        if name not in store_metadata["feature_sets"]:
            store_metadata["feature_sets"][name] = {
                "description": description,
                "tags": tags or [],
                "versions": []
            }
        
        # Add this version to metadata
        version_metadata = {
            "version": version,
            "timestamp": timestamp,
            "num_features": features.shape[1],
            "num_rows": features.shape[0],
            "columns": list(features.columns),
            "metadata": metadata or {}
        }
        
        store_metadata["feature_sets"][name]["versions"].append(version_metadata)
        
        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(store_metadata, f, indent=2)
        
        logger.info(f"Saved feature set '{name}' version '{version}' with {features.shape[1]} features")
        return features_path
    
    def load_feature_set(
        self, 
        name: str,
        version: Optional[str] = None
    ) -> pd.DataFrame:
        """Load a feature set from the feature store.
        
        Args:
            name: Name of the feature set to load
            version: Specific version to load. If None, loads the latest version.
            
        Returns:
            DataFrame containing the features
            
        Raises:
            FileNotFoundError: If the feature set or version does not exist
        """
        # Check if feature set exists
        feature_set_dir = self.config.data.features_path / name
        if not feature_set_dir.exists():
            raise FileNotFoundError(f"Feature set '{name}' not found in feature store")
        
        # Get metadata
        metadata_path = self.config.data.features_path / "metadata.json"
        with open(metadata_path, "r") as f:
            store_metadata = json.load(f)
        
        # Check if feature set exists in metadata
        if name not in store_metadata["feature_sets"]:
            raise FileNotFoundError(f"Feature set '{name}' not found in feature store metadata")
        
        # Get version to load
        if version is None:
            # Find the latest version
            versions = store_metadata["feature_sets"][name]["versions"]
            if not versions:
                raise FileNotFoundError(f"No versions found for feature set '{name}'")
            
            # Sort versions by timestamp (latest first)
            versions.sort(key=lambda v: v["timestamp"], reverse=True)
            version = versions[0]["version"]
        
        # Check if file exists
        features_path = feature_set_dir / f"{version}.parquet"
        if not features_path.exists():
            raise FileNotFoundError(f"Version '{version}' of feature set '{name}' not found")
        
        # Load features
        logger.info(f"Loading feature set '{name}' version '{version}'")
        features = pd.read_parquet(features_path)
        
        return features
    
    def list_feature_sets(self) -> List[Dict]:
        """List all feature sets in the feature store.
        
        Returns:
            List of dictionaries containing information about each feature set
        """
        metadata_path = self.config.data.features_path / "metadata.json"
        with open(metadata_path, "r") as f:
            store_metadata = json.load(f)
        
        feature_sets = []
        for name, metadata in store_metadata["feature_sets"].items():
            # Get the latest version information
            versions = metadata["versions"]
            latest_version = None
            if versions:
                # Sort versions by timestamp (latest first)
                versions.sort(key=lambda v: v["timestamp"], reverse=True)
                latest_version = versions[0]
            
            feature_sets.append({
                "name": name,
                "description": metadata["description"],
                "tags": metadata["tags"],
                "num_versions": len(versions),
                "latest_version": latest_version["version"] if latest_version else None,
                "latest_timestamp": latest_version["timestamp"] if latest_version else None,
                "num_features": latest_version["num_features"] if latest_version else None
            })
        
        return feature_sets
    
    def get_feature_set_versions(self, name: str) -> List[Dict]:
        """Get all versions of a feature set.
        
        Args:
            name: Name of the feature set
            
        Returns:
            List of dictionaries containing information about each version
            
        Raises:
            FileNotFoundError: If the feature set does not exist
        """
        metadata_path = self.config.data.features_path / "metadata.json"
        with open(metadata_path, "r") as f:
            store_metadata = json.load(f)
        
        if name not in store_metadata["feature_sets"]:
            raise FileNotFoundError(f"Feature set '{name}' not found in feature store")
        
        # Sort versions by timestamp (latest first)
        versions = store_metadata["feature_sets"][name]["versions"]
        versions.sort(key=lambda v: v["timestamp"], reverse=True)
        
        return versions
    
    def delete_feature_set(self, name: str) -> None:
        """Delete a feature set from the feature store.
        
        Args:
            name: Name of the feature set to delete
            
        Raises:
            FileNotFoundError: If the feature set does not exist
        """
        # Check if feature set exists
        feature_set_dir = self.config.data.features_path / name
        if not feature_set_dir.exists():
            raise FileNotFoundError(f"Feature set '{name}' not found in feature store")
        
        # Update metadata
        metadata_path = self.config.data.features_path / "metadata.json"
        with open(metadata_path, "r") as f:
            store_metadata = json.load(f)
        
        if name not in store_metadata["feature_sets"]:
            raise FileNotFoundError(f"Feature set '{name}' not found in feature store metadata")
        
        # Remove feature set from metadata
        del store_metadata["feature_sets"][name]
        
        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(store_metadata, f, indent=2)
        
        # Delete all files in feature set directory
        for file in feature_set_dir.glob("*"):
            file.unlink()
        
        # Delete feature set directory
        feature_set_dir.rmdir()
        
        logger.info(f"Deleted feature set '{name}'")
    
    def get_feature_metadata(
        self, 
        name: str,
        version: Optional[str] = None
    ) -> Dict:
        """Get metadata for a feature set.
        
        Args:
            name: Name of the feature set
            version: Specific version to get metadata for. If None, gets metadata for all versions.
            
        Returns:
            Dictionary containing feature set metadata
            
        Raises:
            FileNotFoundError: If the feature set or version does not exist
        """
        metadata_path = self.config.data.features_path / "metadata.json"
        with open(metadata_path, "r") as f:
            store_metadata = json.load(f)
        
        if name not in store_metadata["feature_sets"]:
            raise FileNotFoundError(f"Feature set '{name}' not found in feature store")
        
        feature_set_metadata = store_metadata["feature_sets"][name]
        
        if version is not None:
            # Find specific version
            for ver in feature_set_metadata["versions"]:
                if ver["version"] == version:
                    return ver
            
            raise FileNotFoundError(f"Version '{version}' of feature set '{name}' not found")
        
        return feature_set_metadata 