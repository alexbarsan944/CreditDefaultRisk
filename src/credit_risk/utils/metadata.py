"""
Metadata utilities for model tracking and management
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime

@dataclass
class ModelMetadata:
    """
    Dataclass for storing model metadata
    """
    model_id: str
    model_type: str
    created_at: str
    model_params: Dict[str, Any]
    feature_names: Optional[List[str]] = None
    description: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    feature_selection_timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert metadata to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, filepath: str) -> None:
        """Save metadata to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelMetadata':
        """Create metadata from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelMetadata':
        """Load metadata from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_model_id() -> str:
    """Generate a unique model ID based on timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"model_{timestamp}"


def list_models(models_dir: str) -> List[Dict[str, Any]]:
    """
    List all models in the models directory
    
    Parameters:
    -----------
    models_dir : str
        Path to the models directory
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of model metadata dictionaries
    """
    models = []
    
    if not os.path.exists(models_dir):
        return models
    
    # Iterate through model directories
    for model_dir in os.listdir(models_dir):
        metadata_path = os.path.join(models_dir, model_dir, "metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                metadata = ModelMetadata.load(metadata_path)
                models.append(metadata.to_dict())
            except Exception as e:
                print(f"Error loading metadata from {metadata_path}: {str(e)}")
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return models 