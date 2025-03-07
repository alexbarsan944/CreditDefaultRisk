"""
Data profiling utilities for monitoring data quality throughout the ML pipeline.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging
from typing import Dict, List, Optional, Union, Tuple
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProfiler:
    """
    Tracks and visualizes data statistics throughout the ML pipeline.
    Provides checkpoints at different stages to monitor data quality.
    """
    
    def __init__(self, checkpoint_dir=None):
        """
        Initialize the data profiler.
        
        Parameters:
        -----------
        checkpoint_dir : str, optional
            Directory to save checkpoint data. If None, checkpoints won't be saved to disk.
        """
        self.checkpoints = {}
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # Initialize a timestamp for this profiling session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_checkpoint(self, 
                         stage: str, 
                         X: pd.DataFrame, 
                         y: Optional[pd.Series] = None,
                         description: str = "") -> Dict:
        """
        Create a checkpoint to track data at a specific pipeline stage.
        
        Parameters:
        -----------
        stage : str
            Name of the pipeline stage (e.g., "raw_data", "feature_engineering", "feature_selection")
        X : pd.DataFrame
            Feature data to profile
        y : pd.Series, optional
            Target variable data if available
        description : str, optional
            Additional notes about this checkpoint
            
        Returns:
        --------
        Dict
            Checkpoint statistics
        """
        timestamp = time.time()
        
        # Basic data stats
        stats = {
            "timestamp": timestamp,
            "stage": stage,
            "description": description,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "memory_usage_mb": X.memory_usage(deep=True).sum() / (1024 * 1024),
            "features": {
                "numeric": list(X.select_dtypes(include=np.number).columns),
                "categorical": list(X.select_dtypes(exclude=np.number).columns),
            },
            "missing_values": {
                "total": X.isna().sum().sum(),
                "percentage": (X.isna().sum().sum() / (X.shape[0] * X.shape[1])) * 100,
                "by_column": X.isna().sum().to_dict()
            },
            "column_stats": {}
        }
        
        # Calculate statistics for numeric columns
        for col in stats["features"]["numeric"]:
            stats["column_stats"][col] = {
                "mean": X[col].mean() if not X[col].isna().all() else None,
                "median": X[col].median() if not X[col].isna().all() else None,
                "std": X[col].std() if not X[col].isna().all() else None,
                "min": X[col].min() if not X[col].isna().all() else None,
                "max": X[col].max() if not X[col].isna().all() else None,
                "missing": X[col].isna().sum(),
                "missing_pct": (X[col].isna().sum() / len(X)) * 100,
                "has_infinity": np.isinf(X[col]).any() if X[col].dtype.kind in 'fi' else False,
                "infinity_count": np.isinf(X[col]).sum() if X[col].dtype.kind in 'fi' else 0
            }
            
        # Calculate statistics for categorical columns
        for col in stats["features"]["categorical"]:
            stats["column_stats"][col] = {
                "unique_values": X[col].nunique(),
                "missing": X[col].isna().sum(),
                "missing_pct": (X[col].isna().sum() / len(X)) * 100,
                "most_common": X[col].value_counts().nlargest(1).to_dict() if not X[col].isna().all() else None
            }
            
        # Add target information if available
        if y is not None:
            stats["target"] = {
                "name": y.name,
                "missing": y.isna().sum(),
                "missing_pct": (y.isna().sum() / len(y)) * 100
            }
            
            # Add target distribution for classification
            if y.nunique() <= 10:  # Classification
                stats["target"]["distribution"] = y.value_counts().to_dict()
                
        # Store checkpoint
        self.checkpoints[stage] = stats
        
        # Save checkpoint to disk if directory is provided
        if self.checkpoint_dir:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"checkpoint_{self.session_id}_{stage}.json"
            )
            try:
                with open(checkpoint_path, 'w') as f:
                    # Convert any non-serializable objects (like numpy types) to Python types
                    json_stats = self._make_json_serializable(stats)
                    json.dump(json_stats, f, indent=2)
                logger.info(f"Saved checkpoint for stage '{stage}' to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {str(e)}")
        
        return stats
    
    def compare_checkpoints(self, stage1: str, stage2: str) -> Dict:
        """
        Compare statistics between two checkpoints.
        
        Parameters:
        -----------
        stage1 : str
            First checkpoint stage name
        stage2 : str
            Second checkpoint stage name
            
        Returns:
        --------
        Dict
            Comparison statistics
        """
        if stage1 not in self.checkpoints or stage2 not in self.checkpoints:
            missing = []
            if stage1 not in self.checkpoints:
                missing.append(stage1)
            if stage2 not in self.checkpoints:
                missing.append(stage2)
            raise ValueError(f"Missing checkpoints: {', '.join(missing)}")
        
        cp1 = self.checkpoints[stage1]
        cp2 = self.checkpoints[stage2]
        
        comparison = {
            "stages": [stage1, stage2],
            "sample_count_change": int(cp2["n_samples"] - cp1["n_samples"]),
            "feature_count_change": int(cp2["n_features"] - cp1["n_features"]),
            "missing_values_change": int(cp2["missing_values"]["total"] - cp1["missing_values"]["total"]),
            "columns_added": list(set(cp2["features"]["numeric"] + cp2["features"]["categorical"]) - 
                               set(cp1["features"]["numeric"] + cp1["features"]["categorical"])),
            "columns_removed": list(set(cp1["features"]["numeric"] + cp1["features"]["categorical"]) - 
                                set(cp2["features"]["numeric"] + cp2["features"]["categorical"])),
            "column_stats_changes": {}
        }
        
        # Compare common columns
        common_columns = set(cp1["column_stats"].keys()) & set(cp2["column_stats"].keys())
        for col in common_columns:
            # For numeric columns, compare statistical properties
            if col in cp1["features"]["numeric"] and col in cp2["features"]["numeric"]:
                mean_change = None
                std_change = None
                
                # Calculate mean change if both values are not None
                if cp1["column_stats"][col]["mean"] is not None and cp2["column_stats"][col]["mean"] is not None:
                    mean_change = float(cp2["column_stats"][col]["mean"] - cp1["column_stats"][col]["mean"])
                
                # Calculate std change if both values are not None  
                if cp1["column_stats"][col]["std"] is not None and cp2["column_stats"][col]["std"] is not None:
                    std_change = float(cp2["column_stats"][col]["std"] - cp1["column_stats"][col]["std"])
                
                comparison["column_stats_changes"][col] = {
                    "mean_change": mean_change,
                    "std_change": std_change,
                    "missing_change": int(cp2["column_stats"][col]["missing"] - cp1["column_stats"][col]["missing"])
                }
            # For categorical columns, compare number of unique values
            elif col in cp1["features"]["categorical"] and col in cp2["features"]["categorical"]:
                comparison["column_stats_changes"][col] = {
                    "unique_values_change": int(cp2["column_stats"][col]["unique_values"] - cp1["column_stats"][col]["unique_values"]),
                    "missing_change": int(cp2["column_stats"][col]["missing"] - cp1["column_stats"][col]["missing"])
                }
        
        return comparison
    
    def get_all_checkpoints(self) -> Dict:
        """
        Get all checkpoints created so far.
        
        Returns:
        --------
        Dict
            All checkpoints
        """
        return self.checkpoints
    
    def plot_data_overview(self, stage: str) -> go.Figure:
        """
        Create a visual overview of the data at a specific checkpoint.
        
        Parameters:
        -----------
        stage : str
            The stage to visualize
            
        Returns:
        --------
        go.Figure
            Plotly figure with data overview
        """
        if stage not in self.checkpoints:
            raise ValueError(f"Checkpoint for stage '{stage}' not found")
        
        cp = self.checkpoints[stage]
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Data Composition", 
                "Missing Values by Feature", 
                "Numeric Features Distribution", 
                "Feature Type Breakdown"
            ),
            specs=[
                [{"type": "domain"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "domain"}]
            ]
        )
        
        # Data composition pie chart (numeric vs categorical)
        fig.add_trace(
            go.Pie(
                labels=["Numeric", "Categorical"],
                values=[len(cp["features"]["numeric"]), len(cp["features"]["categorical"])],
                name="Data Composition"
            ),
            row=1, col=1
        )
        
        # Missing values by feature (top 10 with most missing)
        missing_vals = pd.Series(cp["missing_values"]["by_column"]).sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=missing_vals.index, 
                y=missing_vals.values,
                name="Missing Values"
            ),
            row=1, col=2
        )
        
        # Numeric features distribution (box plot)
        # Only plot for a representative sample of numeric columns (max 10)
        numeric_cols = cp["features"]["numeric"][:10] if len(cp["features"]["numeric"]) > 10 else cp["features"]["numeric"]
        
        for col in numeric_cols:
            stats = cp["column_stats"][col]
            fig.add_trace(
                go.Box(
                    name=col,
                    y=[stats["min"], stats["max"]],  # Using min/max as placeholders
                    boxpoints=False,
                    boxmean="sd",  # Show mean and standard deviation
                    q1=[stats["mean"] - stats["std"]] if stats["std"] is not None else [stats["mean"]],
                    median=[stats["median"]],
                    q3=[stats["mean"] + stats["std"]] if stats["std"] is not None else [stats["mean"]],
                ),
                row=2, col=1
            )
            
        # Feature type breakdown (if there are categorical features with different cardinalities)
        cat_grouping = {"Low": 0, "Medium": 0, "High": 0}
        for col in cp["features"]["categorical"]:
            uniques = cp["column_stats"][col]["unique_values"]
            if uniques <= 5:
                cat_grouping["Low"] += 1
            elif uniques <= 20:
                cat_grouping["Medium"] += 1
            else:
                cat_grouping["High"] += 1
        
        fig.add_trace(
            go.Pie(
                labels=list(cat_grouping.keys()),
                values=list(cat_grouping.values()),
                name="Categorical Feature Cardinality"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Data Profile: {stage} (Samples: {cp['n_samples']}, Features: {cp['n_features']})",
            height=700,
            showlegend=False
        )
        
        return fig
    
    def plot_feature_distribution(self, stage: str, feature: str) -> go.Figure:
        """
        Plot distribution of a specific feature.
        
        Parameters:
        -----------
        stage : str
            The stage to visualize
        feature : str
            Feature name to visualize
            
        Returns:
        --------
        go.Figure
            Plotly figure with feature distribution
        """
        if stage not in self.checkpoints:
            raise ValueError(f"Checkpoint for stage '{stage}' not found")
        
        cp = self.checkpoints[stage]
        
        if feature not in cp["column_stats"]:
            raise ValueError(f"Feature '{feature}' not found in checkpoint")
        
        # Determine feature type
        is_numeric = feature in cp["features"]["numeric"]
        
        fig = go.Figure()
        
        if is_numeric:
            # For numeric features, show histogram with statistics overlay
            stats = cp["column_stats"][feature]
            
            # Create a theoretical normal distribution based on mean and std
            if stats["mean"] is not None and stats["std"] is not None and stats["std"] > 0:
                x = np.linspace(stats["mean"] - 3*stats["std"], stats["mean"] + 3*stats["std"], 100)
                y = (1/(stats["std"] * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-stats["mean"])/stats["std"])**2)
                
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red', dash='dash')
                    )
                )
            
            # Add vertical lines for statistics
            for name, value in {
                "Mean": stats["mean"],
                "Median": stats["median"],
                "Min": stats["min"],
                "Max": stats["max"]
            }.items():
                if value is not None:
                    fig.add_vline(
                        x=value,
                        line_width=1,
                        line_dash="dash",
                        annotation_text=name,
                        annotation_position="top right"
                    )
            
            fig.update_layout(
                title=f"Distribution of {feature} (Numeric)",
                xaxis_title=feature,
                yaxis_title="Density",
                showlegend=True
            )
        else:
            # For categorical features, show bar chart of counts
            stats = cp["column_stats"][feature]
            
            fig.update_layout(
                title=f"Distribution of {feature} (Categorical, {stats['unique_values']} unique values)",
                xaxis_title=feature,
                yaxis_title="Count",
                showlegend=False
            )
        
        return fig
    
    def plot_missing_values(self, stage: str) -> go.Figure:
        """
        Plot missing values heatmap for all features.
        
        Parameters:
        -----------
        stage : str
            The stage to visualize
            
        Returns:
        --------
        go.Figure
            Plotly figure with missing values heatmap
        """
        if stage not in self.checkpoints:
            raise ValueError(f"Checkpoint for stage '{stage}' not found")
        
        cp = self.checkpoints[stage]
        
        # Get missing values for all columns
        missing_vals = pd.Series(cp["missing_values"]["by_column"]).sort_values(ascending=False)
        missing_pct = (missing_vals / cp["n_samples"]) * 100
        
        # Create heatmap-style visualization
        fig = go.Figure(data=go.Bar(
            x=missing_pct.index,
            y=missing_pct.values,
            marker_color=missing_pct.values,
            marker_colorscale="Viridis",
            text=missing_vals.values,
            textposition="auto"
        ))
        
        fig.update_layout(
            title=f"Missing Values by Feature ({stage})",
            xaxis_title="Feature",
            yaxis_title="Missing Percentage (%)",
            yaxis=dict(range=[0, 100]),
            height=600,
            margin=dict(l=20, r=20, t=60, b=150),
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_pipeline_comparison(self, metrics: List[str] = ["n_samples", "n_features", "missing_values"]) -> go.Figure:
        """
        Plot metrics across all pipeline stages for comparison.
        
        Parameters:
        -----------
        metrics : List[str], optional
            Metrics to compare across stages
            
        Returns:
        --------
        go.Figure
            Plotly figure with pipeline comparison
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints available for comparison")
        
        stages = list(self.checkpoints.keys())
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for metric in metrics:
            values = []
            for stage in stages:
                if metric == "n_samples":
                    values.append(self.checkpoints[stage]["n_samples"])
                elif metric == "n_features":
                    values.append(self.checkpoints[stage]["n_features"])
                elif metric == "missing_values":
                    values.append(self.checkpoints[stage]["missing_values"]["total"])
            
            # Plot on primary or secondary y-axis based on metric
            secondary_y = metric == "missing_values"
            
            fig.add_trace(
                go.Scatter(
                    x=stages,
                    y=values,
                    mode="lines+markers",
                    name=metric
                ),
                secondary_y=secondary_y
            )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Missing Values", secondary_y=True)
        
        fig.update_layout(
            title_text="Data Metrics Across Pipeline Stages",
            xaxis_title="Pipeline Stage",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _make_json_serializable(self, obj):
        """Convert non-serializable objects to serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        # Convert numpy integer types to Python int
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        # Convert numpy float types to Python float
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj) if not np.isnan(obj) and not np.isinf(obj) else None
        # Convert numpy bool types to Python bool
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        # Handle numpy arrays by converting to list
        elif isinstance(obj, (np.ndarray,)):
            return self._make_json_serializable(obj.tolist())
        # Handle pandas Series and DataFrames
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return self._make_json_serializable(obj.to_dict())
        # Handle NaN values
        elif obj is pd.NA or obj is np.nan:
            return None
        else:
            return obj 