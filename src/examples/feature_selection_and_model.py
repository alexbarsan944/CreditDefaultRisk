#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating feature selection and model training for credit risk prediction.

This script shows how to:
1. Load and preprocess data
2. Perform feature selection using the FeatureSelector
3. Train and evaluate different model types using the CreditRiskModel
4. Cross-validate the models and visualize results
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse

# Add the parent directory to sys.path to import the project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from credit_risk.features.selection import FeatureSelector
from credit_risk.models import CreditRiskModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Feature selection and model training example")
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="data/processed/feature_matrix.csv",
        help="Path to the feature matrix CSV file"
    )
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["xgboost", "lightgbm", "catboost"], 
        default="lightgbm",
        help="Type of model to train"
    )
    parser.add_argument(
        "--n-runs", 
        type=int, 
        default=20,
        help="Number of runs for null importance feature selection"
    )
    parser.add_argument(
        "--n-folds", 
        type=int, 
        default=5,
        help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--split-score-threshold", 
        type=float, 
        default=0.0,
        help="Threshold for feature selection based on split score"
    )
    parser.add_argument(
        "--gain-score-threshold", 
        type=float, 
        default=0.0,
        help="Threshold for feature selection based on gain score"
    )
    parser.add_argument(
        "--correlation-threshold", 
        type=float, 
        default=0.95,
        help="Threshold for removing highly correlated features"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models",
        help="Directory to save model and plots"
    )
    return parser.parse_args()

def main():
    """Main function for the example script."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data_file}")
    try:
        data = pd.read_csv(args.data_file)
        logger.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Check if TARGET column exists
    if "TARGET" not in data.columns:
        logger.error("TARGET column not found in the data")
        return
    
    # Split features and target
    X = data.drop(columns=["TARGET"])
    y = data["TARGET"]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )
    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Feature selection
    logger.info("Starting feature selection...")
    feature_selector = FeatureSelector(random_state=args.random_state)
    
    # Get feature scores
    logger.info(f"Calculating feature scores with {args.n_runs} null importance runs")
    feature_scores = feature_selector.get_feature_scores(X_train, y_train, n_runs=args.n_runs)
    
    # Plot top and bottom features by score
    plt.figure(figsize=(20, 12))
    
    # Top features by split score
    plt.subplot(2, 2, 1)
    top_split = feature_scores.sort_values("split_score", ascending=False).head(20)
    plt.barh(top_split["feature"], top_split["split_score"], color="darkblue")
    plt.title("Top Features by Split Score", fontweight="bold")
    plt.xlabel("Split Score")
    
    # Top features by gain score
    plt.subplot(2, 2, 2)
    top_gain = feature_scores.sort_values("gain_score", ascending=False).head(20)
    plt.barh(top_gain["feature"], top_gain["gain_score"], color="darkblue")
    plt.title("Top Features by Gain Score", fontweight="bold")
    plt.xlabel("Gain Score")
    
    # Bottom features by split score
    plt.subplot(2, 2, 3)
    bottom_split = feature_scores.sort_values("split_score", ascending=True).head(20)
    plt.barh(bottom_split["feature"], bottom_split["split_score"], color="darkred")
    plt.title("Bottom Features by Split Score", fontweight="bold")
    plt.xlabel("Split Score")
    
    # Bottom features by gain score
    plt.subplot(2, 2, 4)
    bottom_gain = feature_scores.sort_values("gain_score", ascending=True).head(20)
    plt.barh(bottom_gain["feature"], bottom_gain["gain_score"], color="darkred")
    plt.title("Bottom Features by Gain Score", fontweight="bold")
    plt.xlabel("Gain Score")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "feature_scores.png"))
    logger.info(f"Feature scores plot saved to {os.path.join(args.output_dir, 'feature_scores.png')}")
    
    # Apply feature selection
    logger.info("Applying feature selection")
    X_train_selected = feature_selector.fit_transform(
        X_train, 
        y_train, 
        n_runs=args.n_runs, 
        split_score_threshold=args.split_score_threshold,
        gain_score_threshold=args.gain_score_threshold,
        correlation_threshold=args.correlation_threshold
    )
    
    X_test_selected = feature_selector.transform(X_test)
    
    logger.info(f"Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]}")
    
    # Save list of selected features
    selected_features = pd.DataFrame({"feature": feature_selector.get_useful_features()})
    selected_features.to_csv(os.path.join(args.output_dir, "selected_features.csv"), index=False)
    logger.info(f"Selected features saved to {os.path.join(args.output_dir, 'selected_features.csv')}")
    
    # Train and evaluate model
    logger.info(f"Training {args.model_type.upper()} model")
    model = CreditRiskModel(model_type=args.model_type, random_state=args.random_state)
    
    # Cross-validate
    logger.info(f"Performing {args.n_folds}-fold cross-validation")
    cv_results = model.cross_validate(
        X_train_selected, y_train, n_folds=args.n_folds, stratified=True
    )
    
    # Plot cross-validation AUC scores
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, args.n_folds + 1), cv_results["fold_metrics"], color="royalblue")
    plt.axhline(y=cv_results["mean_auc"], color="r", linestyle="-", label=f"Mean AUC: {cv_results['mean_auc']:.4f}")
    plt.axhline(y=cv_results["mean_auc"] + cv_results["std_auc"], color="black", linestyle="--")
    plt.axhline(y=cv_results["mean_auc"] - cv_results["std_auc"], color="black", linestyle="--")
    plt.xlabel("Fold")
    plt.ylabel("AUC Score")
    plt.title(f"{args.model_type.upper()} Cross-Validation AUC Scores")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f"{args.model_type}_cv_scores.png"))
    logger.info(f"CV scores plot saved to {os.path.join(args.output_dir, f'{args.model_type}_cv_scores.png')}")
    
    # Train the final model on the entire training set
    logger.info("Training final model on entire training set")
    model.fit(X_train_selected, y_train)
    
    # Evaluate on test set
    logger.info("Evaluating model on test set")
    test_results = model.evaluate(X_test_selected, y_test)
    logger.info(f"Test AUC: {test_results['auc']:.4f}, Accuracy: {test_results['accuracy']:.4f}")
    
    # Plot feature importance
    feature_importance_plot = model.plot_feature_importance(top_n=30)
    feature_importance_plot.savefig(os.path.join(args.output_dir, f"{args.model_type}_feature_importance.png"))
    logger.info(f"Feature importance plot saved to {os.path.join(args.output_dir, f'{args.model_type}_feature_importance.png')}")
    
    # Plot confusion matrix
    confusion_matrix_plot = model.plot_confusion_matrix(X_test_selected, y_test)
    confusion_matrix_plot.savefig(os.path.join(args.output_dir, f"{args.model_type}_confusion_matrix.png"))
    logger.info(f"Confusion matrix plot saved to {os.path.join(args.output_dir, f'{args.model_type}_confusion_matrix.png')}")
    
    # Save the model
    model_path = os.path.join(args.output_dir, f"{args.model_type}_model.joblib")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Example completed successfully")

if __name__ == "__main__":
    main() 