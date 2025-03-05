import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add the src directory to the path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import our model
from credit_risk.models.estimator import CreditRiskModel

print("Testing CreditRiskModel with XGBoost and LightGBM")

# Create a simple dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y = pd.Series(y, name='target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test XGBoost
print("\nTesting XGBoost model:")
try:
    xgb_model = CreditRiskModel(model_type="xgboost", random_state=42)
    xgb_model.fit(X_train, y_train)
    print("XGBoost model trained successfully!")
    
    # Test cross-validation
    cv_results = xgb_model.cross_validate(X_train, y_train, n_folds=3)
    print(f"XGBoost cross-validation AUC: {cv_results.get('mean_auc', 0.0):.4f}")
    
    # Test evaluation
    metrics = xgb_model.evaluate(X_test, y_test)
    print(f"XGBoost test AUC: {metrics.get('auc', 0.0):.4f}")
except Exception as e:
    print(f"Error with XGBoost: {str(e)}")

# Test LightGBM
print("\nTesting LightGBM model:")
try:
    lgb_model = CreditRiskModel(model_type="lightgbm", random_state=42)
    lgb_model.fit(X_train, y_train)
    print("LightGBM model trained successfully!")
    
    # Test cross-validation
    cv_results = lgb_model.cross_validate(X_train, y_train, n_folds=3)
    print(f"LightGBM cross-validation AUC: {cv_results.get('mean_auc', 0.0):.4f}")
    
    # Test evaluation
    metrics = lgb_model.evaluate(X_test, y_test)
    print(f"LightGBM test AUC: {metrics.get('auc', 0.0):.4f}")
except Exception as e:
    print(f"Error with LightGBM: {str(e)}")

print("\nTests completed!") 