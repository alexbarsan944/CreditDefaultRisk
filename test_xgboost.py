import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print(f"XGBoost version: {xgb.__version__}")

# Create a simple dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test 1: Using early_stopping_rounds as a parameter to the model constructor
print("\nTest 1: Using early_stopping_rounds in the model constructor")
try:
    model1 = xgb.XGBClassifier(random_state=42, early_stopping_rounds=10)
    model1.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    print("Success! Model trained with early_stopping_rounds in constructor.")
    print(f"Best iteration: {model1.best_iteration}")
except Exception as e:
    print(f"Error: {str(e)}")

# Test 2: Using early_stopping_rounds as a parameter to fit
print("\nTest 2: Using early_stopping_rounds in fit method")
try:
    model2 = xgb.XGBClassifier(random_state=42)
    model2.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=True)
    print("Success! Model trained with early_stopping_rounds in fit.")
    print(f"Best iteration: {model2.best_iteration}")
except Exception as e:
    print(f"Error: {str(e)}")

# Test 3: Using callbacks
print("\nTest 3: Using callbacks")
try:
    from xgboost.callback import EarlyStopping
    early_stopping = EarlyStopping(rounds=10, save_best=True)
    model3 = xgb.XGBClassifier(random_state=42)
    model3.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[early_stopping], verbose=True)
    print("Success! Model trained with callbacks.")
    print(f"Best iteration: {model3.best_iteration}")
except Exception as e:
    print(f"Error: {str(e)}")

# Test 4: Using callbacks in constructor
print("\nTest 4: Using callbacks in constructor")
try:
    from xgboost.callback import EarlyStopping
    early_stopping = EarlyStopping(rounds=10, save_best=True)
    model4 = xgb.XGBClassifier(random_state=42, callbacks=[early_stopping])
    model4.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    print("Success! Model trained with callbacks in constructor.")
    print(f"Best iteration: {model4.best_iteration}")
except Exception as e:
    print(f"Error: {str(e)}")

# Test 5: Basic fit without early stopping
print("\nTest 5: Basic fit without early stopping")
try:
    model5 = xgb.XGBClassifier(random_state=42)
    model5.fit(X_train, y_train)
    print("Success! Model trained without early stopping.")
except Exception as e:
    print(f"Error: {str(e)}") 