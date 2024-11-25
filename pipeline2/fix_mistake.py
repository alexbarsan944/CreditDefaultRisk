import pandas as pd

# Path to the original data and the feature matrix
original_data_path = "../data/all_cleaned_data.csv"
feature_matrix_path = "feature_engineering_data/feature_matrix.csv"

# Load the original data with only the 'SK_ID_CURR' and 'TARGET' columns
print("Loading the original data...")
original_data = pd.read_csv(original_data_path, usecols=['SK_ID_CURR', 'TARGET'])

# Load the feature matrix
print("Loading the feature matrix...")
feature_matrix = pd.read_csv(feature_matrix_path, index_col='SK_ID_CURR')

# Merge the 'TARGET' variable back into the feature matrix using the index
print("Merging the 'TARGET' variable back into the feature matrix...")
feature_matrix = feature_matrix.merge(original_data, on='SK_ID_CURR', how='left')

# Save the updated feature matrix back to CSV
print("Saving the updated feature matrix with 'TARGET' variable...")
feature_matrix.to_csv("feature_engineering_data/updated_feature_matrix.csv")

print("Updated feature matrix with 'TARGET' variable saved successfully.")
