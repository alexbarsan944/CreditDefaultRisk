import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import featuretools as ft
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier


# Function to clean feature names
def sanitize_feature_names(df):
    clean_names = {
        name: name.replace("{", "_")
        .replace("}", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace('"', "")
        for name in df.columns
    }
    # Rename the columns in the DataFrame
    df.rename(columns=clean_names, inplace=True)
    return df


# Function to display feature importances
def display_importances(feature_importance_df):
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance_df.sort_values(by="importance", ascending=False).head(
            50
        ),
    )
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.show()


def select_features_hist_gb(df, target_column, initial_features=None):
    y = df[target_column]
    X = df.drop(columns=[target_column])

    if initial_features is None:
        # Start with a default feature that is not an index or ID column
        initial_features = [
            col for col in X.columns if col not in ["index", "SK_ID_CURR"]
        ][0]

    print("Using initial features:", initial_features)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train initial model with the initial features using HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier()
    clf.fit(X_train[[initial_features]], y_train)
    baseline_preds = clf.predict_proba(X_test[[initial_features]])[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_preds)

    print(f"Initial AUC with features [{initial_features}]: {baseline_auc:.4f}")

    # Keep track of selected features
    selected_features = [initial_features]

    # Iteratively test each feature
    for feature in X.columns.difference(selected_features):
        test_features = selected_features + [feature]
        clf.fit(X_train[test_features], y_train)
        test_preds = clf.predict_proba(X_test[test_features])[:, 1]
        test_auc = roc_auc_score(y_test, test_preds)

        if test_auc > baseline_auc:
            selected_features.append(feature)
            baseline_auc = test_auc
            print(f"Adding feature '{feature}' improved AUC to {test_auc:.4f}")
        else:
            print(
                f"Feature '{feature}' did not improve AUC; current AUC: {baseline_auc:.4f}"
            )

    return selected_features


# Load and preprocess data
def load_and_preprocess_data(debug=False):
    # Load the dataset
    df = pd.read_csv("../data/all_cleaned_data.csv")

    if debug:
        # Use a smaller sample for debugging
        df = df.sample(n=min(10000, len(df)), random_state=42)
        print("Debug mode: Using a sample of {} rows".format(len(df)))

    # Apply preprocessing steps such as handling missing values
    df = df.dropna(subset=["TARGET"])  # Example preprocessing step
    return df


# Feature generation using Featuretools
def feature_generation(df, debug=False):
    es = ft.EntitySet(id="Data")
    es.entity_from_dataframe(
        entity_id="Data", dataframe=df, make_index=True, index="index"
    )
    feature_matrix, _ = ft.dfs(entityset=es, target_entity="Data", max_depth=2)
    if debug:
        print("Feature Matrix Generated:", feature_matrix.shape)
    return feature_matrix


# Function to train and evaluate model using LightGBM and k-fold cross-validation
def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    df = sanitize_feature_names(df)
    train_df = df[df["TARGET"].notnull()]
    test_df = df[df["TARGET"].isnull()]
    print(
        "Starting LightGBM. Train shape: {}, test shape: {}".format(
            train_df.shape, test_df.shape
        )
    )

    del df
    gc.collect()

    folds = (
        StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
        if stratified
        else KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    )
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [
        f
        for f in train_df.columns
        if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]

    for n_fold, (train_idx, valid_idx) in enumerate(
        folds.split(train_df[feats], train_df["TARGET"])
    ):
        train_x, train_y = (
            train_df[feats].iloc[train_idx],
            train_df["TARGET"].iloc[train_idx],
        )
        valid_x, valid_y = (
            train_df[feats].iloc[valid_idx],
            train_df["TARGET"].iloc[valid_idx],
        )
        clf = LGBMClassifier(
            n_jobs=-1,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbosity=-1,
        )

        clf.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric="auc",
            verbose=100,
            early_stopping_rounds=100,
        )
        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_
        )[:, 1]
        sub_preds += (
            clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1]
            / folds.n_splits
        )

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0
        )
        print(
            "Fold %2d AUC : %.6f"
            % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]))
        )
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print("Full AUC score %.6f" % roc_auc_score(train_df["TARGET"], oof_preds))
    if not debug:
        test_df["TARGET"] = sub_preds
        test_df[["SK_ID_CURR", "TARGET"]].to_csv("submission.csv", index=False)
    display_importances(feature_importance_df)
    return feature_importance_df


def main(debug=False):
    df = load_and_preprocess_data(debug)
    print("Columns in dataset:", df.columns)
    selected_features = select_features_hist_gb(df, "TARGET")
    print("Selected features:", selected_features)


if __name__ == "__main__":
    main(debug=True)
