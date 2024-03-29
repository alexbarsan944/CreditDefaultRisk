import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency, ttest_ind
# automated feature engineering
import featuretools as ft

# Filter out pandas warnings
import warnings 
warnings.filterwarnings('ignore')

start = "\033[1m"  # Bold text
end = "\033[0;0m"  # Reset text
from IPython.core.interactiveshell import InteractiveShell
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
from lightgbm import LGBMClassifier
import lightgbm as lgb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import warnings


def clean_feature_names(dataframe):
    """
    Cleans the feature names in a DataFrame by replacing special characters with underscores.

    Parameters:
    - dataframe: pd.DataFrame, the dataframe to clean.

    Returns:
    - pd.DataFrame, the cleaned dataframe.
    """
    # Replace or remove special JSON characters
    clean_names = {name: name.replace('{', '_').replace('}', '_').replace(':', '_').replace(',', '_').replace('"', '') for name in dataframe.columns}
    # Rename the columns in the DataFrame
    dataframe.rename(columns=clean_names, inplace=True)
    return dataframe




