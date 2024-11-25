import numpy as np
import pandas as pd
import featuretools as ft
import sys
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
from collections import Counter
from featuretools.primitives.base import AggregationPrimitive
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Integer, Double
from contextlib import contextmanager
import time
import cudf.pandas

# Ignore pandas warnings and future warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action="ignore")

# Adding the directory to the path to ensure dependencies are found
sys.path.append("../")

def drop_application_columns(df):
    """Drop features based on permutation feature importance."""
    drop_list = [
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "HOUR_APPR_PROCESS_START",
        "FLAG_EMP_PHONE",
        "FLAG_MOBIL",
        "FLAG_CONT_MOBILE",
        "FLAG_EMAIL",
        "FLAG_PHONE",   
        "FLAG_OWN_REALTY",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_WORK_CITY",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "COMMONAREA_MODE",
        "NONLIVINGAREA_MODE",
        "ELEVATORS_MODE",
        "NONLIVINGAREA_AVG",
        "FLOORSMIN_MEDI",
        "LANDAREA_MODE",
        "NONLIVINGAREA_MEDI",
        "LIVINGAPARTMENTS_MODE",
        "FLOORSMIN_AVG",
        "LANDAREA_AVG",
        "FLOORSMIN_MODE",
        "LANDAREA_MEDI",
        "COMMONAREA_MEDI",
        "YEARS_BUILD_AVG",
        "COMMONAREA_AVG",
        "BASEMENTAREA_AVG",
        "BASEMENTAREA_MODE",
        "NONLIVINGAPARTMENTS_MEDI",
        "BASEMENTAREA_MEDI",
        "LIVINGAPARTMENTS_AVG",
        "ELEVATORS_AVG",
        "YEARS_BUILD_MEDI",
        "ENTRANCES_MODE",
        "NONLIVINGAPARTMENTS_MODE",
        "LIVINGAREA_MODE",
        "LIVINGAPARTMENTS_MEDI",
        "YEARS_BUILD_MODE",
        "YEARS_BEGINEXPLUATATION_AVG",
        "ELEVATORS_MEDI",
        "LIVINGAREA_MEDI",
        "YEARS_BEGINEXPLUATATION_MODE",
        "NONLIVINGAPARTMENTS_AVG",
    ]
    # Drop most flag document columns
    for doc_num in [2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21]:
        drop_list.append("FLAG_DOCUMENT_{}".format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    print(f"Dropped {len(drop_list)} features")
    return df



@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

    
def impute_data(df):
    df = drop_application_columns(df)
    for column in df.columns:
        if df[column].isna().mean() > 0.8:
            df.drop(column, axis=1, inplace=True)
        elif df[column].dtype == 'float64':
            df[column].fillna(df[column].median(), inplace=True)
        elif df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df


# Function to import data and optimize its memory usage
def import_data(file_path, chunk_size, num_rows=None):
    """Generator function to read data in chunks."""
    return pd.read_csv(file_path, parse_dates=True, keep_date_col=True, nrows=num_rows, chunksize=chunk_size)



class NormalizedModeCount(AggregationPrimitive):
    """Return the fraction of total observations that are the most common observation."""
    name = "normalized_mode_count"
    input_types = [ColumnSchema(logical_type=Categorical())]
    return_type = ColumnSchema(logical_type=Double)
    description_template = "the fraction of total observations that are the most common observation"

    def get_function(self):
        def normalized_mode_count(x: pd.Series) -> float:
            if x.mode().shape[0] == 0:
                return np.nan
            counts = dict(Counter(x))
            mode = x.mode().iloc[0]
            return counts[mode] / np.sum(list(counts.values()))
        return normalized_mode_count

class LongestRepetition(AggregationPrimitive):
    """Returns the item with most consecutive occurrences in x."""
    name = "longest_repetition"
    input_types = [ColumnSchema(logical_type=Categorical())]
    return_type = ColumnSchema(logical_type=Categorical)
    description_template = "the item with most consecutive occurrences in series"

    def get_function(self):
        def longest_repetition(x: pd.Series) -> any:
            x = x.dropna()
            if x.shape[0] < 1:
                return None
            longest_element = current_element = None
            longest_repeats = current_repeats = 0
            for element in x:
                if current_element == element:
                    current_repeats += 1
                else:
                    current_element = element
                    current_repeats = 1
                if current_repeats > longest_repeats:
                    longest_repeats = current_repeats
                    longest_element = current_element
            return longest_element
        return longest_repetition
    

def save_schema_to_txt(dataframe, file_path):
    # Convert Woodwork DataFrame to a regular DataFrame and get the dtypes
    schema_df = pd.DataFrame({
        'Column': dataframe.columns,
        'Physical Type': [str(dataframe.dtypes[col]) for col in dataframe.columns],
        'Logical Type': [str(dataframe.ww.logical_types[col]) for col in dataframe.columns],
        'Semantic Tags': [str(dataframe.ww.semantic_tags[col]) for col in dataframe.columns]
    })
    # Save the schema DataFrame to a CSV file
    schema_df.to_csv(file_path, index=False)


def downcast_dtypes(df):
    # Vectorized approach for downcasting data types to reduce memory usage
    # Handle float columns
    cudf.pandas.install()
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Handle integer columns
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        # Ensure no integer overflow during downcast
        c_min = df[col].min()
        c_max = df[col].max()
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)
    return df



@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Function to handle feature generation per chunk
def process_chunk(df_chunk, entity_set, target_dataframe_name, feature_matrix=None):
    # Drop or ignore columns explicitly
    columns_to_use = df_chunk.columns.difference(['index'])
    # Initialize Woodwork only on the columns you want to use
    print("Initializing Woodwork on the chunk...")
    df_chunk = df_chunk[columns_to_use]
    df_chunk.ww.init(
        index="SK_ID_CURR",
        name="app"
    )
    entity_set.add_dataframe(dataframe_name="app", dataframe=df_chunk)
    print("Generating features...")
    # Calculate DFS for the chunk
    fm_chunk, feature_defs = ft.dfs(
        entityset=entity_set,
        target_dataframe_name='app',
        agg_primitives=['mean', 'max', 'min', 'trend', 'mode', 'count', 'sum', 'percent_true'],
        trans_primitives=['diff', 'cum_sum', 'cum_mean', 'percentile'],
        where_primitives=['mean', 'sum', 'count'],
        max_depth=2,
        features_only=False,
        verbose=True,
    )
    # Downcast and concatenate
    # print("Downcasting and concatenating the chunk...")
    # with timer("Downcasted"):
    #     fm_chunk = downcast_dtypes(fm_chunk)

    if feature_matrix is not None:
        feature_matrix = pd.concat([feature_matrix, fm_chunk])
    else:
        feature_matrix = fm_chunk
    
    return feature_matrix, feature_defs


CHUNK_SIZE = 100000
def main(debug=False):
    file_path = "../data/all_cleaned_data.csv"
    es = ft.EntitySet(id="client_data")
    num_rows = 1000 if debug else None

    feature_matrix = None
    chunks = import_data(file_path, CHUNK_SIZE, num_rows=num_rows)
    
    for df_chunk in chunks:
        df_chunk = impute_data(df_chunk)  # Ensure this function handles a dataframe
        feature_matrix, feature_defs = process_chunk(df_chunk, es, "app", feature_matrix)
    
    feature_matrix['EXT_SOURCES_MEAN'] = feature_matrix[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    
    print("Downcasting final matrix...")
    cudf.pandas.install()
    with timer("Downcasted final matrix"):
        feature_matrix = downcast_dtypes(feature_matrix)

    # Save results
    print("Saving results to disk...")
    feature_matrix.to_csv("feature_engineering_data/feature_matrix.csv")
    with open("feature_engineering_data/feature_defs.txt", "w") as f:
        f.write(str(feature_defs))

if __name__ == "__main__":
    with timer("Running full pipeline"):
        main(debug=False)  # Set True to enable debug mode
