import numpy as np
import pandas as pd

# Clearing up memory
import gc

# Featuretools for automated feature engineering
import featuretools as ft

# Suppress pandas warnings
import warnings

warnings.filterwarnings("ignore")


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
        "HOUSETYPE_MODE",
        "FONDKAPREMONT_MODE",
        "EMERGENCYSTATE_MODE",
    ]
    # Drop most flag document columns
    for doc_num in [2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21]:
        drop_list.append("FLAG_DOCUMENT_{}".format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df


print("Reading in data")


# Read in the full datasets
def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


print("-" * 80)
print("train")
app_train = import_data("../data/application_train.csv")

print("-" * 80)
print("test")
app_test = import_data("../data/application_test.csv")

print("-" * 80)
print("bureau_balance")
bureau_balance = import_data("../data/bureau_balance.csv")

print("-" * 80)
print("bureau")
bureau = import_data("../data/bureau.csv")

print("-" * 80)
print("credit_card_balance")
credit = import_data("../data/credit_card_balance.csv")

print("-" * 80)
print("installments_payments")
installments = import_data("../data/installments_payments.csv")

print("-" * 80)
print("pos_cash_balance")
cash = import_data("../data/POS_CASH_balance.csv")

print("-" * 80)
print("previous_application")
previous = import_data("../data/previous_application.csv")


app = app_train

grouped = bureau.groupby("SK_ID_CURR")
app["debt_credit_ratio_None"] = (
    grouped["AMT_CREDIT_SUM_DEBT"].sum() / grouped["AMT_CREDIT_SUM"].sum()
)
app["credit_annuity_ratio"] = app["AMT_CREDIT"] / app["AMT_ANNUITY"]
prev_sorted = previous.sort_values(by=["SK_ID_CURR", "DAYS_DECISION"])
app["prev_PRODUCT_COMBINATION"] = prev_sorted.groupby("SK_ID_CURR")[
    "PRODUCT_COMBINATION"
].last()
app["DAYS_CREDIT_mean"] = bureau.groupby("SK_ID_CURR")["DAYS_CREDIT"].mean()
app["credit_goods_price_ratio"] = app["AMT_CREDIT"] / app["AMT_GOODS_PRICE"]
active_loans = bureau[bureau["CREDIT_ACTIVE"] == "Active"]
app["last_active_DAYS_CREDIT"] = active_loans.groupby("SK_ID_CURR")["DAYS_CREDIT"].max()
app["credit_downpayment"] = app["AMT_GOODS_PRICE"] - app["AMT_CREDIT"]
app["AGE_INT"] = (app["DAYS_BIRTH"] / -365).astype(int)

installments["diff"] = installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]
filtered = installments[installments["DAYS_INSTALMENT"] > -1000]
grouped = (
    filtered.groupby(["SK_ID_PREV", "SK_ID_CURR"])["diff"]
    .mean()
    .groupby("SK_ID_CURR")
    .mean()
)

app["installment_payment_ratio_1000_mean_mean"] = grouped
max_installment = installments.groupby("SK_ID_CURR")["AMT_INSTALMENT"].max()
app["annuity_to_max_installment_ratio"] = app["AMT_ANNUITY"] / max_installment
app["EXT_SOURCES_MEAN"] = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(
    axis=1
)
app["EXT_SOURCES_MAX"] = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(
    axis=1
)

installments["diff"] = installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]
filtered = installments[installments["DAYS_INSTALMENT"] > -1000]
grouped = (
    filtered.groupby(["SK_ID_PREV", "SK_ID_CURR"])["diff"]
    .mean()
    .groupby("SK_ID_CURR")
    .mean()
)

app["installment_payment_ratio_1000_mean_mean"] = grouped
max_installment = installments.groupby("SK_ID_CURR")["AMT_INSTALMENT"].max()
app["annuity_to_max_installment_ratio"] = app["AMT_ANNUITY"] / max_installment


es = ft.EntitySet(id="clients")
# For dataframes with an existing unique index
es = es.add_dataframe(dataframe_name="app", dataframe=app, index="SK_ID_CURR")
es = es.add_dataframe(dataframe_name="bureau", dataframe=bureau, index="SK_ID_BUREAU")
es = es.add_dataframe(dataframe_name="previous", dataframe=previous, index="SK_ID_PREV")

# For dataframes that do not have a unique index, manually add an index column
bureau_balance["bureaubalance_index"] = range(1, len(bureau_balance) + 1)
cash["cash_index"] = range(1, len(cash) + 1)
installments["installments_index"] = range(1, len(installments) + 1)
credit["credit_index"] = range(1, len(credit) + 1)

# Now add these dataframes to the EntitySet
es = es.add_dataframe(
    dataframe_name="bureau_balance",
    dataframe=bureau_balance,
    index="bureaubalance_index",
)
es = es.add_dataframe(dataframe_name="cash", dataframe=cash, index="cash_index")
es = es.add_dataframe(
    dataframe_name="installments", dataframe=installments, index="installments_index"
)
es = es.add_dataframe(dataframe_name="credit", dataframe=credit, index="credit_index")

# Define relationships based on logical connections (foreign keys) between the dataframes
relationships = [
    ("app", "SK_ID_CURR", "bureau", "SK_ID_CURR"),
    ("bureau", "SK_ID_BUREAU", "bureau_balance", "SK_ID_BUREAU"),
    ("app", "SK_ID_CURR", "previous", "SK_ID_CURR"),
    ("previous", "SK_ID_PREV", "cash", "SK_ID_PREV"),
    ("previous", "SK_ID_PREV", "installments", "SK_ID_PREV"),
    ("previous", "SK_ID_PREV", "credit", "SK_ID_PREV"),
]

# Add relationships to the EntitySet
for parent_df, parent_col, child_df, child_col in relationships:
    es.add_relationship(
        parent_dataframe_name=parent_df,
        parent_column_name=parent_col,
        child_dataframe_name=child_df,
        child_column_name=child_col,
    )


print(es)

print("Clearing up memory")

gc.enable()
# Clear up memory
del bureau, bureau_balance, cash, credit, installments, previous
gc.collect()


app = drop_application_columns(app)

print("Deep Feature Synthesis in Progress")

# Default primitives from featuretools
default_agg_primitives = ["sum", "mean", "count", "max", "min", "std"]


# Now you can run dfs with the entity set you've created
feature_matrix, feature_names = ft.dfs(
    entityset=es,
    target_dataframe_name="app",
    agg_primitives=default_agg_primitives,
    max_depth=2,
    features_only=False,
    verbose=True,
)

# Reset the index to make SK_ID_CURR a column again
feature_matrix = feature_matrix.reset_index()

print("Saving features")
feature_matrix.to_csv("feature_matrix_final.csv", index=False)
