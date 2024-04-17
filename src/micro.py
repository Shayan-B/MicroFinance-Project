import os
import re

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import pyarrow.parquet as pq

import matplotlib.pyplot as plt
import seaborn as sns

# STATS
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# ML
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer

from scipy import stats


os.chdir("../")


def read_parquet_file(file_name: str):
    """Read parquet files based on provided name."""
    parquet_file_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", f"{file_name}.parquet"
    )
    # Read the Parquet file into a PyArrow Table
    table = pq.read_table(parquet_file_path)

    # Convert the PyArrow Table to a Pandas DataFrame
    df = table.to_pandas()

    return df


def ChangeType(df: pd.DataFrame, org_type: str, target_type: str) -> pd.DataFrame:
    """function to change the type of columns>

    Args:
        df:
            DataFrame to change the types
        org_type:
            The initial type we are intended to change.
        target_type:
            The target type to change.

    Returns:
        Pandas DataFrame with changed types.
    """
    selected_cols = df.select_dtypes(org_type).columns
    df.loc[:, selected_cols] = df.loc[:, selected_cols].astype(target_type)
    return df


def plot_car_histogram(df: pd.DataFrame, str_value: str):
    """Plot the histogram of 'OWN_CAR_AGE'."""
    ax = df["OWN_CAR_AGE"].plot(kind="hist", bins=150)
    ax.set_title(f"Distribution of OWN_CAR_AGE {str_value} filling NA values")
    plt.show()
    df["OWN_CAR_AGE"].describe()

    return


def select_numerical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Select the numerical values from the given DataFrame.

    Args:
        df:
            The initial DataFrame to select numerical values.

    Returns:
        The DataFrame with only the Numerical columns.
    """
    numerical_dtypes = ["int32", "float32", "int64", "float64"]
    numerical_df = df.select_dtypes(numerical_dtypes)
    numerical_columns = numerical_df.columns

    print("Numerical columns are: \n", numerical_columns.tolist(), "\n")

    return numerical_df


def compute_triu_corr_df(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute the Upper triangle of the correlation DataFrame.

    Args:
        corr_matrix:
            correlation DataFrame.

    Returns:
        A DataFrame containing the upper triangle values of correlation matrix,
        the lower triangle matrix is filled with NA values.
    """
    # Select the upper triangle, not incuding the ones
    upper_triangle_array = np.triu((corr_matrix), k=1)

    # Build the DataFrame
    filtered_corr_df = pd.DataFrame(
        upper_triangle_array,
        index=corr_matrix.index,
        columns=corr_matrix.columns,
    )

    return filtered_corr_df


def compute_entire_correlation(df: pd.DataFrame, threshold: float = 0.7):
    """Find the columns which have high correlation with each other.

    Args:
        df:
            The initila DataFrame containing the variable names as columns and the related
            values for each of them. We wiil select the numerical columns from the gicen DataFrame.
        threshold:
            The threshold value for selecting the high correlation variables.

    """
    # Select Numerical columns
    numerical_df = select_numerical_variables(df)

    # Find the Correlation matrix
    correlation_matrix = numerical_df.corr()

    # use the Upper triangle matrix and create a DataFrame to find the names of the high correlated columns
    filtered_correlation_df = compute_triu_corr_df(correlation_matrix)
    total_list = []

    # Find the Related high correlation values based on threshold value for each column
    for name, row in filtered_correlation_df.iterrows():
        name_list = []
        name_list.append(name)
        col_names = row[(row >= threshold)].index.tolist()
        if len(col_names) != 0:
            name_list.extend(col_names)
            total_list.append(name_list)

    total_correlation_names = pd.DataFrame(total_list)
    print(
        "The final Variables which has higher than threshold correlation are as table below: \n",
        total_correlation_names,
    )

    return


def compute_corr_hist_plot(df: pd.DataFrame, col_list: list):
    """Plot pair wise and show the correlation matrix for specified columns.

    Args:
        df:
            The main DataFrame.
        col_list:
            List of columns to show the pair plot and compute correlation.
    """
    sns.pairplot(df[col_list])
    plt.show()
    print(f"Correlation between {len(col_list)} variables: \n", df[col_list].corr())

    return


def find_outliers(
    data_df: pd.DataFrame, col_name: str, outlier_type: str = None
) -> pd.DataFrame:
    """Find the outliers based on InterQuartile Range."""
    # Finding Quantiles
    Q1 = data_df[col_name].quantile(0.25)
    Q3 = data_df[col_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define the boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    if outlier_type == "upper":
        outliers = data_df[(data_df[col_name] > upper_bound)]
    elif outlier_type == "lower":
        outliers = data_df[(data_df[col_name] < lower_bound)]
    else:
        outliers = data_df[
            (data_df[col_name] < lower_bound) | (data_df[col_name] > upper_bound)
        ]

    return outliers, Q1, Q3


def compute_transform_zscore(data_df: pd.DataFrame, col_name: str):
    data_df[f"{col_name}_LOG"] = np.log(data_df[f"{col_name}"])
    data_df[f"{col_name}_ZS"] = stats.zscore(data_df[f"{col_name}_LOG"])

    # Define COnditions for zscore filtering
    zscore_condition_upper = data_df[f"{col_name}_ZS"] <= 3
    zscore_condition_lower = data_df[f"{col_name}_ZS"] >= -3

    # Find the relative Log values to zscores
    zscore_thresh_upper = (
        data_df.loc[zscore_condition_upper, :]
        .sort_values(by=[f"{col_name}_ZS"], ascending=False)
        .iloc[1, :][f"{col_name}_LOG"]
    )
    zscore_thresh_lower = (
        data_df.loc[zscore_condition_lower, :]
        .sort_values(by=[f"{col_name}_ZS"], ascending=True)
        .iloc[1, :][f"{col_name}_LOG"]
    )

    # Define condition to replace extreme values
    replace_condition_upper = data_df[f"{col_name}_LOG"] > zscore_thresh_upper
    replace_condition_lower = data_df[f"{col_name}_LOG"] < zscore_thresh_lower

    # Replace the values
    data_df.loc[replace_condition_upper, f"{col_name}_LOG"] = zscore_thresh_upper
    data_df.loc[replace_condition_lower, f"{col_name}_LOG"] = zscore_thresh_lower

    return data_df, zscore_thresh_lower, zscore_thresh_upper


def substitute_yn_values(data_df: pd.DataFrame, col_names: list) -> pd.DataFrame:
    """Substitue the specified column Y, N values with 1,0

    Args:
        data_df:
            Main DataFrame containing the 'Y' and 'N' values.
        col_names:
            The name of columns which has 'Y', 'N' values.

    Returns:
        Main DataFrame with 'Y', 'N' values sibstitute for 1 and 0.
    """
    for col in col_names:
        data_df[col] = data_df[col].replace({"Y": 1, "N": 0})
    return data_df


def clean_organization_col(data_df: pd.DataFrame):
    data_df["ORGANIZATION_TYPE"] = (
        data_df["ORGANIZATION_TYPE"]
        .apply(
            lambda x: re.sub(r"(type|Type)\s+\d+", "", x, flags=re.IGNORECASE)
            .replace(":", "")
            .strip()
        )
        .value_counts()
    )
    return data_df


def plot_hist_var_target(data_df: pd.DataFrame, col_name: str):
    """Plot the Histogram of the column with distinct charts for TARGET values."""
    sns.histplot(data_df, x=col_name, hue="TARGET", kde=True).set(
        title=f"Chart of {col_name} based on TARGET values"
    )
    plt.show()
    return

def plot_box_var(data_df: pd.DataFrame, col_name: str, log_scale: bool = False):
    sns.boxplot(
        data_df,
        x=col_name,
        log_scale=log_scale,
        notch=True,
        flierprops={"marker": "x"},
        width=0.3,
    ).set(
        title=f"BoxPlot for distribution of {col_name} values and log_scale: {log_scale}"
    )
    plt.show()

    return

    return
