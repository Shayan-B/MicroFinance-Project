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

import os

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
        Pandas DataFrame with change types.
    """
    selected_cols = df.select_dtypes(org_type).columns
    df.loc[:, selected_cols] = df.loc[:, selected_cols].astype(target_type)
    return df


def plot_car_histogram(df: pd.DataFrame, str_value: str):
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

    print("Numerical columns are:", numerical_columns.tolist())

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
    print("The final Variables which has higher than threshold correlation are as table below: \n", total_correlation_names)
    
    return