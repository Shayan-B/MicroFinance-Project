import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pyarrow.parquet as pq

import matplotlib.pyplot as plt
import seaborn as sns

#STATS
# from statsmodels.stats.outliers_influence import variance_inflation_factor

#ML
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
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))



def read_parquet_file(file_name: str):
    """Read parquet files based on provided name."""
    parquet_file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', f"{file_name}.parquet")
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
    df.loc[:,selected_cols] = df.loc[:,selected_cols].astype(target_type)    
    return df

def plot_car_histogram(df: pd.DataFrame, str_value: str):
    ax = df["OWN_CAR_AGE"].plot(kind="hist", bins=150)
    ax.set_title(f"Distribution of OWN_CAR_AGE {str_value} filling NA values")
    plt.show()
    df["OWN_CAR_AGE"].describe()