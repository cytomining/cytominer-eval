import numpy as np
import pandas as pd
import pandas.api.types as ptypes


def get_upper_matrix(df: pd.DataFrame) -> np.array:
    return np.triu(np.ones(df.shape), k=1).astype(bool)


def convert_pandas_dtypes(df: pd.DataFrame, col_fix: type = np.float64) -> pd.DataFrame:
    try:
        df = df.astype(col_fix)
    except ValueError:
        raise ValueError(
            "Columns cannot be converted to {col}; check input features".format(
                col=col_fix
            )
        )

    return df


def assert_pandas_dtypes(df: pd.DataFrame, col_fix: type = np.float64) -> pd.DataFrame:

    assert col_fix in [np.str, np.float64], "Only np.str and np.float64 are supported"

    df = convert_pandas_dtypes(df=df, col_fix=col_fix)

    assert_error = "Columns not successfully updated, is the dataframe consistent?"
    if col_fix == np.str:
        assert all([ptypes.is_string_dtype(df[x]) for x in df.columns]), assert_error

    if col_fix == np.float64:
        assert all([ptypes.is_numeric_dtype(df[x]) for x in df.columns]), assert_error

    return df
