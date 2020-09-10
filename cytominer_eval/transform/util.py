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


def assert_melt(df: pd.DataFrame, eval_metric: str = "percent_strong") -> None:
    pair_ids = set_pair_ids()
    df = df.loc[:, [pair_ids[x]["index"] for x in pair_ids]]
    index_sums = df.sum().tolist()

    assert_error = "Stop! The eval_metric provided in 'metric_melt()' is incorrect!"
    assert_error = "{err} This is a fatal error providing incorrect results".format(
        err=assert_error
    )
    if eval_metric == "percent_strong":
        assert index_sums[0] != index_sums[1], assert_error
    elif eval_metric == "precision_recall":
        assert index_sums[0] == index_sums[1], assert_error


def set_pair_ids():
    pair_a = "pair_a"
    pair_b = "pair_b"

    return_dict = {
        "pair_a": {
            "index": "{pair_a}_index".format(pair_a=pair_a),
            "suffix": "_{pair_a}".format(pair_a=pair_a),
        },
        "pair_b": {
            "index": "{pair_b}_index".format(pair_b=pair_b),
            "suffix": "_{pair_b}".format(pair_b=pair_b),
        },
    }

    return return_dict
