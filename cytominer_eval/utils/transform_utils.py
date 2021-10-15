import numpy as np
import pandas as pd
from typing import List, Union
import pandas.api.types as ptypes
from collections import OrderedDict

from cytominer_eval.utils.availability_utils import check_eval_metric


def get_upper_matrix(df: pd.DataFrame) -> np.array:
    r"""Helper function to return only an upper matrix of the size of the input

    Parameters
    ----------
    df : pandas.DataFrame
        Any dataframe with a shape

    Returns
    -------
    np.array
        An upper triangle matrix the same shape as the input dataframe
    """
    return np.triu(np.ones(df.shape), k=1).astype(bool)


def convert_pandas_dtypes(df: pd.DataFrame, col_fix: type = float) -> pd.DataFrame:
    r"""Helper funtion to convert pandas column dtypes

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas dataframe to convert columns
    col_fix : {float, str}, optional
        A column type to convert the input dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe with converted columns
    """
    try:
        df = df.astype(col_fix)
    except ValueError:
        raise ValueError(
            "Columns cannot be converted to {col}; check input features".format(
                col=col_fix
            )
        )

    return df


def assert_pandas_dtypes(df: pd.DataFrame, col_fix: type = float) -> pd.DataFrame:
    r"""Helper funtion to ensure pandas columns have compatible columns

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas dataframe to convert columns
    col_fix : {float, str}, optional
        A column type to convert the input dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe with converted columns
    """
    assert col_fix in [str, float], "Only str and float are supported"

    df = convert_pandas_dtypes(df=df, col_fix=col_fix)

    assert_error = "Columns not successfully updated, is the dataframe consistent?"
    if col_fix == str:
        assert all([ptypes.is_string_dtype(df[x]) for x in df.columns]), assert_error

    if col_fix == float:
        assert all([ptypes.is_numeric_dtype(df[x]) for x in df.columns]), assert_error

    return df


def assert_melt(
    df: pd.DataFrame, eval_metric: str = "replicate_reproducibility"
) -> None:
    r"""Helper function to ensure that we properly melted the pairwise correlation
    matrix

    Downstream functions depend on how we process the pairwise correlation matrix. The
    processing is different depending on the evaluation metric.

    Parameters
    ----------
    df : pandas.DataFrame
        A melted pairwise correlation matrix
    eval_metric : str
        The user input eval metric

    Returns
    -------
    None
        Assertion will fail if we incorrectly melted the matrix
    """
    check_eval_metric(eval_metric=eval_metric)

    pair_ids = set_pair_ids()
    df = df.loc[:, [pair_ids[x]["index"] for x in pair_ids]]
    index_sums = df.sum().tolist()

    assert_error = "Stop! The eval_metric provided in 'metric_melt()' is incorrect!"
    assert_error = "{err} This is a fatal error providing incorrect results".format(
        err=assert_error
    )
    if eval_metric == "replicate_reproducibility":
        assert index_sums[0] != index_sums[1], assert_error
    elif eval_metric == "precision_recall":
        assert index_sums[0] == index_sums[1], assert_error
    elif eval_metric == "grit":
        assert index_sums[0] == index_sums[1], assert_error
    elif eval_metric == "hitk":
        assert index_sums[0] == index_sums[1], assert_error


def set_pair_ids():
    r"""Helper function to ensure consistent melted pairiwise column names

    Returns
    -------
    collections.OrderedDict
        A length two dictionary of suffixes and indeces of two pairs.
    """
    pair_a = "pair_a"
    pair_b = "pair_b"

    return_dict = OrderedDict()
    return_dict[pair_a] = {
        "index": "{pair_a}_index".format(pair_a=pair_a),
        "suffix": "_{pair_a}".format(pair_a=pair_a),
    }
    return_dict[pair_b] = {
        "index": "{pair_b}_index".format(pair_b=pair_b),
        "suffix": "_{pair_b}".format(pair_b=pair_b),
    }

    return return_dict


def check_replicate_groups(
    eval_metric: str, replicate_groups: Union[List[str], dict]
) -> None:
    r"""Helper function checking that the user correctly constructed the input replicate
    groups argument

    The package will not calculate evaluation metrics with incorrectly constructed
    replicate_groups. See :py:func:`cytominer_eval.evaluate.evaluate`.

    Parameters
    ----------
    eval_metric : str
        Which evaluation metric to calculate. See
        :py:func:`cytominer_eval.transform.util.get_available_eval_metrics`.
    replicate_groups : {list, dict}
        The tentative data structure listing replicate groups

    Returns
    -------
    None
        Assertion will fail for improperly constructed replicate_groups
    """
    check_eval_metric(eval_metric=eval_metric)

    if eval_metric == "grit":
        assert isinstance(
            replicate_groups, dict
        ), "For grit, replicate_groups must be a dict"

        replicate_key_ids = ["profile_col", "replicate_group_col"]

        assert all(
            [x in replicate_groups for x in replicate_key_ids]
        ), "replicate_groups for grit not formed properly. Must contain {id}".format(
            id=replicate_key_ids
        )
    elif eval_metric == "mp_value":
        assert isinstance(
            replicate_groups, str
        ), "For mp_value, replicate_groups must be a single string."
    else:
        assert isinstance(
            replicate_groups, list
        ), "Replicate groups must be a list for the {op} operation".format(
            op=eval_metric
        )
