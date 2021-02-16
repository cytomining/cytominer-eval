import numpy as np
import pandas as pd
from typing import List, Union
import pandas.api.types as ptypes
from collections import OrderedDict


def get_available_eval_metrics():
    return ["replicate_reproducibility", "precision_recall", "grit", "mp_value"]


def get_available_similarity_metrics():
    return ["pearson", "kendall", "spearman"]


def get_available_grit_summary_methods():
    return ["mean", "median"]


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


def assert_eval_metric(eval_metric: str):
    avail_metrics = get_available_eval_metrics()

    assert (
        eval_metric in avail_metrics
    ), "{eval} not supported. Select one of {avail}".format(
        eval=eval_metric, avail=avail_metrics
    )


def assert_melt(
    df: pd.DataFrame, eval_metric: str = "replicate_reproducibility"
) -> None:

    assert_eval_metric(eval_metric=eval_metric)

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


def set_pair_ids():
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
) -> str:

    assert_eval_metric(eval_metric=eval_metric)

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


def set_grit_column_info(profile_col: str, replicate_group_col: str) -> dict:
    r"""Transform column names to be used in calculating grit

    In calculating grit, the data must have a metadata feature describing the core
    replicate perturbation (profile_col) and a separate metadata feature(s) describing
    the larger group (replicate_group_col) that the perturbation belongs to (e.g. gene,
    MOA).

    Parameters
    ----------
    profile_col : str
        the metadata column storing profile ids. The column can have unique or replicate
        identifiers.
    replicate_group_col : str
        the metadata column indicating a higher order structure (group) than the
        profile column. E.g. target gene vs. guide in a CRISPR experiment.

    Returns
    -------
    dict
        A nested dictionary of renamed columns indicating how to determine replicates
    """
    # Identify column transform names
    pair_ids = set_pair_ids()

    profile_id_with_suffix = [
        "{col}{suf}".format(col=profile_col, suf=pair_ids[x]["suffix"])
        for x in pair_ids
    ]

    group_id_with_suffix = [
        "{col}{suf}".format(col=replicate_group_col, suf=pair_ids[x]["suffix"])
        for x in pair_ids
    ]

    col_info = ["id", "comparison"]
    profile_id_info = dict(zip(col_info, profile_id_with_suffix))
    group_id_info = dict(zip(col_info, group_id_with_suffix))

    column_id_info = {"profile": profile_id_info, "group": group_id_info}
    return column_id_info


def check_grit_replicate_summary_method(replicate_summary_method: str):
    avail_methods = get_available_grit_summary_methods()

    if replicate_summary_method not in avail_methods:
        raise ValueError(
            "{input} method not supported, use one of: {avail}".format(
                input=replicate_summary_method, avail=avail_methods
            )
        )
