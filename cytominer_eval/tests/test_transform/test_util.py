import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from cytominer_eval.transform.util import (
    get_available_eval_metrics,
    get_available_similarity_metrics,
    get_available_grit_summary_methods,
    get_upper_matrix,
    convert_pandas_dtypes,
    assert_pandas_dtypes,
    set_pair_ids,
    set_grit_column_info,
    assert_eval_metric,
    assert_melt,
    check_replicate_groups,
    check_grit_replicate_summary_method,
)

random.seed(123)
tmpdir = tempfile.gettempdir()

data_df = pd.DataFrame(
    {
        "float_a": np.random.normal(1, 1, 4),
        "float_b": np.random.normal(1, 1, 4),
        "string_a": ["a"] * 4,
        "string_b": ["b"] * 4,
    }
)
float_cols = ["float_a", "float_b"]


def test_get_available_eval_metrics():
    expected_result = [
        "replicate_reproducibility",
        "precision_recall",
        "grit",
        "mp_value",
    ]
    assert expected_result == get_available_eval_metrics()


def test_get_available_similarity_metrics():
    expected_result = ["pearson", "kendall", "spearman"]
    assert expected_result == get_available_similarity_metrics()


def test_get_available_grit_summary_methods():
    expected_result = ["mean", "median"]
    assert expected_result == get_available_grit_summary_methods()


def test_assert_eval_metric():
    with pytest.raises(AssertionError) as ae:
        output = assert_eval_metric(eval_metric="NOT SUPPORTED")
    assert "ot supported. Select one of" in str(ae.value)


def test_get_upper_matrix():
    result = get_upper_matrix(data_df)
    assert not result[0, 0]
    assert result[0, 1]
    assert not result[1, 1]

    assert result.sum() == 6


def test_convert_pandas_dtypes():
    with pytest.raises(ValueError) as ve:
        output = convert_pandas_dtypes(data_df)
    assert "check input features" in str(ve.value)

    data_string_type_df = data_df.astype(str)
    output_df = convert_pandas_dtypes(
        data_string_type_df.loc[:, float_cols], col_fix=np.float64
    )
    assert all([ptypes.is_numeric_dtype(output_df[x]) for x in output_df.columns])


def test_assert_pandas_dtypes():
    with pytest.raises(ValueError) as ve:
        output = assert_pandas_dtypes(data_df)
    assert "check input features" in str(ve.value)

    with pytest.raises(AssertionError) as ve:
        output = assert_pandas_dtypes(data_df, col_fix="not supported")
    assert "Only np.str and np.float64 are supported" in str(ve.value)

    output_df = assert_pandas_dtypes(data_df, col_fix=np.str)
    all([ptypes.is_string_dtype(output_df[x]) for x in output_df.columns])

    output_df = convert_pandas_dtypes(output_df.loc[:, float_cols], col_fix=np.float64)
    assert all([ptypes.is_numeric_dtype(output_df[x]) for x in output_df.columns])


def test_set_pair_ids():
    pair_a = "pair_a"
    pair_b = "pair_b"

    result = set_pair_ids()

    assert result[pair_a]["index"] == "{pair_a}_index".format(pair_a=pair_a)
    assert result[pair_a]["index"] == "{pair_a}_index".format(pair_a=pair_a)
    assert result[pair_b]["suffix"] == "_{pair_b}".format(pair_b=pair_b)
    assert result[pair_b]["suffix"] == "_{pair_b}".format(pair_b=pair_b)


def test_set_grit_column_info():
    profile_col = "test_replicate"
    replicate_group_col = "test_group"

    result = set_grit_column_info(
        profile_col=profile_col, replicate_group_col=replicate_group_col
    )

    assert result["profile"]["id"] == "{rep}_pair_a".format(rep=profile_col)
    assert result["profile"]["comparison"] == "{rep}_pair_b".format(rep=profile_col)
    assert result["group"]["id"] == "{group}_pair_a".format(group=replicate_group_col)
    assert result["group"]["comparison"] == "{group}_pair_b".format(
        group=replicate_group_col
    )


def test_check_replicate_groups():
    available_metrics = get_available_eval_metrics()

    replicate_groups = ["Metadata_gene_name", "Metadata_pert_name"]
    replicate_group_dict = {
        "profile_col": "testingA",
        "replicate_group_col": "testingB",
    }
    for operation in available_metrics:
        if operation == "grit":
            check_replicate_groups(
                eval_metric=operation, replicate_groups=replicate_group_dict
            )
            with pytest.raises(AssertionError) as ae:
                output = check_replicate_groups(
                    eval_metric=operation, replicate_groups=replicate_groups
                )
            assert "For grit, replicate_groups must be a dict" in str(ae.value)
        elif operation == "mp_value":
            check_replicate_groups(
                eval_metric=operation, replicate_groups=replicate_groups[0]
            )
            with pytest.raises(AssertionError) as ae:
                output = check_replicate_groups(
                    eval_metric=operation, replicate_groups=replicate_groups
                )
            assert "For mp_value, replicate_groups must be a single string." in str(
                ae.value
            )
        else:
            check_replicate_groups(
                eval_metric=operation, replicate_groups=replicate_groups
            )
            with pytest.raises(AssertionError) as ae:
                output = check_replicate_groups(
                    eval_metric=operation, replicate_groups=replicate_group_dict
                )
            assert "Replicate groups must be a list for the {op} operation".format(
                op=operation
            ) in str(ae.value)

    with pytest.raises(AssertionError) as ae:
        wrong_group_dict = {"MISSING": "nothing here", "MISSING_TOO": "nothing"}
        output = check_replicate_groups(
            eval_metric="grit", replicate_groups=wrong_group_dict
        )
    assert "replicate_groups for grit not formed properly." in str(ae.value)


def test_check_grit_replicate_summary_method():

    # Pass
    for metric in get_available_grit_summary_methods():
        check_grit_replicate_summary_method(metric)

    with pytest.raises(ValueError) as ve:
        output = check_grit_replicate_summary_method("fail")
    assert "method not supported, use one of:" in str(ve.value)
