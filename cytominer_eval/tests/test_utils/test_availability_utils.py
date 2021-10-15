import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pandas.api.types as ptypes


from cytominer_eval.utils.availability_utils import (
    get_available_eval_metrics,
    get_available_similarity_metrics,
    get_available_summary_methods,
    get_available_distribution_compare_methods,
    check_eval_metric,
    check_replicate_summary_method,
    check_similarity_metric,
    check_compare_distribution_method,
)


def test_get_available_summary_methods():
    expected_result = ["mean", "median"]
    assert expected_result == get_available_summary_methods()


def test_get_available_eval_metrics():
    expected_result = [
        "replicate_reproducibility",
        "precision_recall",
        "grit",
        "mp_value",
        "enrichment",
        "hitk",
    ]
    assert expected_result == get_available_eval_metrics()


def test_get_available_similarity_metrics():
    expected_result = ["pearson", "kendall", "spearman"]
    assert expected_result == get_available_similarity_metrics()


def test_get_available_distribution_compare_methods():
    expected_result = ["zscore"]
    assert expected_result == get_available_distribution_compare_methods()


def test_check_eval_metric():
    with pytest.raises(AssertionError) as ae:
        output = check_eval_metric(eval_metric="MISSING")
    assert "MISSING not supported. Select one of" in str(ae.value)


def test_check_replicate_summary_method():
    for metric in get_available_summary_methods():
        check_replicate_summary_method(metric)

    with pytest.raises(ValueError) as ve:
        output = check_replicate_summary_method("fail")
    assert "fail method not supported. Select one of:" in str(ve.value)


def test_check_similarity_metric():
    for metric in get_available_similarity_metrics():
        check_similarity_metric(metric)

    with pytest.raises(AssertionError) as ve:
        output = check_similarity_metric("fail")
    assert "fail not supported. Available similarity metrics:" in str(ve.value)


def test_check_compare_distribution_method():
    for metric in get_available_distribution_compare_methods():
        check_compare_distribution_method(metric)

    with pytest.raises(AssertionError) as ve:
        output = check_compare_distribution_method("fail")
    assert "not supported. Available distribution methods:" in str(ve.value)
