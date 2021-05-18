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
    get_available_grit_summary_methods,
)


def test_get_available_grit_summary_methods():
    expected_result = ["mean", "median"]
    assert expected_result == get_available_grit_summary_methods()


def test_get_available_eval_metrics():
    expected_result = [
        "replicate_reproducibility",
        "precision_recall",
        "grit",
        "mp_value",
        "enrichment",
    ]
    assert expected_result == get_available_eval_metrics()


def test_get_available_similarity_metrics():
    expected_result = ["pearson", "kendall", "spearman"]
    assert expected_result == get_available_similarity_metrics()
