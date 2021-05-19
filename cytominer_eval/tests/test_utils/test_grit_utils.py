import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from cytominer_eval.utils.grit_utils import (
    set_grit_column_info,
    check_grit_replicate_summary_method,
)

from cytominer_eval.utils.availability_utils import get_available_grit_summary_methods


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


def test_check_grit_replicate_summary_method():

    # Pass
    for metric in get_available_grit_summary_methods():
        check_grit_replicate_summary_method(metric)

    with pytest.raises(ValueError) as ve:
        output = check_grit_replicate_summary_method("fail")
    assert "method not supported, use one of:" in str(ve.value)


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
