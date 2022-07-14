import random
import tempfile
import numpy as np
import pandas as pd

from cytominer_eval.utils.grit_utils import set_grit_column_info


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
