import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from cytominer_eval.transform.util import (
    get_upper_matrix,
    convert_pandas_dtypes,
    assert_pandas_dtypes,
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
