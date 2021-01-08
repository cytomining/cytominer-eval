import os
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
from math import isclose
from cytominer_eval.operations import mp_value
from cytominer_eval.operations.util import calculate_mp_value, calculate_mahalanobis

# Load CRISPR dataset
example_file = "SQ00014610_normalized_feature_select.csv.gz"
example_file = pathlib.Path(
    "{file}/../../example_data/gene/{eg}".format(
        file=os.path.dirname(__file__), eg=example_file
    )
)

df = pd.read_csv(example_file)

meta_features = [
    x for x in df.columns if (x.startswith("Metadata_") or x.startswith("Image_"))
]
features = df.drop(meta_features, axis="columns").columns.tolist()

control_perts = ["Luc-2", "LacZ-2", "LacZ-3"]
replicate_id = "Metadata_pert_name"


def test_calculate_mahalanobis():
    sub_df = df[(df.Metadata_WellRow == "A") & (df.Metadata_pert_name == "EMPTY")][
        features
    ]
    control_df = df[df[replicate_id].isin(control_perts)][features]

    maha = calculate_mahalanobis(pert_df=sub_df, control_df=control_df)

    assert isinstance(maha, float)
    # The following value is empirically determined
    # and not theoretically justified but avoids unwanted
    # changes in the implementation of the Mahalanobis distance
    assert isclose(maha, 3.62523778789, abs_tol=1e-09)

    maha = calculate_mahalanobis(pert_df=control_df, control_df=control_df)

    # Distance to itself should be approximately zero
    assert isclose(maha, 0, abs_tol=1e-05)


def test_calculate_mp_value():
    # The mp-values are empirical p-values
    # so they range from 0 to 1, with low values
    # showing a difference to the control condition.

    sub_df = df[(df.Metadata_WellRow == "A") & (df.Metadata_pert_name == "EMPTY")][
        features
    ]
    control_df = df[df[replicate_id].isin(control_perts)][features]

    # Avoid fluctuations in permutations
    np.random.seed(2020)
    result = calculate_mp_value(pert_df=sub_df, control_df=control_df)

    assert isinstance(result, float)
    assert result > 0
    assert result < 1

    # Distance to itself should be approximately zero
    # So mp-value should be 1
    result = calculate_mp_value(
        pert_df=control_df, control_df=control_df, params={"nb_permutations": 2000}
    )

    assert isclose(result, 1, abs_tol=1e-02)

    with pytest.raises(AssertionError) as ae:
        result = calculate_mp_value(
            pert_df=control_df, control_df=control_df, params={"not_a_parameter": 2000}
        )
    assert "Unknown parameters provided. Only" in str(ae.value)


def test_mp_value():
    result = mp_value(
        df=df,
        control_perts=control_perts,
        replicate_id=replicate_id,
        features=features,
    )

    assert "mp_value" in result.columns
    assert all(result.mp_value <= 1)
    assert all(result.mp_value >= 0)
    assert len(np.unique(df[replicate_id])) == len(result)

    with pytest.raises(AssertionError) as ae:
        result = mp_value(
            df=df,
            control_perts=control_perts,
            replicate_id=replicate_id,
            features=features,
            params={"not_a_parameter": 2000},
        )
    assert "Unknown parameters provided. Only" in str(ae.value)
