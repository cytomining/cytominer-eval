import os
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
from cytominer_eval.operations import mp_value
from cytominer_eval.operations.util import (
    calculate_mp_value, 
    calculate_mahalanobis
)

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
    sub_df = df[(df.Metadata_WellRow == "A")&(df.Metadata_pert_name == "EMPTY")][features]
    control_df = df[df[replicate_id].isin(control_perts)][features]
    
    maha = calculate_mahalanobis(pert_df = sub_df,
                                control_df = control_df)
    
    assert isinstance(maha, float)
    assert maha >= 0
    
    maha = calculate_mahalanobis(pert_df = control_df,
                                control_df = control_df)
    
    # Distance to itself should be approximately zero
    assert abs(maha) < 1e-5


def test_calculate_mp_value():
    sub_df = df[(df.Metadata_WellRow == "A")&(df.Metadata_pert_name == "EMPTY")][features]
    control_df = df[df[replicate_id].isin(control_perts)][features]
    
    # Avoid fluctuations in permutations
    np.random.seed(2020)
    result = calculate_mp_value(pert_df = sub_df,
                                control_df = control_df)
    
    assert isinstance(result, float)
    assert result > 0
    assert result < 1
    
    # Distance to itself should be approximately zero
    # So mp-value should be 1
    result = calculate_mp_value(pert_df = control_df,
                                control_df = control_df)

    assert result == 1


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