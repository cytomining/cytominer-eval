import os
import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from cytominer_eval.transform import metric_melt
from cytominer_eval.operations import percent_strong

random.seed(123)
tmpdir = tempfile.gettempdir()

example_file = "SQ00015054_normalized_feature_select.csv.gz"
example_file = pathlib.Path(
    "{file}/../../example_data/compound/{eg}".format(
        file=os.path.dirname(__file__), eg=example_file
    )
)

df = pd.read_csv(example_file)

meta_features = [x for x in df.columns if x.startswith("Metadata_")]
features = df.drop(meta_features, axis="columns").columns.tolist()

similarity_melted_df = metric_melt(
    df=df,
    features=features,
    metadata_features=meta_features,
    similarity_metric="pearson",
)


def test_percent_strong():
    replicate_groups = ["Metadata_broad_sample", "Metadata_mg_per_ml"]
    output = percent_strong(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        quantile=0.95,
    )
    expected_result = 0.4583

    assert np.round(output, 4) == expected_result

    replicate_groups = ["Metadata_moa"]
    output = percent_strong(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        quantile=0.95,
    )
    expected_result = 0.3074

    assert np.round(output, 4) == expected_result


def test_percent_strong_uniquerows():
    with pytest.raises(AssertionError) as err:
        replicate_groups = ["Metadata_pert_well"]
        output = percent_strong(
            similarity_melted_df=similarity_melted_df,
            replicate_groups=replicate_groups,
            quantile=0.95,
        )
    assert "no replicate groups identified in {rep} columns!".format(
        rep=replicate_groups
    ) in str(err.value)

    ### REFERENCE CODE
    # with pytest.raises(AssertionError) as ve:
    # replicate_groups = ["MISSING_COLUMN"]
    # result = percent_strong(
    #     similarity_melted_df=similarity_melted_df,
    #     replicate_groups=replicate_groups,
    #     quantile=0.95,
    # )
    # assert "replicate_group not found in melted dataframe columns" in str(ve.value)
