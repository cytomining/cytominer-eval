import os
import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from cytominer_eval.transform import metric_melt
from cytominer_eval.transform.util import set_pair_ids
from cytominer_eval.operations.util import assign_replicates, calculate_precision_recall

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


def test_assign_replicates():
    replicate_groups = ["Metadata_broad_sample", "Metadata_mg_per_ml"]
    result = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    )

    expected_cols = ["{x}_replicate".format(x=x) for x in replicate_groups] + [
        "group_replicate"
    ]

    # Other functions expect columns to exist
    assert all([x in result.columns.tolist() for x in expected_cols])

    # Given the example data, we expect a certain number of pairwise replicates
    expected_replicates = list(result.loc[:, expected_cols].sum().values)
    assert expected_replicates == [1248, 408, 408]

    # Try with a different number of replicate groups
    replicate_groups = [
        "Metadata_broad_sample",
        "Metadata_mg_per_ml",
        "Metadata_plate_map_name",
    ]
    result = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    )

    expected_cols = ["{x}_replicate".format(x=x) for x in replicate_groups] + [
        "group_replicate"
    ]

    # Other functions expect columns to exist
    assert all([x in result.columns.tolist() for x in expected_cols])

    # Given the example data, we expect a certain number of pairwise replicates
    expected_replicates = list(result.loc[:, expected_cols].sum().values)
    assert expected_replicates == [1248, 408, 73536, 408]

    # This function will fail if a replicate column is given that doesn't belong
    with pytest.raises(AssertionError) as ve:
        replicate_groups = ["MISSING_COLUMN"]
        result = assign_replicates(
            similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
        )
    assert "replicate_group not found in melted dataframe columns" in str(ve.value)


def test_calculate_precision_recall():
    similarity_melted_df = metric_melt(
        df=df,
        features=features,
        metadata_features=meta_features,
        similarity_metric="pearson",
        eval_metric="precision_recall",
    )

    replicate_groups = ["Metadata_broad_sample"]
    result = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    ).sort_values(by="similarity_metric", ascending=False)

    pair_ids = set_pair_ids()
    replicate_group_cols = [
        "{x}{suf}".format(x=x, suf=pair_ids[list(pair_ids)[0]]["suffix"])
        for x in replicate_groups
    ]

    example_group = result.groupby(replicate_group_cols).get_group(
        name=("BRD-A38592941-001-02-7")
    )

    assert example_group.shape[0] == 383 * 6  # number of pairwise comparisons per dose

    # Assert that the similarity metrics are sorted
    assert (example_group.similarity_metric.diff().dropna() > 0).sum() == 0

    # Perform the calculation!
    result = pd.DataFrame(
        calculate_precision_recall(example_group, k=10), columns=["result"]
    )

    expected_result = {"k": 10, "precision": 0.4, "recall": 0.1333}
    expected_result = pd.DataFrame(expected_result, index=["result"]).transpose()

    assert_frame_equal(result, expected_result, check_less_precise=True)

    # Check that recall is 1 when k is maximized
    result = pd.DataFrame(
        calculate_precision_recall(example_group, k=example_group.shape[0]),
        columns=["result"],
    )

    assert result.loc["recall", "result"] == 1
