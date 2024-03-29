import os
import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from cytominer_eval.operations import grit
from cytominer_eval.transform import metric_melt
from cytominer_eval.utils.transform_utils import set_pair_ids
from cytominer_eval.utils.precisionrecall_utils import calculate_precision_recall
from cytominer_eval.utils.availability_utils import get_available_summary_methods
from cytominer_eval.utils.operation_utils import (
    assign_replicates,
    compare_distributions,
)


random.seed(123)
tmpdir = tempfile.gettempdir()

example_file = "SQ00015054_normalized_feature_select.csv.gz"
example_file = pathlib.Path(
    "{file}/../../example_data/compound/{eg}".format(
        file=os.path.dirname(__file__), eg=example_file
    )
)

df = pd.read_csv(example_file)
df = df.assign(
    Metadata_profile_id=[
        "Metadata_profile_{x}".format(x=x) for x in range(0, df.shape[0])
    ]
)

meta_features = [x for x in df.columns if x.startswith("Metadata_")]
features = df.drop(meta_features, axis="columns").columns.tolist()

similarity_melted_df = metric_melt(
    df=df,
    features=features,
    metadata_features=meta_features,
    similarity_metric="pearson",
)

similarity_melted_full_df = metric_melt(
    df=df,
    features=features,
    metadata_features=meta_features,
    similarity_metric="pearson",
    eval_metric="grit",
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


def test_compare_distributions():
    # Define two distributions using a specific compound as an example

    compound = {"compound": "BRD-K07857022-002-01-1"}
    profile_id = {"profile_id": "Metadata_profile_378"}

    target_group = similarity_melted_full_df.query(
        "Metadata_profile_id_pair_a == @profile_id", local_dict=profile_id
    )

    replicate_group_values = target_group.query(
        "Metadata_broad_sample_pair_b == @compound", local_dict=compound
    ).similarity_metric.values.reshape(-1, 1)

    control_group_values = target_group.query(
        "Metadata_broad_sample_pair_b == 'DMSO'"
    ).similarity_metric.values.reshape(-1, 1)

    control_perts = df.query(
        "Metadata_broad_sample == 'DMSO'"
    ).Metadata_profile_id.tolist()

    hardcoded_values_should_not_change = {
        "zscore": {"mean": 5.639379456018854, "median": 5.648269672347573}
    }
    for summary_method in get_available_summary_methods():

        hardcoded = hardcoded_values_should_not_change["zscore"][summary_method]

        result = compare_distributions(
            target_distrib=replicate_group_values,
            control_distrib=control_group_values,
            method="zscore",
            replicate_summary_method=summary_method,
        )
        assert np.round(result, 5) == np.round(hardcoded, 5)

        grit_result = (
            grit(
                similarity_melted_full_df,
                control_perts=control_perts,
                profile_col="Metadata_profile_id",
                replicate_group_col="Metadata_broad_sample",
                replicate_summary_method=summary_method,
            )
            .query("perturbation == @profile_id", local_dict=profile_id)
            .grit.values[0]
        )

        assert result == grit_result
