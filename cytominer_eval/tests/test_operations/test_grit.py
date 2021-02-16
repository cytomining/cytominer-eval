import os
import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from sklearn.preprocessing import StandardScaler

from cytominer_eval.operations import grit
from cytominer_eval.transform import metric_melt
from cytominer_eval.transform.util import (
    assert_melt,
    set_pair_ids,
    set_grit_column_info,
)
from cytominer_eval.operations.util import (
    assign_replicates,
    get_grit_entry,
    calculate_grit,
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

similarity_melted_df = metric_melt(
    df=df,
    features=features,
    metadata_features=meta_features,
    similarity_metric="pearson",
    eval_metric="grit",
)

control_perts = ["Luc-2", "LacZ-2", "LacZ-3"]
profile_col = "Metadata_pert_name"
replicate_group_col = "Metadata_gene_name"

pair_ids = set_pair_ids()
replicate_col_name = "{x}{suf}".format(
    x=profile_col, suf=pair_ids[list(pair_ids)[0]]["suffix"]
)

column_id_info = set_grit_column_info(
    profile_col=profile_col, replicate_group_col=replicate_group_col
)


def test_get_grit_entry():
    with pytest.raises(AssertionError) as ae:
        result = get_grit_entry(df=similarity_melted_df, col=replicate_col_name)
    assert "grit is calculated for each perturbation independently" in str(ae.value)

    expected_result = "EMPTY"
    similarity_subset_df = similarity_melted_df.query(
        "Metadata_pert_name_pair_a == @expected_result"
    )
    result = get_grit_entry(df=similarity_subset_df, col=replicate_col_name)
    assert result == expected_result


def test_calculate_grit():
    result = assign_replicates(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=[profile_col, replicate_group_col],
    )

    assert_melt(result, eval_metric="grit")

    example_group = result.groupby(replicate_col_name).get_group(name=("MTOR-2"))

    # Perform the calculation!
    grit_result = pd.DataFrame(
        calculate_grit(
            example_group, control_perts=control_perts, column_id_info=column_id_info
        ),
        columns=["result"],
    )

    expected_result = {"perturbation": "MTOR-2", "group": "MTOR", "grit": 1.55075}
    expected_result = pd.DataFrame(expected_result, index=["result"]).transpose()

    assert_frame_equal(grit_result, expected_result)

    # Calculate grit will not work with singleton perturbations
    # (no other perts in same group)
    example_group = result.groupby(replicate_col_name).get_group(name=("AURKB-2"))

    grit_result = pd.DataFrame(
        calculate_grit(
            example_group, control_perts=control_perts, column_id_info=column_id_info
        ),
        columns=["result"],
    )

    expected_result = {"perturbation": "AURKB-2", "group": "AURKB", "grit": np.nan}
    expected_result = pd.DataFrame(expected_result, index=["result"]).transpose()

    assert_frame_equal(grit_result, expected_result)

    # Calculate grit will not work with the full dataframe
    with pytest.raises(AssertionError) as ae:
        result = calculate_grit(
            similarity_melted_df,
            control_perts=control_perts,
            column_id_info=column_id_info,
        )
    assert "grit is calculated for each perturbation independently" in str(ae.value)

    # Calculate grit will not work with when control barcodes are missing
    with pytest.raises(AssertionError) as ae:
        result = calculate_grit(
            example_group,
            control_perts=["DOES NOT EXIST", "THIS ONE NEITHER"],
            column_id_info=column_id_info,
        )
    assert "Error! No control perturbations found." in str(ae.value)


def test_grit():
    result = grit(
        similarity_melted_df=similarity_melted_df,
        control_perts=control_perts,
        profile_col=profile_col,
        replicate_group_col=replicate_group_col,
    ).sort_values(by="grit")

    assert all([x in result.columns for x in ["perturbation", "group", "grit"]])

    top_result = pd.DataFrame(
        result.sort_values(by="grit", ascending=False)
        .reset_index(drop=True)
        .iloc[0, :],
    )

    expected_result = {"perturbation": "PTK2-2", "group": "PTK2", "grit": 4.61094}
    expected_result = pd.DataFrame(expected_result, index=[0]).transpose()

    assert_frame_equal(top_result, expected_result)

    # There are six singletons in this dataset
    assert result.grit.isna().sum() == 6

    # No perturbations should be duplicated
    assert result.perturbation.duplicated().sum() == 0

    # With this data, we do not expect the sum of grit to change
    assert np.round(result.grit.sum(), 0) == 152.0


def test_grit_summary_metric():
    result = grit(
        similarity_melted_df=similarity_melted_df,
        control_perts=control_perts,
        profile_col=profile_col,
        replicate_group_col=replicate_group_col,
        replicate_summary_method="median",
    ).sort_values(by="grit")

    assert all([x in result.columns for x in ["perturbation", "group", "grit"]])

    top_result = pd.DataFrame(
        result.sort_values(by="grit", ascending=False)
        .reset_index(drop=True)
        .iloc[0, :],
    )

    expected_result = {"perturbation": "PTK2-2", "group": "PTK2", "grit": 4.715917}
    expected_result = pd.DataFrame(expected_result, index=[0]).transpose()

    assert_frame_equal(
        top_result,
        expected_result,
    )

    with pytest.raises(ValueError) as ve:
        output = grit(
            similarity_melted_df=similarity_melted_df,
            control_perts=control_perts,
            profile_col=profile_col,
            replicate_group_col=replicate_group_col,
            replicate_summary_method="fail",
        )
    assert "method not supported, use one of:" in str(ve.value)
