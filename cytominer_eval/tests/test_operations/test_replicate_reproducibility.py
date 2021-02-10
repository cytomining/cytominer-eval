import os
import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from cytominer_eval.transform import metric_melt
from cytominer_eval.operations import replicate_reproducibility

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


def test_replicate_reproducibility():
    replicate_groups = ["Metadata_broad_sample", "Metadata_mg_per_ml"]
    output = replicate_reproducibility(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        quantile_over_null=0.95,
    )
    expected_result = 0.4583

    assert np.round(output, 4) == expected_result

    replicate_groups = ["Metadata_moa"]
    output = replicate_reproducibility(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        quantile_over_null=0.95,
    )
    expected_result = 0.3074

    assert np.round(output, 4) == expected_result


def test_replicate_reproducibility_uniquerows():
    with pytest.raises(AssertionError) as err:
        replicate_groups = ["Metadata_pert_well"]
        output = replicate_reproducibility(
            similarity_melted_df=similarity_melted_df,
            replicate_groups=replicate_groups,
            quantile_over_null=0.95,
        )
    assert "no replicate groups identified in {rep} columns!".format(
        rep=replicate_groups
    ) in str(err.value)


def test_replicate_reproducibility_return_cor():
    # Confirm that it works with multiple columns
    replicate_groups = ["Metadata_moa"]

    output, med_cor = replicate_reproducibility(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        quantile_over_null=0.95,
        return_median_correlations=True,
    )
    expected_result = 0.3074

    assert np.round(output, 4) == expected_result
    assert (
        med_cor.sort_values(
            by="similarity_metric", ascending=False
        ).Metadata_moa.values[0]
        == "PKC activator"
    )
    assert np.round(med_cor.similarity_metric.max(), 4) == 0.9357

    replicate_groups = ["Metadata_broad_sample", "Metadata_mg_per_ml"]
    output, med_cor = replicate_reproducibility(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        quantile_over_null=0.95,
        return_median_correlations=True,
    )
    expected_result = 0.4583

    assert np.round(output, 4) == expected_result
    assert np.round(med_cor.similarity_metric.mean(), 4) == 0.5407
