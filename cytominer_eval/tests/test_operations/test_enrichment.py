import os
import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd

from cytominer_eval.transform import metric_melt
from cytominer_eval.operations.enrichment import enrichment
from cytominer_eval import evaluate


random.seed(3141)
tmpdir = tempfile.gettempdir()


# Load LINCS dataset
example_file = "SQ00015054_normalized_feature_select.csv.gz"
example_file = pathlib.Path(
    "{file}/../../example_data/compound/{eg}".format(
        file=os.path.dirname(__file__), eg=example_file
    )
)

df = pd.read_csv(example_file)

meta_features = [
    x for x in df.columns if (x.startswith("Metadata_") or x.startswith("Image_"))
]
features = df.drop(meta_features, axis="columns").columns.tolist()

replicate_groups = ["Metadata_broad_sample"]

similarity_melted_df = metric_melt(
    df=df,
    features=features,
    metadata_features=meta_features,
    similarity_metric="pearson",
    eval_metric="enrichment",
)


def test_enrichment():
    percent_list = np.arange(1, 0.97, -0.005)
    result = enrichment(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        percentile=percent_list,
    )

    # check for correct shape and starts with 1.0
    assert result.shape == (7, 4)
    assert result.enrichment_percentile[0] == 1.0
    assert result.enrichment_percentile[1] == 0.995
    # check if the higher percentiles are larger than the small one
    assert result.enrichment_percentile[1] > result.enrichment_percentile.iloc[-1]
    
    result_int = enrichment(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        percentile=0.97,
    )
    
    assert result_int.enrichment_percentile[0] == result.enrichment_percentile[1]


def test_compare_functions():
    percent_list = [0.95, 0.9]
    eval_res = evaluate(
        profiles=df,
        features=features,
        meta_features=meta_features,
        replicate_groups=replicate_groups,
        operation="enrichment",
        similarity_metric="pearson",
        enrichment_percentile=percent_list,
    )
    enr_res = enrichment(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        percentile=percent_list,
    )
    assert enr_res.equals(eval_res)
