import os
import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd

from cytominer_eval.transform import metric_melt
from cytominer_eval.operations import precision_recall

random.seed(123)
tmpdir = tempfile.gettempdir()

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
    eval_metric="precision_recall",
)

replicate_groups = ["Metadata_gene_name", "Metadata_cell_line"]


def test_precision_recall():
    result = precision_recall(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        k=10,
    )

    assert len(result.k.unique()) == 1
    assert result.k.unique()[0] == 10

    # ITGAV has a really strong profile
    assert (
        result.sort_values(by="recall", ascending=False)
        .reset_index(drop=True)
        .iloc[0, :]
        .Metadata_gene_name
        == "ITGAV"
    )

    assert all(x in result.columns for x in replicate_groups)
