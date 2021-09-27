import os
import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/Users/mbornhol/git/mycyto/cytominer-eval")

from cytominer_eval.transform import metric_melt
from cytominer_eval.operations import hitk




random.seed(42)
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
    eval_metric="hitk",
)

def test_hitk():
    result = hitk(similarity_melted_df)
    assert True


test_hitk()