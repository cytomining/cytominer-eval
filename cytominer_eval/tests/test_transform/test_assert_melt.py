import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from cytominer_eval.transform import metric_melt
from cytominer_eval.operations.util import assign_replicates
from cytominer_eval.transform.util import assert_melt


example_file = pathlib.Path(
    f"cytominer_eval/example_data/gene/SQ00014610_normalized_feature_select.csv.gz"
)

df = pd.read_csv(example_file)
meta_features = [
    x for x in df.columns if (x.startswith("Metadata_") or x.startswith("Image_"))
]
features = df.drop(meta_features, axis="columns").columns.tolist()

replicate_groups = ["Metadata_gene_name", "Metadata_cell_line"]


def test_assert_melt():
    for metric in ["precision_recall", "percent_strong"]:
        result = metric_melt(
            df=df,
            features=features,
            metadata_features=meta_features,
            similarity_metric="pearson",
            eval_metric=metric,
        )

        result = assign_replicates(
            similarity_melted_df=result, replicate_groups=replicate_groups
        )

        assert_melt(result, eval_metric=metric)

        if metric == "precision_recall":
            dummy_metrics = ["percent_strong"]
        elif metric == "percent_strong":
            dummy_metrics = ["precision_recall"]

        for dummy_metric in dummy_metrics:
            with pytest.raises(AssertionError) as ve:
                output = assert_melt(result, eval_metric=dummy_metric)
            assert (
                "Stop! The eval_metric provided in 'metric_melt()' is incorrect!"
                in str(ve.value)
            )
