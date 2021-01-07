import os
import random
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from cytominer_eval.transform.transform import get_pairwise_metric, process_melt
from cytominer_eval.transform import metric_melt

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

feature_df = df.loc[:, features]
meta_df = df.loc[:, meta_features]
sample_a = feature_df.iloc[
    0,
].values
sample_b = feature_df.iloc[
    1,
].values
example_sample_corr = np.corrcoef(sample_a, sample_b)[0, 1]

pairwise_metric_df = get_pairwise_metric(feature_df, similarity_metric="pearson")


def test_get_pairwise_metric():
    with pytest.raises(ValueError) as ve:
        output = get_pairwise_metric(df, similarity_metric="pearson")
    assert "check input features" in str(ve.value)

    with pytest.raises(AssertionError) as ve:
        output = get_pairwise_metric(feature_df, similarity_metric="not supported")
    assert "not supported not supported" in str(ve.value)

    result_df = get_pairwise_metric(feature_df, similarity_metric="pearson")

    assert np.diagonal(result_df).sum() == df.shape[0]

    assert round(example_sample_corr, 3) == round(result_df.iloc[0, 1], 3)


def test_process_melt():
    with pytest.raises(AssertionError) as ve:
        output = process_melt(df=feature_df, meta_df=meta_df)
    assert "Matrix must be symmetrical" in str(ve.value)

    melted_df = process_melt(df=pairwise_metric_df, meta_df=meta_df)
    assert round(melted_df.similarity_metric[0], 3) == round(example_sample_corr, 3)
    assert melted_df.shape[0] == 73536


def test_metric_melt():
    result_df = metric_melt(df, features, meta_features, similarity_metric="pearson")
    assert round(result_df.similarity_metric[0], 3) == round(example_sample_corr, 3)
    assert result_df.shape[0] == 73536

    # The index ID is extremely important for aligning the dataframe
    # make sure the method is robust to indeces labeled inconsistently
    same_index_copy = df.copy()
    same_index_copy.index = [3] * same_index_copy.shape[0]

    result_df = metric_melt(
        same_index_copy, features, meta_features, similarity_metric="pearson"
    )

    assert round(result_df.similarity_metric[0], 3) == round(example_sample_corr, 3)
    assert result_df.shape[0] == 73536

    with pytest.raises(AssertionError) as ve:
        output = metric_melt(
            df,
            features,
            meta_features,
            similarity_metric="pearson",
            eval_metric="NOT SUPPORTED",
        )
    assert "not supported. Available evaluation metrics:" in str(ve.value)

    with pytest.raises(AssertionError) as ve:
        output = metric_melt(
            df, features + ["NOT SUPPORTED"], meta_features, similarity_metric="pearson"
        )
    assert "Profile feature not found" in str(ve.value)

    with pytest.raises(AssertionError) as ve:
        output = metric_melt(
            df, features, meta_features + ["NOT SUPPORTED"], similarity_metric="pearson"
        )
    assert "Metadata feature not found" in str(ve.value)

    for full_metric_required in ["precision_recall", "grit"]:
        result_df = metric_melt(
            same_index_copy,
            features,
            meta_features,
            similarity_metric="pearson",
            eval_metric=full_metric_required,
        )

        assert round(result_df.similarity_metric[0], 3) == round(example_sample_corr, 3)
        assert result_df.shape[0] == 147072
