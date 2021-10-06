import os
import random
import pathlib
import tempfile
import numpy as np
import pandas as pd


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
df["Metadata_moa"] = df["Metadata_moa"].fillna("unknown")


meta_features = [
    x for x in df.columns if (x.startswith("Metadata_") or x.startswith("Image_"))
]
features = df.drop(meta_features, axis="columns").columns.tolist()

similarity_melted_df = metric_melt(
    df=df,
    features=features,
    metadata_features=meta_features,
    similarity_metric="pearson",
    eval_metric="hitk",
)

percent_list = [2, 5, 10, 100]
indexes, percent_results = hitk(similarity_melted_df, percent_list)


def test_number_of_hits():
    """Calculates the number of indexes based off the MOA value count in the original df
    """
    s = sum([n * (n - 1) for n in df.Metadata_moa.value_counts()])
    assert s == len(indexes)


def test_hitk_list():
    assert percent_results == {2: 690, 5: 984, 10: 1147, 100: 0}
    assert max(indexes) == 382
    assert min(indexes) == 0


percent_all = "all"
indexes_all, percent_results_all = hitk(similarity_melted_df, percent_all)


def test_hitk_all():
    assert indexes == indexes_all
    assert percent_results_all[0] == 150.75
    assert percent_results_all[383] == 0.0


ran = df.copy()
ran[features] = (
    ran[features].iloc[np.random.permutation(len(df))].reset_index(drop=True)
)
similarity_melted_ran = metric_melt(
    df=ran,
    features=features,
    metadata_features=meta_features,
    similarity_metric="pearson",
    eval_metric="hitk",
)
ran_indexes, ran_percent_results = hitk(similarity_melted_ran, percent_list)


def test_random_input():
    len(ran_indexes) == len(indexes)
    assert ran_percent_results[100] == 0
    median_index = abs(
        np.median(ran_indexes) - len(similarity_melted_ran.pair_b_index.unique()) / 2
    )
    assert median_index < 30
