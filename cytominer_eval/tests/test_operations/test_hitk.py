import os
import random
import pathlib
import numpy as np
import pandas as pd
from math import isclose


from cytominer_eval.transform import metric_melt
from cytominer_eval.operations import hitk


random.seed(42)


# Load LINCS dataset
example_file = "SQ00015054_normalized_feature_select.csv.gz"
example_file = pathlib.Path(
    "{file}/../../example_data/compound/{eg}".format(
        file=os.path.dirname(__file__), eg=example_file
    )
)

df = pd.read_csv(example_file)
# Clean the dataframe for convenience
df.loc[
    (df["Metadata_moa"].isna()) & (df["Metadata_broad_sample"] == "DMSO"),
    "Metadata_moa",
] = "none"
df = df[~df["Metadata_moa"].isna()]


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

# compute the normal index_list
replicate_group = ["Metadata_moa"]
groupby_columns = ["Metadata_broad_sample", "Metadata_Plate", "Metadata_Well"]
percent_list = [2, 5, 10, 100]


index_list, percent_results = hitk(
    similarity_melted_df=similarity_melted_df,
    replicate_groups=replicate_group,
    groupby_columns=groupby_columns,
    percent_list=percent_list,
)


# compute index with percent = all
percent_all = "all"
indexes_all, percent_results_all = hitk(
    similarity_melted_df=similarity_melted_df,
    replicate_groups=replicate_group,
    groupby_columns=groupby_columns,
    percent_list=percent_all,
)

# compute the index with a randomized input
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
percent_list = [2, 5, 10, 100]
ran_index_list, ran_percent_results = hitk(
    similarity_melted_df=similarity_melted_ran,
    replicate_groups=replicate_group,
    groupby_columns=groupby_columns,
    percent_list=percent_list,
)

# if we use a combination of replicate groups that is unique for each index in the original df,
# then no hits will be found.
replicate_group = ["Metadata_moa", "Metadata_broad_sample", "Metadata_pert_well"]
percent_list = [2, 5, 10, 100]

index_list_empty, percent_results_empty = hitk(
    similarity_melted_df=similarity_melted_df,
    replicate_groups=replicate_group,
    groupby_columns=groupby_columns,
    percent_list=percent_list,
)


def test_hitk_list():
    assert percent_results == {2: 679, 5: 928, 10: 1061, 100: 0}
    assert max(index_list) == 364
    assert min(index_list) == 0


def test_number_of_hits():
    """Calculates the number of indexes based off the MOA value count in the original df
    """
    s = sum([n * (n - 1) for n in df["Metadata_moa"].value_counts()])
    assert s == len(index_list)


def test_hitk_all():
    assert index_list == indexes_all
    assert isclose(percent_results_all[0], 150.4, abs_tol=1e-1)
    last_score = percent_results_all[len(percent_results_all) - 1]
    assert isclose(last_score, 0, abs_tol=1e-1)


def test_random_input():
    len(ran_index_list) == len(index_list)
    assert ran_percent_results[100] == 0
    median_index = abs(
        np.median(ran_index_list) - len(similarity_melted_ran.pair_b_index.unique()) / 2
    )
    assert median_index < 30


def test_empty_results():
    assert len(index_list_empty) == 0
    for p in percent_results:
        assert percent_results_empty[p] == 0
