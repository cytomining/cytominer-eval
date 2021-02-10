import numpy as np
import pandas as pd
from typing import List

from .util import (
    assert_pandas_dtypes,
    get_upper_matrix,
    set_pair_ids,
    get_available_eval_metrics,
    get_available_similarity_metrics,
)

available_pairwise_similarity_metrics = get_available_similarity_metrics()
available_evaluation_metrics = get_available_eval_metrics()


def get_pairwise_metric(df: pd.DataFrame, similarity_metric: str) -> pd.DataFrame:
    df = assert_pandas_dtypes(df=df, col_fix=np.float64)

    assert (
        similarity_metric in available_pairwise_similarity_metrics
    ), "{m} not supported. Available similarity metrics: {avail}".format(
        m=similarity_metric, avail=available_pairwise_similarity_metrics
    )

    pair_df = df.transpose().corr(method=similarity_metric)

    # Check if the metric calculation went wrong
    # (Current pandas version makes this check redundant)
    if pair_df.shape == (0, 0):
        raise TypeError(
            "Something went wrong - check that 'features' are profile measurements"
        )

    return pair_df


def process_melt(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    eval_metric: str = "replicate_reproducibility",
) -> pd.DataFrame:

    assert df.shape[0] == df.shape[1], "Matrix must be symmetrical"
    assert (
        eval_metric in available_evaluation_metrics
    ), "{m} not supported. Available evaluation metrics: {avail}".format(
        m=eval_metric, avail=available_evaluation_metrics
    )

    # Get identifiers for pairing metadata
    pair_ids = set_pair_ids()

    # Subset the pairwise similarity metric depending on the eval metric given:
    #   "replicate_reproducibility" - requires only the upper triangle of a symmetric matrix
    #   "precision_recall" - requires the full symmetric matrix (no diagonal)
    # Remove pairwise matrix diagonal and redundant pairwise comparisons
    if eval_metric == "replicate_reproducibility":
        upper_tri = get_upper_matrix(df)
        df = df.where(upper_tri)
    else:
        np.fill_diagonal(df.values, np.nan)

    # Convert pairwise matrix to melted (long) version based on index value
    metric_unlabeled_df = (
        pd.melt(
            df.reset_index(),
            id_vars="index",
            value_vars=df.columns,
            var_name=pair_ids["pair_b"]["index"],
            value_name="similarity_metric",
        )
        .dropna()
        .reset_index(drop=True)
        .rename({"index": pair_ids["pair_a"]["index"]}, axis="columns")
    )

    # Merge metadata on index for both comparison pairs
    output_df = meta_df.merge(
        meta_df.merge(
            metric_unlabeled_df,
            left_index=True,
            right_on=pair_ids["pair_b"]["index"],
        ),
        left_index=True,
        right_on=pair_ids["pair_a"]["index"],
        suffixes=[pair_ids["pair_a"]["suffix"], pair_ids["pair_b"]["suffix"]],
    ).reset_index(drop=True)

    return output_df


def metric_melt(
    df: pd.DataFrame,
    features: List[str],
    metadata_features: List[str],
    eval_metric: str = "replicate_reproducibility",
    similarity_metric: str = "pearson",
) -> pd.DataFrame:
    # Subset dataframes to specific features
    df = df.reset_index(drop=True)

    assert all(
        [x in df.columns for x in metadata_features]
    ), "Metadata feature not found"
    assert all([x in df.columns for x in features]), "Profile feature not found"

    meta_df = df.loc[:, metadata_features]
    df = df.loc[:, features]

    # Convert and assert conversion success
    meta_df = assert_pandas_dtypes(df=meta_df, col_fix=np.str)
    df = assert_pandas_dtypes(df=df, col_fix=np.float64)

    # Get pairwise metric matrix
    pair_df = get_pairwise_metric(df=df, similarity_metric=similarity_metric)

    # Convert pairwise matrix into metadata-labeled melted matrix
    output_df = process_melt(df=pair_df, meta_df=meta_df, eval_metric=eval_metric)

    return output_df
