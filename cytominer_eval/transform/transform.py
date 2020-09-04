import numpy as np
import pandas as pd
from typing import List

from .util import assert_pandas_dtypes, get_upper_matrix

available_pairwise_metrics = ["pearson", "kendall", "spearman"]


def get_pairwise_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = assert_pandas_dtypes(df=df, col_fix=np.float64)

    assert (
        metric in available_pairwise_metrics
    ), f"{metric} not supported. Available metrics: {available_pairwise_metrics}"

    pair_df = df.transpose().corr(method=metric)

    # Check if the metric calculation went wrong
    # (Current pandas version makes this check redundant)
    if pair_df.shape == (0, 0):
        raise TypeError(
            "Something went wrong - check that 'features' are profile measurements"
        )

    return pair_df


def process_melt(df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:

    assert df.shape[0] == df.shape[1], "Matrix must be symmetrical"

    # Remove pairwise matrix diagonal and redundant pairwise comparisons
    upper_tri = get_upper_matrix(df)
    df = df.where(upper_tri)

    # Convert pairwise matrix to melted (long) version based on index value
    metric_unlabeled_df = (
        pd.melt(
            df.reset_index(),
            id_vars="index",
            value_vars=df.columns,
            var_name="pair_b_index",
            value_name="metric",
        )
        .dropna()
        .reset_index(drop=True)
        .rename({"index": "pair_a_index"}, axis="columns")
    )

    # Merge metadata on index for both comparison pairs
    output_df = meta_df.merge(
        meta_df.merge(metric_unlabeled_df, left_index=True, right_on="pair_b_index"),
        left_index=True,
        right_on="pair_a_index",
        suffixes=["_pair_a", "_pair_b"],
    )

    return output_df


def metric_melt(
    df: pd.DataFrame,
    features: List[str],
    metadata_features: List[str],
    metric: str = "pearson",
) -> pd.DataFrame:
    # Subset dataframes to specific features
    df = df.reset_index(drop=True)
    meta_df = df.loc[:, metadata_features]
    df = df.loc[:, features]

    # Convert and assert conversion success
    meta_df = assert_pandas_dtypes(df=meta_df, col_fix=np.str)
    df = assert_pandas_dtypes(df=df, col_fix=np.float64)

    # Get pairwise metric matrix
    pair_df = get_pairwise_metric(df=df, metric=metric)

    # Convert pairwise matrix into metadata-labeled melted matrix
    output_df = process_melt(df=pair_df, meta_df=meta_df)

    return output_df
