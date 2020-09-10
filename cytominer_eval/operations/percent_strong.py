"""
Functions to calculate replicate reproducibility
"""

import numpy as np
import pandas as pd
from typing import List

from .util import assign_replicates


def percent_strong(
    similarity_melted_df: pd.DataFrame,
    replicate_groups: List[str],
    quantile: np.float = 0.95,
) -> np.float:
    """
    Determine the percentage of "strong phenotypes" based on replicate reproducibility

    Arguments:
    similarity_melted_df - a long pandas dataframe output from transform.metric_melt
    replicate_groups - a list of metadata column names in the original profile dataframe
                       to use as replicate columns
    quantile - float between 0 and 1 to indicate the proportion of replicates greater
               the given quantile of non-replicate similarity

    Output:
    A metric describing the proportion of replicates that correlate above the given
    quantile of non-replicate correlation distribution
    """
    assert 0 < quantile and 1 >= quantile, "quantile must be between 0 and 1"

    similarity_melted_df = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    )

    non_replicate_quantile = similarity_melted_df.query(
        "not group_replicate"
    ).metric.quantile(quantile)

    replicate_df = similarity_melted_df.query("group_replicate")
    denom = replicate_df.shape[0]

    percent_strong = (replicate_df.metric > non_replicate_quantile).sum() / denom

    return percent_strong
