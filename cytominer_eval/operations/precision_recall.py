"""
Functions to calculate precision and recall at a given k
"""

import numpy as np
import pandas as pd
from typing import List

from .util import assign_replicates, calculate_precision_recall
from cytominer_eval.transform.util import set_pair_ids, assert_melt


def precision_recall(
    similarity_melted_df: pd.DataFrame,
    replicate_groups: List[str],
    k: int,
) -> pd.DataFrame:
    """Determine the precision and recall at k for all unique replicate groups
    based on a predefined similarity metric (see cytominer_eval.transform.metric_melt)

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        An elongated symmetrical matrix indicating pairwise correlations between
        samples. Importantly, it must follow the exact structure as output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.
    replicate_groups : List
        a list of metadata column names in the original profile dataframe to use as
        replicate columns.
    k : int
        an integer indicating how many pairwise comparisons to threshold.

    Returns
    -------
    pandas.DataFrame
        precision and recall metrics for all replicate groups given k
    """
    # Determine pairwise replicates and make sure to sort based on the metric!
    similarity_melted_df = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    ).sort_values(by="similarity_metric", ascending=False)

    # Check to make sure that the melted dataframe is full
    assert_melt(similarity_melted_df, eval_metric="precision_recall")

    # Extract out specific columns
    pair_ids = set_pair_ids()
    replicate_group_cols = [
        "{x}{suf}".format(x=x, suf=pair_ids[list(pair_ids)[0]]["suffix"])
        for x in replicate_groups
    ]

    # Calculate precision and recall for all groups
    precision_recall_df = similarity_melted_df.groupby(replicate_group_cols).apply(
        lambda x: calculate_precision_recall(x, k=k)
    )

    # Rename the columns back to the replicate groups provided
    rename_cols = dict(zip(replicate_group_cols, replicate_groups))

    return precision_recall_df.reset_index().rename(rename_cols, axis="columns")
