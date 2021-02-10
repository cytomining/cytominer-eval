"""Functions to calculate replicate reproducibility
"""

import numpy as np
import pandas as pd
from typing import List

from .util import assign_replicates, set_pair_ids
from cytominer_eval.transform.util import assert_melt


def replicate_reproducibility(
    similarity_melted_df: pd.DataFrame,
    replicate_groups: List[str],
    quantile_over_null: np.float = 0.95,
    return_median_correlations: bool = False,
) -> np.float:
    r"""Summarize pairwise replicate correlations

    For a given pairwise similarity matrix, replicate information, and specific options,
    output a replicate correlation summary.

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        An elongated symmetrical matrix indicating pairwise correlations between
        samples. Importantly, it must follow the exact structure as output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.
    replicate_groups : list
        A list of metadata column names in the original profile dataframe to indicate
        replicate samples.
    quantile_over_null : float, optional
        A float between 0 and 1 indicating the threshold of nonreplicates to use when
        reporting percent matching or percent replicating. Defaults to 0.95.
    return_median_correlations : bool, optional
        If provided, also return median pairwise correlations per replicate.
        Defaults to False.

    Returns
    -------
    {float, (float, pd.DataFrame)}
        The replicate reproducibility of the profiles according to the replicate
        columns provided. If `return_median_correlations = True` then the function will
        return both the metric and a median pairwise correlation pandas.DataFrame.
    """

    assert (
        0 < quantile_over_null and 1 >= quantile_over_null
    ), "quantile_over_null must be between 0 and 1"

    similarity_melted_df = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    )

    # Check to make sure that the melted dataframe is upper triangle
    assert_melt(similarity_melted_df, eval_metric="replicate_reproducibility")

    # check that there are group_replicates (non-unique rows)
    replicate_df = similarity_melted_df.query("group_replicate")
    denom = replicate_df.shape[0]

    assert denom != 0, "no replicate groups identified in {rep} columns!".format(
        rep=replicate_groups
    )

    non_replicate_quantile = similarity_melted_df.query(
        "not group_replicate"
    ).similarity_metric.quantile(quantile_over_null)

    replicate_reproducibility = (
        replicate_df.similarity_metric > non_replicate_quantile
    ).sum() / denom

    if return_median_correlations:
        pair_ids = set_pair_ids()
        replicate_groups_for_groupby = {
            "{col}{suf}".format(col=x, suf=pair_ids["pair_a"]["suffix"]): x
            for x in replicate_groups
        }

        median_cor_df = (
            replicate_df.groupby(list(replicate_groups_for_groupby))[
                "similarity_metric"
            ]
            .median()
            .reset_index()
            .rename(replicate_groups_for_groupby, axis="columns")
        )

        return (replicate_reproducibility, median_cor_df)

    return replicate_reproducibility
