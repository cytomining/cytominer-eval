"""Enrichtment
"""
import numpy as np
import pandas as pd
from typing import List
import scipy

from .util import assign_replicates, calculate_grit, check_grit_replicate_summary_method
from cytominer_eval.transform.util import (
    set_pair_ids,
    set_grit_column_info,
    assert_melt,
)


def enrichment(
    similarity_melted_df: pd.DataFrame,
    replicate_groups: List[str],
    percentile: float,
) -> dict:
    """Calculate the enrichment score.

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        An elongated symmetrical matrix indicating pairwise correlations between
        samples. Importantly, it must follow the exact structure as output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.
    replicate_groups : List
        a list of metadata column names in the original profile dataframe to use as
        replicate columns.
    percentile :  float
        Determines what percentage of top connections used for the enrichment calculation.

    Returns
    -------
    pandas.DataFrame
        percentile, threshold, ods ration and p value
    """
    # threshold based on percentile of top connections
    threshold = similarity_melted_df.similarity_metric.quantile(percentile)

    replicate_truth_df = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    )
    # calculate the individual components of the contingency tables
    v11 = len(replicate_truth_df.query("group_replicate==True and similarity_metric>@threshold"))
    v12 = len(replicate_truth_df.query("group_replicate==False and similarity_metric>@threshold"))
    v21 = len(replicate_truth_df.query("group_replicate==True and similarity_metric<=@threshold"))
    v22 = len(replicate_truth_df.query("group_replicate==False and similarity_metric<=@threshold"))

    V = np.asarray([[v11, v12], [v21, v22]])
    #print(percentile, threshold)
    #print(V, np.sum(V))
    r = scipy.stats.fisher_exact(V, alternative="greater")
    result = {"percentile": percentile, "threshold": threshold, "ods_ratio": r[0], "p-value": r[1]}
    #df = pd.DataFrame(data=d)
    return result