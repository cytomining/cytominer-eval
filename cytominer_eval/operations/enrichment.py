"""Function to calculate the enrichtment score for a given similarity matrix.
"""
import numpy as np
import pandas as pd
from typing import List
import scipy

from .util import assign_replicates, calculate_precision_recall


def enrichment(
    similarity_melted_df: pd.DataFrame,
    replicate_groups: List[str],
    percentile: float,
) -> dict:
    """Calculate the enrichment score. This score is based on the fisher exact odds score. Similar to the other functions, the closest connections are determined and checked with the replicates.
    This score effectively calculates how much better the distribution of correct connections is from a random sample.

    Parameters
    ----------x`
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
    dict
        percentile, threshold, ods ration and p value
    """
    # threshold based on percentile of top connections
    threshold = similarity_melted_df.similarity_metric.quantile(percentile)

    # adds the column of replicate truth to the elongated similarity df
    replicate_truth_df = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    )
    # calculate the individual components of the contingency tables
    c11 = len(replicate_truth_df.query("group_replicate==True and similarity_metric>@threshold"))
    c12 = len(replicate_truth_df.query("group_replicate==False and similarity_metric>@threshold"))
    c21 = len(replicate_truth_df.query("group_replicate==True and similarity_metric<=@threshold"))
    c22 = len(replicate_truth_df.query("group_replicate==False and similarity_metric<=@threshold"))

    # arrange values into the table and calculate the fisher scores.
    contingency_table = np.asarray([[c11, c12], [c21, c22]])
    r = scipy.stats.fisher_exact(contingency_table, alternative="greater")
    result = {"percentile": percentile, "threshold": threshold, "ods_ratio": r[0], "p-value": r[1]}
    return result
