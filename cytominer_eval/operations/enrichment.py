"""Function to calculate the enrichment score for a given similarity matrix.
"""
import numpy as np
import pandas as pd
from typing import List, Union
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
    enrichment_percentile: Union[float, List[float]],
) -> pd.DataFrame:
    """Calculate the enrichment score. This score is based on the fisher exact odds score. Similar to the other functions, the closest connections are determined and checked with the replicates.
    This score effectively calculates how much better the distribution of correct connections is compared to random.

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        An elongated symmetrical matrix indicating pairwise correlations between
        samples. Importantly, it must follow the exact structure as output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.
    replicate_groups : List
        a list of metadata column names in the original profile dataframe to use as
        replicate columns.
    enrichment_percentile :  List of floats
        Determines what percentage of top connections used for the enrichment calculation.

    Returns
    -------
    dict
        percentile, threshold, odds ratio and p value
    """
    result = []
    replicate_truth_df = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    )
    # loop over all percentiles
    if type(percentile) == float:
        percentile = [percentile]
    for p in percentile:
        # threshold based on percentile of top connections
        threshold = similarity_melted_df.similarity_metric.quantile(p)

        # calculate the individual components of the contingency tables
        v11 = len(
            replicate_truth_df.query(
                "group_replicate==True and similarity_metric>@threshold"
            )
        )
        v12 = len(
            replicate_truth_df.query(
                "group_replicate==False and similarity_metric>@threshold"
            )
        )
        v21 = len(
            replicate_truth_df.query(
                "group_replicate==True and similarity_metric<=@threshold"
            )
        )
        v22 = len(
            replicate_truth_df.query(
                "group_replicate==False and similarity_metric<=@threshold"
            )
        )

        v = np.asarray([[v11, v12], [v21, v22]])
        r = scipy.stats.fisher_exact(v, alternative="greater")
        result.append(
            {
                "enrichment_percentile": p,
                "threshold": threshold,
                "ods_ratio": r[0],
                "p-value": r[1],
            }
        )
    result_df = pd.DataFrame(result)
    return result_df
