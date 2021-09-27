"""Function to calculate the enrichment score for a given similarity matrix.
"""
import numpy as np
import pandas as pd
from typing import List, Union
import scipy

from cytominer_eval.utils.operation_utils import assign_replicates
from cytominer_eval.utils.transform_utils import set_pair_ids, assert_melt


def hitk(
    similarity_melted_df: pd.DataFrame,
    replicate_groups: List[str],
) -> pd.DataFrame:
    """Calculate the enrichment score. This score is based on the fisher exact odds score.
    Similar to the other functions, the closest connections are determined and checked with the replicates.
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


    Returns
    -------
    dict
        percentile, threshold, odds ratio and p value
    """

    return result_df
