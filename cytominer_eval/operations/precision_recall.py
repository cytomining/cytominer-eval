"""Functions to calculate precision and recall at a given k."""

import pandas as pd
from typing import List, Union

from cytominer_eval.utils.precisionrecall_utils import calculate_precision_recall
from cytominer_eval.utils.operation_utils import assign_replicates
from cytominer_eval.utils.transform_utils import set_pair_ids, assert_melt


def precision_recall(
    similarity_melted_df: pd.DataFrame,
    replicate_groups: List[str],
    groupby_columns: List[str],
    k: Union[int, List[int]],
) -> pd.DataFrame:
    """Determine the precision and recall at k for all unique groupby_columns samples
    based on a predefined similarity metric (see cytominer_eval.transform.metric_melt)

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        An elongated symmetrical matrix indicating pairwise correlations between
        samples. Importantly, it must follow the exact structure as output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.
    replicate_groups : List
        a list of metadata column names in the original profile dataframe to use as replicate columns.
    groupby_columns : List of str
        Column by which the similarity matrix is grouped and by which the precision/recall is calculated.
        For example, if groupby_column = Metadata_sample then the precision is calculated for each sample.
        Calculating the precision by sample is the default
        but it is mathematically not incorrect to calculate the precision at the MOA level.
        This is just less intuitive to understand.
    k : List of ints or int
        an integer indicating how many pairwise comparisons to threshold.

    Returns
    -------
    pandas.DataFrame
        precision and recall metrics for all groupby_column groups given k
    """
    # Determine pairwise replicates and make sure to sort based on the metric!
    similarity_melted_df = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    ).sort_values(by="similarity_metric", ascending=False)

    # Check to make sure that the melted dataframe is full
    assert_melt(similarity_melted_df, eval_metric="precision_recall")

    # Extract out specific columns
    pair_ids = set_pair_ids()
    groupby_cols_suffix = [
        "{x}{suf}".format(x=x, suf=pair_ids[list(pair_ids)[0]]["suffix"])
        for x in groupby_columns
    ]
    # iterate over all k
    precision_recall_df = pd.DataFrame()
    if type(k) == int:
        k = [k]
    for k_ in k:
        # Calculate precision and recall for all groups
        precision_recall_df_at_k = similarity_melted_df.groupby(
            groupby_cols_suffix
        ).apply(lambda x: calculate_precision_recall(x, k=k_))
        precision_recall_df = precision_recall_df.append(precision_recall_df_at_k)

    # Rename the columns back to the replicate groups provided
    rename_cols = dict(zip(groupby_cols_suffix, groupby_columns))

    return precision_recall_df.reset_index().rename(rename_cols, axis="columns")
