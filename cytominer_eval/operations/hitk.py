"""Function to calculate the hits at k list and scores for a given similarity matrix.
"""
import pandas as pd
from typing import List, Union


from cytominer_eval.utils.hitk_utils import add_hit_rank, percentage_scores
from cytominer_eval.utils.operation_utils import assign_replicates
from cytominer_eval.utils.transform_utils import set_pair_ids, assert_melt


def hitk(
    similarity_melted_df: pd.DataFrame,
    replicate_groups: List[str],
    groupby_columns: List[str],
    percent_list: Union[int, List[int]],
) -> pd.DataFrame:
    """Calculate the hit@k hits list and percent scores.
    This function groups the similarity matrix by each sample (group_col) and by similarity score. It then determines the rank of each correct hit.
    A correct hit is a connection to another sample with the same replicate attributes (replicate_groups), for example the same MOA.

    Hit@k records all hits/indexes in a long list which can be used to create histogram plots or similar visualizations.

    The percent scores contain the number of hits above the expected random distribution at a given percentage.
    For example, at 5 percent we calculate how many hits are within the first 5 percent of classes (number of neighbors) and then subtract the expected number of hits.

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        An elongated symmetrical matrix indicating pairwise correlations between
        samples. Importantly, it must follow the exact structure as output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.

    replicate_groups : list or int
        a list of metadata column names in the original profile dataframe to use as replicate columns.

    groupby_columns: str
        group columns determine the columns over which the similarity_melted_df is grouped.
        Usually groupby_columns will span the full space of the input data
        such that drop_duplicates by the groupby_cols would not change the data.
        If you group over Metadata_plate for examples, you will get meaningless results.
        This can easily be seen from the fact that the percent score at 100 will be nonzero.

    percent_list : list or "all"
        A list of percentages at which to calculate the percent scores, ie the amount of hits below this percentage.
        If percent_list == "all" a full dict with the length of classes will be created.
        Percentages are given as integers, ie 50 means 50 %.

    Returns
    -------
    hits_list : list
        full list of all hits. Can be used for histogram plotting.
    percent_scores: dict
        dictionary of the percentage list and their corresponding percent scores (see percent score function).
    """
    # make sure percent_list is a list
    if type(percent_list) == int:
        percent_list = [percent_list]
    # check for correct input
    assert type(percent_list) == list or percent_list == "all", "input is incorrect"
    if type(percent_list) == list:
        assert max(percent_list) <= 100, "percentages must be smaller than 100"

    similarity_melted_df = assign_replicates(
        similarity_melted_df=similarity_melted_df, replicate_groups=replicate_groups
    )
    # Check to make sure that the melted dataframe is full
    assert_melt(similarity_melted_df, eval_metric="hitk")

    # Extract the name of the columns in the sim_df
    pair_ids = set_pair_ids()
    groupby_cols_suffix = [
        "{x}{suf}".format(x=x, suf=pair_ids[list(pair_ids)[0]]["suffix"])
        for x in groupby_columns
    ]

    # group the sim_df by the groupby_columns
    grouped = similarity_melted_df.groupby(groupby_cols_suffix)
    nr_of_groups = grouped.ngroups
    # Within each group, add the ranks of each connection to a new column
    similarity_melted_with_rank = grouped.apply(lambda x: add_hit_rank(x))

    # make a list of the ranks of correct connection (hits), ie where the group_replicate is true
    hits_list = similarity_melted_with_rank[
        similarity_melted_with_rank["group_replicate"] == True
    ]["rank"].tolist()

    # calculate the scores at each percentage
    percent_scores = percentage_scores(hits_list, percent_list, nr_of_groups)

    return hits_list, percent_scores
