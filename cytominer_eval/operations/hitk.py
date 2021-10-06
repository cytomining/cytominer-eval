"""Function to calculate the enrichment score for a given similarity matrix.
"""
import numpy as np
import pandas as pd

from cytominer_eval.utils.hitk_utils import index_hits, percentage_scores


def hitk(
        similarity_melted_df: pd.DataFrame,
        percent_list: list,
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

    It is assumed that the similarity_melted_df is unique in Metadata_broad_sample_pair_a and Metadata_pert_well_pair_a

    Returns
    -------
    dict
        percentile, threshold, odds ratio and p value
    """
    # check if input is correct
    if sorted(percent_list) != percent_list:
        print("percent list is not ascending.")

    # group by MOA and then add a rank and a same_moa column to the df
    grouped = similarity_melted_df.groupby('pair_a_index')
    index_df = grouped.apply(index_hits)

    # rename columns for convenience
    index_df.rename(columns={'Metadata_broad_sample_pair_a': 'target_compound',
                             'Metadata_broad_sample_pair_b': 'match_compound', 'Metadata_moa_pair_a': 'target_moa',
                             'Metadata_moa_pair_b': 'match_moa'}, inplace=True)

    # make a list of all index where the MOA was the same
    indexes = index_df[index_df['same_moa'] == True]['rank'].tolist()

    percent_scores = percentage_scores(similarity_melted_df, indexes, percent_list)


    return indexes, percent_scores
