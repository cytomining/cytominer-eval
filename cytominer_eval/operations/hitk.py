"""Function to calculate the enrichment score for a given similarity matrix.
"""
import numpy as np
import pandas as pd

from cytominer_eval.utils.hitk_utils import index_hits, count_hits


def hitk(
    similarity_melted_df: pd.DataFrame,
    only_first = False,
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

    Returns
    -------
    dict
        percentile, threshold, odds ratio and p value
    """
    # MOAs must have more than one compound, so DMSO needs to be deleted from the melted_df
    con = similarity_melted_df[similarity_melted_df["Metadata_broad_sample_pair_a"] != 'DMSO']

    grouped = similarity_melted_df.groupby(['Metadata_broad_sample_pair_a'])

    # add a rank and a same_moa column to the df
    index_df = grouped.apply(lambda x: index_hits(x))
    index_df.reset_index(drop=True, inplace=True)
    # rename columns for convinience
    index_df.rename(columns={'Metadata_broad_sample_pair_a': 'target_compound',
                            'Metadata_broad_sample_pair_b': 'match_compound', 'Metadata_moa_pair_a': 'target_moa',
                            'Metadata_moa_pair_b': 'match_moa'}, inplace=True)

    indexes = count_hits(index_df, only_first=only_first)
    # fully flatten the list
    indexes = [item for sublist in indexes.tolist() for item in sublist]

    # ran_index_flat = random_gen(con, meta_features)

    d = {}
    for n in [50, 75, 100]:
        d[n] = len([i for i in indexes if i < n]) - len([i for i in ran_index_flat if i < n])

    return indexes