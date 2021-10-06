"""Function to calculate the hits at k score for a given similarity matrix.
"""
import pandas as pd


from cytominer_eval.utils.hitk_utils import index_hits, percentage_scores


def hitk(similarity_melted_df: pd.DataFrame, percent_list: list,) -> pd.DataFrame:
    """Calculate the hit@k index and scores.
    The similarity matrix is sorted by compounds and for each compound, the index of the MOA repicates are determined
    All indexes are recorded in a long list which can be used to create histogram plots or similar visualizations.
    The percent scores holds the number of indexes above the expected random distribution at a given percentage.

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        An elongated symmetrical matrix indicating pairwise correlations between
        samples. Importantly, it must follow the exact structure as output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.

    percent_list : list or "all"
        A list of percentages at which to calculate the percent scores, ie the amount of indexes below this percentage.
        If percent_list == "all" a full dict with the length of classes will be created.

    Returns
    -------
    indexes : list
        full list of all indexes. Can be used for histogram plotting
    percent_scores: dict
        percentages or in and their corresponding score
    """
    # check for correct input
    assert type(percent_list) == list or percent_list == "all", "input is incorrect"
    if type(percent_list) == list:
        assert max(percent_list) <= 100, "percentages must be smaller than 100"

    # group by MOA and then add a `rank` and a `same_moa` column to the df
    grouped = similarity_melted_df.groupby("pair_a_index")
    index_df = grouped.apply(index_hits)

    # rename columns for convenience
    index_df.rename(
        columns={
            "Metadata_broad_sample_pair_a": "target_compound",
            "Metadata_broad_sample_pair_b": "match_compound",
            "Metadata_moa_pair_a": "target_moa",
            "Metadata_moa_pair_b": "match_moa",
        },
        inplace=True,
    )

    # make a list of all index where the MOAs was the same
    indexes = index_df[index_df["same_moa"] == True]["rank"].tolist()

    # calculate the scores at each percentage
    percent_scores = percentage_scores(similarity_melted_df, indexes, percent_list)

    return indexes, percent_scores
