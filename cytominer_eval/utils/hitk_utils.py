def index_hits(df):
    """Adds the rank or index of each connection to the dataframe.
    It then further drops and reorders unnecessary columns.

    Parameters
    ----------
    df : grouped sub dataframe from the similarity_melted_df

    Returns
    -------
    slimmed dataframe with rank and same_moa

    """
    # rank all compounds by their similarity
    df = df.sort_values(["similarity_metric"], ascending=False)
    df["rank"] = range(len(df))

    # add a column which is true if pair a and b have the same MOA
    moa = df.Metadata_moa_pair_a.iloc[0]
    df["same_moa"] = df["Metadata_moa_pair_b"] == moa

    # clean and reorder columns
    df = df[
        [
            "Metadata_broad_sample_pair_a",
            "Metadata_moa_pair_a",
            "Metadata_broad_sample_pair_b",
            "Metadata_moa_pair_b",
            "rank",
            "same_moa",
        ]
    ]
    df = df.reset_index(drop=True)
    return df


def percentage_scores(similarity_melted_df, indexes, p_list):
    """Calculates the number of indexes below a certain percentage.
    All of the scores are cleaned with the expected random distribution.
    If p_list = "all" then, instead of percentages, all classes are enumerated and indexes counted.

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        An elongated symmetrical matrix indicating pairwise correlations between
        samples
    indexes : list
        long list of all MOA hit indexes
    p_list : list or 'all'
        list of percentages to score

    Returns
    -------
    d : dict
        dictionary with percentages and scores or a full list of indexes and scores
    """
    # get the number of compounds in this dataset
    nr_of_classes = len(similarity_melted_df.pair_b_index.unique())
    d = {}
    total_hits = len(indexes)

    if p_list == "all":
        # diff is the accumulated difference between the index hits and the average bins
        diff = 0
        average_bin = total_hits / nr_of_classes
        for n in range(nr_of_classes):
            hits_n = indexes.count(n)
            diff += hits_n - average_bin
            d[n] = diff

    else:
        # calculate the accumulated hits at different percentages
        for p in p_list:
            # calculate the integers referring to the percentage of the full range of classes
            p_int = int(p * nr_of_classes / 100)
            # calculate the hits that are expected for a random distribution
            expected_hits = int(p * total_hits / 100)
            d[p] = len([i for i in indexes if i <= p_int]) - expected_hits

    return d
