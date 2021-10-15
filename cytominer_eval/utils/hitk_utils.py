def add_hit_rank(df):
    """Adds the rank/index of each connection to the dataframe.
    This column will later be used to create a full list of hits.

    Parameters
    ----------
    df : sub-grouped dataframe from the similarity_melted_df

    Returns
    -------
    dataframe with added rank column

    """
    # rank all compounds by their similarity
    df = df.sort_values(["similarity_metric"], ascending=False)
    # and assign the index/order to the rank column
    df = df.assign(rank=range(len(df)))
    return df


def percentage_scores(hits_list, p_list, nr_of_groups):
    """Calculates the percent score which is the cumulative number of hits below a given percentage.
    The function counts the number of hits in the hits_list contains below a percentage of the maximum hit score (nr_of_groups).
    It then subtracts the expected value from that accumulated count value.
    such that random input should give scores around zero.
    If p_list = "all" then, instead of percentages, all classes are enumerated and hits counted.

    Parameters
    ----------
    hits_list : list
        long list of hits that correspond to index of the replicate in the list of neighbors
    p_list : list or 'all'
        list of percentages to score. Percentages are given as integers, ie 50 is 50%.
    nr_of_groups : int
        number of groups that add_hit_rank was applied to.
    Returns
    -------
    d : dict
        dictionary with percentages and scores or a full list of indexes and scores
    """
    # get the number of compounds in this dataset
    d = {}
    total_hits = len(hits_list)

    if p_list == "all":
        # diff is the accumulated difference between the amount of hits and the expected hit number per bins
        diff = 0
        # for a random distribution, we expect each bin to have an equal number of hits
        average_bin = total_hits / nr_of_groups
        for n in range(nr_of_groups):
            # count the number of hits that had the index n
            hits_n = hits_list.count(n)
            diff += hits_n - average_bin
            d[n] = diff

    else:
        # calculate the accumulated hit score at different percentages
        # the difference to the code above is that we now calculate the score for certain percentage points
        for p in p_list:
            # calculate the hits that are expected for a random distribution
            expected_hits = int(p * total_hits / 100)

            # calculate the value referring to the percentage of the full range of classes
            # in other words: how many hits are in the top X% of closest neighbors
            p_value = p * nr_of_groups / 100

            # calculate how many hits are below the p_value
            accumulated_hits_n = len([i for i in hits_list if i <= p_value])
            d[p] = accumulated_hits_n - expected_hits

            if p == 100 and d[p] != 0:
                print(
                    "The percent score at 100% is {}, it should be 0 tho. Check your groupby_columns".format(d[p])
                )

    return d
