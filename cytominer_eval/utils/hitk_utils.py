

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
    df = df.sort_values(['similarity_metric'], ascending=False) # inplace=True,
    # rank all compounds by their similarity
    df['rank'] = range(len(df))

    # add a column which is true if pair a and b have the same MOA
    moa = df.Metadata_moa_pair_a.iloc[0]
    df['same_moa'] = df['Metadata_moa_pair_b'] == moa

    # clean and reorder columns
    df = df[['Metadata_broad_sample_pair_a', 'Metadata_moa_pair_a', 'Metadata_broad_sample_pair_b',
             'Metadata_moa_pair_b', 'rank', 'same_moa']]
    df = df.reset_index(drop=True) #inplace=True
    return df


def percentage_scores(similarity_melted_df, indexes, p_list):
    """

    Parameters
    ----------
    similarity_melted_df :
    p_list :
    indexes :

    Returns
    -------

    """
    # get the number of compounds in this dataset
    nr_of_classes = len(similarity_melted_df.pair_b_index.unique())

    d = {}
    total_hits = len(indexes)

    # calculate the accumalated hits at different percentages
    for p in p_list:
        # calculate the integers referring to the percentage of the full range of classes
        p_int = int(p * nr_of_classes / 100)
        # calculate the hits that are expected for a random distribution
        expected_hits = int(p * total_hits / 100)
        d[p] = len([i for i in indexes if i <= p_int]) - expected_hits

    return d
