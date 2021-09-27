import pandas as pd

def index_hits(df):
    df.reset_index(drop=True, inplace=True)
    df['rank'] = range(len(df))
    moa = df.Metadata_moa_pair_a.iloc[0]
    df['same_moa'] = df['Metadata_moa_pair_b'] == moa
    df.drop(columns=['pair_a_index', 'pair_b_index', 'similarity_metric'], inplace=True)
    df = df[['Metadata_broad_sample_pair_a', 'Metadata_moa_pair_a', 'Metadata_broad_sample_pair_b',
             'Metadata_moa_pair_b', 'rank', 'same_moa']]
    return df

def count_first(df: pd.DataFrame):
    """
    Intakes a dataframe with only
    Parameters
    ----------
    df :
    only_first :

    Returns
    -------

    """
    index_ls = df[df['same_moa'] == True]['rank'].tolist()
    if len(index_ls) == 0:
        print('Error. Only one compound for this moa: ', df.target_compound.iloc[0])
    return index_ls[0]

def count_hits(df: pd.DataFrame, only_first):
    if not only_first:
        index_ls = df[df['same_moa'] == True]['rank'].tolist()
        return index_ls
    else:
        group = df.groupby(['target_compound'])
        index_ls = group.apply(lambda x: count_first(x))
        return index_ls


