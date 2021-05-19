import numpy as np
import pandas as pd
from typing import List, Union

from cytominer_eval.utils.transform_utils import set_pair_ids


def assign_replicates(
    similarity_melted_df: pd.DataFrame,
    replicate_groups: List[str],
) -> pd.DataFrame:
    r"""Determine which profiles should be considered replicates.

    Given an elongated pairwise correlation matrix with metadata annotations, determine
    how to assign replicate information.

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        Long pandas DataFrame of annotated pairwise correlations output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.
    replicate_groups : list
        a list of metadata column names in the original profile dataframe used to
        indicate replicate profiles.

    Returns
    -------
    pd.DataFrame
        A similarity_melted_df but with added columns indicating whether or not the
        pairwise similarity metric is comparing replicates or not. Used in most eval
        operations.
    """
    pair_ids = set_pair_ids()
    replicate_col_names = {x: "{x}_replicate".format(x=x) for x in replicate_groups}

    compare_dfs = []
    for replicate_col in replicate_groups:
        replicate_cols_with_suffix = [
            "{col}{suf}".format(col=replicate_col, suf=pair_ids[x]["suffix"])
            for x in pair_ids
        ]

        assert all(
            [x in similarity_melted_df.columns for x in replicate_cols_with_suffix]
        ), "replicate_group not found in melted dataframe columns"

        replicate_col_name = replicate_col_names[replicate_col]

        compare_df = similarity_melted_df.loc[:, replicate_cols_with_suffix]
        compare_df.loc[:, replicate_col_name] = False

        compare_df.loc[
            np.where(compare_df.iloc[:, 0] == compare_df.iloc[:, 1])[0],
            replicate_col_name,
        ] = True
        compare_dfs.append(compare_df)

    compare_df = pd.concat(compare_dfs, axis="columns").reset_index(drop=True)
    compare_df = compare_df.assign(
        group_replicate=compare_df.loc[:, replicate_col_names.values()].min(
            axis="columns"
        )
    ).loc[:, list(replicate_col_names.values()) + ["group_replicate"]]

    similarity_melted_df = similarity_melted_df.merge(
        compare_df, left_index=True, right_index=True
    )
    return similarity_melted_df
