"""Functions to calculate grit

Grit describes phenotype strength of replicate profiles along two distinct axes:

- Similarity to other perturbations that target the same larger group (e.g. gene, MOA)
- Similarity to control perturbations
"""
import numpy as np
import pandas as pd
from typing import List

from .util import assign_replicates, calculate_grit
from cytominer_eval.transform.util import (
    set_pair_ids,
    set_grit_column_info,
    assert_melt,
)


def grit(
    similarity_melted_df: pd.DataFrame,
    control_perts: List[str],
    replicate_id: str,
    group_id: str,
) -> pd.DataFrame:
    r"""Calculate grit

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        a long pandas dataframe output from cytominer_eval.transform.metric_melt
    control_perts : list
        a list of control perturbations to calculate a null distribution
    replicate_id : str
        the metadata identifier marking which column tracks replicate perts
    group_id : str
        the metadata identifier marking which column tracks a higher order groups for
        all perturbations

    Returns
    -------
    A dataframe of grit measurements per perturbation
    """
    # Determine pairwise replicates
    similarity_melted_df = assign_replicates(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=[replicate_id, group_id],
    )

    # Check to make sure that the melted dataframe is full
    assert_melt(similarity_melted_df, eval_metric="grit")

    # Extract out specific columns
    pair_ids = set_pair_ids()
    replicate_col_name = "{x}{suf}".format(
        x=replicate_id, suf=pair_ids[list(pair_ids)[0]]["suffix"]
    )

    # Define the columns to use in the calculation
    column_id_info = set_grit_column_info(replicate_id=replicate_id, group_id=group_id)

    # Calculate grit for each perturbation
    grit_df = (
        similarity_melted_df.groupby(replicate_col_name)
        .apply(lambda x: calculate_grit(x, control_perts, column_id_info))
        .reset_index(drop=True)
    )

    return grit_df
