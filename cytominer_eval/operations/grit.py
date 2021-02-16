"""Grit describes phenotype strength of replicate profiles along two distinct axes:

- Similarity to other perturbations that target the same larger group (e.g. gene, MOA),
  with respect to:
- Similarity to control perturbations
"""
import numpy as np
import pandas as pd
from typing import List

from .util import assign_replicates, calculate_grit, check_grit_replicate_summary_method
from cytominer_eval.transform.util import (
    set_pair_ids,
    set_grit_column_info,
    assert_melt,
)


def grit(
    similarity_melted_df: pd.DataFrame,
    control_perts: List[str],
    profile_col: str,
    replicate_group_col: str,
    replicate_summary_method: str = "mean",
) -> pd.DataFrame:
    r"""Calculate grit

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        a long pandas dataframe output from cytominer_eval.transform.metric_melt
    control_perts : list
        a list of control perturbations to calculate a null distribution
    profile_col : str
        the metadata column storing profile ids. The column can have unique or replicate
        identifiers.
    replicate_group_col : str
        the metadata column indicating a higher order structure (group) than the
        profile column. E.g. target gene vs. guide in a CRISPR experiment.
    replicate_summary_method : {'mean', 'median'}, optional
        how replicate z-scores to control perts are summarized. Defaults to "mean".

    Returns
    -------
    pandas.DataFrame
        A dataframe of grit measurements per perturbation
    """
    # Check if we support the provided summary method
    check_grit_replicate_summary_method(replicate_summary_method)

    # Determine pairwise replicates
    similarity_melted_df = assign_replicates(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=[profile_col, replicate_group_col],
    )

    # Check to make sure that the melted dataframe is full
    assert_melt(similarity_melted_df, eval_metric="grit")

    # Extract out specific columns
    pair_ids = set_pair_ids()
    profile_col_name = "{x}{suf}".format(
        x=profile_col, suf=pair_ids[list(pair_ids)[0]]["suffix"]
    )

    # Define the columns to use in the calculation
    column_id_info = set_grit_column_info(
        profile_col=profile_col, replicate_group_col=replicate_group_col
    )

    # Calculate grit for each perturbation
    grit_df = (
        similarity_melted_df.groupby(profile_col_name)
        .apply(
            lambda x: calculate_grit(
                replicate_group_df=x,
                control_perts=control_perts,
                column_id_info=column_id_info,
                replicate_summary_method=replicate_summary_method,
            )
        )
        .reset_index(drop=True)
    )

    return grit_df
