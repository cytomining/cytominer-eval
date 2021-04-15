"""Calculate evaluation metrics from profiling experiments.

The primary entrypoint into quickly evaluating profile quality.
"""
import warnings
import numpy as np
import pandas as pd
from typing import List, Union

from cytominer_eval.transform import metric_melt
from cytominer_eval.transform.util import check_replicate_groups
from cytominer_eval.operations import (
    replicate_reproducibility,
    precision_recall,
    grit,
    mp_value,
)


def evaluate(
    profiles: pd.DataFrame,
    features: List[str],
    meta_features: List[str],
    replicate_groups: Union[List[str], dict],
    operation: str = "replicate_reproducibility",
    similarity_metric: str = "pearson",
    replicate_reproducibility_quantile: np.float = 0.95,
    replicate_reproducibility_return_median_cor: bool = False,
    precision_recall_k: int = 10,
    grit_control_perts: List[str] = ["None"],
    grit_replicate_summary_method: str = "mean",
    mp_value_params: dict = {},
):
    r"""Evaluate profile quality and strength.

    For a given profile dataframe containing both metadata and feature measurement
    columns, use this function to calculate profile quality metrics. The function
    contains all the necessary arguments for specific evaluation operations.

    Parameters
    ----------
    profiles : pandas.DataFrame
        profiles must be a pandas DataFrame with profile samples as rows and profile
        features as columns. The columns should contain both metadata and feature
        measurements.
    features : list
        A list of strings corresponding to feature measurement column names in the
        `profiles` DataFrame. All features listed must be found in `profiles`.
    meta_features : list
        A list of strings corresponding to metadata column names in the `profiles`
        DataFrame. All features listed must be found in `profiles`.
    replicate_groups : {str, list, dict}
        An important variable indicating which metadata columns denote replicate
        information. All metric operations require replicate profiles.
        `replicate_groups` indicates a str or list of columns to use. For
        `operation="grit"`, `replicate_groups` is a dict with two keys: "profile_col"
        and "replicate_group_col". "profile_col" is the column name that stores
        identifiers for each profile (can be unique), while "replicate_group_col" is the
        column name indicating a higher order replicate information. E.g.
        "replicate_group_col" can be a gene column in a CRISPR experiment with multiple
        guides targeting the same genes. See also
        :py:func:`cytominer_eval.operations.grit` and
        :py:func:`cytominer_eval.transform.util.check_replicate_groups`.
    operation : {'replicate_reproducibility', 'precision_recall', 'grit', 'mp_value'}, optional
        The specific evaluation metric to calculate. The default is
        "replicate_reproducibility".
    similarity_metric: {'pearson', 'spearman', 'kendall'}, optional
        How to calculate pairwise similarity. Defaults to "pearson". We use the input
        in pandas.DataFrame.cor(). The default is "pearson".

    Returns
    -------
    float, pd.DataFrame
        The resulting evaluation metric. The return is either a single value or a pandas
        DataFrame summarizing the metric as specified in `operation`.

    Other Parameters
    -----------------------------
    replicate_reproducibility_quantile : {0.95, ...}, optional
        Only used when `operation='replicate_reproducibility'`. This indicates the
        percentile of the non-replicate pairwise similarity to consider a reproducible
        phenotype. Defaults to 0.95.
    replicate_reproducibility_return_median_cor : bool, optional
        Only used when `operation='replicate_reproducibility'`. If True, then also
        return pairwise correlations as defined by replicate_groups and
        similarity metric
    precision_recall_k : {10, ...}, optional
        Only used when `operation='precision_recall'`. Used to calculate precision and
        recall considering the top k profiles according to pairwise similarity.
    grit_control_perts : {None, ...}, optional
        Only used when `operation='grit'`. Specific profile identifiers used as a
        reference when calculating grit. The list entries must be found in the
        `replicate_groups[replicate_id]` column.
    grit_replicate_summary_method : {"mean", "median"}, optional
        Only used when `operation='grit'`. Defines how the replicate z scores are
        summarized. see
        :py:func:`cytominer_eval.operations.util.calculate_grit`
    mp_value_params : {{}, ...}, optional
        Only used when `operation='mp_value'`. A key, item pair of optional parameters
        for calculating mp value. See also
        :py:func:`cytominer_eval.operations.util.default_mp_value_parameters`
    """
    # Check replicate groups input
    check_replicate_groups(eval_metric=operation, replicate_groups=replicate_groups)

    if operation != "mp_value":
        # Melt the input profiles to long format
        similarity_melted_df = metric_melt(
            df=profiles,
            features=features,
            metadata_features=meta_features,
            similarity_metric=similarity_metric,
            eval_metric=operation,
        )

    # Perform the input operation
    if operation == "replicate_reproducibility":
        metric_result = replicate_reproducibility(
            similarity_melted_df=similarity_melted_df,
            replicate_groups=replicate_groups,
            quantile_over_null=replicate_reproducibility_quantile,
            return_median_correlations=replicate_reproducibility_return_median_cor,
        )
    elif operation == "precision_recall":
        metric_result = precision_recall(
            similarity_melted_df=similarity_melted_df,
            replicate_groups=replicate_groups,
            k=precision_recall_k,
        )
    elif operation == "grit":
        metric_result = grit(
            similarity_melted_df=similarity_melted_df,
            control_perts=grit_control_perts,
            profile_col=replicate_groups["profile_col"],
            replicate_group_col=replicate_groups["replicate_group_col"],
            replicate_summary_method=grit_replicate_summary_method,
        )
    elif operation == "mp_value":
        metric_result = mp_value(
            df=profiles,
            control_perts=grit_control_perts,
            replicate_id=replicate_groups,
            features=features,
            params=mp_value_params,
        )

    return metric_result
