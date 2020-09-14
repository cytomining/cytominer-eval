"""
Extract evaluation metrics from profiling experiments
"""

import warnings
import numpy as np
import pandas as pd
from typing import List, Union

from cytominer_eval.transform import metric_melt
from cytominer_eval.transform.util import check_replicate_groups
from cytominer_eval.operations import percent_strong, precision_recall, grit


def evaluate(
    profiles: pd.DataFrame,
    features: List[str],
    meta_features: List[str],
    replicate_groups: Union[List[str], dict],
    operation: str = "percent_strong",
    similarity_metric: str = "pearson",
    percent_strong_quantile: np.float = 0.95,
    precision_recall_k: int = 10,
    control_perts_grit: List[str] = ["None"],
):
    # Check replicate groups input
    check_replicate_groups(eval_metric=operation, replicate_groups=replicate_groups)

    # Melt the input profiles to long format
    similarity_melted_df = metric_melt(
        df=profiles,
        features=features,
        metadata_features=meta_features,
        similarity_metric=similarity_metric,
        eval_metric=operation,
    )

    # Perform the input operation
    if operation == "percent_strong":
        metric_result = percent_strong(
            similarity_melted_df=similarity_melted_df,
            replicate_groups=replicate_groups,
            quantile=percent_strong_quantile,
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
            control_perts=control_perts_grit,
            replicate_id=replicate_groups["replicate_id"],
            group_id=replicate_groups["group_id"],
        )

    return metric_result
