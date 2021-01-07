"""
Functions to calculate multidimensional perturbation values

This describes the distance in dimensionality-reduced space between a perturbation
and a control.
"""

import numpy as np
import pandas as pd
from typing import List

from .util import calculate_mp_value


def mp_value(
    df: pd.DataFrame,
    control_perts: List[str],
    replicate_id: str,
    features: List[str],
) -> pd.DataFrame:
    """
    Calculate multidimensional perturbation value (mp-value).
    See DOI: 10.1177/1087057112469257

    Arguments:
    df - a pandas dataframe with measurements on each row and features or metadata in
         each column
    control_perts - a list of control perturbations against which the distances will be
                    computed
    replicate_id - the metadata identifier marking which column tracks replicate perts
    features - columns containing numerical features to be used for the mp-value
               computation

    Output:
    A dataframe of mp-values per perturbation.
    """

    assert replicate_id in df.columns, "replicate_id not found in dataframe columns"

    # Extract features for control rows
    control_df = df[df[replicate_id].isin(control_perts)][features]

    # Calculate mp_value for each perturbation
    mp_value_df = pd.DataFrame(
        df.groupby(replicate_id).apply(
            lambda x: calculate_mp_value(x[features], control_df)
        ),
        columns=["mp_value"],
    )

    mp_value_df.reset_index(inplace=True)

    return mp_value_df
