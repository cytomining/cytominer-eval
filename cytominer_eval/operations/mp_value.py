"""Functions to calculate multidimensional perturbation values (mp-value)

mp-value describes the distance, in dimensionality-reduced space, between a perturbation
and a control [1]_.

References
----------

.. [1] Hutz, J. et al. "The Multidimensional Perturbation Value: A Single Metric to
   Measure Similarity and Activity of Treatments in High-Throughput Multidimensional
   Screens" Journal of Biomolecular Screening, Volume: 18 issue: 4, page(s): 367-377.
   doi: 10.1177/1087057112469257
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
    params: dict = {},
) -> pd.DataFrame:
    """Calculate multidimensional perturbation value (mp-value) [1]_.

    Parameters
    ----------
    df : pandas.DataFrame
        profiles with measurements per row and features or metadata per column.
    control_perts : list
        The control perturbations against which the distances will be computed.
    replicate_id : str
        The metadata identifier marking which column tracks control and replicate perts.
    features : list
        columns containing numerical features to be used for the mp-value computation
    params : dict, optional
        Optional parameters provided. See list of parameters in
        :py:func:`cytominer_eval.operations.util.default_mp_value_parameters`

    Returns
    -------
    pd.DataFrame
        mp-values per perturbation.
    """

    assert replicate_id in df.columns, "replicate_id not found in dataframe columns"

    # Extract features for control rows
    control_df = df.loc[df.loc[:, replicate_id].isin(control_perts), features]

    # Calculate mp_value for each perturbation
    mp_value_df = pd.DataFrame(
        df.groupby(replicate_id).apply(
            lambda x: calculate_mp_value(x[features], control_df, params)
        ),
        columns=["mp_value"],
    )

    mp_value_df.reset_index(inplace=True)

    return mp_value_df
