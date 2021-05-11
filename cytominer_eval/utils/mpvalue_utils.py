import numpy as np
import pandas as pd
from typing import List, Union

from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

from cytominer_eval.utils.transform_utils import set_pair_ids


class MahalanobisEstimator:
    """
    Store location and dispersion estimators of the empirical distribution of data
    provided in an array and allow computation of statistical distances.

    Parameters
    ----------
    arr : {pandas.DataFrame, np.ndarray}
        the matrix used to calculate covariance

    Attributes
    ----------
    sigma : np.array
        Fitted covariance matrix of sklearn.covariance.EmpiricalCovariance()

    Methods
    -------
    mahalanobis(X)
        Computes mahalanobis distance between the input array (self.arr) and the X
        array as provided
    """

    def __init__(self, arr: Union[pd.DataFrame, np.ndarray]):
        self.sigma = EmpiricalCovariance().fit(arr)

    def mahalanobis(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute the mahalanobis distance between the empirical distribution described
        by this object and points in an array `X`.

        Parameters
        ----------
        X : {pandas.DataFrame, np.ndarray}
            A samples by features array-like matrix to compute mahalanobis distance
            between self.arr

        Returns
        -------
        numpy.array
            Mahalanobis distance between the input array and the original sigma
        """
        return self.sigma.mahalanobis(X)


def calculate_mahalanobis(pert_df: pd.DataFrame, control_df: pd.DataFrame) -> pd.Series:
    """Given perturbation and control dataframes, calculate mahalanobis distance per
    perturbation

    Usage: Designed to be called within a pandas.DataFrame().groupby().apply(). See
    :py:func:`cytominer_eval.operations.util.calculate_mp_value`.

    Parameters
    ----------
    pert_df : pandas.DataFrame
        A pandas dataframe of replicate perturbations (samples by features)
    control_df : pandas.DataFrame
        A pandas dataframe of control perturbations (samples by features). Must have the
        same feature measurements as pert_df

    Returns
    -------
    float
        The mahalanobis distance between perturbation and control
    """
    assert len(control_df) > 1, "Error! No control perturbations found."

    # Get dispersion and center estimators for the control perturbations
    control_estimators = MahalanobisEstimator(control_df)

    # Distance between mean of perturbation and control
    maha = control_estimators.mahalanobis(np.array(np.mean(pert_df, 0)).reshape(1, -1))[
        0
    ]
    return maha


def default_mp_value_parameters():
    """Set the different default parameters used for mp-values.

    Returns
    -------
    dict
        A default parameter set with keys: rescale_pca (whether the PCA should be
        scaled by variance explained) and nb_permutations (how many permutations to
        calculate empirical p-value). Defaults to True and 100, respectively.
    """
    params = {"rescale_pca": True, "nb_permutations": 100}
    return params


def calculate_mp_value(
    pert_df: pd.DataFrame,
    control_df: pd.DataFrame,
    params: dict = {},
) -> pd.Series:
    """Given perturbation and control dataframes, calculate mp-value per perturbation

    Usage: Designed to be called within a pandas.DataFrame().groupby().apply(). See
    :py:func:`cytominer_eval.operations.mp_value.mp_value`.

    Parameters
    ----------
    pert_df : pandas.DataFrame
        A pandas dataframe of replicate perturbations (samples by features)
    control_df : pandas.DataFrame
        A pandas dataframe of control perturbations (samples by features). Must have the
        same feature measurements as pert_df
    params : {dict}, optional
        the parameters to use when calculating mp value. See
        :py:func:`cytominer_eval.operations.util.default_mp_value_parameters`.

    Returns
    -------
    float
        The mp value for the given perturbation

    """
    assert len(control_df) > 1, "Error! No control perturbations found."

    # Assign parameters
    p = default_mp_value_parameters()

    assert all(
        [x in p.keys() for x in params.keys()]
    ), "Unknown parameters provided. Only {e} are supported.".format(e=p.keys())
    for (k, v) in params.items():
        p[k] = v

    merge_df = pd.concat([pert_df, control_df]).reset_index(drop=True)

    # We reduce the dimensionality with PCA
    # so that 90% of the variance is conserved
    pca = PCA(n_components=0.9, svd_solver="full")
    pca_array = pca.fit_transform(merge_df)
    # We scale columns by the variance explained
    if p["rescale_pca"]:
        pca_array = pca_array * pca.explained_variance_ratio_
    # This seems useless, as the point of using the Mahalanobis
    # distance instead of the Euclidean distance is to be independent
    # of axes scales

    # Distance between mean of perturbation and control
    obs = calculate_mahalanobis(
        pert_df=pca_array[: pert_df.shape[0]],
        control_df=pca_array[-control_df.shape[0] :],
    )
    # In the paper's methods section it mentions the covariance used
    # might be modified to include variation of the perturbation as well.

    # Permutation test
    sim = np.zeros(p["nb_permutations"])
    pert_mask = np.zeros(pca_array.shape[0], dtype=bool)
    pert_mask[: pert_df.shape[0]] = 1
    for i in range(p["nb_permutations"]):
        pert_mask_perm = np.random.permutation(pert_mask)
        pert_perm = pca_array[pert_mask_perm]
        control_perm = pca_array[np.logical_not(pert_mask_perm)]
        sim[i] = calculate_mahalanobis(pert_df=pert_perm, control_df=control_perm)

    return np.mean([x >= obs for x in sim])
