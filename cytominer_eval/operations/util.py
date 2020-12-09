import numpy as np
import pandas as pd
from typing import List

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

from cytominer_eval.transform import metric_melt
from cytominer_eval.transform.util import set_pair_ids


def assign_replicates(
    similarity_melted_df: pd.DataFrame, replicate_groups: List[str],
) -> pd.DataFrame:
    """
    Arguments:
    similarity_melted_df - a long pandas dataframe output from transform.metric_melt
    replicate_groups - a list of metadata column names in the original profile dataframe
                       to use as replicate columns

    Output:
    Adds columns to the similarity metric dataframe to indicate whether or not the
    pairwise similarity metric is comparing replicates or not
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


def calculate_precision_recall(replicate_group_df: pd.DataFrame, k: int) -> pd.Series:
    """
    Usage: Designed to be called within a pandas.DataFrame().groupby().apply()
    """
    assert (
        "group_replicate" in replicate_group_df.columns
    ), "'group_replicate' not found in dataframe; remember to call assign_replicates()."

    recall_denom__total_relevant_items = sum(replicate_group_df.group_replicate)
    precision_denom__num_recommended_items = k

    num_recommended_items_at_k = sum(replicate_group_df.iloc[:k,].group_replicate)

    precision_at_k = num_recommended_items_at_k / precision_denom__num_recommended_items
    recall_at_k = num_recommended_items_at_k / recall_denom__total_relevant_items

    return_bundle = {"k": k, "precision": precision_at_k, "recall": recall_at_k}

    return pd.Series(return_bundle)


def calculate_grit(
    replicate_group_df: pd.DataFrame, control_perts: List[str], column_id_info: dict
) -> pd.Series:
    """
    Usage: Designed to be called within a pandas.DataFrame().groupby().apply()
    """
    group_entry = get_grit_entry(replicate_group_df, column_id_info["group"]["id"])
    pert = get_grit_entry(replicate_group_df, column_id_info["replicate"]["id"])

    # Define distributions for control perturbations
    control_distrib = replicate_group_df.loc[
        replicate_group_df.loc[:, column_id_info["replicate"]["comparison"]].isin(
            control_perts
        ),
        "similarity_metric",
    ].values.reshape(-1, 1)

    assert len(control_distrib) > 1, "Error! No control perturbations found."

    # Define distributions for same group (but not same perturbation)
    same_group_distrib = replicate_group_df.loc[
        (
            replicate_group_df.loc[:, column_id_info["group"]["comparison"]]
            == group_entry
        )
        & (
            replicate_group_df.loc[:, column_id_info["replicate"]["comparison"]] != pert
        ),
        "similarity_metric",
    ].values.reshape(-1, 1)

    if len(same_group_distrib) == 0:
        return_bundle = {"perturbation": pert, "group": group_entry, "grit": np.nan}

    else:
        scaler = StandardScaler()
        scaler.fit(control_distrib)
        grit_z_scores = scaler.transform(same_group_distrib)
        grit = np.mean(grit_z_scores)

        return_bundle = {"perturbation": pert, "group": group_entry, "grit": grit}

    return pd.Series(return_bundle)


def get_grit_entry(df: pd.DataFrame, col: str) -> str:
    entries = df.loc[:, col]
    assert (
        len(entries.unique()) == 1
    ), "grit is calculated for each perturbation independently"
    return str(list(entries)[0])


class DistributionEstimator:
    def __init__(self, arr):
        self.mu = np.array(np.mean(arr, axis = 0))
        self.sigma = EmpiricalCovariance().fit(arr)
        
    def mahalanobis(self, X):
        return(self.sigma.mahalanobis(X - self.mu))


def calculate_mahalanobis(
    pert_df: pd.DataFrame, control_df: pd.DataFrame
    ) -> pd.Series:
    """
    Usage: Designed to be called within a pandas.DataFrame().groupby().apply()
    """
    assert len(control_df) > 1, "Error! No control perturbations found."
        
    # Get dispersion and center estimators for the 
    control_estimators = DistributionEstimator(control_df)
    
    # Distance between mean of perturbation and control
    maha = control_estimators.mahalanobis(
                np.array(np.mean(pert_df, 0)).reshape(1, -1))[0]
    return maha


def default_mp_value_parameters():    
    params = {"rescale_pca": True,
              "nb_permutations": 100}
    return(params)
    
    
def calculate_mp_value(
    pert_df: pd.DataFrame, control_df: pd.DataFrame,
    params = {}
    ) -> pd.Series:
    """
    Usage: Designed to be called within a pandas.DataFrame().groupby().apply()
    """
    assert len(control_df) > 1, "Error! No control perturbations found."
    
    # Assign parameters
    p = default_mp_value_parameters()
    assert (
        all([x in p.keys() for x in params.keys()])
    ), "Unknown parameters provided." 
    for (k,v) in params.items():
        p[k] = v
    
    merge_df = pert_df.append(control_df)
    
    # We reduce the dimensionality with PCA
    # so that 90% of the variance is conserved
    pca = PCA(n_components = 0.9, svd_solver = 'full')
    pca_array = pca.fit_transform(merge_df)
    # We scale columns by the variance explained
    if p["rescale_pca"]:
        pca_array = pca_array * pca.explained_variance_ratio_
    # NB: this seems useless, as the point of using the Mahalanobis
    # distance instead of the Euclidean distance is to be independent
    # of axes scales
    
    # Get dispersion and center estimators for the 
    control_estimators = DistributionEstimator(pca_array[-control_df.shape[0]:])
    
    # Distance between mean of perturbation and control
    obs = calculate_mahalanobis(pert_df = pca_array[:pert_df.shape[0]],
                                control_df = pca_array[-control_df.shape[0]:])
    # NB: in the paper's methods section it mentions the covariance used
    # might be modified to include variation of the perturbation as well.
    
    # Permutation test
    sim = np.zeros(p["nb_permutations"])
    pert_mask = np.zeros(pca_array.shape[0], dtype=bool)
    pert_mask[:pert_df.shape[0]] = 1
    for i in range(p["nb_permutations"]):
        pert_mask_perm = np.random.permutation(pert_mask)
        pert_perm = pca_array[pert_mask_perm]
        control_perm = pca_array[np.logical_not(pert_mask_perm)]
        sim[i] = calculate_mahalanobis(pert_df = pert_perm,
                                       control_df = control_perm)
        
    return np.mean([x >= obs for x in sim])