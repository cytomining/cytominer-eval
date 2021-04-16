import os
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
from cytominer_eval.transform.util import get_available_similarity_metrics
from cytominer_eval import evaluate

example_gene_file = "SQ00014610_normalized_feature_select.csv.gz"
example_gene_file = pathlib.Path(
    "{file}/../example_data/gene/{eg}".format(
        file=os.path.dirname(__file__), eg=example_gene_file
    )
)
gene_profiles = pd.read_csv(example_gene_file)

gene_meta_features = [
    x
    for x in gene_profiles.columns
    if (x.startswith("Metadata_") or x.startswith("Image_"))
]
gene_features = gene_profiles.drop(gene_meta_features, axis="columns").columns.tolist()
gene_groups = ["Metadata_gene_name", "Metadata_pert_name"]

example_compound_file = "SQ00015054_normalized_feature_select.csv.gz"
example_compound_file = pathlib.Path(
    "{file}/../example_data/compound/{eg}".format(
        file=os.path.dirname(__file__), eg=example_compound_file
    )
)
compound_profiles = pd.read_csv(example_compound_file)

compound_meta_features = [
    x for x in compound_profiles.columns if x.startswith("Metadata_")
]
compound_features = compound_profiles.drop(
    compound_meta_features, axis="columns"
).columns.tolist()
compound_groups = ["Metadata_broad_sample", "Metadata_mg_per_ml"]


def test_evaluate_replicate_reproducibility():
    similarity_metrics = get_available_similarity_metrics()
    replicate_reproducibility_quantiles = [0.5, 0.95]

    expected_result = {
        "gene": {
            "pearson": {"0.5": 0.431, "0.95": 0.056},
            "kendall": {"0.5": 0.429, "0.95": 0.054},
            "spearman": {"0.5": 0.429, "0.95": 0.055},
        },
        "compound": {
            "pearson": {"0.5": 0.681, "0.95": 0.458},
            "kendall": {"0.5": 0.679, "0.95": 0.463},
            "spearman": {"0.5": 0.679, "0.95": 0.466},
        },
    }

    for sim_metric in similarity_metrics:
        for quant in replicate_reproducibility_quantiles:
            gene_res = evaluate(
                profiles=gene_profiles,
                features=gene_features,
                meta_features=gene_meta_features,
                replicate_groups=gene_groups,
                operation="replicate_reproducibility",
                replicate_reproducibility_return_median_cor=False,
                similarity_metric=sim_metric,
                replicate_reproducibility_quantile=quant,
            )

            compound_res = evaluate(
                profiles=compound_profiles,
                features=compound_features,
                meta_features=compound_meta_features,
                replicate_groups=compound_groups,
                operation="replicate_reproducibility",
                replicate_reproducibility_return_median_cor=False,
                similarity_metric=sim_metric,
                replicate_reproducibility_quantile=quant,
            )

            assert (
                np.round(gene_res, 3) == expected_result["gene"][sim_metric][str(quant)]
            )
            assert (
                np.round(compound_res, 3)
                == expected_result["compound"][sim_metric][str(quant)]
            )


def test_evaluate_replicate_reprod_return_cor_true():
    reprod, med_cor_df = evaluate(
        profiles=gene_profiles,
        features=gene_features,
        meta_features=gene_meta_features,
        replicate_groups=gene_groups,
        operation="replicate_reproducibility",
        replicate_reproducibility_return_median_cor=True,
        similarity_metric="pearson",
        replicate_reproducibility_quantile=0.95,
    )

    med_cor_df = med_cor_df.sort_values(by="similarity_metric", ascending=False)

    assert np.round(reprod, 3) == 0.056

    top_genes = med_cor_df.Metadata_gene_name[0:5].tolist()
    assert top_genes == ["CDK2", "CCNE1", "ATF4", "KIF11", "CCND1"]

    assert np.round(med_cor_df.similarity_metric.max(), 3) == 0.949
    assert sorted(med_cor_df.columns.tolist()) == sorted(
        [
            "Metadata_gene_name",
            "Metadata_pert_name",
            "similarity_metric",
        ]
    )


def test_evaluate_precision_recall():
    ks = [1, 5, 10, 50, 5000]
    expected_result = {
        "gene": {
            "precision": {"1": 1, "5": 0, "10": 0, "50": 0, "5000": 0},
            "recall": {"1": 0, "5": 3, "10": 4, "50": 14, "5000": 118},
        },
        "compound": {
            "precision": {"1": 18, "5": 9, "10": 5, "50": 2, "5000": 0},
            "recall": {"1": 0, "5": 0, "10": 0, "50": 1, "5000": 58},
        },
    }

    for k in ks:

        result = evaluate(
            profiles=gene_profiles,
            features=gene_features,
            meta_features=gene_meta_features,
            replicate_groups=gene_groups,
            operation="precision_recall",
            similarity_metric="pearson",
            precision_recall_k=k,
        )

        assert (
            result.query("precision == 1").shape[0]
            == expected_result["gene"]["precision"][str(k)]
        )
        assert (
            result.query("recall == 1").shape[0]
            == expected_result["gene"]["recall"][str(k)]
        )

        result = evaluate(
            profiles=compound_profiles,
            features=compound_features,
            meta_features=compound_meta_features,
            replicate_groups=["Metadata_broad_sample"],
            operation="precision_recall",
            similarity_metric="pearson",
            precision_recall_k=k,
        )

        assert (
            result.query("precision == 1").shape[0]
            == expected_result["compound"]["precision"][str(k)]
        )
        assert (
            result.query("recall == 1").shape[0]
            == expected_result["compound"]["recall"][str(k)]
        )


def test_evaluate_grit():
    grit_gene_control_perts = [
        "Chr2-1",
        "Chr2-2",
        "Chr2-3",
        "Chr2-4",
        "Chr2-5",
        "Chr2-6",
        "Luc-1",
        "Luc-2",
        "LacZ-2",
        "LacZ-3",
    ]

    grit_gene_replicate_groups = {
        "profile_col": "Metadata_pert_name",
        "replicate_group_col": "Metadata_gene_name",
    }

    grit_results_df = evaluate(
        profiles=gene_profiles,
        features=gene_features,
        meta_features=gene_meta_features,
        replicate_groups=grit_gene_replicate_groups,
        operation="grit",
        grit_control_perts=grit_gene_control_perts,
        grit_replicate_summary_method="median",
    )

    top_result = (
        grit_results_df.sort_values(by="grit", ascending=False)
        .reset_index(drop=True)
        .iloc[
            0,
        ]
    )
    assert np.round(top_result.grit, 4) == 2.3352
    assert top_result.group == "PTK2"
    assert top_result.perturbation == "PTK2-2"

    grit_compound_replicate_groups = {
        "profile_col": "Metadata_broad_sample",
        "replicate_group_col": "Metadata_moa",
    }

    grit_compound_control_perts = ["DMSO"]

    grit_results_df = evaluate(
        profiles=compound_profiles,
        features=compound_features,
        meta_features=compound_meta_features,
        replicate_groups=grit_compound_replicate_groups,
        operation="grit",
        grit_control_perts=grit_compound_control_perts,
        grit_replicate_summary_method="mean",
    )

    top_result = (
        grit_results_df.sort_values(by="grit", ascending=False)
        .reset_index(drop=True)
        .iloc[
            0,
        ]
    )

    assert np.round(top_result.grit, 4) == 0.9990
    assert top_result.group == "ATPase inhibitor"
    assert top_result.perturbation == "BRD-A94756469-001-04-7"

    with pytest.raises(AssertionError) as ae:
        grit_results_df = evaluate(
            profiles=compound_profiles,
            features=compound_features,
            meta_features=compound_meta_features,
            replicate_groups=compound_groups,
            operation="grit",
            grit_control_perts=grit_compound_control_perts,
        )
    assert "For grit, replicate_groups must be a dict" in str(ae.value)


def test_evaluate_mp_value():
    # Permutations in mp_value could lead to some edge cases
    np.random.seed(2020)

    # Tests on genetic screen dataset
    mp_value_gene_control_perts = [
        "Chr2-1",
        "Chr2-2",
        "Chr2-3",
        "Chr2-4",
        "Chr2-5",
        "Chr2-6",
        "Luc-1",
        "Luc-2",
        "LacZ-2",
        "LacZ-3",
    ]

    mp_value_gene_replicate_groups = "Metadata_pert_name"

    mp_value_results_df = evaluate(
        profiles=gene_profiles,
        features=gene_features,
        meta_features=gene_meta_features,
        replicate_groups=mp_value_gene_replicate_groups,
        operation="mp_value",
        grit_control_perts=mp_value_gene_control_perts,
    )

    # Negative controls should be "close to themselves"
    assert all(
        mp_value_results_df[
            [
                x in mp_value_gene_control_perts
                for x in mp_value_results_df.Metadata_pert_name
            ]
        ].mp_value
        >= 0.05
    )

    # Strong perturbation should differ from controls
    assert "PTK2-2" in list(
        mp_value_results_df[mp_value_results_df.mp_value == 0].Metadata_pert_name
    )

    assert all(
        mp_value_results_df.columns == [mp_value_gene_replicate_groups, "mp_value"]
    )

    # Tests on chemical screen dataset
    mp_value_compound_control_perts = ["DMSO"]
    mp_value_compound_replicate_groups = "Metadata_broad_sample"

    mp_value_results_df = evaluate(
        profiles=compound_profiles,
        features=compound_features,
        meta_features=compound_meta_features,
        replicate_groups=mp_value_compound_replicate_groups,
        operation="mp_value",
        grit_control_perts=mp_value_compound_control_perts,
    )

    # Negative controls should be "close to themselves"
    assert all(
        mp_value_results_df[
            [
                x in mp_value_compound_control_perts
                for x in mp_value_results_df.Metadata_broad_sample
            ]
        ].mp_value
        >= 0.05
    )

    # Strong perturbation should differ from controls
    assert "BRD-A94756469-001-04-7" in list(
        mp_value_results_df[mp_value_results_df.mp_value == 0].Metadata_broad_sample
    )

    assert all(
        mp_value_results_df.columns == [mp_value_compound_replicate_groups, "mp_value"]
    )

    with pytest.raises(AssertionError) as ae:
        mp_value_results_df = evaluate(
            profiles=compound_profiles,
            features=compound_features,
            meta_features=compound_meta_features,
            replicate_groups=mp_value_compound_replicate_groups,
            operation="mp_value",
            grit_control_perts=mp_value_compound_control_perts,
            mp_value_params={"something else": 1},
        )
    assert "Unknown parameters provided. Only" in str(ae.value)
