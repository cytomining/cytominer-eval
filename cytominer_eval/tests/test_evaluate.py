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


def test_evaluate_percent_strong():
    similarity_metrics = get_available_similarity_metrics()
    percent_strong_quantiles = [0.5, 0.95]

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
        for quant in percent_strong_quantiles:
            gene_res = evaluate(
                profiles=gene_profiles,
                features=gene_features,
                meta_features=gene_meta_features,
                replicate_groups=gene_groups,
                operation="percent_strong",
                similarity_metric=sim_metric,
                percent_strong_quantile=quant,
            )

            compound_res = evaluate(
                profiles=compound_profiles,
                features=compound_features,
                meta_features=compound_meta_features,
                replicate_groups=compound_groups,
                operation="percent_strong",
                similarity_metric=sim_metric,
                percent_strong_quantile=quant,
            )

            assert (
                np.round(gene_res, 3) == expected_result["gene"][sim_metric][str(quant)]
            )
            assert (
                np.round(compound_res, 3)
                == expected_result["compound"][sim_metric][str(quant)]
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
