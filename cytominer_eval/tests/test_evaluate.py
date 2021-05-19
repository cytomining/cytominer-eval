import os
import pytest
import pathlib
import tempfile
import numpy as np
import pandas as pd
from cytominer_eval import evaluate

from cytominer_eval.utils.availability_utils import get_available_similarity_metrics


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
