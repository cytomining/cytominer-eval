def get_available_eval_metrics():
    r"""Output the available eval metrics in the cytominer_eval library"""
    return [
        "replicate_reproducibility",
        "precision_recall",
        "grit",
        "mp_value",
        "enrichment",
    ]


def get_available_similarity_metrics():
    r"""Output the available metrics for calculating pairwise similarity in the
    cytominer_eval library
    """
    return ["pearson", "kendall", "spearman", "euclidean"]


def get_available_grit_summary_methods():
    r"""Output the available metrics for calculating pairwise similarity in the
    cytominer_eval library
    """
    return ["mean", "median"]
