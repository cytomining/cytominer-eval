def get_available_eval_metrics():
    """Output the available eval metrics in the cytominer_eval library"""
    return [
        "replicate_reproducibility",
        "precision_recall",
        "grit",
        "mp_value",
        "enrichment",
        "hitk",
    ]


def get_available_similarity_metrics():
    """Output the available metrics for calculating pairwise similarity"""
    return ["pearson", "kendall", "spearman"]


def get_available_summary_methods():
    """Output the available metrics for summarizing output scores"""
    return ["mean", "median"]


def get_available_distribution_compare_methods():
    """Output the available methods to summarize comparison of two distributions"""
    return ["zscore"]


def check_eval_metric(eval_metric: str) -> None:
    """Helper function to ensure that we support the input eval metric

    Parameters
    ----------
    eval_metric : str
        The user input eval metric

    Returns
    -------
    None
        Assertion will fail if we don't support the input eval metric
    """
    avail_metrics = get_available_eval_metrics()

    assert (
        eval_metric in avail_metrics
    ), "{eval} not supported. Select one of {avail}".format(
        eval=eval_metric, avail=avail_metrics
    )


def check_similarity_metric(similarity_metric: str) -> None:
    """Helper function to ensure that we support the input similarity metric

    Parameters
    ----------
    similarity_metric : str
        The user input similarity metric

    Returns
    -------
    None
        Assertion will fail if we don't support the input similarity metric
    """
    avail_metrics = get_available_similarity_metrics()

    assert (
        similarity_metric in avail_metrics
    ), "{m} not supported. Available similarity metrics: {avail}".format(
        m=similarity_metric, avail=avail_metrics
    )


def check_replicate_summary_method(replicate_summary_method: str) -> None:
    """Helper function to ensure that we support the user input replicate summary

    Parameters
    ----------
    replicate_summary_method : str
        The user input replicate summary method

    Returns
    -------
    None
        Assertion will fail if the user inputs an incorrect replicate summary method
    """
    avail_methods = get_available_summary_methods()

    if replicate_summary_method not in avail_methods:
        raise ValueError(
            "{input} method not supported. Select one of: {avail}".format(
                input=replicate_summary_method, avail=avail_methods
            )
        )


def check_compare_distribution_method(distribution_method: str) -> None:
    """Helper function to ensure that we support the user input distribution comparison
    method

    Parameters
    ----------
    distribution_method : str
        The user input distribution comparison method

    Returns
    -------
    None
        Assertion will fail if the user inputs an incorrect replicate summary method
    """
    avail_methods = get_available_distribution_compare_methods()

    assert (
        distribution_method in avail_methods
    ), "{m} not supported. Available distribution methods: {avail}".format(
        m=distribution_method, avail=avail_methods
    )
