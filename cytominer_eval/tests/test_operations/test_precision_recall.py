import os
import random
import pathlib
import pandas as pd


from cytominer_eval.transform import metric_melt
from cytominer_eval.operations import precision_recall

random.seed(42)

# Load CRISPR dataset
example_file = "SQ00014610_normalized_feature_select.csv.gz"
example_file = pathlib.Path(
    "{file}/../../example_data/gene/{eg}".format(
        file=os.path.dirname(__file__), eg=example_file
    )
)

df = pd.read_csv(example_file)

meta_features = [
    x for x in df.columns if (x.startswith("Metadata_") or x.startswith("Image_"))
]
features = df.drop(meta_features, axis="columns").columns.tolist()

similarity_melted_df = metric_melt(
    df=df,
    features=features,
    metadata_features=meta_features,
    similarity_metric="pearson",
    eval_metric="precision_recall",
)

replicate_groups = ["Metadata_gene_name", "Metadata_cell_line"]

groupby_columns = ["Metadata_pert_name"]


def test_precision_recall():
    result_list = precision_recall(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        groupby_columns=groupby_columns,
        k=[5, 10],
    )

    result_int = precision_recall(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=replicate_groups,
        groupby_columns=groupby_columns,
        k=5,
    )

    assert len(result_list.k.unique()) == 2
    assert result_list.k.unique()[0] == 5

    # ITGAV-1 has a really strong profile
    assert (
        result_list.sort_values(by="recall", ascending=False)
        .reset_index(drop=True)
        .iloc[0, :]
        .Metadata_pert_name
        == "ITGAV-1"
    )

    assert all(x in result_list.columns for x in groupby_columns)

    assert result_int.equals(result_list.query("k == 5"))
