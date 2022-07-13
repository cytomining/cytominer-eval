import pandas as pd


def load_cell_health(commit="7d63d4a43014c757fd0c77c0fd1c19540f17cc3d"):
    base_url = f"https://github.com/broadinstitute/cell-health/raw/{commit}"

    url = f"{base_url}/1.generate-profiles/data/processed/cell_health_profiles_merged.tsv.gz"
    df = pd.read_csv(url, sep="\t")

    return df
