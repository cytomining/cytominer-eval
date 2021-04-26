# Cytominer-eval: Evaluating quality of perturbation profiles

[![Actions Status](https://github.com/cytomining/cytominer-eval/workflows/Python%20build/badge.svg)](https://github.com/cytomining/cytominer-eval/actions)
[![Documentation Status](https://readthedocs.org/projects/cytominer-eval/badge/?version=latest)](https://cytominer-eval.readthedocs.io/en/latest/)
[![Coverage Status](https://codecov.io/gh/cytomining/cytominer-eval/branch/master/graph/badge.svg)](https://codecov.io/github/cytomining/cytominer-eval?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Cytominer-eval contains functions to calculate quality metrics for perturbation profiling experiments.

## Installation

Cytominer-eval can be installed via pip:

```bash
pip install cytominer-eval
```

Since the project is actively being developed, to get up to date functionality, you can also install via github commit hash:

```bash
# Example
pip install git+git://github.com/cytomining/cytominer-eval@f7f5b293da54d870e8ba86bacf7dbc874bb79565
```

## Usage

Cytominer-eval uses a simple API for all evaluation metrics.


```python
# Working example
import pandas as pd
from cytominer_eval import evaluate

# Load Data
commit = "6f9d350badd0a18b6c1a76171813aaf9a52f8d9f"
url = f"https://github.com/cytomining/cytominer-eval/raw/{commit}/cytominer_eval/example_data/compound/SQ00015054_normalized_feature_select.csv.gz"

df = pd.read_csv(url)

# Define important function arguments
meta_features = df.columns[df.columns.str.startswith("Metadata_")]
features = df.drop(meta_features, axis="columns").columns.tolist()
replicate_groups = ["Metadata_broad_sample", "Metadata_mg_per_ml"]

# Evaluate profile quality
evaluate(
    profiles=df,
    features=features,
    meta_features=meta_features,
    replicate_groups=replicate_groups,
    replicate_reproducibility_return_median_cor=False,
    operation="replicate_reproducibility",
)
```

## Metrics

Currently, five metric operations are supported:

1. Replicate reproducibility
2. Precision/recall
3. mp-value
4. Grit
5. Enrichment

## Demos

For more in depth tutorials, see https://github.com/cytomining/cytominer-eval/tree/master/demos.
