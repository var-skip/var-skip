# Variable Skipping for Autoregressive Range Density Estimation

This repo contains the code for reproducing the results for the variable
skipping paper.

## Downloading Datasets

IMPORTANT: This repo only includes the first 100 rows of each dataset. This is
sufficient to sanity check if the code runs, but to run real experiments you'll need
to download the original files and replace the samples in `datasets/`.

For Dryad-URLs, see: https://datadryad.org/stash/dataset/doi:10.5061/dryad.p8s0j

For Census, see: https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)

For KDD, see: https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html

For DMV-Full, see: https://catalog.data.gov/dataset/vehicle-snowmobile-and-boat-registrations

## Code Structure

- `datasets/`: folder of actual data.
- `datasets.py`: defines the dataset schemas and data loading code.
- `estimators.py`: defines the progressive sampling algorithm used for inference.
- `made.py`: defines the ResMADE model.
- `transformer.py`: defines the masked transformer model.
- `text_infer.py`: defines the code for pattern matching over text.
- `eval_model.py`: defines random query generation and evaluation.
- `train.py`: main script used to launch experiments and grid sweeps in a Ray cluster.

## Running Experiments

To set up a conda environment, run:

```bash
conda env create -f environment.yml
source activate varskip
```

To run training and evaluation with the natural column order, you can use
`./train.py dmv-full`, `./train.py kdd`, and `./train.py census`.

To run the full grid sweeps from the paper, use `./train.py --run dmv-full-final kdd-final census-final`.
For multi-order training, append `-mo` (e.g., `./train.py --run kdd-final-mo`).

Results are printed to stdout and also stored in `~/ray_results`. To analyze the quantiles of the results,
you can use the `summarize.py` script.
