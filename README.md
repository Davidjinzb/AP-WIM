# AP-WIM Anonymous Code Package

This anonymous package contains the core implementation and experiment drivers for Adaptive-Path Wavelet Interpretation Maps (AP-WIM). It is intended for anonymous review and focuses on the code needed to inspect the method and reproduce the main experiment workflows when the corresponding public datasets and model checkpoints are supplied by the user.

## Contents

- `core/lib/` and `core/src/`: wavelet attribution utilities and supporting model/evaluation code.
- `experiments/wam_experiments/`: AP-WIM attribution methods, baseline attribution methods, evaluation helpers, and experiment runners.
- `scorer_audit_experiments/`: scorer-level retrieval/ranking audit scripts used for the scorer-audit experiment.
- `requirements.txt` and `environment.yml`: Python dependencies.

Generated results, logs, datasets, pretrained checkpoints, local cache files, and machine-specific paths are intentionally excluded.

## Environment

A typical setup is:

```bash
conda env create -f environment.yml
conda activate wam
# or: pip install -r requirements.txt
```

The code expects PyTorch, torchvision, NumPy, SciPy, pandas, OpenCV, timm, ptwt, and matplotlib. GPU execution is optional for inspection but recommended for full attribution runs.

## Data and Checkpoints

Datasets and checkpoints are not included. Set paths with environment variables when running experiments, for example:

```bash
export APWIM_ESC50_ROOT=/path/to/ESC-50-master
export APWIM_TINY_IMAGENET_ROOT=/path/to/tiny-imagenet-200
export APWIM_WAM_EXPERIMENT_DIR=$PWD/experiments/wam_experiments
```

## Example Entrypoints

```bash
bash scorer_audit_experiments/run_tinyimagenet_retrieval_ranking.sh
bash scorer_audit_experiments/run_retrieval_score_audit.sh
python experiments/wam_experiments/generate_attribution.py --help
python experiments/wam_experiments/eval.py --help
```

These commands are templates: users should provide dataset roots, checkpoints, and compute settings appropriate to their environment.
