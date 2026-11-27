# Adaptive-Path Wavelet Interpretation Maps for Auditing Neural Content Scorers

<p align="center">
  <strong>🎓 CIKM 2026</strong><br>
  Zhibo Jin · Zhiyu Zhu · Jiayu Zhang · Huijie Xu · Biaoliang Chen · Nan Yang · Fang Chen · Jianlong Zhou<br>
  <a href="https://doi.org/10.1145/3799682.3841106">📄 Paper</a> &nbsp;·&nbsp;
  <a href="#quick-start">⚡ Quick Start</a> &nbsp;·&nbsp;
  <a href="#evaluation">📊 Evaluation</a> &nbsp;·&nbsp;
  <a href="#citation">📚 Citation</a>
</p>

<p align="center">
  <a href="https://github.com/user-attachments/assets/2450e56b-d522-4e7b-a194-9f96b7b5a605"><img src="https://github.com/user-attachments/assets/2450e56b-d522-4e7b-a194-9f96b7b5a605" width="900" alt="AP-WIM image overlay, wavelet coefficient heatmaps at coarse and fine scales, and attribution summaries by direction and scale."></a><br>
  <em>AP-WIM attribution for an image, shown in the pixel plane and across wavelet scales and directions.</em>
</p>

## Overview

**AP-WIM** identifies image or audio structure that can help restore a selected score from a frozen neural model. It starts from the observed input in a multilevel discrete wavelet representation, updates the coefficients using local score gradients and coefficient-scaled steps, and accumulates contributions along the resulting path. The image implementation projects the attributions onto the pixel grid. The audio implementation analyzes the waveform and maps the attributions onto the log-mel spectrogram used by the classifier.

## Evaluation

| Experiment | Frozen scorer | Evaluation |
| --- | --- | --- |
| 🖼️ **ImageNet classification** | EfficientNet-B0 on a 1,000-image tensor subset | Insertion, deletion, and ROAD |
| 🔊 **ESC-50 classification** | ResNet-50 and MobileNetV2 on single-channel log-mel inputs | Perturbation tests on fold 1 and a five-fold attribution stability check |
| 🔎 **Tiny-ImageNet retrieval** | Cosine similarity between frozen EfficientNet-B0 embeddings | Ranking insertion AUC for target score, Recall@1, Recall@5, MRR, and nDCG@10 |
| 🧪 **ImageNet prototype audit** | Cosine similarity to EfficientNet-B0 classifier prototypes | Insertion, deletion, and optional ROAD |

The paper reports that AP-WIM achieved the highest mean ranking insertion AUC among ten compared attribution methods on all five Tiny-ImageNet endpoints. For target-score insertion, the improvement over the strongest observed baseline was 0.0043 AUC (95% paired-bootstrap interval 0.0022–0.0064). The paper also reports positive insertion gains for both ESC-50 backbones in the five-fold stability check.

## Installation

Run the commands below from the repository root. The [Conda environment](environment.yml) specifies Python 3.9 and CUDA 11.8.

```bash
conda env create -f environment.yml
conda activate apwim-anonymous
```

For an existing Python environment with a compatible PyTorch stack, install the [Python dependencies](requirements.txt) with `python -m pip install -r requirements.txt`.

## Data and model weights

Supply the evaluation data and any task-specific checkpoints locally. The ImageNet and Tiny-ImageNet examples use torchvision's default EfficientNet-B0 weights, which must be available in its cache or downloaded at runtime.

| Input | Configuration |
| --- | --- |
| ImageNet subset | Set `APWIM_IMAGENET_TENSOR_DIR` to the directory containing `img_batch.pt` and `label_batch.pt`. The loader expects preprocessed tensors with aligned sample order. |
| ESC-50 | Set `APWIM_ESC50_ROOT` to the directory containing `meta/esc50.csv` and `audio/`. Set `APWIM_RESNET50_ESC50_CKPT` or `APWIM_MOBILENETV2_ESC50_CKPT` to the corresponding checkpoint for the fold 1 runner. |
| Tiny-ImageNet-200 | Set `APWIM_TINY_IMAGENET_ROOT` to the extracted dataset root with images under `train/<class>/images/`. |

See the [dataset loaders](experiments/wam_experiments/datasets.py) and [experiment configuration](experiments/wam_experiments/apwim_experiment_common.py) for additional supported inputs and checkpoint variables. The repository contains the source and experiment drivers. Datasets, task-specific checkpoints, logs, and generated scores are supplied or produced separately.

## Quick Start

This 32-image example generates AP-WIM maps and evaluates insertion and deletion on the same samples. The first command saves a NumPy attribution array. The second saves per-example perturbation curves and prints the aggregate AUCs.

> [!TIP]
> Use a fresh output directory when changing settings. The runners can reuse existing files.

```bash
export APWIM_IMAGENET_TENSOR_DIR=/path/to/imagenet-tensors
RUN_DIR=results/imagenet-example

python experiments/wam_experiments/generate_attribution.py \
  --dataset imagenet --model efficient_b0 --attr_method ap_wim \
  --max_samples 32 --batch_size 8 --attribution_dir "$RUN_DIR/attributions"

python experiments/wam_experiments/eval.py \
  --dataset imagenet --model efficient_b0 --attr_method ap_wim \
  --max_samples 32 --batch_size 8 --attribution_dir "$RUN_DIR/attributions" \
  --score_dir "$RUN_DIR/scores"
```

The runners accept `ap_wim` and the legacy name `wam_mfaba`. See the [ROAD evaluator](experiments/wam_experiments/test_road.py) and the [ESC-50 multifold driver](experiments/wam_experiments/run_esc50_multifold_minimal.sh) for those protocols.

## Retrieval and scorer audits

### Tiny-ImageNet query–gallery ranking

The Tiny-ImageNet audit constructs disjoint query and gallery sets from the training split. The default configuration samples 10 gallery images and 3 query images per class across 200 classes. Each query's attribution target is its highest-scoring same-class gallery image. The evaluator progressively inserts the pixels ranked highest by the attribution and records the score and ranking curves.

```bash
export APWIM_TINY_IMAGENET_ROOT=/path/to/tiny-imagenet-200

python scorer_audit_experiments/tinyimagenet_retrieval_ranking_audit.py \
  --output_root results/tinyimagenet-ranking --methods all
```

### ImageNet prototype scoring

The prototype audit uses the ImageNet tensor subset configured above. It computes relevance to classifier prototypes and evaluates perturbation curves for those scores.

```bash
python scorer_audit_experiments/retrieval_score_audit.py \
  --output_root results/imagenet-prototype-audit --methods all
```

Both audits write `attributions/`, `scores/`, and `tables/` under their output directory. Each Python entry point accepts `--help` for its full configuration.

## Repository layout

| Path | Contents |
| --- | --- |
| [`core/lib/`](core/lib) and [`core/src/`](core/src) | Wavelet attribution utilities and supporting model, evaluation, and visualization code |
| [`experiments/wam_experiments/`](experiments/wam_experiments) | Image and audio attribution, baselines, and classification evaluations |
| [`scorer_audit_experiments/`](scorer_audit_experiments) | Query-gallery ranking and prototype scorer audits |

## Citation

```bibtex
@inproceedings{jin2026apwim,
  author    = {Jin, Zhibo and Zhu, Zhiyu and Zhang, Jiayu and Xu, Huijie and
               Chen, Biaoliang and Yang, Nan and Chen, Fang and Zhou, Jianlong},
  title     = {Adaptive-Path Wavelet Interpretation Maps for Auditing Neural Content Scorers},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026},
  pages     = {1952--1962},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3799682.3841106}
}
```

See [LICENSE](LICENSE) for the license notice.
