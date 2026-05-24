#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

WAM_EXPERIMENT_DIR = Path(os.environ.get("APWIM_WAM_EXPERIMENT_DIR", Path(__file__).resolve().parents[1] / "experiments" / "wam_experiments"))
if str(WAM_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(WAM_EXPERIMENT_DIR))

from apwim_experiment_common import (  # noqa: E402
    NormalizeLayer,
    default_ap_wim_steps,
    dispatch_attr_method,
    get_attr_dict,
    is_wavelet_method,
    save_npy_atomic,
    save_npz_atomic,
)
from evaluation import auc_curve  # noqa: E402
from utils import device, setup_seed  # noqa: E402


METHODS_ALL = [
    "ap_wim",
    "wam_ig",
    "wam_sg",
    "ig",
    "fast_ig",
    "sg",
    "saliencymap",
    "mfaba",
    "la",
    "agi",
]


class ImageListDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), int(label), path


class EfficientNetEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.normalize = NormalizeLayer(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.features = backbone.features
        self.avgpool = backbone.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return F.normalize(feat, dim=1)


class GalleryRetrievalScorer(nn.Module):
    """Frozen query-to-gallery retrieval scorer.

    The model returns shifted cosine scores between a query image embedding and
    fixed gallery embeddings. Each output dimension is a gallery item, so
    attribution targets a real retrieval item rather than a classifier logit.
    """

    def __init__(self, encoder: EfficientNetEmbedding, gallery_embeddings: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("gallery_embeddings", F.normalize(gallery_embeddings.float(), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return (feat @ self.gallery_embeddings.t() + 1.0) * 0.5


def parse_methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return METHODS_ALL
    return [x.strip() for x in value.replace(" ", ",").split(",") if x.strip()]


def tiny_imagenet_splits(
    root: Path,
    *,
    gallery_per_class: int,
    query_per_class: int,
    max_classes: int,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[str]]:
    train_root = root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing Tiny-ImageNet train folder: {train_root}")
    class_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])
    if max_classes > 0:
        class_dirs = class_dirs[:max_classes]
    rng = random.Random(seed)
    gallery: list[tuple[str, int]] = []
    query: list[tuple[str, int]] = []
    class_names: list[str] = []
    for label, class_dir in enumerate(class_dirs):
        image_dir = class_dir / "images"
        files = sorted(str(p) for p in image_dir.glob("*.JPEG"))
        rng.shuffle(files)
        need = gallery_per_class + query_per_class
        if len(files) < need:
            raise ValueError(f"Class {class_dir.name} has {len(files)} images, need {need}")
        gallery.extend((p, label) for p in files[:gallery_per_class])
        query.extend((p, label) for p in files[gallery_per_class:need])
        class_names.append(class_dir.name)
    return gallery, query, class_names


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


@torch.no_grad()
def compute_embeddings(encoder: nn.Module, dataset: Dataset, batch_size: int) -> tuple[torch.Tensor, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    embeddings = []
    labels = []
    encoder.eval()
    for x, y, _ in tqdm(loader, desc="embed"):
        emb = encoder(x.to(device)).detach().cpu()
        embeddings.append(emb)
        labels.append(y.numpy())
    return torch.cat(embeddings, dim=0), np.concatenate(labels).astype(np.int64)


@torch.no_grad()
def choose_positive_targets(
    model: GalleryRetrievalScorer,
    query_dataset: Dataset,
    gallery_labels: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    target_indices = []
    query_labels = []
    recall1 = []
    recall5 = []
    mrr = []
    ndcg10 = []
    for x, y, _ in tqdm(loader, desc="targets"):
        scores = model(x.to(device)).detach().cpu().numpy()
        y_np = y.numpy().astype(np.int64)
        query_labels.append(y_np)
        for i, label in enumerate(y_np):
            positive = np.flatnonzero(gallery_labels == label)
            pos_scores = scores[i, positive]
            target_indices.append(int(positive[int(np.argmax(pos_scores))]))
            metrics = retrieval_metrics(scores[i], gallery_labels, int(label), k_values=(1, 5), ndcg_k=10)
            recall1.append(metrics["recall@1"])
            recall5.append(metrics["recall@5"])
            mrr.append(metrics["mrr"])
            ndcg10.append(metrics["ndcg@10"])
    meta = {
        "full_recall@1": float(np.mean(recall1)),
        "full_recall@5": float(np.mean(recall5)),
        "full_mrr": float(np.mean(mrr)),
        "full_ndcg@10": float(np.mean(ndcg10)),
    }
    return np.asarray(target_indices, dtype=np.int64), np.concatenate(query_labels), meta


def call_image_attribution(
    method_name: str,
    method_fn,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    wavelet: str,
    J: int,
    steps: int | None,
):
    dispatch_name = dispatch_attr_method(method_name)
    x = x.to(device)
    y = y.to(device)
    if dispatch_name in {"wam_ig", "wam_sg", "wam_mfaba"}:
        kwargs = {"wavelet": wavelet, "J": J, "approx_coeffs": False}
        if dispatch_name == "wam_mfaba":
            kwargs["steps"] = default_ap_wim_steps("imagenet") if steps is None else steps
        elif dispatch_name == "wam_ig" and steps is not None:
            kwargs["n_steps"] = steps
        return method_fn(model, x, y, **kwargs)
    return method_fn(model, x, y)


def path_for(root: Path, kind: str, method: str) -> Path:
    prefix = "tinyimagenet_retrieval_efficient_b0_gallery"
    if kind == "attr":
        return root / "attributions" / f"{prefix}_{method}_attributions.npy"
    if kind == "ranking":
        return root / "scores" / f"{prefix}_{method}_ranking_insertion_scores.npz"
    raise ValueError(kind)


def run_attribution(args, model: nn.Module, query_dataset: Dataset, target_indices: np.ndarray, method: str) -> Path:
    out_path = path_for(args.output_root, "attr", method)
    if out_path.exists() and not args.force:
        print(f"[ATTR] exists: {out_path}")
        return out_path
    attr_dict = get_attr_dict("imagenet")
    if method not in attr_dict:
        raise ValueError(f"Unknown method={method}; available={sorted(attr_dict)}")
    method_fn = attr_dict[method]
    loader = DataLoader(query_dataset, batch_size=args.attr_batch_size, shuffle=False, num_workers=2)
    chunks = []
    offset = 0
    print(f"[ATTR] method={method} n={len(query_dataset)} batch={args.attr_batch_size} output={out_path}")
    if is_wavelet_method(method):
        print(f"[ATTR] wavelet={args.wavelet} J={args.J} steps={args.steps}")
    for x, _, _ in tqdm(loader, desc=f"attr:{method}"):
        y = torch.from_numpy(target_indices[offset : offset + len(x)]).long()
        attr = call_image_attribution(
            method,
            method_fn,
            model,
            x,
            y,
            wavelet=args.wavelet,
            J=args.J,
            steps=args.steps,
        )
        chunks.append(attr)
        offset += len(x)
    arr = np.concatenate(chunks, axis=0).astype(np.float32)
    save_npy_atomic(out_path, arr)
    print(f"[ATTR] saved {out_path} shape={arr.shape} min={arr.min():.6g} max={arr.max():.6g}")
    return out_path


def retrieval_metrics(
    scores: np.ndarray,
    gallery_labels: np.ndarray,
    query_label: int,
    *,
    k_values: tuple[int, ...] = (1, 5),
    ndcg_k: int = 10,
) -> dict[str, float]:
    order = np.argsort(-scores)
    rel = (gallery_labels[order] == query_label).astype(np.float64)
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = float(rel[:k].max() if rel.size else 0.0)
    hit_positions = np.flatnonzero(rel > 0)
    out["mrr"] = float(1.0 / (hit_positions[0] + 1)) if hit_positions.size else 0.0
    gains = rel[:ndcg_k] / np.log2(np.arange(2, min(ndcg_k, rel.size) + 2))
    positives = int((gallery_labels == query_label).sum())
    ideal_len = min(ndcg_k, positives)
    if ideal_len > 0:
        ideal = np.ones(ideal_len, dtype=np.float64) / np.log2(np.arange(2, ideal_len + 2))
        out[f"ndcg@{ndcg_k}"] = float(gains.sum() / ideal.sum())
    else:
        out[f"ndcg@{ndcg_k}"] = 0.0
    return out


def masks_from_attr(attr: torch.Tensor, fractions: list[float]) -> torch.Tensor:
    # attr: [B, 1, H, W]. Returns [S, B, 1, H, W].
    b, _, h, w = attr.shape
    flat = attr.reshape(b, -1)
    n = flat.shape[1]
    order = torch.argsort(flat, dim=1, descending=True)
    masks = []
    for frac in fractions:
        k = int(round(frac * n))
        mask = torch.zeros_like(flat)
        if k > 0:
            idx = order[:, : min(k, n)]
            mask.scatter_(1, idx, 1.0)
        masks.append(mask.reshape(b, 1, h, w))
    return torch.stack(masks, dim=0)


@torch.no_grad()
def run_ranking_insertion(
    args,
    model: GalleryRetrievalScorer,
    query_dataset: Dataset,
    gallery_labels: np.ndarray,
    query_labels: np.ndarray,
    target_indices: np.ndarray,
    method: str,
) -> Path:
    out_path = path_for(args.output_root, "ranking", method)
    if out_path.exists() and not args.force:
        print(f"[RANK] exists: {out_path}")
        return out_path

    attr_path = path_for(args.output_root, "attr", method)
    attributions = np.load(attr_path)
    fractions = [float(x) for x in args.fractions.split(",")]
    loader = DataLoader(query_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2)
    metrics = {
        "target_score": [],
        "recall@1": [],
        "recall@5": [],
        "mrr": [],
        "ndcg@10": [],
    }
    print(f"[RANK] method={method} attr={attr_path} output={out_path} fractions={fractions}")
    offset = 0
    for x, _, _ in tqdm(loader, desc=f"rank:{method}"):
        sl = slice(offset, offset + len(x))
        attr = torch.from_numpy(attributions[sl]).float().to(device)
        x = x.to(device)
        masks = masks_from_attr(attr, fractions)
        batch_scores = {name: [] for name in metrics}
        for step_idx in range(len(fractions)):
            perturbed = x * masks[step_idx]
            scores = model(perturbed).detach().cpu().numpy()
            for i in range(scores.shape[0]):
                label = int(query_labels[offset + i])
                target = int(target_indices[offset + i])
                rm = retrieval_metrics(scores[i], gallery_labels, label, k_values=(1, 5), ndcg_k=10)
                batch_scores["target_score"].append(float(scores[i, target]))
                batch_scores["recall@1"].append(rm["recall@1"])
                batch_scores["recall@5"].append(rm["recall@5"])
                batch_scores["mrr"].append(rm["mrr"])
                batch_scores["ndcg@10"].append(rm["ndcg@10"])
        for name in metrics:
            arr = np.asarray(batch_scores[name], dtype=np.float32).reshape(len(fractions), len(x)).T
            metrics[name].append(arr)
        offset += len(x)

    arrays = {name: np.concatenate(parts, axis=0).astype(np.float32) for name, parts in metrics.items()}
    save_npz_atomic(
        out_path,
        fractions=np.asarray(fractions, dtype=np.float32),
        gallery_labels=gallery_labels.astype(np.int64),
        query_labels=query_labels.astype(np.int64),
        target_indices=target_indices.astype(np.int64),
        **arrays,
    )
    print(
        "[RANK] saved "
        f"{out_path} target_score_auc={per_example_auc(arrays['target_score'], fractions).mean():.6f} "
        f"mrr_auc={per_example_auc(arrays['mrr'], fractions).mean():.6f}"
    )
    return out_path


def per_example_auc(arr: np.ndarray, fractions: list[float] | np.ndarray) -> np.ndarray:
    fractions = np.asarray(fractions, dtype=np.float64)
    if fractions.size == arr.shape[1]:
        return np.trapz(arr.astype(np.float64), x=fractions, axis=1).astype(np.float64)
    return np.apply_along_axis(auc_curve, 1, arr).astype(np.float64)


def bootstrap_diff_ci(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(args, methods: list[str]) -> None:
    tables = args.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    values = {}
    for method in methods:
        score_path = path_for(args.output_root, "ranking", method)
        if not score_path.exists():
            continue
        data = np.load(score_path)
        fractions = data["fractions"]
        for metric in ["target_score", "recall@1", "recall@5", "mrr", "ndcg@10"]:
            auc = per_example_auc(data[metric], fractions)
            key = f"rank_insertion_{metric}_auc"
            values[(method, key)] = auc
            rows.append(
                {
                    "method": method,
                    "metric": key,
                    "mean": float(auc.mean()),
                    "std": float(auc.std(ddof=1)),
                    "n": int(auc.size),
                }
            )

    csv_path = tables / "tinyimagenet_retrieval_ranking_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(rows)

    comparisons = []
    for metric in sorted({m for _, m in values}):
        ap = values.get(("ap_wim", metric))
        if ap is None:
            continue
        candidates = [
            (method, values[(method, metric)])
            for method in methods
            if method != "ap_wim" and (method, metric) in values
        ]
        if not candidates:
            continue
        best_method, best = max(candidates, key=lambda item: item[1].mean())
        n = min(ap.size, best.size)
        diff = ap[:n] - best[:n]
        lo, hi = bootstrap_diff_ci(diff, args.n_boot, args.seed)
        comparisons.append(
            {
                "metric": metric,
                "ap_wim_mean": float(ap.mean()),
                "best_baseline": best_method,
                "baseline_mean": float(best.mean()),
                "delta_ap_minus_best": float(diff.mean()),
                "ci95_low": lo,
                "ci95_high": hi,
                "favorable_for_apwim": bool(diff.mean() > 0 and lo > 0),
            }
        )

    comp_path = tables / "tinyimagenet_retrieval_ranking_ap_vs_best.csv"
    with comp_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "metric",
            "ap_wim_mean",
            "best_baseline",
            "baseline_mean",
            "delta_ap_minus_best",
            "ci95_low",
            "ci95_high",
            "favorable_for_apwim",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)

    config_path = args.output_root / "retrieval_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    md_path = tables / "tinyimagenet_retrieval_ranking_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Tiny-ImageNet Content Retrieval Ranking Audit\n\n")
        f.write(
            "This experiment evaluates AP-WIM on a query-gallery content-based image retrieval task. "
            "Tiny-ImageNet train images are split per class into a fixed gallery and query set; relevance is same-class membership. "
            "A frozen EfficientNet-B0 embedding scorer ranks gallery items by cosine similarity to the perturbed query image. "
            "Attribution quality is measured by how quickly rank metrics recover under insertion of top-ranked query pixels.\n\n"
        )
        if config:
            f.write("## Retrieval setup\n\n")
            for key in [
                "num_classes",
                "gallery_per_class",
                "query_per_class",
                "num_gallery",
                "num_query",
                "full_recall@1",
                "full_recall@5",
                "full_mrr",
                "full_ndcg@10",
            ]:
                if key in config:
                    f.write(f"- {key}: {config[key]}\n")
            f.write("\n")
        f.write("## Mean ranking-insertion AUC metrics\n\n")
        f.write("| Method | Metric | Mean | Std | n |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['method']} | {row['metric']} | {row['mean']:.4f} | {row['std']:.4f} | {row['n']} |\n")
        f.write("\n## AP-WIM vs strongest baseline\n\n")
        f.write("| Metric | AP-WIM | Best baseline | Baseline | Delta | 95% CI | Paper-use |\n")
        f.write("|---|---:|---|---:|---:|---:|---|\n")
        for row in comparisons:
            use = "include" if row["favorable_for_apwim"] else "do not emphasize"
            f.write(
                f"| {row['metric']} | {row['ap_wim_mean']:.4f} | {row['best_baseline']} | "
                f"{row['baseline_mean']:.4f} | {row['delta_ap_minus_best']:.4f} | "
                f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {use} |\n"
            )
    print(f"[SUMMARY] wrote {csv_path}, {comp_path}, {md_path}")


def build_state(args):
    transform = build_transform()
    gallery_samples, query_samples, class_names = tiny_imagenet_splits(
        args.tiny_root,
        gallery_per_class=args.gallery_per_class,
        query_per_class=args.query_per_class,
        max_classes=args.max_classes,
        seed=args.seed,
    )
    gallery_dataset = ImageListDataset(gallery_samples, transform)
    query_dataset = ImageListDataset(query_samples, transform)
    encoder = EfficientNetEmbedding().eval().to(device)
    gallery_embeddings, gallery_labels = compute_embeddings(encoder, gallery_dataset, args.embed_batch_size)
    model = GalleryRetrievalScorer(encoder, gallery_embeddings.to(device)).eval().to(device)
    target_indices, query_labels, full_meta = choose_positive_targets(
        model,
        query_dataset,
        gallery_labels,
        args.eval_batch_size,
    )
    config = {
        "dataset": "tiny-imagenet-200",
        "tiny_root": str(args.tiny_root),
        "num_classes": len(class_names),
        "gallery_per_class": args.gallery_per_class,
        "query_per_class": args.query_per_class,
        "num_gallery": len(gallery_dataset),
        "num_query": len(query_dataset),
        "methods_requested": parse_methods(args.methods),
        "fractions": args.fractions,
        "wavelet": args.wavelet,
        "J": args.J,
        "steps": args.steps,
        "seed": args.seed,
        "device": str(device),
        **full_meta,
    }
    config_tmp = args.output_root / f"retrieval_config.{os.getpid()}.tmp"
    config_tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(config_tmp, args.output_root / "retrieval_config.json")
    split_tmp = args.output_root / f"retrieval_split_metadata.{os.getpid()}.npz"
    np.savez(
        split_tmp,
        gallery_labels=gallery_labels.astype(np.int64),
        query_labels=query_labels.astype(np.int64),
        target_indices=target_indices.astype(np.int64),
        class_names=np.asarray(class_names),
        gallery_paths=np.asarray([p for p, _ in gallery_samples]),
        query_paths=np.asarray([p for p, _ in query_samples]),
    )
    os.replace(split_tmp, args.output_root / "retrieval_split_metadata.npz")
    return model, query_dataset, gallery_labels, query_labels, target_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--tiny_root", type=Path, default=Path(os.environ.get("APWIM_TINY_IMAGENET_ROOT", "data/tiny-imagenet-200")))
    parser.add_argument("--methods", type=str, default="ap_wim,wam_ig,ig,fast_ig,saliencymap")
    parser.add_argument("--gallery_per_class", type=int, default=10)
    parser.add_argument("--query_per_class", type=int, default=3)
    parser.add_argument("--max_classes", type=int, default=200)
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--attr_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--J", type=int, default=3)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--fractions", type=str, default="0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()

    setup_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "attributions").mkdir(exist_ok=True)
    (args.output_root / "scores").mkdir(exist_ok=True)
    (args.output_root / "tables").mkdir(exist_ok=True)
    methods = parse_methods(args.methods)

    if not args.summarize_only:
        model, query_dataset, gallery_labels, query_labels, target_indices = build_state(args)
        for method in methods:
            run_attribution(args, model, query_dataset, target_indices, method)
            run_ranking_insertion(args, model, query_dataset, gallery_labels, query_labels, target_indices, method)
    summarize(args, methods)


if __name__ == "__main__":
    main()
