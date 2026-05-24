#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

WAM_EXPERIMENT_DIR = Path(os.environ.get("APWIM_WAM_EXPERIMENT_DIR", Path(__file__).resolve().parents[1] / "experiments" / "wam_experiments"))
if str(WAM_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(WAM_EXPERIMENT_DIR))

from apwim_experiment_common import (
    NormalizeLayer,
    build_dataset,
    default_ap_wim_steps,
    dispatch_attr_method,
    get_attr_dict,
    is_wavelet_method,
    save_npy_atomic,
    save_npz_atomic,
)
from evaluation import CausalMetric, auc_curve
from test_road import make_baseline_like, road_eval_one_batch
from utils import device, setup_seed


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


class EfficientNetPrototypeRetrievalScorer(nn.Module):
    """Frozen image-to-prototype relevance scorer.

    The scorer maps an image to an EfficientNet-B0 feature vector and compares it
    with the normalized classifier prototypes. It returns shifted cosine
    relevance scores in [0, 1] for all ImageNet prototypes. This is a lightweight
    proxy for a differentiable retrieval/ranking scorer and uses the same image
    tensor subset as the main ImageNet attribution experiment.
    """

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
        proto = backbone.classifier[1].weight.detach().clone()
        self.register_buffer("prototypes", F.normalize(proto, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = F.normalize(feat, dim=1)
        score = feat @ self.prototypes.t()
        return (score + 1.0) * 0.5


def parse_methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return METHODS_ALL
    return [x.strip() for x in value.replace(" ", ",").split(",") if x.strip()]


def path_for(root: Path, kind: str, method: str) -> Path:
    if kind == "attr":
        return root / "attributions" / f"retrieval_imagenet_efficient_b0_prototype_{method}_attributions.npy"
    if kind == "causal":
        return root / "scores" / f"retrieval_imagenet_efficient_b0_prototype_{method}_causal_scores.npz"
    if kind == "road":
        return root / "scores" / f"retrieval_imagenet_efficient_b0_prototype_{method}_road_scores_baseline-blur.npz"
    raise ValueError(kind)


def substrate_fn(mode: str):
    if mode == "zeros":
        return torch.zeros_like

    def _fn(x: torch.Tensor) -> torch.Tensor:
        return make_baseline_like(x, mode)

    return _fn


def compute_pred_targets(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x.to(device)).argmax(dim=1).long()


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


def run_attribution(args, model: nn.Module, dataset, method: str) -> Path:
    out_path = path_for(args.output_root, "attr", method)
    if out_path.exists() and not args.force:
        print(f"[ATTR] exists: {out_path}")
        return out_path

    attr_dict = get_attr_dict("imagenet")
    if method not in attr_dict:
        raise ValueError(f"Unknown method={method}; available={sorted(attr_dict)}")
    method_fn = attr_dict[method]
    loader = DataLoader(dataset, batch_size=args.attr_batch_size, shuffle=False)
    chunks = []
    print(f"[ATTR] method={method} samples={len(dataset)} batch={args.attr_batch_size} output={out_path}")
    if is_wavelet_method(method):
        print(f"[ATTR] wavelet={args.wavelet} J={args.J} steps={args.steps}")
    for batch in tqdm(loader, desc=f"attr:{method}"):
        x = batch[0]
        y = compute_pred_targets(model, x)
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
    arr = np.concatenate(chunks, axis=0).astype(np.float32)
    save_npy_atomic(out_path, arr)
    print(f"[ATTR] saved {out_path} shape={arr.shape} min={arr.min():.6g} max={arr.max():.6g}")
    return out_path


def run_causal(args, model: nn.Module, dataset, method: str) -> Path:
    out_path = path_for(args.output_root, "causal", method)
    if out_path.exists() and not args.force:
        print(f"[CAUSAL] exists: {out_path}")
        return out_path

    attr_path = path_for(args.output_root, "attr", method)
    attributions = np.load(attr_path)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)
    insertion = CausalMetric(
        model,
        mode="ins",
        step=224,
        substrate_fn=substrate_fn("zeros"),
        device=device,
    )
    deletion = CausalMetric(
        model,
        mode="del",
        step=224,
        substrate_fn=substrate_fn("zeros"),
        device=device,
    )
    all_ins = []
    all_del = []
    print(f"[CAUSAL] method={method} attr={attr_path} output={out_path}")
    for idx, batch in enumerate(tqdm(loader, desc=f"causal:{method}")):
        x = batch[0].cpu()
        sl = slice(idx * args.eval_batch_size, idx * args.eval_batch_size + len(x))
        attr = attributions[sl]
        all_ins.append(insertion.evaluate(x, attr, len(x)))
        all_del.append(deletion.evaluate(x, attr, len(x)))
    ins = np.concatenate(all_ins, axis=0).astype(np.float32)
    dele = np.concatenate(all_del, axis=0).astype(np.float32)
    save_npz_atomic(
        out_path,
        insertion_scores=ins,
        deletion_scores=dele,
        dataset=np.array("retrieval_imagenet"),
        model=np.array("efficient_b0_prototype"),
        attr_method=np.array(method),
        target=np.array("predicted_prototype"),
    )
    print(
        f"[CAUSAL] saved {out_path} insertion={auc_curve(ins.mean(axis=0)):.6f} "
        f"deletion={auc_curve(dele.mean(axis=0)):.6f}"
    )
    return out_path


def run_road(args, model: nn.Module, dataset, method: str) -> Path:
    out_path = path_for(args.output_root, "road", method)
    if out_path.exists() and not args.force:
        print(f"[ROAD] exists: {out_path}")
        return out_path

    attr_path = path_for(args.output_root, "attr", method)
    attributions = np.load(attr_path)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)
    percentiles = tuple(int(x) for x in args.percentiles.split(","))
    morf_all = []
    lerf_all = []
    print(f"[ROAD] method={method} attr={attr_path} output={out_path} percentiles={percentiles}")
    for idx, batch in enumerate(tqdm(loader, desc=f"road:{method}")):
        x = batch[0]
        y = compute_pred_targets(model, x).cpu()
        sl = slice(idx * args.eval_batch_size, idx * args.eval_batch_size + len(x))
        attr = attributions[sl]
        morf, lerf = road_eval_one_batch(
            model,
            x,
            y,
            attr,
            percentiles=percentiles,
            baseline_mode="blur",
            use_pred_as_target=True,
        )
        morf_all.append(morf)
        lerf_all.append(lerf)
    morf_scores = np.concatenate(morf_all, axis=0).astype(np.float32)
    lerf_scores = np.concatenate(lerf_all, axis=0).astype(np.float32)
    road_values = np.apply_along_axis(auc_curve, 1, lerf_scores) - np.apply_along_axis(auc_curve, 1, morf_scores)
    save_npz_atomic(
        out_path,
        morf_scores=morf_scores,
        lerf_scores=lerf_scores,
        road_score=np.array(float(road_values.mean()), dtype=np.float32),
        percentiles=np.array(percentiles),
        baseline=np.array("blur"),
        target=np.array("predicted_prototype"),
    )
    print(f"[ROAD] saved {out_path} road={road_values.mean():.6f}")
    return out_path


def per_example_auc(arr: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(auc_curve, 1, arr).astype(np.float64)


def bootstrap_diff_ci(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=np.float64)
    if diff.size == 0:
        return float("nan"), float("nan")
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
        causal_path = path_for(args.output_root, "causal", method)
        road_path = path_for(args.output_root, "road", method)
        if not causal_path.exists():
            continue
        causal = np.load(causal_path)
        insertion = per_example_auc(causal["insertion_scores"])
        deletion = per_example_auc(causal["deletion_scores"])
        metrics = {
            "insertion_auc": insertion,
            "deletion_auc": deletion,
            "i_minus_d": insertion - deletion,
        }
        if road_path.exists():
            road = np.load(road_path)
            morf = per_example_auc(road["morf_scores"])
            lerf = per_example_auc(road["lerf_scores"])
            metrics["road_auc"] = lerf - morf
        for metric, arr in metrics.items():
            values[(method, metric)] = arr
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "n": int(arr.size),
                }
            )

    csv_path = tables / "retrieval_score_audit_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(rows)

    comparisons = []
    for metric in sorted({m for _, m in values}):
        ap = values.get(("ap_wim", metric))
        if ap is None:
            continue
        candidates = [(method, values[(method, metric)]) for method in methods if method != "ap_wim" and (method, metric) in values]
        if not candidates:
            continue
        if metric == "deletion_auc":
            best_method, best = min(candidates, key=lambda item: item[1].mean())
            diff = best[: ap.size] - ap[: best.size]
            favorable = diff.mean() > 0
        else:
            best_method, best = max(candidates, key=lambda item: item[1].mean())
            diff = ap[: best.size] - best[: ap.size]
            favorable = diff.mean() > 0
        lo, hi = bootstrap_diff_ci(diff, args.n_boot, args.seed)
        comparisons.append(
            {
                "metric": metric,
                "ap_wim_mean": float(ap.mean()),
                "best_baseline": best_method,
                "baseline_mean": float(best.mean()),
                "delta_ap_minus_best_or_best_minus_ap_for_deletion": float(diff.mean()),
                "ci95_low": lo,
                "ci95_high": hi,
                "favorable_for_apwim": bool(favorable and lo > 0),
            }
        )

    comp_path = tables / "retrieval_score_audit_ap_vs_best.csv"
    with comp_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "metric",
            "ap_wim_mean",
            "best_baseline",
            "baseline_mean",
            "delta_ap_minus_best_or_best_minus_ap_for_deletion",
            "ci95_low",
            "ci95_high",
            "favorable_for_apwim",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)

    md_path = tables / "retrieval_score_audit_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Prototype Retrieval-Score Audit\n\n")
        f.write(
            "This experiment explains a frozen EfficientNet-B0 prototype relevance scorer. "
            "The scalar score is shifted cosine similarity between the image embedding and ImageNet classifier prototypes, "
            "a lightweight differentiable proxy for content retrieval/ranking score auditing.\n\n"
        )
        f.write("## Mean metrics\n\n")
        f.write("| Method | Metric | Mean | Std | n |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['metric']} | {row['mean']:.4f} | "
                f"{row['std']:.4f} | {row['n']} |\n"
            )
        f.write("\n## AP-WIM vs strongest baseline\n\n")
        f.write("| Metric | AP-WIM | Best baseline | Baseline | Delta | 95% CI | Paper-use |\n")
        f.write("|---|---:|---|---:|---:|---:|---|\n")
        for row in comparisons:
            use = "include if review-fit story" if row["favorable_for_apwim"] else "treat cautiously"
            f.write(
                f"| {row['metric']} | {row['ap_wim_mean']:.4f} | {row['best_baseline']} | "
                f"{row['baseline_mean']:.4f} | {row['delta_ap_minus_best_or_best_minus_ap_for_deletion']:.4f} | "
                f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {use} |\n"
            )
    print(f"[SUMMARY] wrote {csv_path}, {comp_path}, {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="ap_wim,wam_ig,ig,fast_ig,saliencymap")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--attr_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--J", type=int, default=3)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--percentiles", type=str, default="10,20,30,40,50,60,70,80,90")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--skip_road", action="store_true")
    args = parser.parse_args()

    setup_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "attributions").mkdir(exist_ok=True)
    (args.output_root / "scores").mkdir(exist_ok=True)
    (args.output_root / "tables").mkdir(exist_ok=True)
    methods = parse_methods(args.methods)
    with (args.output_root / "retrieval_score_audit_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "methods": methods,
                "max_samples": args.max_samples,
                "attr_batch_size": args.attr_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "wavelet": args.wavelet,
                "J": args.J,
                "steps": args.steps,
                "device": str(device),
            },
            f,
            indent=2,
        )

    if not args.summarize_only:
        dataset = build_dataset("imagenet", max_samples=args.max_samples)
        model = EfficientNetPrototypeRetrievalScorer().eval().to(device)
        for method in methods:
            run_attribution(args, model, dataset, method)
            run_causal(args, model, dataset, method)
            if not args.skip_road:
                run_road(args, model, dataset, method)
    summarize(args, methods)


if __name__ == "__main__":
    main()
