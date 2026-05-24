from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from apwim_experiment_common import causal_score_path, method_aliases, road_score_path
from evaluation import auc_curve


DEFAULT_TASKS = "imagenet:efficient_b0,esc50:resnet50,esc50:mobilenet_v2"
DEFAULT_BASELINES = "wam_ig,wam_sg,ig,fast_ig,sg,saliencymap,mfaba,la,agi"


def parse_tasks(value: str) -> list[tuple[str, str]]:
    tasks = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        dataset, model = item.split(":", 1)
        tasks.append((dataset.strip(), model.strip()))
    return tasks


def first_existing(paths):
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def load_causal(score_dir, dataset, model, method):
    candidates = [causal_score_path(score_dir, dataset, model, alias) for alias in method_aliases(method)]
    path = first_existing(candidates)
    if path is None:
        return None, None
    data = np.load(path)
    insertion = np.apply_along_axis(auc_curve, 1, data["insertion_scores"])
    deletion = np.apply_along_axis(auc_curve, 1, data["deletion_scores"])
    return {
        "insertion": insertion.astype(np.float64),
        "deletion_reduction": (-deletion).astype(np.float64),
        "i_and_d_combined": (insertion - deletion).astype(np.float64),
    }, path


def load_road(score_dir, dataset, model, method, baseline, target):
    candidates = []
    for alias in method_aliases(method):
        candidates.append(road_score_path(score_dir, dataset, model, alias, baseline=baseline, target=target))
        candidates.append(road_score_path(score_dir, dataset, model, alias, baseline=baseline, target=target, legacy=True))
    path = first_existing(candidates)
    if path is None:
        return None, None
    data = np.load(path)
    morf = np.apply_along_axis(auc_curve, 1, data["morf_scores"])
    lerf = np.apply_along_axis(auc_curve, 1, data["lerf_scores"])
    return (lerf - morf).astype(np.float64), path


def bootstrap(diff: np.ndarray, n_boot: int, seed: int, ci: float):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    mean = float(diff.mean())
    p_two = float(2.0 * min((means <= 0).mean(), (means >= 0).mean()))
    p_one = float((means <= 0).mean())
    return mean, float(lo), float(hi), min(p_two, 1.0), p_one


def holm_bonferroni(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    if valid.size == 0:
        return out.tolist()
    order = valid[np.argsort(p[valid])]
    m = valid.size
    adjusted_sorted = []
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min((m - rank) * p[idx], 1.0)
        running = max(running, adj)
        adjusted_sorted.append(running)
    for idx, adj in zip(order, adjusted_sorted):
        out[idx] = min(adj, 1.0)
    return out.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str, default="scores")
    parser.add_argument("--output_csv", type=str, default="bootstrap_ci_results.csv")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--ap_method", type=str, default="ap_wim")
    parser.add_argument("--baselines", type=str, default=DEFAULT_BASELINES)
    parser.add_argument("--road_baseline", type=str, default="blur")
    parser.add_argument("--road_target", type=str, default="pred")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baselines = [m.strip() for m in args.baselines.split(",") if m.strip()]
    rows = []

    for dataset, model in parse_tasks(args.tasks):
        ap_causal, ap_causal_path = load_causal(args.score_dir, dataset, model, args.ap_method)
        ap_road, ap_road_path = load_road(args.score_dir, dataset, model, args.ap_method, args.road_baseline, args.road_target)
        if ap_causal is None:
            print(f"[MISS] causal AP-WIM missing for {dataset}/{model}")
            continue

        metric_values = {
            "insertion": (ap_causal["insertion"], ap_causal_path),
            "i_and_d_combined": (ap_causal["i_and_d_combined"], ap_causal_path),
            "deletion_reduction": (ap_causal["deletion_reduction"], ap_causal_path),
        }
        if ap_road is not None:
            metric_values["road_score"] = (ap_road, ap_road_path)

        baseline_cache = {}
        for baseline in baselines:
            causal, causal_path = load_causal(args.score_dir, dataset, model, baseline)
            road, road_path = load_road(args.score_dir, dataset, model, baseline, args.road_baseline, args.road_target)
            baseline_cache[baseline] = {
                "causal": causal,
                "causal_path": causal_path,
                "road": road,
                "road_path": road_path,
            }

        for metric, (ap_values, ap_path) in metric_values.items():
            candidates = []
            for baseline, loaded in baseline_cache.items():
                if metric == "road_score":
                    values = loaded["road"]
                    path = loaded["road_path"]
                else:
                    causal = loaded["causal"]
                    values = None if causal is None else causal[metric]
                    path = loaded["causal_path"]
                if values is None:
                    continue
                candidates.append((baseline, float(np.nanmean(values)), values, path))
            if not candidates:
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "baseline": "",
                        "n": 0,
                        "ap_mean": float(np.nanmean(ap_values)),
                        "baseline_mean": np.nan,
                        "mean_diff": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_two_sided": np.nan,
                        "p_one_sided_gt0": np.nan,
                        "ap_path": str(ap_path),
                        "baseline_path": "",
                    }
                )
                continue

            baseline, baseline_mean, baseline_values, baseline_path = max(candidates, key=lambda item: item[1])
            n = min(len(ap_values), len(baseline_values))
            diff = ap_values[:n] - baseline_values[:n]
            mean, lo, hi, p_two, p_one = bootstrap(diff, args.n_boot, args.seed, args.ci)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "baseline": baseline,
                    "n": int(n),
                    "ap_mean": float(np.nanmean(ap_values[:n])),
                    "baseline_mean": baseline_mean,
                    "mean_diff": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_two_sided": p_two,
                    "p_one_sided_gt0": p_one,
                    "ap_path": str(ap_path),
                    "baseline_path": str(baseline_path),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_two_sided_holm"] = holm_bonferroni(df["p_two_sided"].tolist())
        df["p_one_sided_holm"] = holm_bonferroni(df["p_one_sided_gt0"].tolist())
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(df)
    print("saved:", args.output_csv)


if __name__ == "__main__":
    main()
