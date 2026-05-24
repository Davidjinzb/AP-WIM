from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from apwim_experiment_common import causal_score_path, road_score_path
from evaluation import auc_curve


HIGHER_IS_BETTER = {
    "insertion_auc": True,
    "minus_deletion_auc": True,
    "i_minus_d": True,
    "road_auc": True,
}


def parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in value.replace(" ", ",").split(",") if x.strip()]


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or n_boot <= 0:
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_causal_values(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    insertion = np.apply_along_axis(auc_curve, 1, data["insertion_scores"])
    deletion = np.apply_along_axis(auc_curve, 1, data["deletion_scores"])
    return {
        "insertion_auc": insertion.astype(np.float64),
        "minus_deletion_auc": (-deletion).astype(np.float64),
        "i_minus_d": (insertion - deletion).astype(np.float64),
    }


def load_road_values(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    morf = np.apply_along_axis(auc_curve, 1, data["morf_scores"])
    lerf = np.apply_along_axis(auc_curve, 1, data["lerf_scores"])
    return {"road_auc": (lerf - morf).astype(np.float64)}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--models", type=str, default="resnet50,mobilenet_v2")
    parser.add_argument("--methods", type=str, default="ap_wim,wam_ig,ig")
    parser.add_argument("--folds", type=str, default="1,2,3,4,5")
    parser.add_argument("--road_baseline", type=str, default="blur")
    parser.add_argument("--road_target", type=str, default="pred")
    parser.add_argument("--substrate", type=str, default="zeros")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    score_dir = Path(args.score_dir)
    out_dir = Path(args.out_dir)
    models = parse_csv_arg(args.models)
    methods = parse_csv_arg(args.methods)
    folds = [int(x) for x in parse_csv_arg(args.folds)]

    fold_rows: list[dict[str, object]] = []
    values: dict[tuple[str, str, str], list[np.ndarray]] = {}
    missing: list[str] = []

    for model in models:
        for method in methods:
            for fold in folds:
                tag = f"fold{fold}"
                causal_path = causal_score_path(
                    score_dir,
                    "esc50",
                    model,
                    method,
                    output_tag=tag,
                    substrate=args.substrate,
                )
                if causal_path.exists():
                    causal_values = load_causal_values(causal_path)
                    for metric, arr in causal_values.items():
                        values.setdefault((model, method, metric), []).append(arr)
                        fold_rows.append(
                            {
                                "model": model,
                                "method": method,
                                "fold": fold,
                                "metric": metric,
                                "mean": arr.mean(),
                                "std": arr.std(ddof=1) if arr.size > 1 else 0.0,
                                "n": arr.size,
                                "source": causal_path.name,
                            }
                        )
                else:
                    missing.append(str(causal_path))

                road_path = road_score_path(
                    score_dir,
                    "esc50",
                    model,
                    method,
                    output_tag=tag,
                    baseline=args.road_baseline,
                    target=args.road_target,
                )
                if road_path.exists():
                    road_values = load_road_values(road_path)
                    for metric, arr in road_values.items():
                        values.setdefault((model, method, metric), []).append(arr)
                        fold_rows.append(
                            {
                                "model": model,
                                "method": method,
                                "fold": fold,
                                "metric": metric,
                                "mean": arr.mean(),
                                "std": arr.std(ddof=1) if arr.size > 1 else 0.0,
                                "n": arr.size,
                                "source": road_path.name,
                            }
                        )
                else:
                    missing.append(str(road_path))

    aggregate_rows: list[dict[str, object]] = []
    concat_values: dict[tuple[str, str, str], np.ndarray] = {}
    for key, arrs in values.items():
        model, method, metric = key
        fold_means = np.array([arr.mean() for arr in arrs], dtype=np.float64)
        concat = np.concatenate(arrs, axis=0)
        concat_values[key] = concat
        lo, hi = bootstrap_ci(concat, args.n_boot, args.seed)
        aggregate_rows.append(
            {
                "model": model,
                "method": method,
                "metric": metric,
                "direction": "higher" if HIGHER_IS_BETTER.get(metric, True) else "lower",
                "mean": concat.mean(),
                "std": concat.std(ddof=1) if concat.size > 1 else 0.0,
                "ci95_low": lo,
                "ci95_high": hi,
                "fold_mean": fold_means.mean(),
                "fold_std": fold_means.std(ddof=1) if fold_means.size > 1 else 0.0,
                "n": concat.size,
                "n_folds": len(arrs),
            }
        )

    comparison_rows: list[dict[str, object]] = []
    for model in models:
        metrics = sorted({metric for (m, _, metric) in concat_values if m == model})
        for metric in metrics:
            ap = concat_values.get((model, "ap_wim", metric))
            if ap is None:
                continue
            baseline_candidates = []
            for method in methods:
                if method in {"ap_wim", "wam_mfaba"}:
                    continue
                arr = concat_values.get((model, method, metric))
                if arr is not None:
                    baseline_candidates.append((method, arr))
            if not baseline_candidates:
                continue
            best_method, best_arr = max(baseline_candidates, key=lambda item: item[1].mean())
            n = min(ap.size, best_arr.size)
            diff = ap[:n] - best_arr[:n]
            lo, hi = bootstrap_ci(diff, args.n_boot, args.seed)
            comparison_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "ap_mean": ap.mean(),
                    "best_baseline": best_method,
                    "baseline_mean": best_arr.mean(),
                    "delta_ap_minus_baseline": diff.mean(),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n": n,
                    "favorable_for_apwim": bool(diff.mean() > 0.0 and lo > 0.0),
                }
            )

    write_csv(
        out_dir / "esc50_multifold_fold_summary.csv",
        fold_rows,
        ["model", "method", "fold", "metric", "mean", "std", "n", "source"],
    )
    write_csv(
        out_dir / "esc50_multifold_aggregate.csv",
        aggregate_rows,
        [
            "model",
            "method",
            "metric",
            "direction",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "fold_mean",
            "fold_std",
            "n",
            "n_folds",
        ],
    )
    write_csv(
        out_dir / "esc50_multifold_ap_vs_best.csv",
        comparison_rows,
        [
            "model",
            "metric",
            "ap_mean",
            "best_baseline",
            "baseline_mean",
            "delta_ap_minus_baseline",
            "ci95_low",
            "ci95_high",
            "n",
            "favorable_for_apwim",
        ],
    )

    md = out_dir / "esc50_multifold_summary.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# ESC-50 5-Fold Attribution Summary\n\n")
        f.write("This run evaluates AP-WIM on all five ESC-50 folds using the same trained fold checkpoints.\n\n")
        f.write("## AP-WIM vs strongest non-AP baseline\n\n")
        if comparison_rows:
            f.write("| Model | Metric | AP-WIM | Best baseline | Baseline | Delta | 95% CI | Paper-use |\n")
            f.write("|---|---:|---:|---|---:|---:|---:|---|\n")
            for row in comparison_rows:
                use = "include as supportive" if row["favorable_for_apwim"] else "treat cautiously"
                f.write(
                    f"| {row['model']} | {row['metric']} | {fmt(float(row['ap_mean']))} | "
                    f"{row['best_baseline']} | {fmt(float(row['baseline_mean']))} | "
                    f"{fmt(float(row['delta_ap_minus_baseline']))} | "
                    f"[{fmt(float(row['ci95_low']))}, {fmt(float(row['ci95_high']))}] | {use} |\n"
                )
        else:
            f.write("No complete AP-WIM comparison was available.\n")
        f.write("\n## Missing score files\n\n")
        if missing:
            for item in missing:
                f.write(f"- `{item}`\n")
        else:
            f.write("None.\n")
    print("wrote:", out_dir)
    if missing:
        print(f"missing score files: {len(missing)}")


if __name__ == "__main__":
    main()
