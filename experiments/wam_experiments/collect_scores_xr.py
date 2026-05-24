import pandas as pd
import numpy as np
from evaluation import auc_curve
import os
import argparse
from apwim_experiment_common import causal_score_path, method_aliases, road_score_path
attr_methods = [
    # "wam_ig",
    # "wam_sg",
    "ap_wim",
    # "ig",
    # "fast_ig",
    # "sg",
    # "saliencymap",
    # "mfaba",
    # "la",
    # "agi"
]
datasets = [
    "imagenet",
    "esc50",
    # "esc50",
    "esc50",
    # "fma",
    # "audiomnist",
    # "cub"
]
models = [
    "efficient_b0",
    # "resnet18",
    # "resnet18",
    # "resnet18",
    "resnet50",
    "mobilenet_v2"
]
steps = [1,3,5,7,10,13,15,17,20]


def _first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _causal_candidates(score_dir, dataset, model, method, step):
    return [
        causal_score_path(score_dir, dataset, model, alias, steps=step)
        for alias in method_aliases(method)
    ]


def _road_candidates(score_dir, dataset, model, method, step, baseline, target):
    paths = []
    for alias in method_aliases(method):
        paths.append(
            road_score_path(
                score_dir,
                dataset,
                model,
                alias,
                steps=step,
                baseline=baseline,
                target=target,
            )
        )
        paths.append(
            road_score_path(
                score_dir,
                dataset,
                model,
                alias,
                steps=step,
                baseline=baseline,
                target=target,
                legacy=True,
            )
        )
    return paths


parser = argparse.ArgumentParser()
parser.add_argument("--score_dir", type=str, default="scores")
parser.add_argument("--output_csv", type=str, default="causal_evaluation_results_xr.csv")
parser.add_argument("--road_baseline", type=str, default="blur")
parser.add_argument("--road_target", type=str, default="pred")
parser.add_argument("--include_missing", action="store_true", help="Keep rows whose causal and ROAD files are both missing.")
args = parser.parse_args()

results = []
# for model in models:
#     for dataset in datasets:
for model,dataset in zip(models,datasets):
        for attr_method in attr_methods:
            for step in steps:
                # if os.path.exists(f'scores/{dataset}_{model}_{attr_method}_causal_scores.npz'):
                causal_path = _first_existing(_causal_candidates(args.score_dir, dataset, model, attr_method, step))
                if causal_path is not None:
                    scores_insert_deletion = np.load(causal_path)
                    insertion_auc = auc_curve(scores_insert_deletion['insertion_scores'].mean(0))
                    deletion_auc = auc_curve(scores_insert_deletion['deletion_scores'].mean(0))
                else:
                    insertion_auc = np.nan
                    deletion_auc = np.nan
                road_path = _first_existing(_road_candidates(args.score_dir, dataset, model, attr_method, step, args.road_baseline, args.road_target))
                if road_path is not None:
                    scores_road = np.load(road_path)
                    road_score = scores_road['road_score']
                else:
                    road_score = np.nan
                if causal_path is None and road_path is None and not args.include_missing:
                    continue
                # scores_infid = np.load(f'scores/{dataset}_{model}_{attr_method}_infid_scores.npy')
                # insertion_auc = scores['insertion_scores'].mean(0)
                # deletion_auc = scores['deletion_scores'].mean(0)
                
                results.append({
                    "model": model,
                    "dataset": dataset,
                    "attr_method": attr_method,
                    "insertion": insertion_auc,
                    "deletion": deletion_auc,
                    "road_score": road_score,
                    "causal_path": "" if causal_path is None else str(causal_path),
                    "road_path": "" if road_path is None else str(road_path),
                    # "infid_score":scores_infid.mean(),
                    "faithfulness_gap": insertion_auc - deletion_auc,
                    "steps": step
                })
df = pd.DataFrame(results)
# df.to_csv('causal_evaluation_results.csv',index=False)
# keep .4f
df['insertion'] = df['insertion'].map(lambda x: f"{x:.4f}")
df['deletion'] = df['deletion'].map(lambda x: f"{x:.4f}")
df['faithfulness_gap'] = df['faithfulness_gap'].map(lambda x: f"{x:.4f}")
# df['road_score'] = df['road_score'].map(lambda x: f"{x:.4f}")
# df['infid_score'] = df['infid_score'].map(lambda x: f"{x:.4f}")
print(df)
df.to_csv(args.output_csv,index=False)
