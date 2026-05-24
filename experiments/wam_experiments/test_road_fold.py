from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from apwim_experiment_common import (
    clean_tag,
    resolve_existing_attribution,
    road_score_path,
    save_npz_atomic,
)
from datasets import ESC50
from evaluation import auc_curve
from models import build_mobilenetv2_esc50, build_resnet50_esc50
from test_road import road_eval_one_batch
from utils import device, setup_seed
import torch


def load_state_dict(path: str | Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def checkpoint_path(model_name: str, fold: int, ckpt_root: str | Path) -> Path:
    return Path(ckpt_root) / f"{model_name}_fold{fold}_seed42" / "best.pth"


def build_fold_model(model_name: str, fold: int, ckpt_root: str | Path) -> nn.Module:
    if model_name == "resnet50":
        model = build_resnet50_esc50()
    elif model_name == "mobilenet_v2":
        model = build_mobilenetv2_esc50()
    else:
        raise ValueError(f"Unsupported ESC-50 model for multifold ROAD: {model_name}")
    model.load_state_dict(load_state_dict(checkpoint_path(model_name, fold, ckpt_root)))
    return nn.Sequential(model, nn.Softmax(dim=1)).eval().to(device)


def main() -> None:
    setup_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet50", "mobilenet_v2"])
    parser.add_argument("--fold", required=True, type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--attr_method", required=True)
    parser.add_argument("--baseline", type=str, default="blur", choices=["zeros", "noise", "blur"])
    parser.add_argument("--target", type=str, default="pred", choices=["pred", "true"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--attribution_dir", type=str, default="attributions_multifold")
    parser.add_argument("--score_dir", type=str, default="scores_multifold")
    parser.add_argument("--output_tag", type=str, default="")
    parser.add_argument("--ckpt_root", type=str, default="runs/E1_imagenet_20260509_200917/runs_esc50")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--percentiles", type=str, default="10,20,30,40,50,60,70,80,90")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tag = clean_tag(args.output_tag or f"fold{args.fold}")
    percentiles = tuple(int(x) for x in args.percentiles.split(",") if x)
    dataset = ESC50(mode="test", num_FOLD=args.fold, root_dir=os.environ.get("APWIM_ESC50_ROOT"))
    if args.max_samples is not None and args.max_samples > 0:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = build_fold_model(args.model, args.fold, args.ckpt_root)

    out_path = road_score_path(
        args.score_dir,
        "esc50",
        args.model,
        args.attr_method,
        steps=args.steps,
        output_tag=tag,
        baseline=args.baseline,
        target=args.target,
    )
    if out_path.exists() and not args.force:
        print("exists:", out_path)
        return

    attr_path = resolve_existing_attribution(
        args.attribution_dir,
        "esc50",
        args.model,
        args.attr_method,
        steps=args.steps,
        output_tag=tag,
    )
    attributions = np.load(attr_path)

    print(
        f"[ROAD-FOLD] model={args.model} fold={args.fold} method={args.attr_method} "
        f"samples={len(dataset)} baseline={args.baseline} target={args.target} tag={tag}"
    )
    print(f"[ROAD-FOLD] attribution={attr_path}")
    print(f"[ROAD-FOLD] output={out_path}")

    all_morf = []
    all_lerf = []
    for idx, batch in enumerate(tqdm(loader, desc=f"road fold{args.fold} {args.model} {args.attr_method}")):
        x = batch[0].to(device)
        y = batch[1].to(device).long()
        start = idx * args.batch_size
        end = start + x.shape[0]
        attr = attributions[start:end]
        morf, lerf = road_eval_one_batch(
            model,
            x,
            y,
            attr,
            percentiles=percentiles,
            baseline_mode=args.baseline,
            use_pred_as_target=(args.target == "pred"),
        )
        all_morf.append(morf)
        all_lerf.append(lerf)

    morf_scores = np.concatenate(all_morf, axis=0)
    lerf_scores = np.concatenate(all_lerf, axis=0)
    morf_auc = auc_curve(morf_scores.mean(axis=0))
    lerf_auc = auc_curve(lerf_scores.mean(axis=0))
    road_score = lerf_auc - morf_auc
    print(f"MORF AUC: {morf_auc:.4f}")
    print(f"LERF AUC: {lerf_auc:.4f}")
    print(f"ROAD: {road_score:.4f}")
    save_npz_atomic(
        out_path,
        percentiles=np.array((0,) + percentiles, dtype=np.int32),
        morf_scores=morf_scores,
        lerf_scores=lerf_scores,
        morf_auc=np.array(morf_auc, dtype=np.float32),
        lerf_auc=np.array(lerf_auc, dtype=np.float32),
        road_score=np.array(road_score, dtype=np.float32),
        baseline=np.array(args.baseline),
        target=np.array(args.target),
        dataset=np.array("esc50"),
        model=np.array(args.model),
        fold=np.array(args.fold, dtype=np.int32),
        attr_method=np.array(args.attr_method),
    )
    print("saved:", out_path)


if __name__ == "__main__":
    main()
