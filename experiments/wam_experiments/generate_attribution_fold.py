from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from apwim_experiment_common import (
    attribution_path,
    call_attribution,
    clean_tag,
    default_ap_wim_steps,
    dispatch_attr_method,
    get_attr_dict,
    is_wavelet_method,
    save_npy_atomic,
)
from datasets import ESC50
from models import build_mobilenetv2_esc50, build_resnet50_esc50
from utils import device, setup_seed


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
        raise ValueError(f"Unsupported ESC-50 model for multifold eval: {model_name}")

    ckpt = checkpoint_path(model_name, fold, ckpt_root)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model.load_state_dict(load_state_dict(ckpt))
    return nn.Sequential(model, nn.Softmax(dim=1)).eval().to(device)


def main() -> None:
    setup_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet50", "mobilenet_v2"])
    parser.add_argument("--fold", required=True, type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--attr_method", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--attribution_dir", type=str, default="attributions_multifold")
    parser.add_argument("--output_tag", type=str, default="")
    parser.add_argument("--ckpt_root", type=str, default="runs/E1_imagenet_20260509_200917/runs_esc50")
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--J", type=int, default=3)
    parser.add_argument("--approx_coeffs", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tag = clean_tag(args.output_tag or f"fold{args.fold}")
    dataset = ESC50(mode="test", num_FOLD=args.fold, root_dir=os.environ.get("APWIM_ESC50_ROOT"))
    if args.max_samples is not None and args.max_samples > 0:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = build_fold_model(args.model, args.fold, args.ckpt_root)

    out_path = attribution_path(
        args.attribution_dir,
        "esc50",
        args.model,
        args.attr_method,
        steps=args.steps,
        output_tag=tag,
    )
    if out_path.exists() and not args.force:
        print("exists:", out_path)
        return

    dispatch_method = dispatch_attr_method(args.attr_method)
    steps = args.steps
    if dispatch_method == "wam_mfaba" and steps is None:
        steps = default_ap_wim_steps("esc50")

    attr_dict = get_attr_dict("esc50")
    if args.attr_method not in attr_dict:
        raise ValueError(f"Unknown attr_method={args.attr_method}. Available: {sorted(attr_dict)}")
    attr_fn = attr_dict[args.attr_method]

    print(
        f"[ATTR-FOLD] model={args.model} fold={args.fold} method={args.attr_method} "
        f"samples={len(dataset)} batch_size={args.batch_size} tag={tag}"
    )
    print(f"[ATTR-FOLD] checkpoint={checkpoint_path(args.model, args.fold, args.ckpt_root)}")
    if is_wavelet_method(args.attr_method):
        print(f"[ATTR-FOLD] wavelet={args.wavelet} J={args.J} steps={steps}")
    print(f"[ATTR-FOLD] output={out_path}")

    attributions = []
    for batch in tqdm(loader, desc=f"attr fold{args.fold} {args.model} {args.attr_method}"):
        attr = call_attribution(
            args.attr_method,
            attr_fn,
            model,
            "esc50",
            batch,
            wavelet=args.wavelet,
            J=args.J,
            approx_coeffs=args.approx_coeffs,
            steps=steps,
        )
        attributions.append(attr)

    arr = np.concatenate(attributions, axis=0)
    print(arr.shape, arr.min(), arr.max(), arr.mean())
    save_npy_atomic(out_path, arr)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
