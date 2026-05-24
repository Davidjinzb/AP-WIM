#!/usr/bin/env python3
# benchmark_samples_fps_fixedbatch.py

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import device, setup_seed

from attr_methods_baseline import ig, fast_ig, sg, saliencymap, mfaba, la, agi
from attr_methods_2d import wam_ig as wam_ig_2d, wam_sg as wam_sg_2d, wam_mfaba as wam_mfaba_2d
from attr_methods_1d import wam_ig as wam_ig_1d, wam_sg as wam_sg_1d, wam_mfaba as wam_mfaba_1d

from datasets import IMAGENETSUBSET1000
from torchvision.models import efficientnet_b0


def cuda_sync():
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


class NormalizeModule(nn.Module):
    """Vectorized ImageNet normalization for [B,C,H,W]."""
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


def build_dataset(args):
    if args.dataset == "imagenet":
        return IMAGENETSUBSET1000()
    elif args.dataset == "esc50":
        from datasets import ESC50
        return ESC50(mode="test", num_FOLD=1)
    elif args.dataset == "fma":
        from datasets import FMA_SMALL
        return FMA_SMALL(mode="test")
    elif args.dataset == "audiomnist":
        from datasets import AudioMNIST
        return AudioMNIST(mode="test")
    elif args.dataset == "cub":
        from datasets import CUBDATASET
        return CUBDATASET(root="data/CUB/test")
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")


def build_model(args):
    sfmx = nn.Softmax(dim=1)

    if args.model == "efficient_b0" and args.dataset == "imagenet":
        norm = NormalizeModule(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        backbone = efficientnet_b0(pretrained=True)
        model = nn.Sequential(norm, backbone, sfmx).eval().to(device)
        return model

    elif args.model == "resnet18" and args.dataset == "esc50":
        from models import build_resnet18_esc50
        model = build_resnet18_esc50()
        model.load_state_dict(torch.load("resnet18_esc50_vanilla.pth"))
        model = nn.Sequential(model, sfmx).eval().to(device)
        return model

    elif args.model == "resnet18" and args.dataset == "fma":
        from models import build_resnet18_fma
        model = build_resnet18_fma()
        model.load_state_dict(torch.load("resnet18_fma_vanilla.pth"))
        model = nn.Sequential(model, sfmx).eval().to(device)
        return model

    elif args.model == "resnet18" and args.dataset == "audiomnist":
        from models import build_resnet18_audiomnist
        model = build_resnet18_audiomnist()
        model.load_state_dict(torch.load("resnet18_audiomnist_vanilla.pth"))
        model = nn.Sequential(model, sfmx).eval().to(device)
        return model

    elif args.model == "resnet50" and args.dataset == "cub":
        from models import build_resnet50_cub
        model = build_resnet50_cub()
        model.load_state_dict(torch.load("resnet50_cub_vanilla.pth"))
        model = nn.Sequential(model, sfmx).eval().to(device)
        return model

    elif args.model == "resnet50" and args.dataset == "esc50":
        from models import build_resnet50_esc50
        model = build_resnet50_esc50()
        model.load_state_dict(torch.load("runs_esc50/resnet50_fold1_seed42_1768497163/best.pth")["state_dict"])
        model = nn.Sequential(model, sfmx).eval().to(device)
        return model

    elif args.model == "mobilenet_v2" and args.dataset == "esc50":
        from models import build_mobilenetv2_esc50
        model = build_mobilenetv2_esc50()
        model.load_state_dict(torch.load("runs_esc50/mobilenet_v2_fold1_seed42_1768497160/best.pth")["state_dict"])
        model = nn.Sequential(model, sfmx).eval().to(device)
        return model

    else:
        raise NotImplementedError(f"Unsupported model/dataset: {args.model}/{args.dataset}")


def get_attr_dict(args):
    return {
        "ig": ig,
        "fast_ig": fast_ig,
        "sg": sg,
        "saliencymap": saliencymap,
        "mfaba": mfaba,
        "la": la,
        "agi": agi,
        "wam_ig": wam_ig_2d if args.dataset in ["imagenet", "cub"] else wam_ig_1d,
        "wam_sg": wam_sg_2d if args.dataset in ["imagenet", "cub"] else wam_sg_1d,
        "wam_mfaba": wam_mfaba_2d if args.dataset in ["imagenet", "cub"] else wam_mfaba_1d,
        "ap_wim": wam_mfaba_2d if args.dataset in ["imagenet", "cub"] else wam_mfaba_1d,
    }


def move_batch_to_target(batch, args):
    """只做一次 batch 准备，把需要上 GPU 的张量提前放上去；避免 benchmark 里含数据搬运。"""
    if args.dataset in ["imagenet", "cub"]:
        x, y = batch
        return (x.to(device), y.to(device))
    else:
        Xlgmel, y, waveform = batch
        # Xlgmel/y 上 GPU；waveform 保持 CPU（wam_1d 里通常说 cpu ok）
        return (Xlgmel.to(device), y.long().to(device), waveform)


def batch_size_of(batch, args):
    if args.dataset in ["imagenet", "cub"]:
        x, _y = batch
        return int(x.shape[0])
    else:
        _Xlgmel, y, _waveform = batch
        return int(y.shape[0])


def compute_attribution(method_name, method_fn, model, batch, args):
    if args.dataset in ["imagenet", "cub"]:
        x, y = batch
        if method_name == "ap_wim":
            return method_fn(model, x, y, steps=7)
        return method_fn(model, x, y)

    Xlgmel, y, waveform = batch

    # 1D WAM uses waveform
    if method_fn in [wam_ig_1d, wam_sg_1d, wam_mfaba_1d]:
        if method_name == "ap_wim":
            return method_fn(model, waveform, y, steps=3)
        return method_fn(model, waveform, y)

    # baseline special args
    if method_name == "mfaba":
        return method_fn(model, Xlgmel, y, data_min=None, data_max=None)
    if method_name in ["la", "agi"]:
        return method_fn(model, Xlgmel, y, clip_min=None, clip_max=None)

    return method_fn(model, Xlgmel, y)


def benchmark_fixed_batch(method_name, method_fn, model, batch, args):
    bs = batch_size_of(batch, args)

    # warmup（同一个 batch）
    torch.set_grad_enabled(True)
    for _ in range(args.warmup_iters):
        _ = compute_attribution(method_name, method_fn, model, batch, args)
    cuda_sync()

    # timed（同一个 batch）
    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = compute_attribution(method_name, method_fn, model, batch, args)
    cuda_sync()
    t1 = time.perf_counter()

    elapsed = max(t1 - t0, 1e-12)
    total_samples = bs * args.iters
    samples_per_sec = total_samples / elapsed
    ms_per_sample = (elapsed / total_samples) * 1000.0 if total_samples > 0 else float("inf")
    return samples_per_sec, ms_per_sample, elapsed, total_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="efficient_b0")
    ap.add_argument("--dataset", type=str, default="imagenet")
    ap.add_argument("--batch_size", type=int, default=8)

    # 关键：不遍历 dataloader 全部，只取一个 batch，然后重复 iters 次
    ap.add_argument("--iters", type=int, default=2, help="timed iterations on the SAME cached batch")
    ap.add_argument("--warmup_iters", type=int, default=2, help="warmup iterations on the SAME cached batch")

    ap.add_argument("--methods", type=str, default="all",
                    help='comma-separated, e.g. "ig,sg,wam_mfaba" or "all"')
    ap.add_argument("--output_dir", type=str, default=".")
    args = ap.parse_args()

    setup_seed(42)
    torch.backends.cudnn.benchmark = True

    dataset = build_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args)
    attr_dict = get_attr_dict(args)

    if args.methods.strip().lower() == "all":
        method_list = list(attr_dict.keys())
    else:
        method_list = [m.strip() for m in args.methods.split(",") if m.strip()]
        for m in method_list:
            if m not in attr_dict:
                raise ValueError(f"Unknown method: {m}. Available: {list(attr_dict.keys())}")

    # 只取一个 batch（一次性搬到需要的设备上），后续 benchmark 不再碰 dataloader
    first_batch = next(iter(dataloader))
    cached_batch = move_batch_to_target(first_batch, args)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} | Model: {args.model} | Batch size: {args.batch_size}")
    print(f"Warmup iters: {args.warmup_iters} | Timed iters: {args.iters}")
    print("NOTE: benchmark reuses ONE cached batch; DataLoader overhead is excluded.")
    print("-" * 96)
    print(f"{'method':12s} | {'samples/s':>12s} | {'ms/sample':>10s} | {'elapsed(s)':>10s} | {'N(samples)':>10s}")
    print("-" * 96)

    results = []
    for m in method_list:
        sps, msps, elapsed, n = benchmark_fixed_batch(m, attr_dict[m], model, cached_batch, args)
        results.append((m, sps, msps, elapsed, n))
        print(f"{m:12s} | {sps:12.2f} | {msps:10.3f} | {elapsed:10.3f} | {n:10d}")
    import pandas as pd
    df = pd.DataFrame(results, columns=["method", "samples_per_sec", "ms_per_sample", "elapsed_sec", "num_samples"])
    # 只保留前3列
    df = df[["method", "samples_per_sec", "ms_per_sample"]]
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"benchmark_{args.dataset}_{args.model}_bs{args.batch_size}.csv", index=False)
    print("-" * 96)
    results.sort(key=lambda x: x[1], reverse=True)
    print("Sorted by samples/s (desc):")
    for m, sps, msps, elapsed, n in results:
        print(f"{m:12s} | {sps:12.2f} samples/s | {msps:10.3f} ms/sample")


if __name__ == "__main__":
    main()
