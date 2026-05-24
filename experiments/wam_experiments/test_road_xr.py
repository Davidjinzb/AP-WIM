# road_eval.py
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import DataLoader
from utils import device, setup_seed

setup_seed(42)


# =========================================================
# AUC
# =========================================================
def auc_curve(arr_1d: np.ndarray) -> float:
    """Normalized AUC for a curve y over uniform x in [0,1]."""
    arr = np.asarray(arr_1d, dtype=np.float32)
    return float((arr.sum() - arr[0] / 2.0 - arr[-1] / 2.0) / (len(arr) - 1))


# =========================================================
# Baseline (fast) for ROAD perturb
# =========================================================
@torch.no_grad()
def gaussian_blur_nd(x: torch.Tensor, k=11, sigma=5.0):
    """
    Depthwise gaussian blur for 2D-like tensors:
      - images:  [B,C,H,W]
      - esc50:   [B,1,T,M]   (treat as H=T, W=M)
    """
    assert x.dim() == 4
    B, C, H, W = x.shape
    dtype = x.dtype
    dev = x.device

    coords = torch.arange(k, device=dev, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g_col = g.view(1, 1, k, 1)
    g_row = g.view(1, 1, 1, k)

    x = F.conv2d(x, g_col.expand(C, 1, k, 1), padding=(k // 2, 0), groups=C)
    x = F.conv2d(x, g_row.expand(C, 1, 1, k), padding=(0, k // 2), groups=C)
    return x


@torch.no_grad()
def make_baseline_like(x: torch.Tensor, mode: str):
    """
    x: [B,C,H,W] or [B,1,T,M]
    """
    if mode == "zeros":
        return torch.zeros_like(x)

    if mode == "noise":
        # per-sample per-channel mean/std noise (fits any domain better than U[0,1])
        B = x.shape[0]
        flat = x.view(B, x.shape[1], -1)  # [B,C,HW]
        mu = flat.mean(dim=2, keepdim=True)
        std = flat.std(dim=2, keepdim=True).clamp_min(1e-6)
        noise = torch.randn_like(flat) * std + mu
        return noise.view_as(x)

    if mode == "blur":
        return gaussian_blur_nd(x, k=11, sigma=5.0)

    raise ValueError(f"Unknown baseline mode: {mode}")


# =========================================================
# Attr shape helpers
# =========================================================
def _ensure_attr_1ch_4d(attr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Make attribution shape to [B,1,H,W] (or [B,1,T,M]).
    不随意加 abs，不改梯度，只做形状对齐 + reduce to 1channel（保留你原来意思）。
    """
    if attr.dim() == 2:
        # [B,L] -> [B,1,L] (not expected here)
        return attr.unsqueeze(1)
    if attr.dim() == 3:
        # [B,H,W] or [B,T,M] -> [B,1,H,W]
        return attr.unsqueeze(1)
    if attr.dim() == 4:
        if attr.shape[1] == 1:
            return attr
        # [B,C,H,W] -> [B,1,H,W]
        return attr.mean(dim=1, keepdim=True)
    raise ValueError(f"Unsupported attr dim={attr.dim()}, shape={tuple(attr.shape)}")


def _percentile_masks(attr_1ch: torch.Tensor, percentiles, largest=True):
    """
    attr_1ch: [B,1,H,W]
    returns list of bool masks [B,1,H,W]
    """
    B = attr_1ch.shape[0]
    flat = attr_1ch.view(B, -1)

    masks = []
    for p in percentiles:
        q = p / 100.0
        if largest:
            thr = torch.quantile(flat, 1.0 - q, dim=1, keepdim=True)
            m = flat >= thr
        else:
            thr = torch.quantile(flat, q, dim=1, keepdim=True)
            m = flat <= thr
        masks.append(m.view_as(attr_1ch))
    return masks


# =========================================================
# ROAD eval (one batch)
# =========================================================
@torch.no_grad()
def road_eval_one_batch(
    model,
    x: torch.Tensor,          # [B,C,H,W] or [B,1,T,M]
    y: torch.Tensor,          # [B] long
    attr_np: np.ndarray,      # slice from saved attributions
    percentiles=(10, 20, 30, 40, 50),
    baseline_mode="noise",
    use_pred_as_target=True,
):
    """
    model 输出已经是 softmax 概率（你说的“softmax 已经包括在模型里”）。
    Returns:
      morf_scores: [B,P+1]  (0 + each percentile)
      lerf_scores: [B,P+1]
    """
    model.eval()
    dev = next(model.parameters()).device

    x = x.to(dev).detach()
    y = y.to(dev).long()

    attr = torch.from_numpy(attr_np).to(dev, dtype=x.dtype)
    attr = _ensure_attr_1ch_4d(attr, x)  # [B,1,*,*]

    probs0 = model(x)  # [B,K] probabilities
    if use_pred_as_target:
        target = probs0.argmax(dim=1)
    else:
        target = y

    base_score = probs0.gather(1, target.view(-1, 1)).squeeze(1)  # [B]

    baseline = make_baseline_like(x, baseline_mode)

    morf_masks = _percentile_masks(attr, percentiles, largest=True)
    lerf_masks = _percentile_masks(attr, percentiles, largest=False)

    morf = [base_score]
    lerf = [base_score]

    for m in morf_masks:
        x_pert = x * (~m) + baseline * m
        p = model(x_pert).gather(1, target.view(-1, 1)).squeeze(1)
        morf.append(p)

    for m in lerf_masks:
        x_pert = x * (~m) + baseline * m
        p = model(x_pert).gather(1, target.view(-1, 1)).squeeze(1)
        lerf.append(p)

    morf = torch.stack(morf, dim=1).cpu().numpy().astype(np.float32)
    lerf = torch.stack(lerf, dim=1).cpu().numpy().astype(np.float32)
    return morf, lerf


# =========================================================
# main
# =========================================================
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='efficient_b0', help='efficient_b0 | resnet18')
    argparser.add_argument('--attr_method', type=str, default='wam_mfaba', help='attribution method name in filename')
    argparser.add_argument('--dataset', type=str, default='imagenet', help='imagenet | esc50')
    argparser.add_argument('--baseline', type=str, default='blur', choices=['zeros', 'noise', 'blur'])
    argparser.add_argument('--use_pred_target', action='store_true', default=True)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--percentiles', type=str, default='10,20,30,40,50,60,70,80,90')
    argparser.add_argument("--debug",action="store_true")
    argparser.add_argument('--steps', type=int, default=5)
    args = argparser.parse_args()

    percentiles = tuple(int(x) for x in args.percentiles.split(","))

    # ------------------ dataset / loader ------------------
    if args.dataset == 'imagenet':
        from datasets import IMAGENETSUBSET1000
        dataset = IMAGENETSUBSET1000()
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'esc50':
        from datasets import ESC50
        dataset = ESC50(mode='test', num_FOLD=1)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'fma':
        from datasets import FMA_SMALL
        dataset = FMA_SMALL(mode='test')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == "cub":
        from datasets import CUBDATASET
        dataset = CUBDATASET(root="data/CUB/test")
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == "audiomnist":
        from datasets import AudioMNIST
        dataset = AudioMNIST(mode='test')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    else:
        raise NotImplementedError

    # ------------------ model ------------------
    if args.dataset == 'imagenet' and args.model == 'efficient_b0':
        from torchvision.models import efficientnet_b0
        import torchvision.transforms as transforms

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        sfmx = nn.Softmax(dim=1)

        try:
            from torchvision.models import EfficientNet_B0_Weights
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            backbone = efficientnet_b0(pretrained=True)

        model = nn.Sequential(normalize, backbone, sfmx).eval().to(device)

    elif args.dataset == 'esc50' and args.model == 'resnet18':
        from models import build_resnet18_esc50
        model_core = build_resnet18_esc50()
        model_core.load_state_dict(torch.load("resnet18_esc50_vanilla.pth", map_location="cpu"))
        model = nn.Sequential(model_core, nn.Softmax(dim=1)).eval().to(device)
    elif args.dataset == 'fma' and args.model == 'resnet18':
        from models import build_resnet18_fma
        model_core = build_resnet18_fma()
        model_core.load_state_dict(torch.load("resnet18_fma_vanilla.pth", map_location="cpu"))
        model = nn.Sequential(model_core, nn.Softmax(dim=1)).eval().to(device)
    elif args.dataset == "audiomnist" and args.model == "resnet18":
        from models import build_resnet18_audiomnist
        model_core = build_resnet18_audiomnist()
        model_core.load_state_dict(torch.load("resnet18_audiomnist_vanilla.pth", map_location="cpu"))
        model = nn.Sequential(model_core, nn.Softmax(dim=1)).eval().to(device)
    elif args.dataset == "cub" and args.model == "resnet50":
        from models import build_resnet50_cub
        model_core = build_resnet50_cub()
        model_core.load_state_dict(torch.load("resnet50_cub_vanilla.pth", map_location="cpu"))
        model = nn.Sequential(model_core, nn.Softmax(dim=1)).eval().to(device)
    elif args.dataset == "esc50" and args.model == "resnet50":
        from models import build_resnet50_esc50
        model_core = build_resnet50_esc50()
        model_core.load_state_dict(torch.load("runs_esc50/resnet50_fold1_seed42_1768497163/best.pth", map_location="cpu")['state_dict'])
        model = nn.Sequential(model_core, nn.Softmax(dim=1)).eval().to(device)
    elif args.dataset == "esc50" and args.model == "mobilenet_v2":
        from models import build_mobilenetv2_esc50
        model_core = build_mobilenetv2_esc50()
        model_core.load_state_dict(torch.load("runs_esc50/mobilenet_v2_fold1_seed42_1768497160/best.pth", map_location="cpu")['state_dict'])
        model = nn.Sequential(model_core, nn.Softmax(dim=1)).eval().to(device)
    else:
        raise NotImplementedError(f"Unsupported: dataset={args.dataset}, model={args.model}")

    # ------------------ load attributions ------------------
    attr_path = f'attributions/{args.dataset}_{args.model}_{args.attr_method}_attributions_{args.steps}.npy'
    attributions = np.load(attr_path)

    assert attributions.shape[0] == len(dataset), f"attr N={attributions.shape[0]} != dataset N={len(dataset)}"

    # ------------------ output ------------------
    os.makedirs('scores', exist_ok=True)
    out_path = f'scores/{args.dataset}_{args.model}_{args.attr_method}_road_scores_{args.steps}.npz'
    # if os.path.exists(out_path) and not args.debug:
    #     print("exists:", out_path)
    #     return
    if not os.path.exists(out_path) or args.debug:

        # ------------------ eval ------------------
        all_morf, all_lerf = [], []

        for idx, batch in enumerate(tqdm(dataloader)):
            if args.dataset == 'imagenet' or args.dataset == "cub":
                x, y = batch
                x = x.to(device)
                y = y.to(device).long()

            else:  # esc50
                # 兼容 (Xlgmel, y, waveform, ...) 或 (Xlgmel, y, ...)
                Xlgmel = batch[0]
                y = batch[1]
                x = Xlgmel.to(device)          # [B,1,T,M]
                y = y.to(device).long()

            s = idx * args.batch_size
            e = s + x.shape[0]
            attr_b = attributions[s:e]

            morf, lerf = road_eval_one_batch(
                model, x, y, attr_b,
                percentiles=percentiles,
                baseline_mode=args.baseline,
                use_pred_as_target=args.use_pred_target,
            )
            all_morf.append(morf)
            all_lerf.append(lerf)

        all_morf = np.concatenate(all_morf, axis=0)  # [N,P+1]
        all_lerf = np.concatenate(all_lerf, axis=0)

        morf_auc = auc_curve(all_morf.mean(axis=0))
        lerf_auc = auc_curve(all_lerf.mean(axis=0))
        road_score = lerf_auc - morf_auc

        print(f"[ROAD] dataset={args.dataset}, model={args.model}, attr={args.attr_method}, baseline={args.baseline}")
        print(f"  percentiles: {(0,) + percentiles}")
        print(f"  MORF AUC: {morf_auc:.4f}")
        print(f"  LERF AUC: {lerf_auc:.4f}")
        print(f"  ROAD score (LERF - MORF): {road_score:.4f}")

        np.savez(
            out_path,
            percentiles=np.array((0,) + percentiles, dtype=np.int32),
            morf_scores=all_morf,
            lerf_scores=all_lerf,
            morf_auc=np.array(morf_auc, dtype=np.float32),
            lerf_auc=np.array(lerf_auc, dtype=np.float32),
            road_score=np.array(road_score, dtype=np.float32),
            baseline=np.array(args.baseline),
            use_pred_target=np.array(int(args.use_pred_target)),
            dataset=np.array(args.dataset),
            model=np.array(args.model),
            attr_method=np.array(args.attr_method),
        )
        print("saved:", out_path)


if __name__ == "__main__":
    main()
