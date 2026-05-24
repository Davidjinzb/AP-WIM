# calc_infid_like_causal.py
# 按你 causal metric 的方式加载 dataset / dataloader / model
# 计算 scale-invariant INFID（输入域高斯噪声扰动）
#
# 输出：
#   scores/{dataset}_{model}_{attr_method}_infid_scores.npy   每个样本一个分数
# 同时打印 mean/std

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import device, setup_seed
from apwim_experiment_common import (
    build_dataset,
    build_model,
    infid_score_path,
    resolve_existing_attribution,
    save_npy_atomic,
)

FORWARD_BZ = 32  # perturb forward 的小 batch，显存不够就调小


def _match_attr_to_input(attr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    infidelity 里需要 <noise, attr>，要求 attr 跟 x 同 shape（用于 dot-product）
    你保存的通常是 [B,1,...]：
      - ImageNet: x=[B,3,H,W], attr=[B,1,H,W] -> repeat 到 3 通道
      - ESC50   : x=[B,1,T,M], attr=[B,1,T,M] -> 原样
    """
    if x.dim() != attr.dim():
        raise ValueError(f"attr.dim={attr.dim()} != x.dim={x.dim()}")

    if x.dim() == 4:
        if attr.shape[1] == 1 and x.shape[1] != 1:
            return attr.repeat(1, x.shape[1], 1, 1)
        return attr

    raise ValueError(f"Unsupported input dim: {x.dim()}")


@torch.no_grad()
def _forward_in_chunks(model: nn.Module, x: torch.Tensor, bz: int) -> torch.Tensor:
    outs = []
    n = x.shape[0]
    for s in range(0, n, bz):
        outs.append(model(x[s:s + bz]))
    return torch.cat(outs, dim=0)


def infid_scale_invariant_one(
    model: nn.Module,
    x1: torch.Tensor,          # [1,C,...]
    y1: int,
    attr1: torch.Tensor,       # [1,C,...] after match
    *,
    n_perturb: int = 100,
    sigma_mode: str = "per_sample_std",  # fixed / per_sample_std / per_sample_range
    sigma: float = 0.01,
    sigma_ratio: float = 0.2,
    use_prob: bool = False,     # True: 用 softmax 概率作为 f(x)；False: 用 logits（推荐）
    denom: str = "energy",      # energy(推荐) / none
    clip_inputs: bool = False,  # 仅建议 ImageNet 输入域确实在[0,1]时开启
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    forward_bz: int = FORWARD_BZ,
    eps: float = 1e-12,
) -> float:
    """
    Scale-invariant INFID（稳定版）：
      df = f(x) - f(x - I)
      e  = <I, attr>
      beta = E[df*e] / E[e^2]
      mse  = E[(df - beta*e)^2]
      score = mse / E[df^2]   (denom="energy")  ✅避免 df 很小导致爆炸
    """
    dev = x1.device
    x1 = x1.float()
    attr1 = attr1.float()

    # f(x)
    out_x = model(x1)  # [1,K] logits 或 probs（取决于你模型是否带 softmax）
    if use_prob:
        fx = torch.softmax(out_x, dim=1)[0, y1]
    else:
        fx = out_x[0, y1]

    # per-sample sigma
    flat = x1.view(1, -1)
    if sigma_mode == "fixed":
        s = torch.tensor(float(sigma), device=dev)
    elif sigma_mode == "per_sample_std":
        s = flat.std(dim=1, unbiased=False).clamp_min(eps)[0] * float(sigma_ratio)
    elif sigma_mode == "per_sample_range":
        s = (flat.max(dim=1).values - flat.min(dim=1).values).clamp_min(eps)[0] * float(sigma_ratio)
    else:
        raise ValueError(f"unknown sigma_mode={sigma_mode}")

    # noise & perturbed inputs
    noise = torch.randn((n_perturb,) + tuple(x1.shape[1:]), device=dev) * s
    x_pert = x1.expand(n_perturb, *x1.shape[1:]) - noise

    if clip_inputs:
        x_pert = x_pert.clamp(clip_min, clip_max)

    out_pert = _forward_in_chunks(model, x_pert, forward_bz)  # [n_perturb,K]
    if use_prob:
        f_pert = torch.softmax(out_pert, dim=1)[:, y1]
    else:
        f_pert = out_pert[:, y1]

    df = (fx - f_pert).double()  # [N]

    # e = <noise, attr>
    attr_flat = attr1.view(-1).double()
    e = (noise.view(n_perturb, -1).double() @ attr_flat)  # [N]

    # optimal scaling beta
    e2 = (e * e).mean().clamp_min(eps)
    beta = (df * e).mean() / e2

    mse = ((df - beta * e) ** 2).mean()

    if denom == "none":
        return float(mse.item())
    elif denom == "energy":
        df2 = (df * df).mean().clamp_min(eps)
        return float((mse / df2).item())
    else:
        raise ValueError(f"unknown denom={denom}")


def main():
    setup_seed(42)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='efficient_b0')
    argparser.add_argument('--attr_method', type=str, default='wam_mfaba')
    argparser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'esc50'])

    # INFID params
    argparser.add_argument('--n_perturb', type=int, default=100)
    argparser.add_argument('--sigma_mode', type=str, default='per_sample_std',
                           choices=['fixed', 'per_sample_std', 'per_sample_range'])
    argparser.add_argument('--sigma', type=float, default=0.01)
    argparser.add_argument('--sigma_ratio', type=float, default=0.2)
    argparser.add_argument('--use_prob', action='store_true', help='用 softmax 概率当 f(x)，默认用 logits')
    argparser.add_argument('--denom', type=str, default='energy', choices=['energy', 'none'])
    argparser.add_argument('--clip_inputs', action='store_true', help='只建议 ImageNet 输入在[0,1]时开')
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--forward_bz', type=int, default=FORWARD_BZ)
    argparser.add_argument('--max_samples', type=int, default=None)
    argparser.add_argument('--attribution_dir', type=str, default='attributions')
    argparser.add_argument('--score_dir', type=str, default='scores')
    argparser.add_argument('--output_tag', type=str, default='')
    argparser.add_argument('--resume', action='store_true')
    argparser.add_argument('--force', action='store_true')
    argparser.add_argument('--save_every', type=int, default=25)

    args = argparser.parse_args()

    # -----------------------------
    # dataset / dataloader (按你 causal 的写法)
    # -----------------------------
    dataset = build_dataset(args.dataset, max_samples=args.max_samples)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # -----------------------------
    # model (按你 causal 的写法)
    # -----------------------------
    # 为了 INFID 更稳，默认使用 logits；--use_prob 会在 infid 函数内部转 softmax。
    model = build_model(args.dataset, args.model, softmax=False)

    model = nn.DataParallel(model)

    # -----------------------------
    # attributions (按你 causal 的路径)
    # -----------------------------
    os.makedirs(args.score_dir, exist_ok=True)
    out_path = infid_score_path(
        args.score_dir,
        args.dataset,
        args.model,
        args.attr_method,
        output_tag=args.output_tag,
    )
    partial_path = out_path.with_name(out_path.name + ".partial.npy")
    if out_path.exists() and not args.force:
        print("exists:", out_path)
        return

    attr_path = resolve_existing_attribution(
        args.attribution_dir,
        args.dataset,
        args.model,
        args.attr_method,
        output_tag=args.output_tag,
    )
    attributions = np.load(attr_path)
    print(f'[LOAD] {attr_path} shape={attributions.shape} '
          f'min={attributions.min():.6f} max={attributions.max():.6f} mean={attributions.mean():.6f}')

    # -----------------------------
    # compute INFID
    # -----------------------------
    all_scores = []
    completed = 0
    if args.resume and partial_path.exists() and not args.force:
        all_scores = list(np.load(partial_path).astype(np.float32))
        completed = len(all_scores)
        print(f"[RESUME] loaded {completed} partial INFID scores from {partial_path}")
    ptr = 0

    for batch in tqdm(dataloader, desc="INFID(scale-invariant)"):
        # 取 x,y（按你 causal 风格：esc50 可能有多余返回）
        if args.dataset == 'imagenet':
            x, y = batch
        else:
            x = batch[0]
            y = batch[1]

        bsz = x.shape[0]
        attr_np = attributions[ptr:ptr + bsz]
        batch_start = ptr
        ptr += bsz

        # to torch
        x = x.to(device)
        y = y.to(device).long()
        attr = torch.from_numpy(attr_np).to(device=device, dtype=torch.float32)

        # 对齐 attr -> x 的 shape（ImageNet: 1ch -> 3ch）
        attr = _match_attr_to_input(attr, x)

        # 逐样本算（避免 [B,N,...] 爆显存）
        for i in range(bsz):
            global_index = batch_start + i
            if global_index < completed:
                continue
            score = infid_scale_invariant_one(
                model=model,
                x1=x[i:i+1],
                y1=int(y[i].item()),
                attr1=attr[i:i+1],
                n_perturb=args.n_perturb,
                sigma_mode=args.sigma_mode,
                sigma=args.sigma,
                sigma_ratio=args.sigma_ratio,
                use_prob=args.use_prob,
                denom=args.denom,
                clip_inputs=(args.clip_inputs and args.dataset == 'imagenet'),
                forward_bz=args.forward_bz,
            )
            all_scores.append(score)
            if args.save_every > 0 and len(all_scores) % args.save_every == 0:
                save_npy_atomic(partial_path, np.array(all_scores, dtype=np.float32))

    scores = np.array(all_scores, dtype=np.float32)
    save_npy_atomic(out_path, scores)
    if partial_path.exists():
        os.remove(partial_path)
    print(f'[INFID] mean={scores.mean():.6f} std={scores.std():.6f} n={scores.shape[0]} saved: {out_path}')


if __name__ == '__main__':
    main()
