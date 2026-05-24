# -*- coding: utf-8 -*-
"""
这个脚本做的事（只跑 1 个 batch）：
1) 用 EfficientNetB0 在 IMAGENETSUBSET1000 上取一个 batch 图片
2) 对 batch 做一次 WAM-MFABA（小波域的 MFABA 归因）
3) 保存 5 张图到文件夹 OUT_DIR：
   (1) mfaba_s0_4dirs_bar.png      : 四个方向(Approx/H/V/D)的“总体重要性”(raw + normalized)
   (2) mfaba_s0_4dirs_perlevel.png : 每个小波尺度(level)上 H/V/D 的重要性曲线(raw + normalized)
   (3) heatmaps_s0_level-1.png     : 最细尺度(level=-1)的 A/H/V/D 四张热力图
   (4) heatmaps_s0_level0.png      : 最粗细节尺度(level=0)的 A/H/V/D 四张热力图
   (5) overlay_s0.png              : 最终 pixel_attr 叠加到原图上的可视化

重要概念（你画的图要解释清楚）：
- WAM-MFABA 产生的是“小波系数空间”的 attribution（attr_np），它按层(level)存了：
    attr_np[0]        : Approx(低频/粗略结构)系数块
    attr_np[1:], 每层 : Detail 系数三方向 (H,V,D)
- H/V/D 在二维小波里通常对应：
    H: 对“水平细节”敏感（更像检测水平边/横向纹理）
    V: 对“垂直细节”敏感（更像检测竖直边/纵向纹理）
    D: 对“对角细节”敏感（更像检测斜边/斜纹理）
  （注意：这是对二维图像小波的常见解释，和你用的 wavelet="haar" 一致）

- 你脚本里还有一步：wavelet_layout_to_pixel_attr
  它把“小波布局空间”的图（layout）重新投影回像素域，得到 pixel_attr，
  这样才能做 overlay（把归因热力图叠在原图上）
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import ptwt
import cv2

from torchvision import transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader

from datasets import IMAGENETSUBSET1000
from utils import device
from utils import wavelet_layout_to_pixel_attr


# =========================
# Wavelet WAM core
# =========================
def _assert_same_device(model, x, y):
    """
    这里强制保证：
    - 输入 x 和标签 y 在同一个 device 上（GPU/CPU），否则会 silent 变慢或报错
    - y 必须是 long，因为 gather 需要 long index
    """
    dev_m = next(model.parameters()).device
    if x.device != dev_m:
        raise RuntimeError(f"x.device={x.device} != model.device={dev_m}")
    if y.device != dev_m:
        raise RuntimeError(f"y.device={y.device} != model.device={dev_m}")
    if y.dtype != torch.long:
        raise RuntimeError(f"y.dtype must be torch.long, got {y.dtype}")


def _require_grad_coeffs(coeffs):
    """
    wavedec2 返回的 coeffs 结构是：
      coeffs[0] : approx tensor
      coeffs[1:] : detail tuple (horizontal, vertical, diagonal)

    我们需要对每个系数张量 requires_grad=True，
    这样反传时能得到 loss 对小波系数的梯度。
    """
    grad_coeffs = [coeffs[0].requires_grad_()]
    for c in coeffs[1:]:
        grad_coeffs.append(
            ptwt.constants.WaveletDetailTuple2d(
                c.horizontal.requires_grad_(),
                c.vertical.requires_grad_(),
                c.diagonal.requires_grad_(),
            )
        )
    return grad_coeffs


def _coeffs_value_numpy(coeffs):
    """
    把 coeffs（torch tensor / detail tuple）变成 numpy 结构（同样的层级结构不变）
    输出：
      out[0] : approx np array  [B,C,h,w]
      out[1:] : detail tuple，每个有 .horizontal/.vertical/.diagonal -> np array [B,C,h,w]
    """
    out = [coeffs[0].detach().cpu().numpy()]
    for c in coeffs[1:]:
        out.append(
            ptwt.constants.WaveletDetailTuple2d(
                c.horizontal.detach().cpu().numpy(),
                c.vertical.detach().cpu().numpy(),
                c.diagonal.detach().cpu().numpy(),
            )
        )
    return out


def _visualize_grad_wam_layout(grad_coeffs_np, normalize_coeffs=True):
    """
    这一步做的是：把“小波系数结构”拼成一个 layout 图（B,H,W）
    layout 的意义是“小波布局空间”：
      - 左上角是 approx block（低频，最粗尺度）
      - 右上、左下、右下分别是各层的 H/V/D 细节块（从粗到细）

    注意：
    - 这个 layout 不是像素空间，不可直接叠在原图上
    - normalize_coeffs=True 时，每一块会各自 max-norm（更适合看结构，但会扭曲大小对比）
    """
    B = grad_coeffs_np[0].shape[0]
    H = int(2 * grad_coeffs_np[-1].horizontal.shape[-1])  # 由最细层尺寸推回原图 H
    layout = np.zeros((B, H, H), dtype=np.float32)

    # approx block (top-left)
    approx = np.abs(grad_coeffs_np[0].mean(axis=1))  # channel 平均 -> [B,h,w]
    if normalize_coeffs and approx.max() > 0:
        approx = approx / (approx.max() + 1e-12)
    layout[:, :approx.shape[1], :approx.shape[2]] = approx

    # details blocks (from coarse->fine layout placing)
    for i, c in enumerate(grad_coeffs_np[1:][::-1]):
        end = int(H / 2**i)
        start = int(H / 2**(i + 1))

        h = np.abs(c.horizontal.mean(axis=1))
        v = np.abs(c.vertical.mean(axis=1))
        d = np.abs(c.diagonal.mean(axis=1))

        if normalize_coeffs:
            if h.max() > 0: h = h / (h.max() + 1e-12)
            if v.max() > 0: v = v / (v.max() + 1e-12)
            if d.max() > 0: d = d / (d.max() + 1e-12)

        # layout 的摆放规则：
        # 右上 = horizontal, 左下 = vertical, 右下 = diagonal（对应常见小波排布）
        layout[:, start:end, start:end] = d
        layout[:, start:end, :start] = v
        layout[:, :start, start:end] = h

    return layout


def wam_mfaba(
    model,
    x,                     # [B,C,H,W] 输入图像（像素域）
    y,                     # [B] long 目标类别
    wavelet="haar",
    J=3,
    mode="reflect",
    steps=10,
):
    """
    WAM-MFABA：
    - 在“小波系数空间”里做一个多步迭代（steps 次）
    - 每步：
        1) 对 coeffs 反传，得到 grad wrt coeffs
        2) 使用 MFABA 的累计规则累积 attributions
        3) 使用 sign(grad) 更新 coeffs（类似迭代式/对抗式更新）
    - 最终：
        attr_np  : 小波系数归因（结构化保存：A + 每层(H,V,D)）
        pixel_attr: 把小波布局投影回像素域后的归因 [B,1,H,W]（可叠图）
    """
    _assert_same_device(model, x, y)

    # 1) 图像 -> 小波系数
    coeffs = ptwt.wavedec2(x, wavelet, level=J, mode=mode)

    # 2) learning rate: 每步走 coeff/steps（保持在同 device 上）
    lrs = []
    for c in coeffs:
        if isinstance(c, ptwt.constants.WaveletDetailTuple2d):
            lrs.append(
                ptwt.constants.WaveletDetailTuple2d(
                    c.horizontal / steps,
                    c.vertical / steps,
                    c.diagonal / steps,
                )
            )
        else:
            lrs.append(c / steps)

    attributions = None

    for _ in range(steps):
        # ---- compute grads wrt coeffs ----
        grad_coeffs = _require_grad_coeffs(coeffs)
        img = ptwt.waverec2(grad_coeffs, wavelet)  # coeffs -> 重建图像

        logits = model(img)
        loss = logits.gather(1, y.view(-1, 1)).mean()

        model.zero_grad(set_to_none=True)
        loss.backward()

        # ---- accumulate attr and update coeffs ----
        new_coeffs = []
        if attributions is None:
            attributions = []

        # approx 更新
        g0 = grad_coeffs[0].grad
        lr0 = lrs[0]
        if len(attributions) == 0:
            attributions.append(lr0 * torch.sign(g0) * g0)
        else:
            attributions[0] = attributions[0] + lr0 * torch.sign(g0) * g0
        new_coeffs.append((coeffs[0] + lr0 * torch.sign(g0)).detach())

        # detail 三方向更新
        for idx, (c, gc, lr) in enumerate(zip(coeffs[1:], grad_coeffs[1:], lrs[1:]), start=1):
            gh, gv, gd = gc.horizontal.grad, gc.vertical.grad, gc.diagonal.grad

            if len(attributions) <= idx:
                attributions.append(
                    ptwt.constants.WaveletDetailTuple2d(
                        lr.horizontal * torch.sign(gh) * gh,
                        lr.vertical   * torch.sign(gv) * gv,
                        lr.diagonal   * torch.sign(gd) * gd,
                    )
                )
            else:
                a = attributions[idx]
                attributions[idx] = ptwt.constants.WaveletDetailTuple2d(
                    a.horizontal + lr.horizontal * torch.sign(gh) * gh,
                    a.vertical   + lr.vertical   * torch.sign(gv) * gv,
                    a.diagonal   + lr.diagonal   * torch.sign(gd) * gd,
                )

            new_coeffs.append(
                ptwt.constants.WaveletDetailTuple2d(
                    (c.horizontal + lr.horizontal * torch.sign(gh)).detach(),
                    (c.vertical   + lr.vertical   * torch.sign(gv)).detach(),
                    (c.diagonal   + lr.diagonal   * torch.sign(gd)).detach(),
                )
            )

        coeffs = new_coeffs

    # 3) 平均化（让 attributions 不随 steps 放大）
    attributions = [
        (a / steps) if not isinstance(a, ptwt.constants.WaveletDetailTuple2d)
        else ptwt.constants.WaveletDetailTuple2d(a.horizontal / steps, a.vertical / steps, a.diagonal / steps)
        for a in attributions
    ]

    # 4) 转 numpy coeff-struct
    attr_np = _coeffs_value_numpy(attributions)  # [approx, (h,v,d), ...]

    # 5) 生成 wavelet-layout（只用于小波布局可视化 / 再投影）
    wam_layout = _visualize_grad_wam_layout(attr_np, normalize_coeffs=True)  # [B,H,W]
    wam_layout = np.expand_dims(wam_layout, axis=1)  # [B,1,H,W]

    # 6) 投影回像素域：pixel_attr 可用于叠在原图上看
    pixel_attr = wavelet_layout_to_pixel_attr(
        wam_layout,
        J=J,
        approx_coeffs=False,
        reduce="sum",
    )  # [B,H,W]

    return np.expand_dims(pixel_attr, axis=1), attr_np  # pixel_attr:[B,1,H,W], attr_np coeff-struct


# =========================
# Importance + plot helpers
# =========================

def _reduce_map(x: np.ndarray, reduce="mean"):
    """
    把一个系数块（或方向块）汇聚成一个标量“重要性”
    - mean: 平均幅值（常用）
    - sum : 总能量/总量
    - l2  : L2 能量
    """
    if reduce == "mean": return float(np.mean(x))
    if reduce == "sum":  return float(np.sum(x))
    if reduce == "l2":   return float(np.sqrt(np.sum(x * x)))
    raise ValueError("reduce must be one of: mean/sum/l2")

def _safe_norm_vec(v: np.ndarray, eps=1e-12):
    """
    向量 max 归一化（用于 4 dirs 的 normalized 版本）
    目的：只比较“相对占比”，不看绝对大小
    """
    m = float(np.max(v))
    if m < eps:
        return v
    return v / (m + eps)

def _safe_norm_map(m: np.ndarray, eps=1e-12):
    """
    热力图 min-max 归一化（方便显示）
    """
    m = np.asarray(m, dtype=np.float32)
    mn, mx = float(m.min()), float(m.max())
    return (m - mn) / (mx - mn + eps)

def wavelet_importance_4dirs(attr_np, B_select=0, per_level=False,
                             agg="mean", use_abs=True, normalize=False):
    """
    从 MFABA 的 coeff 归因结构 attr_np，提取 “四方向重要性”：
      A: Approx（低频/粗结构）
      H: Horizontal detail（水平细节方向）
      V: Vertical detail（垂直细节方向）
      D: Diagonal detail（对角细节方向）

    两种输出：
    - per_level=False:
        返回总体标量：A,H,V,D 各一个
    - per_level=True:
        返回每层(level)上的 H/V/D 序列
        A 只有一个 approx 块（只属于最粗尺度），因此放在 level=1，其它 level=0

    normalize=True 表示把 A/H/V/D 这四个方向做 max-norm，得到“占比/相对重要性”
    """
    # A: approx block
    A = attr_np[0][B_select]  # [C,h,w]
    if use_abs: A = np.abs(A)
    A_map = A.mean(axis=0)    # [h,w] (channel 聚合)

    # H/V/D: per level
    H_list, V_list, D_list = [], [], []
    for det in attr_np[1:]:
        h = det.horizontal[B_select]
        v = det.vertical[B_select]
        d = det.diagonal[B_select]
        if use_abs:
            h, v, d = np.abs(h), np.abs(v), np.abs(d)
        H_list.append(_reduce_map(h, agg))
        V_list.append(_reduce_map(v, agg))
        D_list.append(_reduce_map(d, agg))

    A_val = _reduce_map(A_map, agg)

    if per_level:
        J = len(attr_np) - 1
        A_levels = [A_val] + [0.0] * (J - 1)
        out = {
            "A": np.array(A_levels, dtype=np.float32),
            "H": np.array(H_list, dtype=np.float32),
            "V": np.array(V_list, dtype=np.float32),
            "D": np.array(D_list, dtype=np.float32),
        }
    else:
        out = {
            "A": np.array([A_val], dtype=np.float32),
            "H": np.array([np.sum(H_list)], dtype=np.float32),
            "V": np.array([np.sum(V_list)], dtype=np.float32),
            "D": np.array([np.sum(D_list)], dtype=np.float32),
        }

    if normalize:
        vals = np.array([out["A"].sum(), out["H"].sum(), out["V"].sum(), out["D"].sum()], dtype=np.float32)
        vals = _safe_norm_vec(vals)  # max-norm
        scale = {k: vals[i] for i, k in enumerate(["A", "H", "V", "D"])}
        if per_level:
            out["A"] = out["A"] * scale["A"]
            out["H"] = out["H"] * scale["H"]
            out["V"] = out["V"] * scale["V"]
            out["D"] = out["D"] * scale["D"]
        else:
            out["A"][0] = scale["A"]
            out["H"][0] = scale["H"]
            out["V"][0] = scale["V"]
            out["D"][0] = scale["D"]

    return out


def save_wavelet_importance_4dirs(attr_np, out_dir, sample_idx=0, agg="mean",
                                  use_abs=True, per_level=False,
                                  title_prefix="WAM-MFABA",
                                  figsize=(10,4), dpi=200):
    """
    保存“重要性图”：
    - per_level=False: 保存 1 张图（左右两个子图）
        左: raw 绝对重要性（A/H/V/D）
        右: normalized 相对占比（A/H/V/D）
      含义：告诉你模型主要依赖低频(A) 还是某种方向细节(H/V/D)

    - per_level=True: 保存 1 张图（左右两个子图）
        左: raw 的每层重要性曲线
        右: normalized 的每层重要性曲线
      含义：告诉你模型偏向粗尺度(level=1) 还是细尺度(level=J)
    """
    os.makedirs(out_dir, exist_ok=True)
    raw = wavelet_importance_4dirs(attr_np, B_select=sample_idx, per_level=per_level,
                                   agg=agg, use_abs=use_abs, normalize=False)
    nor = wavelet_importance_4dirs(attr_np, B_select=sample_idx, per_level=per_level,
                                   agg=agg, use_abs=use_abs, normalize=True)

    if not per_level:
        dirs = ["A","H","V","D"]
        raw_vals = [raw[d][0] for d in dirs]
        nor_vals = [nor[d][0] for d in dirs]

        fig = plt.figure(figsize=figsize)
        plt.subplot(1,2,1)
        plt.bar(dirs, raw_vals)
        plt.title(f"{title_prefix} raw (s={sample_idx})")
        plt.ylabel("importance")

        plt.subplot(1,2,2)
        plt.bar(dirs, nor_vals)
        plt.title(f"{title_prefix} normalized (s={sample_idx})")
        plt.ylabel("importance (max-norm)")
        plt.tight_layout()

        path = os.path.join(out_dir, f"{title_prefix}_s{sample_idx}_4dirs_bar.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    J = len(attr_np) - 1
    xs = np.arange(1, J + 1)

    fig = plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(xs, raw["H"], marker="o", label="H")
    plt.plot(xs, raw["V"], marker="o", label="V")
    plt.plot(xs, raw["D"], marker="o", label="D")
    plt.plot(xs, raw["A"], marker="o", label="A (approx)")
    plt.title(f"{title_prefix} per-level raw (s={sample_idx})")
    plt.xlabel("level (1=coarse -> J=fine)")
    plt.ylabel(f"importance ({agg})")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(xs, nor["H"], marker="o", label="H")
    plt.plot(xs, nor["V"], marker="o", label="V")
    plt.plot(xs, nor["D"], marker="o", label="D")
    plt.plot(xs, nor["A"], marker="o", label="A (approx)")
    plt.title(f"{title_prefix} per-level normalized (s={sample_idx})")
    plt.xlabel("level (1=coarse -> J=fine)")
    plt.ylabel("importance (norm)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, f"{title_prefix}_s{sample_idx}_4dirs_perlevel.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_wavelet_heatmaps_4dirs(attr_np, out_dir, sample_idx=0, level=-1,
                               use_abs=True, normalize_each=True,
                               figsize=(10,8), dpi=200):
    """
    保存 “某一层(level) 的 A/H/V/D 热力图”（2x2）：
    - Approx (A) : 低频块（最粗结构）
    - H/V/D     : 当前 level 对应的方向细节块

    level 的解释：
    - level=-1 : 最细一层（最高频细节）
    - level=0  : 最粗细节层（低频边缘）
    作用：你能看见“在哪些空间位置”，该方向/该频带在起作用
    """
    os.makedirs(out_dir, exist_ok=True)

    approx = attr_np[0][sample_idx].mean(axis=0)
    det = attr_np[1:][level]
    H = det.horizontal[sample_idx].mean(axis=0)
    V = det.vertical[sample_idx].mean(axis=0)
    D = det.diagonal[sample_idx].mean(axis=0)

    if use_abs:
        approx, H, V, D = np.abs(approx), np.abs(H), np.abs(V), np.abs(D)

    if normalize_each:
        approx = _safe_norm_map(approx)
        H = _safe_norm_map(H)
        V = _safe_norm_map(V)
        D = _safe_norm_map(D)

    maps = [approx, H, V, D]
    names = ["Approx (A)", "Horizontal detail (H)", "Vertical detail (V)", "Diagonal detail (D)"]

    fig = plt.figure(figsize=figsize)
    for i, (m, name) in enumerate(zip(maps, names), start=1):
        plt.subplot(2,2,i)
        plt.imshow(m, aspect="auto")
        plt.title(name)
        plt.axis("off")
    plt.suptitle(f"Wavelet coeff maps (s={sample_idx}, level={level})")
    plt.tight_layout()

    path = os.path.join(out_dir, f"heatmaps_s{sample_idx}_level{level}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_overlay_pixel_attr(x_batch, pixel_attr, out_dir, sample_idx=0,
                            alpha=0.5, dpi=200):
    """
    保存 pixel_attr 叠图：
    - pixel_attr 是已经从小波布局空间投影回像素域的归因 [B,1,H,W]
    - 叠到原图上后，你能直观看到“模型关注的像素区域”
    """
    os.makedirs(out_dir, exist_ok=True)

    img = x_batch[sample_idx].detach().cpu().permute(1,2,0).clamp(0,1).numpy()

    if isinstance(pixel_attr, torch.Tensor):
        m = pixel_attr[sample_idx, 0].detach().cpu().numpy()
    else:
        m = pixel_attr[sample_idx, 0]

    m = _safe_norm_map(m)

    fig = plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Input image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.imshow(m, alpha=alpha)
    plt.title("Overlay pixel_attr")
    plt.axis("off")
    plt.tight_layout()

    path = os.path.join(out_dir, f"overlay_s{sample_idx}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# =========================
# Main: one batch -> save figs
# =========================
if __name__ == "__main__":
    """
    运行后你会得到 5 张图，它们分别回答：
    1) mfaba_s0_4dirs_bar.png
       -> 这张样本整体是靠低频(A) 还是细节(H/V/D)？
       -> normalized 让你看“占比”，raw 让你看“绝对强度”

    2) mfaba_s0_4dirs_perlevel.png
       -> H/V/D 在不同尺度(level)上的贡献
       -> 看模型更依赖粗尺度还是细尺度的边缘/纹理

    3) heatmaps_s0_level-1.png
       -> 最细一层（高频）里 A/H/V/D 的空间分布
       -> “高频纹理/细小边缘”主要出现在哪些区域

    4) heatmaps_s0_level0.png
       -> 最粗细节层（低频边缘）里 A/H/V/D 的空间分布
       -> “大边界/大结构边缘”主要出现在哪些区域

    5) overlay_s0.png
       -> 最终像素域归因叠到原图上
       -> 直观看模型在图像哪些像素区域最敏感
    """
    OUT_DIR = "vis_wam_mfaba_batch0"
    os.makedirs(OUT_DIR, exist_ok=True)

    # dataset / loader
    dataset = IMAGENETSUBSET1000()
    BATCH_SIZE = 4
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    sfmx = nn.Softmax(dim=1)
    model = efficientnet_b0(pretrained=True)
    model = nn.Sequential(normalize, model, sfmx).eval().to(device)

    # one batch
    x, y = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device).long()

    # wam mfaba
    pixel_attr, attr_np = wam_mfaba(model, x, y, wavelet="haar", J=3, mode="reflect", steps=10)

    print("x:", tuple(x.shape))                 # [B,3,H,W]
    print("pixel_attr:", tuple(pixel_attr.shape))  # [B,1,H,W]

    # save figures for sample 0
    p_bar = save_wavelet_importance_4dirs(attr_np, OUT_DIR, sample_idx=0, per_level=False, title_prefix="mfaba")
    p_lvl = save_wavelet_importance_4dirs(attr_np, OUT_DIR, sample_idx=0, per_level=True,  title_prefix="mfaba")
    p_hm_fine = save_wavelet_heatmaps_4dirs(attr_np, OUT_DIR, sample_idx=0, level=-1)
    p_hm_coarse = save_wavelet_heatmaps_4dirs(attr_np, OUT_DIR, sample_idx=0, level=0)
    p_overlay = save_overlay_pixel_attr(x, pixel_attr, OUT_DIR, sample_idx=0)

    print("saved:")
    print(p_bar)
    print(p_lvl)
    print(p_hm_fine)
    print(p_hm_coarse)
    print(p_overlay)
