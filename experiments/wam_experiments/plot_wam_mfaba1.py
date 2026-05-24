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
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# ====== IJCAI Style Settings ======
def setup_ijcai_style():
    """
    Configure matplotlib to produce IJCAI-quality figures.
    - Font: Times New Roman (Serif)
    - Sizes: Readable labels and titles
    - Layout: Tight and clean
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # LaTeX-like math font
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "figure.autolayout": True,
    })

setup_ijcai_style()

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
                                  figsize=(10,4), dpi=300):
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
        ax1 = plt.subplot(1,2,1)
        # Use a qualitative palette suitable for academic papers
        colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44'] # Paul Tol's bright scheme
        bars = ax1.bar(dirs, raw_vals, color=colors, edgecolor='black', linewidth=1.0)
        ax1.set_title(f"{title_prefix} Raw Importance", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Attribution Score")
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax2 = plt.subplot(1,2,2)
        bars2 = ax2.bar(dirs, nor_vals, color=colors, edgecolor='black', linewidth=1.0)
        ax2.set_title(f"{title_prefix} Normalized Ratio", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Normalized Score (Max-Norm)")
        ax2.set_ylim(0, 1.15)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()

        path = os.path.join(out_dir, f"{title_prefix}_s{sample_idx}_4dirs_bar.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        path_pdf = os.path.join(out_dir, f"{title_prefix}_s{sample_idx}_4dirs_bar.pdf")
        fig.savefig(path_pdf, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    J = len(attr_np) - 1
    xs = np.arange(1, J + 1)

    fig = plt.figure(figsize=(12,5)) # Slightly taller for better aspect ratio
    
    # Styles for lines: distinct markers and colors
    line_styles = [
        {'label': 'H (Horizontal)', 'color': '#EE6677', 'marker': 'o', 'linestyle': '-'},
        {'label': 'V (Vertical)', 'color': '#228833', 'marker': 's', 'linestyle': '--'},
        {'label': 'D (Diagonal)', 'color': '#CCBB44', 'marker': '^', 'linestyle': '-.'},
        {'label': 'A (Approx)', 'color': '#4477AA', 'marker': 'D', 'linestyle': ':'}
    ]

    ax1 = plt.subplot(1,2,1)
    ax1.plot(xs, raw["H"], **line_styles[0])
    ax1.plot(xs, raw["V"], **line_styles[1])
    ax1.plot(xs, raw["D"], **line_styles[2])
    ax1.plot(xs, raw["A"], **line_styles[3])
    
    ax1.set_title(f"{title_prefix} Raw (Per Level)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Wavelet Level (Coarse $\\to$ Fine)")
    ax1.set_ylabel(f"Importance ({agg})")
    ax1.legend(frameon=True, fancybox=False, edgecolor='black')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xticks(xs)

    ax2 = plt.subplot(1,2,2)
    ax2.plot(xs, nor["H"], **line_styles[0])
    ax2.plot(xs, nor["V"], **line_styles[1])
    ax2.plot(xs, nor["D"], **line_styles[2])
    ax2.plot(xs, nor["A"], **line_styles[3])
    
    ax2.set_title(f"{title_prefix} Normalized (Per Level)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Wavelet Level (Coarse $\\to$ Fine)")
    ax2.set_ylabel("Normalized Score")
    ax2.legend(frameon=True, fancybox=False, edgecolor='black')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xticks(xs)
    
    plt.tight_layout()

    path = os.path.join(out_dir, f"{title_prefix}_s{sample_idx}_4dirs_perlevel.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    path_pdf = os.path.join(out_dir, f"{title_prefix}_s{sample_idx}_4dirs_perlevel.pdf")
    fig.savefig(path_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_wavelet_heatmaps_4dirs(attr_np, out_dir, sample_idx=0, level=-1,
                               use_abs=True, normalize_each=True,
                               figsize=(10, 8), dpi=300):
    """
    保存 “某一层(level) 的 A/H/V/D 热力图”（2x2）：
    - 优化版：紧凑布局，共享一个 Colorbar
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
    names = ["Approx (A)", "Horizontal (H)", "Vertical (V)", "Diagonal (D)"]

    # Use subplots with constrained_layout for compact packing
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    im = None
    for i, (m, name) in enumerate(zip(maps, names)):
        ax = axes[i]
        im = ax.imshow(m, aspect="equal", cmap="magma", vmin=0, vmax=1)
        ax.set_title(name, fontsize=24, fontweight='bold')
        ax.axis("off")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, location='right', aspect=30)
    cbar.ax.tick_params(labelsize=12)
    # cbar.set_label("Normalized Intensity", fontsize=14, rotation=270, labelpad=20)
        
    plt.suptitle(f"Wavelet Coeff Maps (Level={level})", fontsize=16, fontweight='bold')
    
    # Save
    path = os.path.join(out_dir, f"heatmaps_s{sample_idx}_level{level}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    path_pdf = os.path.join(out_dir, f"heatmaps_s{sample_idx}_level{level}.pdf")
    fig.savefig(path_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_overlay_pixel_attr(x_batch, pixel_attr, out_dir, sample_idx=0,
                            alpha=0.5, dpi=300):
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

    fig = plt.figure(figsize=(12,5))
    
    ax1 = plt.subplot(1,2,1)
    ax1.imshow(img)
    ax1.set_title("Input Image", fontsize=14, fontweight='bold')
    ax1.axis("off")

    ax2 = plt.subplot(1,2,2)
    ax2.imshow(img)
    im = ax2.imshow(m, alpha=alpha, cmap="magma")
    ax2.set_title("Overlay Pixel Attribution", fontsize=14, fontweight='bold')
    ax2.axis("off")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Attribution Intensity", rotation=270, labelpad=15)
    
    plt.tight_layout()

    path = os.path.join(out_dir, f"overlay_s{sample_idx}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    path_pdf = os.path.join(out_dir, f"overlay_s{sample_idx}.pdf")
    fig.savefig(path_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_summary_figure(x_batch, pixel_attr, attr_np, out_dir, sample_idx=0, dpi=300):
    """
    Save a comprehensive summary figure with new layout:
    Left Col:
      Top: (a) Input + Overlay
      Bottom: (c) Level 0 Heatmaps | (d) Level -1 Heatmaps
    Right Col:
      Top: (b) Bar Charts (Raw & Norm)
      Bottom: (e) Line Charts (Raw & Norm)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup Figure
    fig = plt.figure(figsize=(32, 14), constrained_layout=True)
    
    # Main Grid: 1 Row, 2 Cols
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1.6], wspace=0.25)
    
    # ====== Left Column ======
    # Top: Images (a), Bottom: Heatmaps (c, d)
    # Equal height ratios to ensure top images match bottom grids in size
    gs_left = gs_main[0].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.08)
    
    # --- (a) Top-Left: Input + Overlay ---
    # Split top row to match bottom row structure (wspace=0.15) for vertical alignment
    gs_top_cont = gs_left[0].subgridspec(2, 1, height_ratios=[1.0, 0.05], hspace=0.0)
    gs_top_split = gs_top_cont[0].subgridspec(1, 2, wspace=0.15)

    # Caption (Bottom)
    ax_ta = fig.add_subplot(gs_top_cont[1])
    ax_ta.text(0.5, 0.95, "(a) AP-WIM attribution result", ha='center', va='top', fontsize=28, fontweight='bold', fontfamily='serif')
    ax_ta.axis("off")
    
    # Left & Right Containers: [Image, Gap, Cbar]
    # Use exact same ratios as bottom grids [1.0, 0.02, 0.04] to ensure perfect width alignment
    ratios = [1.0, 0.02, 0.04]
    gs_input_box = gs_top_split[0].subgridspec(1, 3, width_ratios=ratios)
    gs_overlay_box = gs_top_split[1].subgridspec(1, 3, width_ratios=ratios)
    
    img = x_batch[sample_idx].detach().cpu().permute(1,2,0).clamp(0,1).numpy()
    if isinstance(pixel_attr, torch.Tensor):
        m = pixel_attr[sample_idx, 0].detach().cpu().numpy()
    else:
        m = pixel_attr[sample_idx, 0]
    m = _safe_norm_map(m)
    
    # --- Overlay (Plot first to define im_over for dummy cbar) ---
    ax_over = fig.add_subplot(gs_overlay_box[0])
    ax_over.imshow(img, aspect='equal')
    im_over = ax_over.imshow(m, alpha=0.8, cmap="magma", aspect='equal')
    ax_over.set_title("Overlay Pixel Attribution", fontsize=32, fontweight='bold', fontfamily='serif', pad=10)
    ax_over.axis("off")
    
    # Colorbar in its own slot (Right)
    ax_cbar = fig.add_subplot(gs_overlay_box[2])
    cbar_over = fig.colorbar(im_over, cax=ax_cbar)
    cbar_over.ax.tick_params(labelsize=18)

    # --- Input Image ---
    ax_orig = fig.add_subplot(gs_input_box[0])
    ax_orig.imshow(img, aspect='equal')
    ax_orig.set_title("Input Image", fontsize=32, fontweight='bold', fontfamily='serif', pad=10)
    ax_orig.axis("off")
    
    # Dummy Cbar Slot to match spacing (Invisible but takes space)
    ax_cbar_dummy = fig.add_subplot(gs_input_box[2])
    cbar_dummy = fig.colorbar(im_over, cax=ax_cbar_dummy)
    cbar_dummy.ax.tick_params(labelsize=18, labelcolor='none', color='none')
    cbar_dummy.solids.set_alpha(0)
    cbar_dummy.outline.set_visible(False)
    cbar_over.set_label("Attribution Score", rotation=270, labelpad=25, fontsize=20, fontfamily='serif')
    
    # --- (c) & (d) Bottom-Left: Heatmaps ---
    # Split into Left (c) and Right (d)
    gs_cd = gs_left[1].subgridspec(1, 2, wspace=0.15)
    
    # (c) Level 0 Heatmaps
    gs_c_cont = gs_cd[0].subgridspec(2, 1, height_ratios=[1.0, 0.05], hspace=0.0)
    
    # Split (c) area into: [2x2 Grid | Gap | Colorbar]
    # Ratios match Overlay: [1.0 (for grid), 0.02, 0.04]
    gs_c_main = gs_c_cont[0].subgridspec(1, 3, width_ratios=[1.0, 0.02, 0.04])
    
    gs_c = gs_c_main[0].subgridspec(2, 2, wspace=0.02, hspace=0.02)
    
    # Caption (Bottom)
    ax_tc = fig.add_subplot(gs_c_cont[1])
    ax_tc.text(0.5, 0.95, "(c) Coefficient heatmaps level 0.", ha='center', va='top', fontsize=28, fontweight='bold', fontfamily='serif')
    ax_tc.axis("off")
    
    approx = attr_np[0][sample_idx].mean(axis=0)
    det0 = attr_np[1:][0] # Level 0
    H0 = det0.horizontal[sample_idx].mean(axis=0)
    V0 = det0.vertical[sample_idx].mean(axis=0)
    D0 = det0.diagonal[sample_idx].mean(axis=0)
    maps0 = [_safe_norm_map(np.abs(m)) for m in [approx, H0, V0, D0]]
    names = ["Approx (A)", "Horizontal (H)", "Vertical (V)", "Diagonal (D)"]
    
    axes0 = []
    im0 = None
    for i, (mp, name) in enumerate(zip(maps0, names)):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs_c[row, col])
        im0 = ax.imshow(mp, aspect="equal", cmap="magma", vmin=0, vmax=1)
        ax.set_title(name, fontsize=26, fontweight='bold', fontfamily='serif', pad=2)
        ax.axis("off")
        axes0.append(ax)
        
    # Add dummy colorbar for Level 0 Heatmaps to maintain symmetry (invisible)
    # We allocate the slot in gs_c_main[2] but leave it empty/invisible
    ax_cbar0_dummy = fig.add_subplot(gs_c_main[2])
    ax_cbar0_dummy.axis("off")
    
    # (d) Level -1 Heatmaps
    gs_d_cont = gs_cd[1].subgridspec(2, 1, height_ratios=[1.0, 0.05], hspace=0.0)
    
    # Split (d) area into: [2x2 Grid | Gap | Colorbar]
    gs_d_main = gs_d_cont[0].subgridspec(1, 3, width_ratios=[1.0, 0.02, 0.04])
    
    gs_d = gs_d_main[0].subgridspec(2, 2, wspace=0.02, hspace=0.02)
    
    # Caption (Bottom)
    ax_td = fig.add_subplot(gs_d_cont[1])
    ax_td.text(0.5, 0.95, "(d) Coefficient heatmaps level -1.", ha='center', va='top', fontsize=28, fontweight='bold', fontfamily='serif')
    ax_td.axis("off")
    
    det1 = attr_np[1:][-1] # Level -1
    H1 = det1.horizontal[sample_idx].mean(axis=0)
    V1 = det1.vertical[sample_idx].mean(axis=0)
    D1 = det1.diagonal[sample_idx].mean(axis=0)
    maps1 = [_safe_norm_map(np.abs(m)) for m in [approx, H1, V1, D1]] # Reuse approx for viz
    
    axes1 = []
    im1 = None
    for i, (mp, name) in enumerate(zip(maps1, names)):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs_d[row, col])
        im1 = ax.imshow(mp, aspect="equal", cmap="magma", vmin=0, vmax=1)
        ax.set_title(name, fontsize=26, fontweight='bold', fontfamily='serif', pad=2)
        ax.axis("off")
        axes1.append(ax)
        
    # Add colorbar for Level -1 Heatmaps in its own slot
    if im1:
        ax_cbar1 = fig.add_subplot(gs_d_main[2])
        cbar1 = fig.colorbar(im1, cax=ax_cbar1)
        cbar1.ax.tick_params(labelsize=18)
        cbar1.set_label("Attribution Score", rotation=270, labelpad=25, fontsize=20, fontfamily='serif')
    
    # ====== Right Column ======
    # Top: (b), Bottom: (e)
    gs_right = gs_main[1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.2)
    
    # --- (b) Top-Right: Bar Charts ---
    gs_b_cont = gs_right[0].subgridspec(2, 1, height_ratios=[1.0, 0.1], hspace=0.05)
    gs_b = gs_b_cont[0].subgridspec(1, 2, wspace=0.05)

    # Caption (Bottom)
    ax_tb = fig.add_subplot(gs_b_cont[1])
    ax_tb.text(0.5, 0.9, "(b) Raw importance and normalized ratio across wavelet directions", ha='center', va='center', fontsize=28, fontweight='bold', fontfamily='serif')
    ax_tb.axis("off")
    
    raw_bar = wavelet_importance_4dirs(attr_np, B_select=sample_idx, per_level=False, agg="mean", use_abs=True, normalize=False)
    nor_bar = wavelet_importance_4dirs(attr_np, B_select=sample_idx, per_level=False, agg="mean", use_abs=True, normalize=True)
    dirs = ["A","H","V","D"]
    colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44']
    
    ax_bar_raw = fig.add_subplot(gs_b[0])
    ax_bar_raw.bar(dirs, [raw_bar[d][0] for d in dirs], color=colors, edgecolor='black')
    ax_bar_raw.set_title("AP-WIM Raw Importance", fontsize=28, fontweight='bold', fontfamily='serif')
    ax_bar_raw.set_ylabel("Attribution Score", fontsize=24, fontfamily='serif')
    ax_bar_raw.grid(axis='y', linestyle=':', alpha=0.5)
    ax_bar_raw.tick_params(labelsize=18)
    ax_bar_raw.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_bar_raw.yaxis.get_offset_text().set_fontsize(18)
    ax_bar_raw.spines['top'].set_visible(False)
    ax_bar_raw.spines['right'].set_visible(False)
    
    ax_bar_norm = fig.add_subplot(gs_b[1])
    ax_bar_norm.bar(dirs, [nor_bar[d][0] for d in dirs], color=colors, edgecolor='black')
    ax_bar_norm.set_title("AP-WIM Normalized Ratio", fontsize=28, fontweight='bold', fontfamily='serif')
    ax_bar_norm.set_ylabel("Normalized Score (Max-Norm)", fontsize=24, fontfamily='serif')
    ax_bar_norm.set_ylim(0, 1.1)
    ax_bar_norm.grid(axis='y', linestyle=':', alpha=0.5)
    ax_bar_norm.tick_params(labelsize=18)
    ax_bar_norm.spines['top'].set_visible(False)
    ax_bar_norm.spines['right'].set_visible(False)

    # Add legend with colored patches
    legend_elements = [
        Patch(facecolor='#4477AA', edgecolor='black', label='Approximation'),
        Patch(facecolor='#EE6677', edgecolor='black', label='Horizontal'),
        Patch(facecolor='#228833', edgecolor='black', label='Vertical'),
        Patch(facecolor='#CCBB44', edgecolor='black', label='Diagonal')
    ]
    ax_bar_norm.legend(handles=legend_elements, fontsize=20, loc='upper right')

    # --- (e) Bottom-Right: Line Charts ---
    gs_e_cont = gs_right[1].subgridspec(2, 1, height_ratios=[1.0, 0.1], hspace=0.05)
    gs_e = gs_e_cont[0].subgridspec(1, 2, wspace=0.05)

    # Caption (Bottom)
    ax_te = fig.add_subplot(gs_e_cont[1])
    ax_te.text(0.5, 0.9, "(e) Raw and normalized attribution scores across wavelet levels", ha='center', va='center', fontsize=28, fontweight='bold', fontfamily='serif')
    ax_te.axis("off")
    
    raw_lvl = wavelet_importance_4dirs(attr_np, B_select=sample_idx, per_level=True, agg="mean", use_abs=True, normalize=False)
    nor_lvl = wavelet_importance_4dirs(attr_np, B_select=sample_idx, per_level=True, agg="mean", use_abs=True, normalize=True)
    J = len(attr_np) - 1
    xs = np.arange(1, J + 1)
    line_styles = [
        {'label': 'H (Horizontal)', 'color': '#EE6677', 'marker': 'o', 'linestyle': '-', 'linewidth': 4, 'markersize': 12},
        {'label': 'V (Vertical)', 'color': '#228833', 'marker': 's', 'linestyle': '--', 'linewidth': 4, 'markersize': 12},
        {'label': 'D (Diagonal)', 'color': '#CCBB44', 'marker': '^', 'linestyle': '-.', 'linewidth': 4, 'markersize': 12},
        {'label': 'A (Approx)', 'color': '#4477AA', 'marker': 'D', 'linestyle': ':', 'linewidth': 4, 'markersize': 12}
    ]
    
    ax_line_raw = fig.add_subplot(gs_e[0])
    for i, k in enumerate(["H", "V", "D", "A"]):
        ax_line_raw.plot(xs, raw_lvl[k], **line_styles[i])
    ax_line_raw.set_title("AP-WIM Raw (Per Level)", fontsize=28, fontweight='bold', fontfamily='serif')
    ax_line_raw.set_xlabel("Wavelet Level (Coarse -> Fine)", fontsize=24, fontfamily='serif')
    ax_line_raw.set_ylabel("Importance Score", fontsize=24, fontfamily='serif')
    ax_line_raw.grid(True, linestyle=':', alpha=0.5)
    ax_line_raw.set_xticks(xs)
    ax_line_raw.tick_params(labelsize=18)
    ax_line_raw.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_line_raw.yaxis.get_offset_text().set_fontsize(18)
    ax_line_raw.spines['top'].set_visible(False)
    ax_line_raw.spines['right'].set_visible(False)
    # Legend removed for Raw plot as requested
    
    ax_line_norm = fig.add_subplot(gs_e[1])
    for i, k in enumerate(["H", "V", "D", "A"]):
        ax_line_norm.plot(xs, nor_lvl[k], **line_styles[i])
    ax_line_norm.set_title("AP-WIM Normalized (Per Level)", fontsize=28, fontweight='bold', fontfamily='serif')
    ax_line_norm.set_xlabel("Wavelet Level (Coarse -> Fine)", fontsize=24, fontfamily='serif')
    ax_line_norm.set_ylabel("Normalized Score", fontsize=24, fontfamily='serif')
    # ax_line_norm.set_ylim(0, 1.1)
    ax_line_norm.legend(fontsize=20, loc='upper right')
    ax_line_norm.grid(True, linestyle=':', alpha=0.5)
    ax_line_norm.set_xticks(xs)
    ax_line_norm.tick_params(labelsize=18)
    ax_line_norm.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_line_norm.yaxis.get_offset_text().set_fontsize(18)
    ax_line_norm.spines['top'].set_visible(False)
    ax_line_norm.spines['right'].set_visible(False)

    path = os.path.join(out_dir, f"summary_s{sample_idx}_layout_v2.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    path_pdf = os.path.join(out_dir, f"summary_s{sample_idx}_layout_v2.pdf")
    fig.savefig(path_pdf, dpi=dpi, bbox_inches="tight")
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
    OUT_DIR = "vis_wam_mfaba_batch0_new"
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
    p_bar = save_wavelet_importance_4dirs(attr_np, OUT_DIR, sample_idx=0, per_level=False, title_prefix="AP-WIM")
    p_lvl = save_wavelet_importance_4dirs(attr_np, OUT_DIR, sample_idx=0, per_level=True,  title_prefix="AP-WIM")
    p_hm_fine = save_wavelet_heatmaps_4dirs(attr_np, OUT_DIR, sample_idx=0, level=-1)
    p_hm_coarse = save_wavelet_heatmaps_4dirs(attr_np, OUT_DIR, sample_idx=0, level=0)
    
    # Combined heatmap (Coarse + Fine) - Disabled per user request
    # p_hm_combined = save_combined_wavelet_heatmaps(attr_np, OUT_DIR, sample_idx=0, levels=[0, -1])

    p_overlay = save_overlay_pixel_attr(x, pixel_attr, OUT_DIR, sample_idx=0)
    
    # Comprehensive Summary Figure
    p_summary = save_summary_figure(x, pixel_attr, attr_np, OUT_DIR, sample_idx=0)

    print("saved:")
    print(p_bar)
    print(p_lvl)
    # print(p_hm_combined)
    print(p_hm_fine)
    print(p_hm_coarse)
    print(p_overlay)
    print(p_summary)
