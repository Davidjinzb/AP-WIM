# %% [markdown]
# # WAM (2D) 可视化 Demo（ImageNet 子集，EfficientNet-B0，单个 batch）
# 
# 这个 notebook 会：
# - 加载 `IMAGENETSUBSET1000`，取 **第一个 batch**（默认 B=16）
# - 使用 `efficientnet_b0`（带 softmax）
# - 计算 `wam_sg / wam_ig / wam_mfaba` 的归因
# - 同时输出两种可视化：
#   1) **wavelet-layout 拼图**（类似论文右上）
#   2) **multiscale heatmap decomposition**（每个尺度上采样到原图大小并可叠加）
# - 对 `wam_ig`（alpha 路径）和 `wam_mfaba`（迭代步）输出**中间样本重建图**（用于体现 OOD）
# 

# %%

# ====== 0. Imports & env ======
import os
os.makedirs("results", exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import ptwt
import cv2

from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

# 你的工程里已有这些
from datasets import IMAGENETSUBSET1000
from utils import device  # 你项目里的 device
try:
    from utils import setup_seed
    setup_seed(42)
except Exception:
    torch.manual_seed(42)
    np.random.seed(42)

# baseline methods（可选对比）
from attr_methods_baseline import ig, fast_ig, sg, saliencymap, mfaba, la

print("device:", device)


# %%

# ====== 1. Build model: NormalizeLayer + EfficientNet-B0 + Softmax ======

class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std

# ImageNet mean/std
norm_layer = NormalizeLayer(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])

# torchvision 版本兼容：优先 weights
try:
    from torchvision.models import EfficientNet_B0_Weights
    backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
except Exception:
    backbone = efficientnet_b0(pretrained=True)

model = nn.Sequential(
    norm_layer,
    backbone,
    nn.Softmax(dim=1),
).eval().to(device)

print("model ready:", type(model))


# %%

# ====== 2. Load dataset & take ONE batch ======
BATCH_SIZE = 16
dataset = IMAGENETSUBSET1000()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

x, y = next(iter(dataloader))
x = x.to(device)
y = y.to(device).long()

with torch.no_grad():
    p = model(x)
    top1 = p.argmax(dim=1)

print("x:", x.shape, x.dtype)
print("y:", y.shape, y.dtype)
print("top1 (first 5):", top1[:5].tolist())


# %% [markdown]
# ## 3) WAM (2D) + 两种可视化 + 中间样本输出
# 
# 下面实现的是 **WAM-SG / WAM-IG / WAM-MFABA** 的“debug 版”。
# - 默认输出：`pixel_attr`（[B,1,H,W]）
# - 可选输出：`layout_attr`（[B,H,W]，wavelet-layout 拼图）
# - 可选输出：路径中间样本 `debug`（IG: alphas; MFABA: steps），包含
#   - `imgs`: 每个中间点的 **重建图**（原域，用于 OOD）
#   - `attr_layouts`: 每个中间点的 attribution layout（用于拼图/多尺度可视化）
# 

# %%

# ====== 3.1 Helpers (layout <-> pixel multiscale + plotting) ======

import matplotlib.patches as patches
from utils import wavelet_layout_to_pixel_attr  # 你项目里已有

def _assert_same_device(model, x, y):
    dev_m = next(model.parameters()).device
    if x.device != dev_m:
        raise RuntimeError(f"x.device={x.device} != model.device={dev_m}")
    if y.device != dev_m:
        raise RuntimeError(f"y.device={y.device} != model.device={dev_m}")
    if y.dtype != torch.long:
        raise RuntimeError(f"y.dtype must be torch.long, got {y.dtype}")

def _norm01(x, eps=1e-12):
    x = x.astype(np.float32)
    m, M = np.min(x), np.max(x)
    if (M - m) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (M - m + eps)

def plot_rgb(img_chw, title=None, save_path=None):
    if hasattr(img_chw, "detach"):
        img = img_chw.detach().cpu().numpy()
    else:
        img = img_chw
    img = np.moveaxis(img, 0, 2)
    fig = plt.figure(figsize=(4,4))
    if title: plt.title(title)
    plt.imshow(_norm01(img))
    plt.axis("off")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def overlay_heatmap(img_chw, heat_hw, title=None, cmap="magma", alpha=0.45, save_path=None):
    if hasattr(img_chw, "detach"):
        img = img_chw.detach().cpu().numpy()
    else:
        img = img_chw
    img = np.moveaxis(img, 0, 2)
    hm = _norm01(heat_hw)

    fig = plt.figure(figsize=(4,4))
    if title: plt.title(title)
    plt.imshow(_norm01(img))
    plt.imshow(hm, cmap=cmap, alpha=alpha)
    plt.axis("off")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_wavelet_layout(layout_map, J, b=0, cmap="magma", title="Wavelet layout", save_path=None):
    L = layout_map[b]
    H, W = L.shape
    assert H == W

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(_norm01(L), cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

    endA = H // (2**J)
    ax.add_patch(patches.Rectangle((0, 0), endA, endA, linewidth=2, edgecolor="white", facecolor="none"))

    for i in range(J):
        end = int(H / 2**i)
        start = int(H / 2**(i + 1))
        ax.add_patch(patches.Rectangle((start, 0), end-start, start, linewidth=2, edgecolor="white", facecolor="none"))        # H
        ax.add_patch(patches.Rectangle((0, start), start, end-start, linewidth=2, edgecolor="white", facecolor="none"))        # V
        ax.add_patch(patches.Rectangle((start, start), end-start, end-start, linewidth=2, edgecolor="white", facecolor="none")) # D

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return fig

def layout_to_multiscale_pixel(layout_map, J, return_dirs=True):
    B, H, W = layout_map.shape
    scales_sum = np.zeros((B, J, H, W), dtype=np.float32)
    dirs = np.zeros((B, J, 3, H, W), dtype=np.float32) if return_dirs else None

    for j in range(J):
        end = int(H / 2**j)
        start = int(H / 2**(j + 1))
        d = layout_map[:, start:end, start:end]
        v = layout_map[:, start:end, :start]
        h = layout_map[:, :start, start:end]

        for b in range(B):
            h_up = cv2.resize(h[b], (W, H), interpolation=cv2.INTER_LINEAR)
            v_up = cv2.resize(v[b], (W, H), interpolation=cv2.INTER_LINEAR)
            d_up = cv2.resize(d[b], (W, H), interpolation=cv2.INTER_LINEAR)

            scales_sum[b, j] = h_up + v_up + d_up
            if return_dirs:
                dirs[b, j, 0] = h_up
                dirs[b, j, 1] = v_up
                dirs[b, j, 2] = d_up

    return scales_sum, dirs

def plot_multiscale(scales_sum, img_chw=None, b=0, overlay=True, alpha=0.45, cmap="magma", title_prefix="Scale", save_path=None):
    J = scales_sum.shape[1]
    fig, axs = plt.subplots(1, J, figsize=(4*J, 4))
    if J == 1:
        axs = [axs]

    base = None
    if img_chw is not None:
        if hasattr(img_chw, "detach"):
            base = img_chw.detach().cpu().numpy()
        else:
            base = img_chw
        base = np.moveaxis(base, 0, 2)
        base = _norm01(base)

    for j in range(J):
        hm = _norm01(scales_sum[b, j])
        axs[j].set_title(f"{title_prefix} {j+1}")
        axs[j].axis("off")
        if overlay and base is not None:
            axs[j].imshow(base)
            axs[j].imshow(hm, cmap=cmap, alpha=alpha)
        else:
            axs[j].imshow(hm, cmap=cmap)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return fig


# %%

# ====== 3.2 Wavelet coeff helpers (2D) ======

def _detach_coeffs(coeffs):
    out = []
    for c in coeffs:
        if isinstance(c, ptwt.constants.WaveletDetailTuple2d):
            out.append(ptwt.constants.WaveletDetailTuple2d(c.horizontal.detach(), c.vertical.detach(), c.diagonal.detach()))
        else:
            out.append(c.detach())
    return out

def _scale_coeffs(coeffs, alpha: float):
    out = [coeffs[0] * alpha]
    for c in coeffs[1:]:
        out.append(ptwt.constants.WaveletDetailTuple2d(c.horizontal * alpha, c.vertical * alpha, c.diagonal * alpha))
    return out

def _require_grad_coeffs(coeffs):
    grad_coeffs = [coeffs[0].requires_grad_()]
    for c in coeffs[1:]:
        grad_coeffs.append(ptwt.constants.WaveletDetailTuple2d(
            c.horizontal.requires_grad_(),
            c.vertical.requires_grad_(),
            c.diagonal.requires_grad_(),
        ))
    return grad_coeffs

def _coeffs_value_numpy(coeffs):
    out = [coeffs[0].detach().cpu().numpy()]
    for c in coeffs[1:]:
        out.append(ptwt.constants.WaveletDetailTuple2d(
            c.horizontal.detach().cpu().numpy(),
            c.vertical.detach().cpu().numpy(),
            c.diagonal.detach().cpu().numpy(),
        ))
    return out

def _coeffs_grad_numpy(coeffs_with_grad):
    out = [coeffs_with_grad[0].grad.detach().cpu().numpy()]
    for c in coeffs_with_grad[1:]:
        out.append(ptwt.constants.WaveletDetailTuple2d(
            c.horizontal.grad.detach().cpu().numpy(),
            c.vertical.grad.detach().cpu().numpy(),
            c.diagonal.grad.detach().cpu().numpy(),
        ))
    return out

def _grad_coeffs_from_coeffs(model, coeffs, y, wavelet="haar"):
    grad_coeffs = _require_grad_coeffs(coeffs)
    img = ptwt.waverec2(grad_coeffs, wavelet)  # [B,C,H,W]
    out = model(img)  # [B,1000] probs
    loss = out.gather(1, y.view(-1, 1)).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    return _coeffs_grad_numpy(grad_coeffs)

def _visualize_grad_wam_layout(grad_coeffs_np, normalize_coeffs=True):
    B = grad_coeffs_np[0].shape[0]
    H = int(2 * grad_coeffs_np[-1].horizontal.shape[-1])
    layout = np.zeros((B, H, H), dtype=np.float32)

    approx = np.abs(grad_coeffs_np[0].mean(axis=1))
    if normalize_coeffs and approx.max() > 0:
        approx = approx / (approx.max() + 1e-12)
    layout[:, :approx.shape[1], :approx.shape[2]] = approx

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

        layout[:, start:end, start:end] = d
        layout[:, start:end, :start] = v
        layout[:, :start, start:end] = h

    return layout


# %%

# ====== 3.3 WAM debug implementations ======

def wam_sg_debug(
    model, x, y,
    wavelet="haar", J=3, mode="reflect",
    n_samples=25, stdev_spread=0.25, seed=42,
    normalize_coeffs=True,
    return_layout=True,
):
    _assert_same_device(model, x, y)
    torch.manual_seed(seed)

    B = x.shape[0]
    layout_sum = None

    for _ in range(n_samples):
        noisy = torch.empty_like(x)
        for i in range(B):
            max_x = x[i].max()
            min_x = x[i].min()
            stdev = stdev_spread * (max_x - min_x)
            noisy[i] = x[i] + torch.randn_like(x[i]) * stdev

        coeffs = ptwt.wavedec2(noisy, wavelet, level=J, mode=mode)
        grad_np = _grad_coeffs_from_coeffs(model, coeffs, y, wavelet=wavelet)
        layout = _visualize_grad_wam_layout(grad_np, normalize_coeffs=normalize_coeffs)
        layout_sum = layout if layout_sum is None else (layout_sum + layout)

    layout_avg = layout_sum / float(n_samples)  # [B,H,W]
    pixel_attr = wavelet_layout_to_pixel_attr(
        np.expand_dims(layout_avg, axis=1), J=J, approx_coeffs=False, reduce="sum"
    )  # [B,H,W]
    pixel_attr = np.expand_dims(pixel_attr, axis=1)  # [B,1,H,W]

    return (pixel_attr, layout_avg) if return_layout else pixel_attr


def wam_ig_debug(
    model, x, y,
    wavelet="haar", J=3, mode="reflect",
    n_steps=25,
    normalize_grad_layout=True,
    return_layout=True,
    return_path=True,
    path_alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
    clamp_img=(0.0, 1.0),
):
    _assert_same_device(model, x, y)

    coeffs = ptwt.wavedec2(x, wavelet, level=J, mode=mode)
    baseline_z = _visualize_grad_wam_layout(_coeffs_value_numpy(coeffs), normalize_coeffs=True)

    alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    B, H, W = baseline_z.shape
    grad_path = np.zeros((B, n_steps, H, W), dtype=np.float32)

    debug = {"alphas": path_alphas, "imgs": [], "attr_layouts": []} if return_path else None
    sel = set([float(np.round(a, 6)) for a in path_alphas]) if return_path else set()

    for i, a in enumerate(alphas):
        path_coeffs = _scale_coeffs(coeffs, float(a))
        grad_np = _grad_coeffs_from_coeffs(model, path_coeffs, y, wavelet=wavelet)
        grad_layout = _visualize_grad_wam_layout(grad_np, normalize_coeffs=normalize_grad_layout)
        grad_path[:, i] = grad_layout

        if return_path and float(np.round(a, 6)) in sel:
            with torch.no_grad():
                img_a = ptwt.waverec2(_detach_coeffs(path_coeffs), wavelet)
                if clamp_img is not None:
                    lo, hi = clamp_img
                    img_a = img_a.clamp(lo, hi)
            debug["imgs"].append(img_a.detach().cpu())
            debug["attr_layouts"].append(grad_layout.copy())

    integral = np.trapz(np.nan_to_num(grad_path), x=alphas, axis=1)
    layout_attr = baseline_z * integral  # [B,H,W]

    pixel_attr = wavelet_layout_to_pixel_attr(
        np.expand_dims(layout_attr, axis=1), J=J, approx_coeffs=False, reduce="sum"
    )  # [B,H,W]
    pixel_attr = np.expand_dims(pixel_attr, axis=1)

    if return_layout and return_path:
        return pixel_attr, layout_attr, debug
    if return_layout:
        return pixel_attr, layout_attr
    if return_path:
        return pixel_attr, debug
    return pixel_attr


def wam_mfaba_debug(
    model, x, y,
    wavelet="haar", J=3, mode="reflect",
    steps=10,
    normalize_coeffs=True,
    return_layout=True,
    return_path=True,
    path_steps=(0, 1, 2, 5, 9),
    clamp_img=(0.0, 1.0),
):
    _assert_same_device(model, x, y)

    coeffs = ptwt.wavedec2(x, wavelet, level=J, mode=mode)

    # lr
    lrs = []
    for c in coeffs:
        if isinstance(c, ptwt.constants.WaveletDetailTuple2d):
            lrs.append(ptwt.constants.WaveletDetailTuple2d(c.horizontal / steps, c.vertical / steps, c.diagonal / steps))
        else:
            lrs.append(c / steps)

    attributions = None
    debug = {"steps": path_steps, "imgs": [], "attr_layouts": []} if return_path else None
    sel_steps = set(path_steps) if return_path else set()

    for t in range(steps):
        grad_coeffs = _require_grad_coeffs(coeffs)
        img = ptwt.waverec2(grad_coeffs, wavelet)

        out = model(img)
        loss = out.gather(1, y.view(-1, 1)).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()

        new_coeffs = []
        if attributions is None:
            attributions = []

        # approx
        g0 = grad_coeffs[0].grad
        lr0 = lrs[0]
        if len(attributions) == 0:
            attributions.append(lr0 * torch.sign(g0) * g0)
        else:
            attributions[0] = attributions[0] + lr0 * torch.sign(g0) * g0
        new_coeffs.append((coeffs[0] + lr0 * torch.sign(g0)).detach())

        # details
        for idx, (c, gc, lr) in enumerate(zip(coeffs[1:], grad_coeffs[1:], lrs[1:]), start=1):
            gh, gv, gd = gc.horizontal.grad, gc.vertical.grad, gc.diagonal.grad

            if len(attributions) <= idx:
                attributions.append(ptwt.constants.WaveletDetailTuple2d(
                    lr.horizontal * torch.sign(gh) * gh,
                    lr.vertical   * torch.sign(gv) * gv,
                    lr.diagonal   * torch.sign(gd) * gd,
                ))
            else:
                a = attributions[idx]
                attributions[idx] = ptwt.constants.WaveletDetailTuple2d(
                    a.horizontal + lr.horizontal * torch.sign(gh) * gh,
                    a.vertical   + lr.vertical   * torch.sign(gv) * gv,
                    a.diagonal   + lr.diagonal   * torch.sign(gd) * gd,
                )

            new_coeffs.append(ptwt.constants.WaveletDetailTuple2d(
                (c.horizontal + lr.horizontal * torch.sign(gh)).detach(),
                (c.vertical   + lr.vertical   * torch.sign(gv)).detach(),
                (c.diagonal   + lr.diagonal   * torch.sign(gd)).detach(),
            ))

        coeffs = new_coeffs

        if return_path and t in sel_steps:
            with torch.no_grad():
                img_t = ptwt.waverec2(_detach_coeffs(coeffs), wavelet)
                if clamp_img is not None:
                    lo, hi = clamp_img
                    img_t = img_t.clamp(lo, hi)
            debug["imgs"].append(img_t.detach().cpu())

            attr_np_t = _coeffs_value_numpy([
                (a / (t+1)) if not isinstance(a, ptwt.constants.WaveletDetailTuple2d)
                else ptwt.constants.WaveletDetailTuple2d(a.horizontal/(t+1), a.vertical/(t+1), a.diagonal/(t+1))
                for a in attributions
            ])
            layout_t = _visualize_grad_wam_layout(attr_np_t, normalize_coeffs=normalize_coeffs)
            debug["attr_layouts"].append(layout_t)

    # final avg
    attributions = [
        (a / steps) if not isinstance(a, ptwt.constants.WaveletDetailTuple2d)
        else ptwt.constants.WaveletDetailTuple2d(a.horizontal/steps, a.vertical/steps, a.diagonal/steps)
        for a in attributions
    ]
    layout_attr = _visualize_grad_wam_layout(_coeffs_value_numpy(attributions), normalize_coeffs=normalize_coeffs)

    pixel_attr = wavelet_layout_to_pixel_attr(
        np.expand_dims(layout_attr, axis=1), J=J, approx_coeffs=False, reduce="sum"
    )
    pixel_attr = np.expand_dims(pixel_attr, axis=1)

    if return_layout and return_path:
        return pixel_attr, layout_attr, debug
    if return_layout:
        return pixel_attr, layout_attr
    if return_path:
        return pixel_attr, debug
    return pixel_attr


# %%

# ====== 4. Run WAM on ONE batch (pick one sample index to visualize) ======
J = 3
b = 0  # visualize sample #0 in the batch

pixel_sg, layout_sg = wam_sg_debug(model, x, y, J=J, n_samples=16, stdev_spread=0.25, return_layout=True)

pixel_ig, layout_ig, dbg_ig = wam_ig_debug(
    model, x, y, J=J, n_steps=25,
    return_layout=True, return_path=True,
    path_alphas=(0.0, 0.25, 0.5, 0.75, 1.0)
)

pixel_mf, layout_mf, dbg_mf = wam_mfaba_debug(
    model, x, y, J=J, steps=10,
    return_layout=True, return_path=True,
    path_steps=(0, 1, 2, 5, 9)
)

print("pixel_sg:", pixel_sg.shape, "layout_sg:", layout_sg.shape)
print("pixel_ig:", pixel_ig.shape, "layout_ig:", layout_ig.shape, "dbg_ig imgs:", len(dbg_ig["imgs"]))
print("pixel_mf:", pixel_mf.shape, "layout_mf:", layout_mf.shape, "dbg_mf imgs:", len(dbg_mf["imgs"]))


# %% [markdown]
# ## 5) 最终归因：两种可视化（layout拼图 + multiscale分解）

# %%

# ====== 5.1 show original image ======
plot_rgb(x[b], title=f"Original image (batch idx={b})", save_path="results/original_image.pdf")


# %%

# ====== 5.2 WAM-SG visuals ======
overlay_heatmap(x[b], pixel_sg[b,0], title="WAM-SG pixel attribution (overlay)", save_path="results/wam_sg_pixel.pdf")
plot_wavelet_layout(layout_sg, J=J, b=b, title="WAM-SG attribution (wavelet-layout)", save_path="results/wam_sg_layout.pdf")
scales_sg, _ = layout_to_multiscale_pixel(layout_sg, J=J, return_dirs=False)
plot_multiscale(scales_sg, img_chw=x[b], b=b, overlay=True, title_prefix="WAM-SG scale", save_path="results/wam_sg_scale.pdf")


# %%

# ====== 5.3 WAM-IG visuals ======
overlay_heatmap(x[b], pixel_ig[b,0], title="WAM-IG pixel attribution (overlay)", save_path="results/wam_ig_pixel.pdf")
plot_wavelet_layout(layout_ig, J=J, b=b, title="WAM-IG attribution (wavelet-layout)", save_path="results/wam_ig_layout.pdf")
scales_ig, _ = layout_to_multiscale_pixel(layout_ig, J=J, return_dirs=False)
plot_multiscale(scales_ig, img_chw=x[b], b=b, overlay=True, title_prefix="WAM-IG scale", save_path="results/wam_ig_scale.pdf")


# %%

# ====== 5.4 WAM-MFABA visuals ======
overlay_heatmap(x[b], pixel_mf[b,0], title="WAM-MFABA pixel attribution (overlay)", save_path="results/wam_mfaba_pixel.pdf")
plot_wavelet_layout(layout_mf, J=J, b=b, title="WAM-MFABA attribution (wavelet-layout)", save_path="results/wam_mfaba_layout.pdf")
scales_mf, _ = layout_to_multiscale_pixel(layout_mf, J=J, return_dirs=False)
plot_multiscale(scales_mf, img_chw=x[b], b=b, overlay=True, title_prefix="WAM-MFABA scale", save_path="results/wam_mfaba_scale.pdf")


# %% [markdown]
# ## 6) IG / MFABA 中间样本（转回原域）用于体现 OOD

# %%

# ====== 6.1 IG path images + intermediate attribution layouts ======
for a, img_a, lay_a in zip(dbg_ig["alphas"], dbg_ig["imgs"], dbg_ig["attr_layouts"]):
    plot_rgb(img_a[b], title=f"IG path reconstructed image (alpha={a})", save_path=f"results/ig_path_img_alpha_{a}.pdf")
    plot_wavelet_layout(lay_a, J=J, b=b, title=f"IG path attribution layout (alpha={a})", save_path=f"results/ig_path_layout_alpha_{a}.pdf")
    scales_a, _ = layout_to_multiscale_pixel(lay_a, J=J, return_dirs=False)
    plot_multiscale(scales_a, img_chw=img_a[b], b=b, overlay=True, title_prefix=f"IG alpha={a} scale", save_path=f"results/ig_path_scale_alpha_{a}.pdf")


# %%

# ====== 6.2 MFABA path images + intermediate attribution layouts ======
for t, img_t, lay_t in zip(dbg_mf["steps"], dbg_mf["imgs"], dbg_mf["attr_layouts"]):
    plot_rgb(img_t[b], title=f"MFABA path reconstructed image (step={t})", save_path=f"results/mfaba_path_img_step_{t}.pdf")
    plot_wavelet_layout(lay_t, J=J, b=b, title=f"MFABA path attribution layout (step={t})", save_path=f"results/mfaba_path_layout_step_{t}.pdf")
    scales_t, _ = layout_to_multiscale_pixel(lay_t, J=J, return_dirs=False)
    plot_multiscale(scales_t, img_chw=img_t[b], b=b, overlay=True, title_prefix=f"MFABA step={t} scale", save_path=f"results/mfaba_path_scale_step_{t}.pdf")


# # %% [markdown]
# # ## 7) （可选）baseline 方法对比：IG / SG / Saliency / FastIG / MFABA / LA

# # %%

# # ====== 7. baseline (optional) ======
# def _to_hw_np(a):
#     if isinstance(a, torch.Tensor):
#         a = a.detach().cpu().numpy()
#     return a

# base_ig = ig(model, x, y, n_steps=25)
# base_sg = sg(model, x, y, nt_samples=16, stdevs=0.25)
# base_sal = saliencymap(model, x, y)
# base_fast = fast_ig(model, x, y)

# try:
#     base_mfaba = mfaba(model, x, y, data_min=None, data_max=None)
# except TypeError:
#     base_mfaba = mfaba(model, x, y)

# try:
#     base_la = la(model, x, y, clip_min=None, clip_max=None)
# except TypeError:
#     base_la = la(model, x, y)

# for name, a in [
#     ("baseline_ig", base_ig),
#     ("baseline_sg", base_sg),
#     ("baseline_saliency", base_sal),
#     ("baseline_fastig", base_fast),
#     ("baseline_mfaba", base_mfaba),
#     ("baseline_la", base_la),
# ]:
#     a = _to_hw_np(a)  # expect [B,1,H,W]
#     overlay_heatmap(x[b], a[b,0], title=f"{name} (overlay)")



