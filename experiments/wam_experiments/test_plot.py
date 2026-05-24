# =========================================================
# WAM (2D) Visualize & SAVE ALL FIGS (ImageNet subset, EfficientNet-B0, ONE batch)
# Put this into a single .ipynb cell sequence (or a .py script run in notebook)
# =========================================================

# %% [markdown]
# # WAM (2D) 可视化 + 全部保存到文件夹
# - 数据：IMAGENETSUBSET1000
# - 模型：efficientnet_b0 + Normalize + Softmax（你代码一致）
# - 只跑一个 batch
# - 计算 wam_sg / wam_ig / wam_mfaba
# - 保存两种可视化：
#   1) wavelet-layout 拼图（带白框）
#   2) multiscale heatmap decomposition（每个尺度上采样到原图大小并叠加）
# - 保存 IG path / MFABA path 的中间重建图（体现 OOD）+ 中间 layout + multiscale

# %%
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import ptwt

from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from datasets import IMAGENETSUBSET1000
from utils import device, wavelet_layout_to_pixel_attr
try:
    from utils import setup_seed
    setup_seed(42)
except Exception:
    torch.manual_seed(42)
    np.random.seed(42)

print("device:", device)

# =========================================================
# 1) Model: Normalize + EfficientNet-B0 + Softmax
# =========================================================
class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    def forward(self, x):
        return (x - self.mean) / self.std

norm_layer = NormalizeLayer(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])

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

print("model ready")

# =========================================================
# 2) Load ONE batch
# =========================================================
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

# =========================================================
# 3) Plot helpers (SAVE version)
# =========================================================
def _norm01(arr, eps=1e-12):
    arr = arr.astype(np.float32)
    m, M = np.min(arr), np.max(arr)
    if (M - m) < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - m) / (M - m + eps)

def _to_hwc(img_chw):
    if hasattr(img_chw, "detach"):
        img = img_chw.detach().cpu().numpy()
    else:
        img = img_chw
    return np.moveaxis(img, 0, 2)

def _save_fig(fig, path, dpi=160):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def plot_rgb_fig(img_chw, title=None):
    img = _norm01(_to_hwc(img_chw))
    fig = plt.figure(figsize=(4, 4))
    if title:
        plt.title(title)
    plt.imshow(img)
    plt.axis("off")
    return fig

def overlay_heatmap_fig(img_chw, heat_hw, title=None, cmap="magma", alpha=0.45):
    img = _norm01(_to_hwc(img_chw))
    hm = _norm01(heat_hw)
    fig = plt.figure(figsize=(4, 4))
    if title:
        plt.title(title)
    plt.imshow(img)
    plt.imshow(hm, cmap=cmap, alpha=alpha)
    plt.axis("off")
    return fig

def plot_wavelet_layout_fig(layout_map, J, b=0, cmap="magma", title="Wavelet layout"):
    L = layout_map[b]
    H, W = L.shape
    assert H == W

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(_norm01(L), cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

    # approx
    endA = H // (2**J)
    ax.add_patch(patches.Rectangle((0, 0), endA, endA, linewidth=2, edgecolor="white", facecolor="none"))

    # details boxes
    for i in range(J):
        end = int(H / 2**i)
        start = int(H / 2**(i + 1))
        ax.add_patch(patches.Rectangle((start, 0), end-start, start, linewidth=2, edgecolor="white", facecolor="none"))        # H
        ax.add_patch(patches.Rectangle((0, start), start, end-start, linewidth=2, edgecolor="white", facecolor="none"))        # V
        ax.add_patch(patches.Rectangle((start, start), end-start, end-start, linewidth=2, edgecolor="white", facecolor="none")) # D

    return fig

def layout_to_multiscale_pixel(layout_map, J):
    """
    layout_map: [B,H,W] wavelet-layout
    returns: scales_sum [B,J,H,W] where each j is upsample(h+v+d) to original size
    """
    B, H, W = layout_map.shape
    scales_sum = np.zeros((B, J, H, W), dtype=np.float32)

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

    return scales_sum

def plot_multiscale_fig(scales_sum, img_chw=None, b=0, overlay=True, alpha=0.45, cmap="magma", title_prefix="Scale"):
    """
    scales_sum: [B,J,H,W]
    """
    J = scales_sum.shape[1]
    fig, axs = plt.subplots(1, J, figsize=(4 * J, 4))
    if J == 1:
        axs = [axs]

    base = None
    if img_chw is not None:
        base = _norm01(_to_hwc(img_chw))

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
    return fig

# =========================================================
# 4) Wavelet coeff helpers
# =========================================================
def _assert_same_device(model, x, y):
    dev_m = next(model.parameters()).device
    if x.device != dev_m:
        raise RuntimeError(f"x.device={x.device} != model.device={dev_m}")
    if y.device != dev_m:
        raise RuntimeError(f"y.device={y.device} != model.device={dev_m}")
    if y.dtype != torch.long:
        raise RuntimeError(f"y.dtype must be torch.long, got {y.dtype}")

def _detach_coeffs(coeffs):
    out = []
    for c in coeffs:
        if isinstance(c, ptwt.constants.WaveletDetailTuple2d):
            out.append(ptwt.constants.WaveletDetailTuple2d(
                c.horizontal.detach(), c.vertical.detach(), c.diagonal.detach()
            ))
        else:
            out.append(c.detach())
    return out

def _scale_coeffs(coeffs, alpha: float):
    out = [coeffs[0] * alpha]
    for c in coeffs[1:]:
        out.append(ptwt.constants.WaveletDetailTuple2d(
            c.horizontal * alpha, c.vertical * alpha, c.diagonal * alpha
        ))
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
    out = model(img)                           # [B,1000] (prob)
    loss = out.gather(1, y.view(-1, 1)).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    return _coeffs_grad_numpy(grad_coeffs)

def _visualize_grad_wam_layout(grad_coeffs_np, normalize_coeffs=True):
    """
    grad_coeffs_np: [approx_np, detail_tuple_np, ...]
    returns: layout [B,H,H]
    """
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

# =========================================================
# 5) WAM debug: SG / IG / MFABA
# =========================================================
def wam_sg_debug(
    model, x, y,
    wavelet="haar", J=3, mode="reflect",
    n_samples=25, stdev_spread=0.25, seed=42,
    normalize_coeffs=True,
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
    return pixel_attr, layout_avg

def wam_ig_debug(
    model, x, y,
    wavelet="haar", J=3, mode="reflect",
    n_steps=25,
    normalize_grad_layout=True,
    path_alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
    clamp_img=(0.0, 1.0),
):
    _assert_same_device(model, x, y)

    coeffs = ptwt.wavedec2(x, wavelet, level=J, mode=mode)
    baseline_z = _visualize_grad_wam_layout(_coeffs_value_numpy(coeffs), normalize_coeffs=True)

    alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    B, H, W = baseline_z.shape
    grad_path = np.zeros((B, n_steps, H, W), dtype=np.float32)

    # debug store
    debug = {"alphas": list(path_alphas), "imgs": [], "attr_layouts": []}
    sel = set([float(np.round(a, 6)) for a in path_alphas])

    for i, a in enumerate(alphas):
        a_ = float(a)
        path_coeffs = _scale_coeffs(coeffs, a_)
        grad_np = _grad_coeffs_from_coeffs(model, path_coeffs, y, wavelet=wavelet)
        grad_layout = _visualize_grad_wam_layout(grad_np, normalize_coeffs=normalize_grad_layout)
        grad_path[:, i] = grad_layout

        if float(np.round(a_, 6)) in sel:
            with torch.no_grad():
                img_a = ptwt.waverec2(_detach_coeffs(path_coeffs), wavelet)
                if clamp_img is not None:
                    lo, hi = clamp_img
                    img_a = img_a.clamp(lo, hi)
            debug["imgs"].append(img_a.detach().cpu())
            debug["attr_layouts"].append(grad_layout.copy())

    integral = np.trapz(np.nan_to_num(grad_path), x=alphas, axis=1)  # [B,H,W]
    layout_attr = baseline_z * integral

    pixel_attr = wavelet_layout_to_pixel_attr(
        np.expand_dims(layout_attr, axis=1), J=J, approx_coeffs=False, reduce="sum"
    )  # [B,H,W]
    pixel_attr = np.expand_dims(pixel_attr, axis=1)
    return pixel_attr, layout_attr, debug

def wam_mfaba_debug(
    model, x, y,
    wavelet="haar", J=3, mode="reflect",
    steps=10,
    normalize_coeffs=True,
    path_steps=(0, 1, 2, 5, 9),
    clamp_img=(0.0, 1.0),
):
    _assert_same_device(model, x, y)

    coeffs = ptwt.wavedec2(x, wavelet, level=J, mode=mode)

    # lr
    lrs = []
    for c in coeffs:
        if isinstance(c, ptwt.constants.WaveletDetailTuple2d):
            lrs.append(ptwt.constants.WaveletDetailTuple2d(
                c.horizontal / steps, c.vertical / steps, c.diagonal / steps
            ))
        else:
            lrs.append(c / steps)

    attributions = None
    debug = {"steps": list(path_steps), "imgs": [], "attr_layouts": []}
    sel_steps = set(path_steps)

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

        # save path step
        if t in sel_steps:
            with torch.no_grad():
                img_t = ptwt.waverec2(_detach_coeffs(coeffs), wavelet)
                if clamp_img is not None:
                    lo, hi = clamp_img
                    img_t = img_t.clamp(lo, hi)
            debug["imgs"].append(img_t.detach().cpu())

            # interim attribution avg (up to t)
            attr_np_t = _coeffs_value_numpy([
                (a / (t + 1)) if not isinstance(a, ptwt.constants.WaveletDetailTuple2d)
                else ptwt.constants.WaveletDetailTuple2d(a.horizontal/(t+1), a.vertical/(t+1), a.diagonal/(t+1))
                for a in attributions
            ])
            lay_t = _visualize_grad_wam_layout(attr_np_t, normalize_coeffs=normalize_coeffs)
            debug["attr_layouts"].append(lay_t)

    # final avg
    attributions = [
        (a / steps) if not isinstance(a, ptwt.constants.WaveletDetailTuple2d)
        else ptwt.constants.WaveletDetailTuple2d(a.horizontal/steps, a.vertical/steps, a.diagonal/steps)
        for a in attributions
    ]

    layout_attr = _visualize_grad_wam_layout(_coeffs_value_numpy(attributions), normalize_coeffs=normalize_coeffs)

    pixel_attr = wavelet_layout_to_pixel_attr(
        np.expand_dims(layout_attr, axis=1), J=J, approx_coeffs=False, reduce="sum"
    )  # [B,H,W]
    pixel_attr = np.expand_dims(pixel_attr, axis=1)
    return pixel_attr, layout_attr, debug

# =========================================================
# 6) RUN: one batch, and SAVE EVERYTHING
# =========================================================
SAVE_DIR = "wam_visuals_effb0_imagenet_onebatch"
os.makedirs(SAVE_DIR, exist_ok=True)

J = 3
b = 0  # visualize sample index within batch

# run
pixel_sg, layout_sg = wam_sg_debug(model, x, y, J=J, n_samples=16, stdev_spread=0.25)
pixel_ig, layout_ig, dbg_ig = wam_ig_debug(model, x, y, J=J, n_steps=25, path_alphas=(0.0,0.25,0.5,0.75,1.0))
pixel_mf, layout_mf, dbg_mf = wam_mfaba_debug(model, x, y, J=J, steps=10, path_steps=(0,1,2,5,9))

def save_bundle(prefix, base_img_chw, pixel_attr_hw, layout_attr_bhw):
    fig = overlay_heatmap_fig(base_img_chw, pixel_attr_hw, title=f"{prefix} overlay")
    _save_fig(fig, os.path.join(SAVE_DIR, f"{prefix}__overlay.png"))

    fig = plot_wavelet_layout_fig(layout_attr_bhw, J=J, b=b, title=f"{prefix} wavelet-layout")
    _save_fig(fig, os.path.join(SAVE_DIR, f"{prefix}__layout.png"))

    scales = layout_to_multiscale_pixel(layout_attr_bhw, J=J)
    fig = plot_multiscale_fig(scales, img_chw=base_img_chw, b=b, overlay=True, title_prefix=f"{prefix} scale")
    _save_fig(fig, os.path.join(SAVE_DIR, f"{prefix}__multiscale.png"))

# original
fig = plot_rgb_fig(x[b], title=f"Original (b={b})")
_save_fig(fig, os.path.join(SAVE_DIR, f"00__original__b{b}.png"))

# final attributions
save_bundle(f"01__wam_sg__b{b}", x[b], pixel_sg[b,0], layout_sg)
save_bundle(f"02__wam_ig__b{b}", x[b], pixel_ig[b,0], layout_ig)
save_bundle(f"03__wam_mfaba__b{b}", x[b], pixel_mf[b,0], layout_mf)

# IG path (OOD)
for k, (a, img_a, lay_a) in enumerate(zip(dbg_ig["alphas"], dbg_ig["imgs"], dbg_ig["attr_layouts"])):
    fig = plot_rgb_fig(img_a[b], title=f"IG recon alpha={a} (b={b})")
    _save_fig(fig, os.path.join(SAVE_DIR, f"10__ig_path__k{k:02d}__alpha{a:.2f}__recon__b{b}.png"))

    fig = plot_wavelet_layout_fig(lay_a, J=J, b=b, title=f"IG layout alpha={a} (b={b})")
    _save_fig(fig, os.path.join(SAVE_DIR, f"11__ig_path__k{k:02d}__alpha{a:.2f}__layout__b{b}.png"))

    scales_a = layout_to_multiscale_pixel(lay_a, J=J)
    fig = plot_multiscale_fig(scales_a, img_chw=img_a[b], b=b, overlay=True, title_prefix=f"IG alpha={a:.2f} scale")
    _save_fig(fig, os.path.join(SAVE_DIR, f"12__ig_path__k{k:02d}__alpha{a:.2f}__multiscale__b{b}.png"))

# MFABA path (OOD)
for k, (t, img_t, lay_t) in enumerate(zip(dbg_mf["steps"], dbg_mf["imgs"], dbg_mf["attr_layouts"])):
    fig = plot_rgb_fig(img_t[b], title=f"MFABA recon step={t} (b={b})")
    _save_fig(fig, os.path.join(SAVE_DIR, f"20__mfaba_path__k{k:02d}__step{t:02d}__recon__b{b}.png"))

    fig = plot_wavelet_layout_fig(lay_t, J=J, b=b, title=f"MFABA layout step={t} (b={b})")
    _save_fig(fig, os.path.join(SAVE_DIR, f"21__mfaba_path__k{k:02d}__step{t:02d}__layout__b{b}.png"))

    scales_t = layout_to_multiscale_pixel(lay_t, J=J)
    fig = plot_multiscale_fig(scales_t, img_chw=img_t[b], b=b, overlay=True, title_prefix=f"MFABA step={t:02d} scale")
    _save_fig(fig, os.path.join(SAVE_DIR, f"22__mfaba_path__k{k:02d}__step{t:02d}__multiscale__b{b}.png"))

print("✅ Saved all visuals into:", os.path.abspath(SAVE_DIR))
