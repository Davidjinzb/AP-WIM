# -*- coding: utf-8 -*-

import numpy as np
import torch
import ptwt
import cv2
from utils import wavelet_layout_to_pixel_attr


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
            out.append(
                ptwt.constants.WaveletDetailTuple2d(
                    c.horizontal.detach(),
                    c.vertical.detach(),
                    c.diagonal.detach(),
                )
            )
        else:
            out.append(c.detach())
    return out

def _scale_coeffs(coeffs, alpha: float):
    """coeffs * alpha (for IG path)"""
    out = [coeffs[0] * alpha]
    for c in coeffs[1:]:
        out.append(
            ptwt.constants.WaveletDetailTuple2d(
                c.horizontal * alpha,
                c.vertical * alpha,
                c.diagonal * alpha,
            )
        )
    return out

def _require_grad_coeffs(coeffs):
    """clone structure but enable requires_grad on each tensor"""
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
    coeffs -> numpy list, preserving wavedec2 structure:
      [approx_np, detail_tuple_np, detail_tuple_np, ...]
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

def _coeffs_grad_numpy(coeffs_with_grad):
    """same as above but read .grad"""
    out = [coeffs_with_grad[0].grad.detach().cpu().numpy()]
    for c in coeffs_with_grad[1:]:
        out.append(
            ptwt.constants.WaveletDetailTuple2d(
                c.horizontal.grad.detach().cpu().numpy(),
                c.vertical.grad.detach().cpu().numpy(),
                c.diagonal.grad.detach().cpu().numpy(),
            )
        )
    return out


def _grad_coeffs_from_coeffs(model, coeffs, y, wavelet="haar"):
    """
    输入：
      coeffs: wavedec2 输出格式的 list（tensor 或 detail tuple tensor）
      y: [B] long tensor
    输出：
      grad_coeffs_np: numpy list（approx + 每层(h/v/d) 的梯度）
    """
    grad_coeffs = _require_grad_coeffs(coeffs)

    # reconstruct image from coeffs
    img = ptwt.waverec2(grad_coeffs, wavelet)  # shape [B, C, H, W]

    logits = model(img)
    # target logit mean: gather 比 diag 更简单
    loss = logits.gather(1, y.view(-1, 1)).mean()

    model.zero_grad(set_to_none=True)
    loss.backward()

    return _coeffs_grad_numpy(grad_coeffs)


def _fit_layout_block(block, target_h, target_w):
    """Resize a [B,h,w] coefficient block to the target layout slot."""
    block = np.asarray(block, dtype=np.float32)
    if block.shape[1] == target_h and block.shape[2] == target_w:
        return block
    out = np.zeros((block.shape[0], target_h, target_w), dtype=np.float32)
    for b in range(block.shape[0]):
        out[b] = cv2.resize(block[b], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return out


def _visualize_grad_wam_layout(grad_coeffs_np, normalize_coeffs=True, layout_shape=None):
    """
    把“小波系数梯度”拼成 wavelet-layout map（不是原图像素坐标系）
    输入：
      grad_coeffs_np: numpy list（approx + detail tuples）
    输出：
      layout: np.ndarray, shape [B, H, W]  （默认用原图尺寸）
    """
    B = grad_coeffs_np[0].shape[0]
    if layout_shape is None:
        H = int(2 * grad_coeffs_np[-1].horizontal.shape[-2])
        W = int(2 * grad_coeffs_np[-1].horizontal.shape[-1])
    else:
        H, W = int(layout_shape[0]), int(layout_shape[1])
    J = len(grad_coeffs_np) - 1

    layout = np.zeros((B, H, W), dtype=np.float32)

    # approx block (top-left)
    approx = np.abs(grad_coeffs_np[0].mean(axis=1))
    if normalize_coeffs and approx.max() > 0:
        approx = approx / (approx.max() + 1e-12)
    approx_h = max(1, int(H / 2**J))
    approx_w = max(1, int(W / 2**J))
    approx = _fit_layout_block(approx, approx_h, approx_w)
    layout[:, :approx_h, :approx_w] = approx

    # details blocks (from coarse->fine layout placing)
    for i, c in enumerate(grad_coeffs_np[1:][::-1]):
        end_h = int(H / 2**i)
        end_w = int(W / 2**i)
        start_h = int(H / 2**(i + 1))
        start_w = int(W / 2**(i + 1))
        block_h = end_h - start_h
        block_w = end_w - start_w

        h = np.abs(c.horizontal.mean(axis=1))
        v = np.abs(c.vertical.mean(axis=1))
        d = np.abs(c.diagonal.mean(axis=1))

        if normalize_coeffs:
            if h.max() > 0: h = h / (h.max() + 1e-12)
            if v.max() > 0: v = v / (v.max() + 1e-12)
            if d.max() > 0: d = d / (d.max() + 1e-12)

        h = _fit_layout_block(h, start_h, block_w)
        v = _fit_layout_block(v, block_h, start_w)
        d = _fit_layout_block(d, block_h, block_w)

        layout[:, start_h:end_h, start_w:end_w] = d
        layout[:, start_h:end_h, :start_w] = v
        layout[:, :start_h, start_w:end_w] = h

    return layout


def reproject_layout_to_pixel_scales(layout_map, J, approx_coeffs=False):
    """
    输入：
      layout_map: np.ndarray, shape [B, H, W]  (wavelet-layout)
    输出：
      approx_coeffs=False: np.ndarray [B, J, H, W]
      approx_coeffs=True : np.ndarray [B, J+1, H, W] (最后一层 approx)
    """
    B, H, W = layout_map.shape
    out = np.zeros((B, J + (1 if approx_coeffs else 0), H, W), dtype=np.float32)

    for j in range(J):
        end = int(H / 2**j)
        start = int(H / 2**(j + 1))

        diagonal = layout_map[:, start:end, start:end]
        vertical = layout_map[:, start:end, :start]
        horizontal = layout_map[:, :start, start:end]

        for b in range(B):
            tmp_h = cv2.resize(horizontal[b], (W, H), interpolation=cv2.INTER_LINEAR)
            tmp_v = cv2.resize(vertical[b], (W, H), interpolation=cv2.INTER_LINEAR)
            tmp_d = cv2.resize(diagonal[b], (W, H), interpolation=cv2.INTER_LINEAR)
            out[b, j] = tmp_h + tmp_v + tmp_d

    if approx_coeffs:
        end = int(H / 2**J)
        for b in range(B):
            tmp_a = layout_map[b, :end, :end]
            out[b, J] = cv2.resize(tmp_a, (W, H), interpolation=cv2.INTER_LINEAR)

    return out


# =========================================================
# 1) wam_sg: SmoothGrad-WAM
# =========================================================

def wam_sg(
    model,
    x,                     # [B, C, H, W]
    y,                     # [B] long
    wavelet="haar",
    J=3,
    mode="reflect",
    n_samples=25,
    stdev_spread=0.25,
    seed=42,
    normalize_coeffs=True,
    approx_coeffs=False,
    return_scales=False,
):
    """
    输出：
      - layout_avg: np.ndarray [B, H, W] (wavelet-layout map)
    """
    _assert_same_device(model, x, y)
    torch.manual_seed(seed)

    B = x.shape[0]
    layout_sum = None

    for _ in range(n_samples):
        # 用 torch 噪声，避免 numpy/device 来回
        noisy = torch.empty_like(x)
        for i in range(B):
            max_x = x[i].max()
            min_x = x[i].min()
            stdev = stdev_spread * (max_x - min_x)
            noisy[i] = x[i] + torch.randn_like(x[i]) * stdev

        coeffs = ptwt.wavedec2(noisy, wavelet, level=J, mode=mode)
        grad_np = _grad_coeffs_from_coeffs(model, coeffs, y, wavelet=wavelet)
        layout = _visualize_grad_wam_layout(
            grad_np,
            normalize_coeffs=normalize_coeffs,
            layout_shape=x.shape[-2:],
        )

        layout_sum = layout if layout_sum is None else (layout_sum + layout)

    layout_avg = layout_sum / float(n_samples)

    pixel_attr = wavelet_layout_to_pixel_attr(
        np.expand_dims(layout_avg, axis=1),  # [B,1,H,W]
        J=J,
        approx_coeffs=approx_coeffs,
        reduce="sum",
    )  # [B,H,W]
    pixel_attr = np.expand_dims(pixel_attr, axis=1)  # [B,1,H,W]

    return pixel_attr


# =========================================================
# 2) wam_ig: IntegratedGrad-WAM
# =========================================================

def wam_ig(
    model,
    x,                     # [B, C, H, W]
    y,                     # [B] long
    wavelet="haar",
    J=3,
    mode="reflect",
    n_steps=25,
    normalize_grad_layout=True,
    approx_coeffs=False,
    return_scales=False,
):
    """
    IntegratedGrad-WAM（按你的原实现思路）：

      coeffs = wavedec2(x)
      baseline_z = layout(coeffs 的“值”)  (用 coeff 值投成 wavelet-layout)
      alpha ∈ [0,1]：
         path_coeffs = alpha * coeffs
         grad_layout(alpha) = gradWAM(path_coeffs)
      integral = trapz(grad_layout, alpha)
      attribution_layout = baseline_z * integral

    输出：
      - attribution_layout: np.ndarray [B, H, W] (wavelet-layout map)
    """
    _assert_same_device(model, x, y)

    coeffs = ptwt.wavedec2(x, wavelet, level=J, mode=mode)

    # baseline_z：用 coeff “值”拼成 wavelet-layout（与你原 intergrated_wam 保持一致）
    coeff_vals_np = _coeffs_value_numpy(coeffs)
    baseline_z = _visualize_grad_wam_layout(
        coeff_vals_np,
        normalize_coeffs=True,
        layout_shape=x.shape[-2:],
    )  # [B,H,W]

    alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    B, H, W = baseline_z.shape
    grad_path = np.zeros((B, n_steps, H, W), dtype=np.float32)

    for i, a in enumerate(alphas):
        path_coeffs = _scale_coeffs(coeffs, float(a))
        grad_np = _grad_coeffs_from_coeffs(model, path_coeffs, y, wavelet=wavelet)
        grad_layout = _visualize_grad_wam_layout(
            grad_np,
            normalize_coeffs=normalize_grad_layout,
            layout_shape=x.shape[-2:],
        )
        grad_path[:, i] = grad_layout

    integral = np.trapz(np.nan_to_num(grad_path), x=alphas, axis=1)  # [B,H,W]
    attribution_layout = baseline_z * integral
    pixel_attr = wavelet_layout_to_pixel_attr(
        np.expand_dims(attribution_layout, axis=1),  # [B,1,H,W]
        J=J,
        approx_coeffs=approx_coeffs,
        reduce="sum",
    )  # [B,H,W]
    attribution_layout = np.expand_dims(pixel_attr, axis=1)  # [B,1,H,W]
    return attribution_layout


# =========================================================
# 3) wam_mfaba: MFABA-WAM (pure torch update, no to(device))
# =========================================================

def wam_mfaba(
    model,
    x,                     # [B, C, H, W]
    y,                     # [B] long
    wavelet="haar",
    J=3,
    mode="reflect",
    steps=10,
    approx_coeffs=False,
    return_scales=False,
):
    """

    输出：
      - wam_layout: np.ndarray [B, H, W] (wavelet-layout)
    """
    _assert_same_device(model, x, y)

    coeffs = ptwt.wavedec2(x, wavelet, level=J, mode=mode)

    # lr（保持 tensor/device 不动）
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

    attributions = None  # coeff-structured tensors

    for _ in range(steps):
        # ---- compute grads wrt coeffs (in torch) ----
        grad_coeffs = _require_grad_coeffs(coeffs)
        img = ptwt.waverec2(grad_coeffs, wavelet)

        logits = model(img)
        loss = logits.gather(1, y.view(-1, 1)).mean()

        model.zero_grad(set_to_none=True)
        loss.backward()

        # ---- accumulate attr and update coeffs ----
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

    # 平均
    attributions = [
        (a / steps) if not isinstance(a, ptwt.constants.WaveletDetailTuple2d)
        else ptwt.constants.WaveletDetailTuple2d(a.horizontal / steps, a.vertical / steps, a.diagonal / steps)
        for a in attributions
    ]

    # 转成 numpy coeff-structure，再拼 layout
    attr_np = _coeffs_value_numpy(attributions)
    wam_layout = _visualize_grad_wam_layout(
        attr_np,
        normalize_coeffs=True,
        layout_shape=x.shape[-2:],
    )
    wam_layout = np.expand_dims(wam_layout, axis=1)  # [B,1,H,W]
    pixel_attr = wavelet_layout_to_pixel_attr(
        wam_layout,
        J=J,
        approx_coeffs=approx_coeffs,
        reduce="sum",
    )  # [B,H,W]
    return np.expand_dims(pixel_attr, axis=1) # [B,1,H,W]
