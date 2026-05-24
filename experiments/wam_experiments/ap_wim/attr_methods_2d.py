# -*- coding: utf-8 -*-

import numpy as np
import torch
import ptwt

from utils import wavelet_layout_to_pixel_attr


def _assert_same_device(model, x, y):
    dev_m = next(model.parameters()).device
    if x.device != dev_m:
        raise RuntimeError(f"x.device={x.device} != model.device={dev_m}")
    if y.device != dev_m:
        raise RuntimeError(f"y.device={y.device} != model.device={dev_m}")
    if y.dtype != torch.long:
        raise RuntimeError(f"y.dtype must be torch.long, got {y.dtype}")


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


def _visualize_grad_wam_layout(coeffs_np, normalize_coeffs=True):
    B = coeffs_np[0].shape[0]
    H = int(2 * coeffs_np[-1].horizontal.shape[-1])

    layout = np.zeros((B, H, H), dtype=np.float32)

    # approx block (top-left)
    approx = np.abs(coeffs_np[0].mean(axis=1))
    if normalize_coeffs and approx.max() > 0:
        approx = approx / (approx.max() + 1e-12)
    layout[:, :approx.shape[1], :approx.shape[2]] = approx

    # details blocks (from coarse->fine layout placing)
    for i, c in enumerate(coeffs_np[1:][::-1]):
        end = int(H / 2**i)
        start = int(H / 2**(i + 1))

        h = np.abs(c.horizontal.mean(axis=1))
        v = np.abs(c.vertical.mean(axis=1))
        d = np.abs(c.diagonal.mean(axis=1))

        if normalize_coeffs:
            if h.max() > 0:
                h = h / (h.max() + 1e-12)
            if v.max() > 0:
                v = v / (v.max() + 1e-12)
            if d.max() > 0:
                d = d / (d.max() + 1e-12)

        layout[:, start:end, start:end] = d
        layout[:, start:end, :start] = v
        layout[:, :start, start:end] = h

    return layout



def ap_wim(
    model,
    x,                     # [B, C, H, W]
    y,                     # [B] long
    wavelet="haar",
    J=3,
    mode="reflect",
    steps=10,
):

    _assert_same_device(model, x, y)

    coeffs = ptwt.wavedec2(x, wavelet, level=J, mode=mode)

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

    attributions = [
        (a / steps) if not isinstance(a, ptwt.constants.WaveletDetailTuple2d)
        else ptwt.constants.WaveletDetailTuple2d(a.horizontal / steps, a.vertical / steps, a.diagonal / steps)
        for a in attributions
    ]

    attr_np = _coeffs_value_numpy(attributions)
    wam_layout = _visualize_grad_wam_layout(attr_np, normalize_coeffs=True)
    wam_layout = np.expand_dims(wam_layout, axis=1)  # [B,1,H,W]

    pixel_attr = wavelet_layout_to_pixel_attr(
        wam_layout,
        J=J,
        approx_coeffs=False,
        reduce="sum",
    )  # [B,H,W]

    return np.expand_dims(pixel_attr, axis=1)  # [B,1,H,W]
