# -*- coding: utf-8 -*-

import numpy as np
import torch
import ptwt
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


def coeff_attr_to_input_melspec(
    coeff_attr,                 # list length (J+1), each np.ndarray [B, L_j]
    ref_waveform,               # torch.Tensor [B, W]
    wavelet="haar",
    J=3,
    n_fft=1024,
    hop_length=512,
    sample_rate=44100,
    n_mels=128,
    power=2.0,
    use_abs=False,
    eps=1e-8,
    return_waveform_attr=False,
):
    if not isinstance(coeff_attr, list) or len(coeff_attr) != (J + 1):
        raise ValueError(f"coeff_attr must be list of length J+1={J+1}")

    if not isinstance(ref_waveform, torch.Tensor) or ref_waveform.ndim != 2:
        raise ValueError("ref_waveform must be torch.Tensor [B, W]")

    device = ref_waveform.device
    dtype = ref_waveform.dtype
    B, W = ref_waveform.shape

    coeffs_t = []
    for lvl, c in enumerate(coeff_attr):
        if not isinstance(c, np.ndarray):
            c = np.asarray(c)
        ct = torch.from_numpy(c).to(device=device, dtype=dtype)
        if ct.shape[0] != B:
            raise ValueError(f"coeff_attr[{lvl}] batch size {ct.shape[0]} != ref batch {B}")
        coeffs_t.append(ct)

    waveform_attr = ptwt.waverec(coeffs_t, wavelet)  # [B, W'] (may differ slightly)

    if waveform_attr.shape[1] > W:
        waveform_attr = waveform_attr[:, :W]
    elif waveform_attr.shape[1] < W:
        pad = W - waveform_attr.shape[1]
        waveform_attr = torch.nn.functional.pad(waveform_attr, (0, pad))

    if use_abs:
        waveform_attr = waveform_attr.abs()

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        center=True,
    ).to(device)

    to_db = torchaudio.transforms.AmplitudeToDB().to(device)

    ms = mel(waveform_attr)      # [B, n_mels, T]
    ms = to_db(ms)               # [B, n_mels, T]
    ms = ms.transpose(1, 2)      # [B, T, n_mels]
    ms = ms.unsqueeze(1)         # [B, 1, T, n_mels]

    if return_waveform_attr:
        return ms, waveform_attr
    return ms


def _to_numpy_1d(coeffs, grads=True):
    if grads:
        return [c.grad.detach().cpu().numpy() for c in coeffs]
    return [c.detach().cpu().numpy() for c in coeffs]


def compute_melspec_torch(
    waveforms: torch.Tensor,     # [B, W]
    n_fft=1024,
    sample_rate=44100,
    n_mels=128,
):
    device = waveforms.device
    mel = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels).to(device)
    power_to_db = AmplitudeToDB().to(device)

    ms = mel(waveforms)          # [B, n_mels, T]
    ms = power_to_db(ms)         # [B, n_mels, T]
    ms = ms.transpose(1, 2)      # [B, T, n_mels]
    return ms.unsqueeze(1)       # [B, 1, T, n_mels]


def _ensure_x_tensor(x, normalize: bool = False):
    if isinstance(x, list):
        x = np.array(x, dtype=np.float32)
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x.float()
    else:
        raise TypeError("x must be list or torch.Tensor")

    if normalize:
        denom = t.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        t = t / denom
    return t


def _ensure_y_tensor(y, device):
    if isinstance(y, torch.Tensor):
        return y.to(device=device, dtype=torch.long)
    return torch.tensor(y, device=device, dtype=torch.long)


def _clone_leaf_list(coeffs):
    return [c.detach().clone().requires_grad_(True) for c in coeffs]


def _base_wam1d_once(
    model,
    x,                      # torch.Tensor [B,W] if waveform=True, else list[tensor]
    y,                      # torch.LongTensor [B]
    wavelet="haar",
    J=3,
    mode="reflect",
    n_fft=1024,
    sample_rate=44100,
    n_mels=128,
    waveform=True,
):
    device = next(model.parameters()).device
    model.eval()

    y = _ensure_y_tensor(y, device)

    if waveform:
        x = x.to(device)
        coeffs = ptwt.wavedec(x, wavelet, level=J, mode=mode)  # list[tensor]
    else:
        coeffs = x
        if not isinstance(coeffs, list):
            raise TypeError("waveform=False expects x to be a list of coeff tensors")

    grad_coeffs = _clone_leaf_list(coeffs)

    x_rec = ptwt.waverec(grad_coeffs, wavelet)  # [B,W]

    melspec = compute_melspec_torch(
        x_rec, n_fft=n_fft, sample_rate=sample_rate, n_mels=n_mels
    )  # [B,1,T,n_mels]
    melspec.retain_grad()

    out = model(melspec)
    loss = out.gather(1, y.view(-1, 1)).mean()

    model.zero_grad(set_to_none=True)
    loss.backward()

    melspec_grad_np = melspec.grad.detach().cpu().numpy()[:, 0, :, :].astype(np.float32)
    grad_coeffs_np_list = _to_numpy_1d(grad_coeffs, grads=True)

    return melspec_grad_np, grad_coeffs_np_list


def ap_wim(
    model,
    x,                      # list[wav] or torch.Tensor [B,W]
    y,                      # [B]
    wavelet="haar",
    J=3,
    mode="reflect",
    n_fft=1024,
    sample_rate=44100,
    n_mels=128,
    steps=3,
    normalize_waveform=False,
):
    device = next(model.parameters()).device
    x = _ensure_x_tensor(x, normalize=normalize_waveform).to(device)
    y = _ensure_y_tensor(y, device)

    coeffs = ptwt.wavedec(x, wavelet, level=J, mode=mode)
    lrs = [c / steps for c in coeffs]

    attributions = None

    for _ in range(steps):
        _, grads = _base_wam1d_once(
            model, coeffs, y,
            wavelet=wavelet, J=J, mode=mode,
            n_fft=n_fft, sample_rate=sample_rate, n_mels=n_mels,
            waveform=False,
        )

        if attributions is None:
            attributions = [
                lr.detach().cpu().numpy() * np.sign(g) * g
                for lr, g in zip(lrs, grads)
            ]
        else:
            attributions = [
                a + lr.detach().cpu().numpy() * np.sign(g) * g
                for a, lr, g in zip(attributions, lrs, grads)
            ]

        new_coeffs = []
        for c, lr, g in zip(coeffs, lrs, grads):
            g_t = torch.from_numpy(g).to(device=device, dtype=c.dtype)
            new_coeffs.append((c + lr * torch.sign(g_t)).detach())
        coeffs = new_coeffs

    coeff_attr = [a / steps for a in attributions]

    return coeff_attr_to_input_melspec(
        coeff_attr, x,
        wavelet=wavelet, J=J,
        n_fft=n_fft, sample_rate=sample_rate,
        n_mels=n_mels,
        use_abs=False,
    ).cpu().detach().numpy()
