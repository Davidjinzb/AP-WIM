import numpy as np
import torch
import ptwt
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np
import torch
import ptwt
import torchaudio

def coeff_attr_to_input_melspec(
    coeff_attr,                 # list length (J+1), each np.ndarray [B, L_j]
    ref_waveform,               # torch.Tensor [B, W]  (用来对齐长度 + 可选设备)
    wavelet="haar",
    J=3,
    n_fft=1024,
    hop_length=512,
    sample_rate=44100,
    n_mels=128,
    power=2.0,
    use_abs=False,              # 可选：是否对重建后的归因取 abs（可视化常用）
    eps=1e-8,
    return_waveform_attr=False, # True: 同时返回 waveform_attr
):
    """
    返回：
      melspec_attr_recon: torch.Tensor [B, 1, T, n_mels]  (模型输入域)
      (可选) waveform_attr: torch.Tensor [B, W]
    """

    if not isinstance(coeff_attr, list) or len(coeff_attr) != (J + 1):
        raise ValueError(f"coeff_attr must be list of length J+1={J+1}")

    if not isinstance(ref_waveform, torch.Tensor) or ref_waveform.ndim != 2:
        raise ValueError("ref_waveform must be torch.Tensor [B, W]")

    device = ref_waveform.device
    dtype = ref_waveform.dtype
    B, W = ref_waveform.shape

    # 1) numpy coeff_attr -> torch coeffs
    coeffs_t = []
    for lvl, c in enumerate(coeff_attr):
        if not isinstance(c, np.ndarray):
            c = np.asarray(c)
        ct = torch.from_numpy(c).to(device=device, dtype=dtype)
        if ct.shape[0] != B:
            raise ValueError(f"coeff_attr[{lvl}] batch size {ct.shape[0]} != ref batch {B}")
        coeffs_t.append(ct)

    # 2) waverec: coeffs -> waveform attribution
    waveform_attr = ptwt.waverec(coeffs_t, wavelet)   # [B, W']（W' 可能略不同）

    # 3) 对齐长度到 ref_waveform 的 W
    if waveform_attr.shape[1] > W:
        waveform_attr = waveform_attr[:, :W]
    elif waveform_attr.shape[1] < W:
        pad = W - waveform_attr.shape[1]
        waveform_attr = torch.nn.functional.pad(waveform_attr, (0, pad))

    if use_abs:
        waveform_attr = waveform_attr.abs()

    # （可选）避免极小值导致 dB 里出现 -inf
    # torchaudio.transforms.AmplitudeToDB 自己会做 clamp，但这里也可保险一下
    # waveform_attr = waveform_attr.clamp(min=-1.0, max=1.0)

    # 4) waveform_attr -> MelSpectrogram -> dB，得到模型输入域
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        center=True,
    ).to(device)

    to_db = torchaudio.transforms.AmplitudeToDB().to(device)

    ms = mel(waveform_attr)          # [B, n_mels, T]
    ms = to_db(ms)                   # [B, n_mels, T]
    ms = ms.transpose(1, 2)          # [B, T, n_mels]
    ms = ms.unsqueeze(1)             # [B, 1, T, n_mels]

    if return_waveform_attr:
        return ms, waveform_attr
    return ms


def _to_numpy_1d(coeffs, grads=True):
    """coeffs: list[tensor] for wavedec; grads reads .grad."""
    if grads:
        return [c.grad.detach().cpu().numpy() for c in coeffs]
    return [c.detach().cpu().numpy() for c in coeffs]


def compute_melspec_torch(
    waveforms: torch.Tensor,         # [B, W] float
    n_fft=1024,
    sample_rate=44100,
    n_mels=128,
):
    """
    waveforms: torch.Tensor [B, W] on SAME device you want to run on
    return: torch.Tensor [B, 1, T, n_mels]
    """
    device = waveforms.device
    mel = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels).to(device)
    power_to_db = AmplitudeToDB().to(device)

    # mel: [B, n_mels, T]
    ms = mel(waveforms)
    # db: [B, n_mels, T]
    ms = power_to_db(ms)
    # -> [B, T, n_mels]
    ms = ms.transpose(1, 2)
    # -> [B, 1, T, n_mels]
    return ms.unsqueeze(1)


def _ensure_x_tensor(x, normalize: bool = False):
    """
    x:
      - list of 1D arrays length B, each [W]
      - or torch.Tensor [B,W]
    return torch.Tensor [B,W] float32 (CPU or GPU都行，后面会搬到 model device)

    normalize=False: 直接用你给的 waveform（你 dataset 已经做过归一化就用这个）
    normalize=True : 额外做一次 max-abs 归一化（保底）
    """
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
    """make each coeff a fresh leaf tensor requiring grad"""
    return [c.detach().clone().requires_grad_(True) for c in coeffs]


def _base_wam1d_once(
    model,
    x,                      # torch.Tensor [B,W]   (if waveform=True)
    y,                      # torch.LongTensor [B]
    wavelet="haar",
    J=3,
    mode="reflect",
    n_fft=1024,
    sample_rate=44100,
    n_mels=128,
    waveform=True,
):
    """
    One forward/backward pass like BaseWAM1D.__call__.

    Returns:
      melspec_grad_np: np.ndarray [B, T, n_mels]
      grad_coeffs_np_list: list length J+1, each np.ndarray [B, L_j]
    """
    device = next(model.parameters()).device
    model.eval()

    y = _ensure_y_tensor(y, device)

    # 1) wavelet coeffs
    if waveform:
        x = x.to(device)
        coeffs = ptwt.wavedec(x, wavelet, level=J, mode=mode)  # list[tensor]
    else:
        coeffs = x  # x is coeff list already
        if not isinstance(coeffs, list):
            raise TypeError("waveform=False expects x to be a list of coeff tensors")

    # 2) require grads on coeffs (fresh leaves)
    grad_coeffs = _clone_leaf_list(coeffs)

    # 3) reconstruct
    x_rec = ptwt.waverec(grad_coeffs, wavelet)  # [B,W]

    # 4) melspec, keep grad
    melspec = compute_melspec_torch(
        x_rec, n_fft=n_fft, sample_rate=sample_rate, n_mels=n_mels
    )  # [B,1,T,n_mels]
    melspec.retain_grad()

    # 5) forward/backward
    out = model(melspec)  # model expects [B,1,T,n_mels]
    loss = out.gather(1, y.view(-1, 1)).mean()

    model.zero_grad(set_to_none=True)
    loss.backward()

    # melspec grad: [B,1,T,n_mels] -> [B,T,n_mels]
    melspec_grad_np = melspec.grad.detach().cpu().numpy()[:, 0, :, :].astype(np.float32)

    grad_coeffs_np_list = _to_numpy_1d(grad_coeffs, grads=True)
    return melspec_grad_np, grad_coeffs_np_list


def _scale_coeffs_1d(coeffs, alpha: float):
    return [alpha * c for c in coeffs]


# =========================================================
# 1) wam_sg
# =========================================================

def wam_sg(
    model,
    x,                      # list[wav] or torch.Tensor [B,W]
    y,                      # [B]
    wavelet="haar",
    J=3,
    mode="reflect",
    n_fft=1024,
    sample_rate=44100,
    n_mels=128,
    n_samples=25,
    stdev_spread=0.001,
    seed=42,
    normalize_waveform=False,  # ★默认不normalize（你 dataset 已经做了）
):
    """
    SmoothGrad-WAM (1D)

    returns:
      melspec_grad_avg: np.ndarray [B, T, n_mels]
      grad_coeffs_avg: list length (J+1), each np.ndarray [B, L_j]
    """
    device = next(model.parameters()).device
    x = _ensure_x_tensor(x, normalize=normalize_waveform).to(device)
    y = _ensure_y_tensor(y, device)

    torch.manual_seed(seed)

    melspec_grads = []
    all_grads = []

    for _ in range(n_samples):
        # noise on device
        # stdev per-sample based on range (max-min)
        maxv = x.amax(dim=1, keepdim=True)
        minv = x.amin(dim=1, keepdim=True)
        stdev = (stdev_spread * (maxv - minv)).clamp_min(0.0)

        noisy = x + torch.randn_like(x) * stdev

        ms_grad, grad_coeffs = _base_wam1d_once(
            model, noisy, y,
            wavelet=wavelet, J=J, mode=mode,
            n_fft=n_fft, sample_rate=sample_rate, n_mels=n_mels,
            waveform=True,
        )
        melspec_grads.append(ms_grad)
        all_grads.append(grad_coeffs)

    melspec_grad_avg = np.mean(np.stack(melspec_grads, axis=0), axis=0)

    grad_coeffs_avg = []
    for lvl in range(J + 1):
        lvl_stack = np.stack([g[lvl] for g in all_grads], axis=0)  # [n_samples,B,L]
        grad_coeffs_avg.append(np.mean(lvl_stack, axis=0))

    # return melspec_grad_avg.astype(np.float32), grad_coeffs_avg
    return coeff_attr_to_input_melspec(grad_coeffs_avg, x,
                                      wavelet=wavelet, J=J,
                                      n_fft=n_fft, sample_rate=sample_rate,
                                      n_mels=n_mels,
                                      use_abs=False).cpu().detach().numpy()


# =========================================================
# 2) wam_ig
# =========================================================

def wam_ig(
    model,
    x,                      # list[wav] or torch.Tensor [B,W]
    y,                      # [B]
    wavelet="haar",
    J=3,
    mode="reflect",
    n_fft=1024,
    sample_rate=44100,
    n_mels=128,
    n_steps=25,
    normalize_waveform=False,  # ★默认不normalize
):
    """
    returns:
      melspec_attr: np.ndarray [B, T, n_mels]
      coeff_attr: list length (J+1), each np.ndarray [B, L_j]
    """
    device = next(model.parameters()).device
    x = _ensure_x_tensor(x, normalize=normalize_waveform).to(device)
    y = _ensure_y_tensor(y, device)

    coeffs = ptwt.wavedec(x, wavelet, level=J, mode=mode)
    baseline_coeff_np = _to_numpy_1d(coeffs, grads=False)

    baseline_melspec = compute_melspec_torch(
        x, n_fft=n_fft, sample_rate=sample_rate, n_mels=n_mels
    ).detach().cpu().numpy()[:, 0, :, :].astype(np.float32)  # [B,T,M]

    alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)

    path_melspec_grads = []
    path_coeff_grads = []

    for a in alphas:
        path_coeffs = _scale_coeffs_1d(coeffs, float(a))
        ms_grad, grad_coeffs = _base_wam1d_once(
            model, path_coeffs, y,
            wavelet=wavelet, J=J, mode=mode,
            n_fft=n_fft, sample_rate=sample_rate, n_mels=n_mels,
            waveform=False,
        )
        path_melspec_grads.append(ms_grad)
        path_coeff_grads.append(grad_coeffs)

    path_melspec_grads = np.stack(path_melspec_grads, axis=1)  # [B,n_steps,T,M]
    integral_melspec = np.trapz(path_melspec_grads, x=alphas, axis=1)  # [B,T,M]
    melspec_attr = baseline_melspec * integral_melspec

    coeff_attr = []
    for lvl in range(J + 1):
        lvl_path = np.stack([pcg[lvl] for pcg in path_coeff_grads], axis=0)  # [n_steps,B,L]
        lvl_int = np.trapz(lvl_path, x=alphas, axis=0)  # [B,L]
        coeff_attr.append(baseline_coeff_np[lvl] * lvl_int)

    # return melspec_attr.astype(np.float32), coeff_attr
    return coeff_attr_to_input_melspec(coeff_attr, x,
                                      wavelet=wavelet, J=J,
                                      n_fft=n_fft, sample_rate=sample_rate,
                                      n_mels=n_mels,
                                      use_abs=False).cpu().detach().numpy()


# =========================================================
# 3) wam_mfaba
# =========================================================

def wam_mfaba(
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
    normalize_waveform=False,  # ★默认不normalize
):
    """
    returns:
      last_melspec_grad: np.ndarray [B,T,n_mels]
      coeff_attr: list length (J+1), each np.ndarray [B,L_j]
    """
    device = next(model.parameters()).device
    x = _ensure_x_tensor(x, normalize=normalize_waveform).to(device)
    y = _ensure_y_tensor(y, device)

    coeffs = ptwt.wavedec(x, wavelet, level=J, mode=mode)
    lrs = [c / steps for c in coeffs]

    attributions = None
    last_ms_grad = None

    for _ in range(steps):
        ms_grad, grads = _base_wam1d_once(
            model, coeffs, y,
            wavelet=wavelet, J=J, mode=mode,
            n_fft=n_fft, sample_rate=sample_rate, n_mels=n_mels,
            waveform=False,
        )
        last_ms_grad = ms_grad

        # accumulate attr in numpy space
        if attributions is None:
            attributions = [lr.detach().cpu().numpy() * np.sign(g) * g for lr, g in zip(lrs, grads)]
        else:
            attributions = [a + lr.detach().cpu().numpy() * np.sign(g) * g for a, lr, g in zip(attributions, lrs, grads)]

        # update coeffs (torch) using sign of grads
        new_coeffs = []
        for c, lr, g in zip(coeffs, lrs, grads):
            g_t = torch.from_numpy(g).to(device=device, dtype=c.dtype)
            new_coeffs.append((c + lr * torch.sign(g_t)).detach())
        coeffs = new_coeffs

    coeff_attr = [a / steps for a in attributions]
    # return last_ms_grad.astype(np.float32), coeff_attr
    return coeff_attr_to_input_melspec(coeff_attr, x,
                                      wavelet=wavelet, J=J,
                                      n_fft=n_fft, sample_rate=sample_rate,
                                      n_mels=n_mels,
                                      use_abs=False).cpu().detach().numpy()

