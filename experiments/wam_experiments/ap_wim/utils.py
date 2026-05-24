import torch
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    """Set up random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
import numpy as np
import cv2

def wavelet_layout_to_pixel_attr(attr, J, approx_coeffs=False, reduce="sum"):
    # torch -> numpy
    if hasattr(attr, "detach"):
        attr = attr.detach()
    if hasattr(attr, "cpu"):
        attr = attr.cpu()
    attr = np.asarray(attr, dtype=np.float32)

    if attr.ndim == 4:   # [B,1,H,W]
        attr = attr[:, 0]
    if attr.ndim != 3:
        raise ValueError(f"attr must be [B,H,W] or [B,1,H,W], got {attr.shape}")

    B, H, W = attr.shape
    if H != W:
        raise ValueError("expect square layout [H,W]")

    n_levels = J + (1 if approx_coeffs else 0)
    scales = np.zeros((B, n_levels, H, W), dtype=np.float32)

    for j in range(J):
        end = int(H / (2 ** j))
        start = int(H / (2 ** (j + 1)))

        diagonal   = attr[:, start:end, start:end]
        vertical   = attr[:, start:end, :start]
        horizontal = attr[:, :start, start:end]

        for b in range(B):
            tmp_h = cv2.resize(horizontal[b], (W, H), interpolation=cv2.INTER_LINEAR)
            tmp_v = cv2.resize(vertical[b],   (W, H), interpolation=cv2.INTER_LINEAR)
            tmp_d = cv2.resize(diagonal[b],   (W, H), interpolation=cv2.INTER_LINEAR)
            scales[b, j] = tmp_h + tmp_v + tmp_d

    if approx_coeffs:
        end = int(H / (2 ** J))
        for b in range(B):
            approx = attr[b, :end, :end]
            scales[b, J] = cv2.resize(approx, (W, H), interpolation=cv2.INTER_LINEAR)

    if reduce == "sum":
        return scales.sum(axis=1)  # [B,H,W]
    elif reduce == "none":
        return scales              # [B,J(,+1),H,W]
    else:
        raise ValueError("reduce must be 'sum' or 'none'")