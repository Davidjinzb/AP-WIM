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
    """
    把 wavelet-layout attribution 转成像素域 attribution（对齐原图坐标）。

    输入:
      attr: np.ndarray 或 torch.Tensor
            支持 shape:
              - [B, 1, H, W]
              - [B, H, W]
            这里的 H=W 通常是 224
      J: 小波分解层数（你用的 J=3）
      approx_coeffs: 是否把 approx 那块也作为一层输出（一般不需要）
      reduce: "sum" / "none"
            - "sum": 返回 [B, H, W] 像素域总热图（各尺度相加）
            - "none": 返回 [B, J, H, W] 每个尺度一张热图

    输出:
      - if reduce=="sum":  pixel_attr [B, H, W]
      - if reduce=="none": scales    [B, J(+1), H, W]
    """
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

    # 每一层 j：从 layout 中取出该层的三块（h/v/d），resize 回 HxW 再相加
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