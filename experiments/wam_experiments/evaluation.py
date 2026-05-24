import numpy as np
import torch
from tqdm import tqdm

def _prep_exp_to_hw(exp_batch: np.ndarray, B: int, H: int, W: int) -> np.ndarray:
    exp = np.asarray(exp_batch)
    if exp.ndim == 4:
        exp = exp.mean(axis=1)      # (B,H,W)
    elif exp.ndim == 3:
        pass
    else:
        raise ValueError(f"exp_batch must be (B,C,H,W) or (B,1,H,W) or (B,H,W), got {exp.shape}")

    if exp.shape != (B, H, W):
        raise ValueError(f"exp_batch shape mismatch, expect (B,H,W)=({B},{H},{W}), got {exp.shape}")

    return exp.reshape(B, H * W).astype(np.float32)

def auc_curve(scores_1d: np.ndarray) -> float:
    scores_1d = np.asarray(scores_1d, dtype=np.float32).reshape(-1)
    if scores_1d.size < 2:
        return float(scores_1d.mean()) if scores_1d.size == 1 else 0.0
    x = np.linspace(0.0, 1.0, scores_1d.size, dtype=np.float32)
    return float(np.trapz(scores_1d, x=x))

class CausalMetric:
    def __init__(self, model, mode: str, step: int, substrate_fn, device=None):
        assert mode in ["del", "ins"]
        self.model = model
        self.mode = mode
        self.step = int(step)
        self.substrate_fn = substrate_fn
        self.device = device if device is not None else next(model.parameters()).device

    @torch.no_grad()
    def evaluate(self, img_batch: torch.Tensor, exp_batch: np.ndarray, batch_size: int):
        """
        img_batch: torch.Tensor (CPU) [B,C,H,W]
        exp_batch: np.ndarray         [B,C,H,W] or [B,1,H,W] or [B,H,W]
        return: scores np.ndarray [B,T]
        """
        if img_batch.ndim != 4:
            raise ValueError(f"img_batch must be [B,C,H,W], got {img_batch.shape}")

        B, C, H, W = img_batch.shape
        HW = H * W

        # --- 1) predict top class once ---
        preds_all = []
        for s in range(0, B, batch_size):
            e = min(B, s + batch_size)
            preds = self.model(img_batch[s:e].to(self.device))
            preds_all.append(preds.detach().cpu())
        preds_all = torch.cat(preds_all, dim=0)            # [B,num_classes] on CPU
        top_cpu = torch.argmax(preds_all, dim=1).long()    # [B] CPU
        top = top_cpu.to(self.device)                      # ✅ move to device once

        # --- 2) saliency order (descending) ---
        exp_flat = _prep_exp_to_hw(exp_batch, B=B, H=H, W=W)         # [B,HW]
        salient_order = np.argsort(exp_flat, axis=1)[:, ::-1].copy() # [B,HW]

        # --- 3) substrate and start/finish on device ---
        img0 = img_batch.to(self.device)
        substrate = torch.empty_like(img0)
        for s in range(0, B, batch_size):
            e = min(B, s + batch_size)
            substrate[s:e] = self.substrate_fn(img0[s:e])

        if self.mode == "del":
            start = img0.clone()
            finish = substrate
            caption = "Deleting  "
        else:
            start = substrate.clone()
            finish = img0
            caption = "Inserting "

        # --- 4) iterate steps ---
        n_steps = (HW + self.step - 1) // self.step
        T = n_steps + 1
        scores = np.empty((B, T), dtype=np.float32)

        for i in range(T):
            # forward scores
            step_preds = []
            for s in range(0, B, batch_size):
                e = min(B, s + batch_size)
                out = self.model(start[s:e])  # on device
                sel = out.gather(1, top[s:e].view(-1, 1)).squeeze(1)  # ✅ top is on device now
                step_preds.append(sel.detach().cpu().numpy())
            scores[:, i] = np.concatenate(step_preds, axis=0)

            if i == n_steps:
                break

            # update pixels
            a = i * self.step
            b = min(HW, (i + 1) * self.step)
            idx = torch.from_numpy(salient_order[:, a:b]).to(self.device, dtype=torch.long)  # [B,k]

            start_flat = start.view(B, C, HW)
            finish_flat = finish.view(B, C, HW)

            values = torch.gather(finish_flat, 2, idx.unsqueeze(1).expand(-1, C, -1))
            start_flat.scatter_(2, idx.unsqueeze(1).expand(-1, C, -1), values)

        print(f"AUC(mean curve): {auc_curve(scores.mean(axis=0)):.6f}")
        return scores
