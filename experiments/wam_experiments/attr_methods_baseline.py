import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel
import random

# =========================================================
# helpers
# =========================================================

def _ensure_target_tensor(y, device):
    if isinstance(y, torch.Tensor):
        return y.to(device=device, dtype=torch.long)
    return torch.tensor(y, device=device, dtype=torch.long)

def _reduce_to_1channel(attr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    attr: same shape as x
    return:
      if x is [B,L]      -> [B,1,L]
      if x is [B,C,L]    -> [B,1,L]  (mean over C)
      if x is [B,C,H,W]  -> [B,1,H,W] (mean over C)
    """
    if x.dim() == 2:  # [B,L]
        return attr.unsqueeze(1)
    if x.dim() == 3:  # [B,C,L]
        return attr.mean(dim=1, keepdim=True)
    if x.dim() == 4:  # [B,C,H,W]
        return attr.mean(dim=1, keepdim=True)
    raise ValueError(f"Unsupported input dim {x.dim()}, expect 2/3/4D.")

def _assert_supported_input(x):
    if x.dim() not in (2, 3, 4):
        raise ValueError(f"x must be [B,L] or [B,C,L] or [B,C,H,W], got {tuple(x.shape)}")


# =========================================================
# 1) Saliency Map (captum)
# =========================================================

def saliencymap(model, x, y):
    """
    Captum Saliency

    x:
      - 1D: [B,L] or [B,C,L]
      - 2D: [B,C,H,W]
    y: [B] long class index
    return: np.float32
      - 1D: [B,1,L]
      - 2D: [B,1,H,W]
    """
    _assert_supported_input(x)
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device).detach().requires_grad_(True)
    y = _ensure_target_tensor(y, device)

    sal = Saliency(model)
    attr = sal.attribute(x, target=y)                 # same shape as x
    attr = attr.abs()
    attr = _reduce_to_1channel(attr, x)               # [B,1,*]
    return attr.detach().cpu().numpy().astype(np.float32)


# =========================================================
# 2) Integrated Gradients (captum)
# =========================================================

def ig(model, x, y, n_steps=25, baselines=None, internal_batch_size=None):
    """
    Captum IntegratedGradients

    x:
      - 1D: [B,L] or [B,C,L]
      - 2D: [B,C,H,W]
    y: [B] long
    baselines:
      - None -> zeros_like(x)
      - otherwise tensor broadcastable to x
    return: np.float32 [B,1,*]
    """
    _assert_supported_input(x)
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device).detach()
    y = _ensure_target_tensor(y, device)

    if baselines is None:
        baselines = torch.zeros_like(x)
    else:
        baselines = baselines.to(device)

    explainer = IntegratedGradients(model)
    attr = explainer.attribute(
        x,
        baselines=baselines,
        target=y,
        n_steps=n_steps,
        method="riemann_trapezoid",
        internal_batch_size=internal_batch_size,
        return_convergence_delta=False,
    )
    attr = attr.abs()
    attr = _reduce_to_1channel(attr, x)
    return attr.detach().cpu().numpy().astype(np.float32)


# =========================================================
# 3) SmoothGrad (captum NoiseTunnel + Saliency)
# =========================================================

def sg(model, x, y, nt_samples=25, stdevs=0.25):
    """
    Captum SmoothGrad implemented by NoiseTunnel(Saliency)

    x:
      - 1D: [B,L] or [B,C,L]
      - 2D: [B,C,H,W]
    y: [B] long
    nt_samples: number of noisy samples
    stdevs: noise std in input scale
    return: np.float32 [B,1,*]
    """
    _assert_supported_input(x)
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device).detach().requires_grad_(True)
    y = _ensure_target_tensor(y, device)

    base = Saliency(model)
    nt = NoiseTunnel(base)

    attr = nt.attribute(
        x,
        target=y,
        nt_type="smoothgrad",
        nt_samples=nt_samples,
        stdevs=stdevs,
    )
    attr = attr.abs()
    attr = _reduce_to_1channel(attr, x)
    return attr.detach().cpu().numpy().astype(np.float32)


# =========================================================
# 4) FastIG
# =========================================================

class FastIG:
    """
    FastIG (single-step)
    Original: data * grad(data)
    Your model outputs already-softmax probabilities -> use NLL: -log(p_y)
    Works for 1D and 2D inputs.

    __call__(x,y) returns np.float32 [B,1,*]
    """
    def __init__(self, model, eps=1e-8):
        self.model = model
        self.eps = eps

    def __call__(self, x, y):
        _assert_supported_input(x)
        self.model.eval()
        device = next(self.model.parameters()).device

        x = x.to(device).detach().requires_grad_(True)
        y = _ensure_target_tensor(y, device)

        out = self.model(x)  # expected probs [B,num_classes] because you fused softmax
        # NLL on prob: -log(p_y)
        p = out.gather(1, y.view(-1, 1)).clamp_min(self.eps)  # [B,1]
        loss = (-torch.log(p)).mean()

        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        attr = (x * grad)  # keep same spirit as your original fast_ig
        attr = attr.abs()
        attr = _reduce_to_1channel(attr, x)
        return attr.detach().cpu().numpy().astype(np.float32)


def fast_ig(model, x, y, eps=1e-8):
    """
    Convenience function wrapper.
    """
    return FastIG(model, eps=eps)(x, y)


import torch
import numpy as np

class FGSMGradProbSign:
    def __init__(self, data_min=0.0, data_max=1.0):
        self.data_min = data_min
        self.data_max = data_max

    @torch.no_grad()
    def _project(self, x_adv, x0):
        if self.data_min is None and self.data_max is None:
            return x_adv
        delta = x_adv - x0
        return (x0 + delta).clamp(self.data_min, self.data_max)

    def __call__(self, model, data, target, num_steps=50):
        """
        data:   torch.Tensor [B,*]
        target: torch.LongTensor [B]
        returns:
          adv:     [B,*]
          success: [B] bool  (adv_pred != target)
          hats:    list of [B,*] (len = steps+1)
          grads:   list of [B,*] (len = steps)
        """
        device = next(model.parameters()).device
        x0 = data.to(device).detach()
        y = target.to(device).long()
        alpha = data / num_steps
        dt = x0.clone().detach().requires_grad_(True)

        hats = [dt.detach().clone()]
        grads = []

        for _ in range(num_steps):
            probs = model(dt)  # [B,K] probabilities
            score = probs.gather(1, y.view(-1, 1)).squeeze(1)  # [B]
            score.sum().backward()
            g = dt.grad.detach()
            grads.append(g.clone())
            dt_next = dt + alpha * g.sign()
            dt = self._project(dt_next.detach(), x0).requires_grad_(True)
            hats.append(dt.detach().clone())
            if dt.grad is not None:
                dt.grad.zero_()
                
        with torch.no_grad():
            success = model(dt).argmax(dim=1) != y

        return dt.detach(), success.detach(), hats, grads


class MFABA:
    def __call__(self, hats, grads):
        hats = torch.stack(hats, dim=0)
        grads = torch.stack(grads, dim=0)
        t_list = hats[1:] - hats[:-1]
        attribution = torch.sum(t_list * grads, dim=0)  # [B,*]
        return attribution.detach().cpu()

def mfaba(model,x, y, num_steps=10, data_min=0.0, data_max=1.0):
    fgsm = FGSMGradProbSign(data_min=data_min, data_max=data_max)
    adv, success, hats, grads = fgsm(model, x, y, num_steps=num_steps)
    mfaba_explainer = MFABA()
    attr = mfaba_explainer(hats, grads)
    attr = _reduce_to_1channel(attr, x)
    return attr.detach().cpu().numpy().astype(np.float32)


def clamp(x, min_value, max_value):
    if min_value is not None and max_value is not None:
        return x.clamp(min_value, max_value)
    else:
        return x

def la(model, data, target, epsilon=10, max_iter=30, seed=42,
       clip_min=0.0, clip_max=1.0):
    """
    输入:
      data: torch.Tensor [B,C,H,W] (建议按你原来就用 4D；也能兼容 2/3D，但你主要是图像)
      target: torch.Tensor/array/list [B]  (这里只用于接口一致，不参与 init_pred 逻辑)
    输出:
      np.ndarray [B,1,H,W] (或 [B,1,*] 对应输入维度)
    """
    device = next(model.parameters()).device
    model.eval()

    # 固定随机
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = data.to(device).detach()
    B = data.shape[0]

    with torch.no_grad():
        out = model(data)                 # 已是 prob
        init_pred = out.argmax(dim=-1)    # [B]
        num_classes = out.shape[-1]

    all_cls = np.arange(num_classes)
    # if select_num >= num_classes:
    #     selected = all_cls
    # else:
    #     selected = np.random.choice(all_cls, select_num, replace=False)
    select_num = max(2, num_classes // 50)  # 至少选 2 个类别
    selected = np.random.choice(all_cls, select_num, replace=False)

    step = (data.clone() / float(epsilon)).detach()

    attribution_result = torch.zeros_like(data)

    # ========== targeted 部分（对应你 selected[:-1]） ==========
    for l in selected[:-1]:
        targeted = torch.full((B,), int(l), device=device, dtype=torch.long)

        if int(l) < num_classes - 1:
            targeted = torch.where(targeted == init_pred, targeted + 1, targeted)
        else:
            targeted = torch.where(targeted == init_pred, targeted - 1, targeted)

        ori_image = data.clone()
        adv_image = data.clone()
        result = torch.zeros_like(data)

        for _ in range(max_iter):
            adv_image = adv_image.detach().requires_grad_(True)

            prob = model(adv_image)  # 已是 softmax prob

            model.zero_grad(set_to_none=True)
            if adv_image.grad is not None:
                adv_image.grad.zero_()
            loss_target = prob.gather(1, targeted.view(-1, 1)).sum()
            loss_target.backward(retain_graph=True)
            target_grad = adv_image.grad.detach().clone()

            model.zero_grad(set_to_none=True)
            adv_image.grad.zero_()
            loss_untarget = prob.gather(1, init_pred.view(-1, 1)).sum()
            loss_untarget.backward()
            untarget_grad = adv_image.grad.detach().clone()

            # adv_image = (ori_image + step * target_grad.sign()).clamp(clip_min, clip_max).detach()
            adv_image = clamp(ori_image + step * target_grad.sign(), clip_min, clip_max).detach()

            result = result + (-untarget_grad * (adv_image - ori_image))

        attribution_result = attribution_result + result

    ori_image = data.clone()
    adv_image = data.clone()
    result = torch.zeros_like(data)

    for _ in range(max_iter):
        adv_image = adv_image.detach().requires_grad_(True)
        prob = model(adv_image)

        model.zero_grad(set_to_none=True)
        if adv_image.grad is not None:
            adv_image.grad.zero_()
        loss_untarget = prob.gather(1, init_pred.view(-1, 1)).sum()
        loss_untarget.backward()
        untarget_grad = adv_image.grad.detach().clone()

        # adv_image = (ori_image - step * untarget_grad.sign()).clamp(clip_min, clip_max).detach()
        adv_image = clamp(ori_image - step * untarget_grad.sign(), clip_min, clip_max).detach()

        result = result + (-untarget_grad * (adv_image - ori_image))

    attribution_result = attribution_result + result

    attribution_result = _reduce_to_1channel(attribution_result, data)

    return attribution_result.detach().cpu().numpy().astype(np.float32)


def fgsm_step_prob(image, epsilon, data_grad_adv, data_grad_lab, clip_min=0.0, clip_max=1.0):
    """
    你的原版：
      delta = epsilon * sign(grad_adv)
      perturbed = clamp(image + delta)
      delta = perturbed - image
      return perturbed, (- data_grad_lab * delta)
    """
    delta = epsilon * data_grad_adv.sign()
    perturbed = clamp(image + delta, clip_min, clip_max)
    delta = perturbed - image
    delta = -data_grad_lab * delta
    return perturbed, delta


def pgd_step_prob(image, epsilon, model, init_pred, targeted, max_iter,
                  clip_min=0.0, clip_max=1.0):
    """
    你的原 pgd_step，但把 softmax 去掉（模型已输出 prob）
    loss 计算按你原来：
      loss = prob[:, targeted].sum()
      loss_lab = prob[:, init_pred].sum()
    并保留 leave_index 的累积方式。
    """
    perturbed = image.clone()
    leave_index = list(range(image.shape[0]))
    c_delta = None

    for i in range(max_iter):
        perturbed = perturbed.detach().requires_grad_(True)

        prob = model(perturbed)              # [B,K] 已是 prob
        pred = prob.argmax(dim=-1)           # [B]

        # 成功就移除（你的原逻辑）
        for j in leave_index.copy():
            if pred[j] == targeted[j]:
                leave_index.remove(j)
        if len(leave_index) == 0:
            break

        # targeted prob 梯度
        model.zero_grad(set_to_none=True)
        if perturbed.grad is not None:
            perturbed.grad.zero_()
        loss = prob.gather(1, targeted.view(-1, 1)).sum()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed.grad.detach().clone()

        # init prob 梯度
        model.zero_grad(set_to_none=True)
        perturbed.grad.zero_()
        loss_lab = prob.gather(1, init_pred.view(-1, 1)).sum()
        loss_lab.backward()
        data_grad_lab = perturbed.grad.detach().clone()

        # FGSM step（按你原来的 “+delta because we are ascending”）
        perturbed, delta = fgsm_step_prob(
            image, epsilon, data_grad_adv, data_grad_lab,
            clip_min=clip_min, clip_max=clip_max
        )

        # 累积 delta（只对仍未成功的样本累积）
        if c_delta is None:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    if c_delta is None:
        c_delta = torch.zeros_like(image)

    return c_delta, perturbed.detach()


def agi(model, data, target, epsilon=0.05, max_iter=10, seed=42,
        clip_min=0.0, clip_max=1.0):
    """
    按你原版 agi 改成你这套 baseline 风格：
    - 支持 2/3/4D 输入，但你的原始 AGI 是 4D，这里仍然要求 4D（保持原意）
    - 模型输出已 softmax prob：不再额外 softmax
    - loss 完全按你原来（target prob / init prob）
    - 不随便加 abs，不改梯度，不改 reduce_to_1channel（保留）
    返回: np.float32 [B,1,H,W]
    """
    _assert_supported_input(data)

    model.eval()
    device = next(model.parameters()).device

    x = data.to(device).detach()
    y = _ensure_target_tensor(target, device)  # 这里仅用于接口一致，init_pred 用模型算
    B = x.shape[0]

    # 固定随机，与你一致
    random.seed(seed)
    with torch.no_grad():
        prob = model(x)                 # [B,K] 已是 prob
        init_pred = prob.argmax(dim=-1) # [B]
        num_classes = prob.shape[-1]

    # 你的原版：random.sample(range(0, 999), topk)
    # 这里改成用 num_classes 自适应，但不改变“随机挑 topk 类”的含义
    # max_cls = min(num_classes, 1000)  # 兼容你原来写死 999/1000 的习惯
    # top_ids = random.sample(list(range(0, max_cls - 1)), k=min(topk, max_cls - 1))
    topk = max(2, num_classes // 50)  # 至少选 2 个类别
    top_ids = random.sample(list(range(0, num_classes - 1)), k=min(topk, num_classes - 1))

    step_grad = 0.0
    for l in top_ids:
        targeted = torch.full((B,), int(l), device=device, dtype=torch.long)
        # 你的原逻辑：避免 targeted == init_pred
        if int(l) < (num_classes - 1):
            targeted = torch.where(targeted == init_pred, targeted + 1, targeted)
        else:
            targeted = torch.where(targeted == init_pred, targeted - 1, targeted)

        delta, _ = pgd_step_prob(
            x, epsilon, model,
            init_pred=init_pred,
            targeted=targeted,
            max_iter=max_iter,
            clip_min=clip_min, clip_max=clip_max
        )
        step_grad = step_grad + delta

    attr = step_grad  # 不做 abs，不除 topk（你原来注释了 / topk）
    attr = _reduce_to_1channel(attr, x)  # 保留你的 reduce 方式

    return attr.detach().cpu().numpy().astype(np.float32)