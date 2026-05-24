import os
import json
import csv
import random
import traceback
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# 你的工程内接口
from src.helpers import load_vision_model
from src.evaluators import EvalImageBaselines, Eval2DWAM

# -----------------------
# Config
# -----------------------
TINY_ROOT = "data/tiny-imagenet-200"  # <<--- 修改成你的实际路径
VAL_IMG_DIR = os.path.join(TINY_ROOT, "val", "images")

N_SAMPLES = 1000                # 抽取多少张做平均
N_ITER = 128                  # insertion/deletion 步数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "./outputs_tiny"
os.makedirs(OUT_DIR, exist_ok=True)

# 模型键：'resnet18' 或 'efficientnet_b0'
MODEL_KEY = "resnet18"

# SmoothGrad / WAM 参数
SMOOTH_N_SAMPLES = 25
SMOOTH_STDEV_SPREAD = 0.25
WAVELET_J = 3
WAVELET_NAME = "haar"

# 复现实验
random.seed(42)

# -----------------------
# Preprocess
# -----------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_image(img_path: str) -> Image.Image:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    return Image.open(img_path).convert('RGB')

def preprocess(img: Image.Image, device: str):
    return get_transform()(img).unsqueeze(0).to(device)

def predict_label(model: torch.nn.Module, x: torch.Tensor) -> tuple:
    """返回预测标签和置信度"""
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        confidence = float(probs[0, pred_idx].item())
        return pred_idx, confidence

# -----------------------
# Helpers
# -----------------------
def get_last_conv_layer(model: torch.nn.Module) -> nn.Module:
    """自动抓取最后一个 Conv2d，供 Grad-CAM 使用。"""
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv

def _nan_safe(v):
    return float(v) if np.isfinite(v) else float('nan')

def _retry_if_nan(fn, *args, **kwargs):
    """
    先按原参数计算；若出现 NaN，再把 n_iter 减半重算一次（最多一次重试）。
    """
    try:
        ins, de, fa = fn(*args, **kwargs)
        print(f"    Method result: ins={ins:.4f}, del={de:.4f}, faith={fa}")
    except Exception as e:
        print(f"    Method failed with error: {e}")
        return float('nan'), float('nan'), None

    if not (np.isfinite(ins) and np.isfinite(de)):
        print(f"    Got NaN values, retrying with reduced n_iter...")
        n_iter = kwargs.get("n_iter", 128)
        if n_iter > 32:
            try:
                ins2, de2, fa2 = fn(*args, **{**kwargs, "n_iter": n_iter // 2})
                print(f"    Retry result: ins={ins2:.4f}, del={de2:.4f}, faith={fa2}")
                if np.isfinite(ins2): ins = ins2
                if np.isfinite(de2):  de  = de2
                if (fa is None) and (fa2 is not None): fa = fa2
            except Exception as e:
                print(f"    Retry also failed: {e}")
    return _nan_safe(ins), _nan_safe(de), (None if fa is None else _nan_safe(fa))

# -----------------------
# Evaluators wrapper
# -----------------------
def run_baseline(method_name: str, model, x_batch, y_idx: int, device: str, n_iter: int = 128):
    # 确保输入带梯度
    x_batch = x_batch.clone().detach().to(device)
    x_batch.requires_grad_(True)
    y = torch.tensor([y_idx], dtype=torch.long, device=device)

    if method_name == 'gradcam':
        # 对 ResNet 有 layer4，其他模型可用 get_last_conv_layer(model)
        layers = model.layer4[-1] if hasattr(model, 'layer4') else get_last_conv_layer(model)
        evaluator = EvalImageBaselines(method_name='gradcam', model=model, layers=layers, batch_size=8)
    elif method_name == 'integratedgrad':
        evaluator = EvalImageBaselines(method_name='integratedgrad', model=model, batch_size=8)
    elif method_name == 'saliency':
        evaluator = EvalImageBaselines(method_name='saliency', model=model, batch_size=8)
    elif method_name == 'smoothgrad':
        evaluator = EvalImageBaselines(method_name='smoothgrad', model=model, batch_size=8)
    else:
        raise ValueError(f"Unsupported baseline method: {method_name}")

    ins_scores = evaluator.insertion(x_batch, y, n_iter=n_iter)
    del_scores = evaluator.deletion(x_batch, y, n_iter=n_iter)
    
    # 计算faithfulness = insertion - deletion
    ins_auc = float(ins_scores[0])
    del_auc = float(del_scores[0])
    faith_score = ins_auc - del_auc
    
    return ins_auc, del_auc, faith_score

def run_wam(method: str, model, x_batch, y_idx: int, device: str, n_iter: int = 128):
    # method in {'smooth', 'integratedgrad'} -> WAM-SG / WAM-IG
    wam = Eval2DWAM(model=model, method=method, J=WAVELET_J, batch_size=16)
    y = torch.tensor([y_idx], dtype=torch.long, device=device)
    ins_scores = wam.insertion(x_batch, y, n_iter=n_iter)
    del_scores = wam.deletion(x_batch, y, n_iter=n_iter)
    
    # 计算faithfulness = insertion - deletion
    ins_auc = float(ins_scores[0])
    del_auc = float(del_scores[0])
    faith_score = ins_auc - del_auc
    
    # print(ins_auc, del_auc, faith_score)
    # raise Exception
    
    return ins_auc, del_auc, faith_score

# -----------------------
# Data
# -----------------------
def sample_images_from_val(n: int) -> List[str]:
    all_imgs = [os.path.join(VAL_IMG_DIR, f) for f in os.listdir(VAL_IMG_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(all_imgs) < n:
        raise RuntimeError(f"VAL images not enough: found {len(all_imgs)} < {n}")
    random.shuffle(all_imgs)
    return all_imgs[:n]

# -----------------------
# Main
# -----------------------
def main():
    # 1) 加载 ImageNet 预训练模型
    model = load_vision_model(model_key=MODEL_KEY, device=DEVICE)
    model.eval()

    # 2) 随机抽 5 张 val 图
    img_paths = sample_images_from_val(N_SAMPLES)

    # baseline_methods = ['integratedgrad', 'saliency', 'gradcam', 'smoothgrad']  # baselines
    # wam_methods = [('integratedgrad', 'wam_ig'), ('smooth', 'wam_sg')]
    wam_methods = [('integratedgrad', 'wam_ig'), ('mfaba', 'wam_mfaba')]

    per_image_results: List[Dict[str, Any]] = []
    # agg = {m: {'insertion': [], 'deletion': [], 'faithfulness': []} for m in baseline_methods + [wm[1] for wm in wam_methods]}
    agg = {m: {'insertion': [], 'deletion': [], 'faithfulness': []} for m in ['wam_ig', 'wam_mfaba']}
    # agg = {"integratedgrad": {'insertion': [], 'deletion': [], 'faithfulness': []},
    #        "mfaba": {'insertion': [], 'deletion': [], 'faithfulness': []}}
    # print(agg)
    # raise Exception

    for i, img_path in enumerate(img_paths, 1):
        rec = {'image': img_path, 'target_class_index': None, 'scores': {}}
        # try:
        img = load_image(img_path)
        x = preprocess(img, DEVICE)

        # 清除任何现有的 hooks（防止多次注册造成冲突）
        model.eval()
        for module in model.modules():
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()
            if hasattr(module, '_backward_hooks'):
                module._backward_hooks.clear()

        y_idx, confidence = predict_label(model, x)
        rec['target_class_index'] = int(y_idx)
        rec['prediction_confidence'] = confidence

        print(f"[{i}/{N_SAMPLES}] Processing: {os.path.basename(img_path)}")
        print(f"  Predicted class: {y_idx}, Confidence: {confidence:.4f}")

        # Baseline 方法
        # for name in baseline_methods:
        #     print(f"  Running method: {name}")
        #     ins, de, faith = run_baseline(name, model, x, y_idx, DEVICE, n_iter=N_ITER)
        #     rec['scores'][name] = {'insertion_auc': ins, 'deletion_auc': de, 'faithfulness': faith}
        #     agg[name]['insertion'].append(ins)
        #     agg[name]['deletion'].append(de)
        #     agg[name]['faithfulness'].append(faith)

        # WAM 变体
        for method, alias in wam_methods:
            print(f"  Running WAM method: {alias}")
            ins, de, faith = run_wam(method, model, x, y_idx, DEVICE, n_iter=N_ITER)
            rec['scores'][alias] = {'insertion_auc': ins, 'deletion_auc': de, 'faithfulness': faith}
            agg[alias]['insertion'].append(ins)
            agg[alias]['deletion'].append(de)
            agg[alias]['faithfulness'].append(faith)

        # except Exception as e:
        #     rec['error'] = f"{type(e).__name__}: {e}"
        #     rec['traceback'] = traceback.format_exc()

        per_image_results.append(rec)
        print(f"[{i}/{N_SAMPLES}] Done: {os.path.basename(img_path)}")

    def _mean_or_nan(vals): 
        return float(np.mean(vals)) if len(vals) > 0 else float('nan')

    mean_results = {}
    for name in agg.keys():
        mean_results[name] = {
            'insertion_auc_mean': _mean_or_nan(agg[name]['insertion']),
            'deletion_auc_mean': _mean_or_nan(agg[name]['deletion']),
            'faithfulness_mean': _mean_or_nan(agg[name]['faithfulness'])
        }

    print("\n=== Tiny ImageNet 5-image Averages ===")
    for k, v in mean_results.items():
        print(f"{k:12s} | ins={v['insertion_auc_mean']:.4f} | del={v['deletion_auc_mean']:.4f} | faith={v['faithfulness_mean']:.4f}")

    out_json = os.path.join(OUT_DIR, "tiny5_scores.json")
    with open(out_json, "w") as f:
        json.dump({'images': img_paths, 'per_image': per_image_results, 'mean_over_5': mean_results}, f, indent=2)
    print(f"Saved JSON: {out_json}")

    out_csv = os.path.join(OUT_DIR, "tiny5_scores_mean.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "insertion_auc_mean", "deletion_auc_mean", "faithfulness_mean"])
        for k, v in mean_results.items():
            w.writerow([k, v["insertion_auc_mean"], v["deletion_auc_mean"], v["faithfulness_mean"]])
    print(f"Saved CSV:  {out_csv}")

if __name__ == "__main__":
    main()
