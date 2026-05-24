import os
import json
import torch
import numpy as np
from PIL import Image

from src.helpers import load_vision_model
from src.evaluators import EvalImageBaselines, Eval2DWAM
import torchvision.transforms as transforms


def load_image(img_path: str) -> Image.Image:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    return Image.open(img_path).convert('RGB')


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess(img: Image.Image, device: str):
    tfm = get_transform()
    xt = tfm(img).unsqueeze(0).to(device)
    return xt


def predict_label(model: torch.nn.Module, x: torch.Tensor) -> int:
    with torch.no_grad():
        logits = model(x)
        return int(torch.argmax(logits, dim=1).item())


def run_baseline(method_name: str, model, x_batch, y_idx: int, device: str, n_iter: int = 128):
    if method_name == 'gradcam':
        layers = model.layer4[-1] if hasattr(model, 'layer4') else None
        evaluator = EvalImageBaselines(method_name='gradcam', model=model, layers=layers, batch_size=8)
    elif method_name == 'integratedgrad':
        evaluator = EvalImageBaselines(method_name='integratedgrad', model=model, batch_size=8)
    elif method_name == 'saliency':
        evaluator = EvalImageBaselines(method_name='saliency', model=model, batch_size=8)
    elif method_name == 'smoothgrad':
        evaluator = EvalImageBaselines(method_name='smoothgrad', model=model, batch_size=8)
    else:
        raise ValueError(f"Unsupported baseline method: {method_name}")

    y = torch.tensor([y_idx], dtype=torch.long, device=device)
    ins_scores = evaluator.insertion(x_batch, y, n_iter=n_iter)
    del_scores = evaluator.deletion(x_batch, y, n_iter=n_iter)
    return float(ins_scores[0]), float(del_scores[0])


def run_wam(method: str, model, x_batch, y_idx: int, device: str, n_iter: int = 128):
    # method in {'smooth', 'integratedgrad'} -> WAM-SG / WAM-IG
    wam = Eval2DWAM(model=model, method=method, J=3, batch_size=16)
    y = torch.tensor([y_idx], dtype=torch.long, device=device)
    ins_scores = wam.insertion(x_batch, y, n_iter=n_iter)
    del_scores = wam.deletion(x_batch, y, n_iter=n_iter)
    return float(ins_scores[0]), float(del_scores[0])


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(repo_root, 'assets', 'elephant.jpg')
    out_dir = os.path.join(repo_root, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_vision_model(model_key='resnet18', device=device)

    img = load_image(img_path)
    x_batch = preprocess(img, device)
    y_idx = predict_label(model, x_batch)

    results = {}

    # Baselines
    for name in ['integratedgrad', 'saliency', 'gradcam', 'smoothgrad']:
        ins, de = run_baseline(name, model, x_batch, y_idx, device)
        results[name] = {'insertion_auc': ins, 'deletion_auc': de}

    # WAM variants
    ins, de = run_wam('integratedgrad', model, x_batch, y_idx, device)
    results['wam_ig'] = {'insertion_auc': ins, 'deletion_auc': de}

    ins, de = run_wam('smooth', model, x_batch, y_idx, device)
    results['wam_sg'] = {'insertion_auc': ins, 'deletion_auc': de}

    # Save and print
    out_json = os.path.join(out_dir, 'baselines_scores.json')
    with open(out_json, 'w') as f:
        json.dump({
            'image': img_path,
            'target_class_index': int(y_idx),
            'scores': results
        }, f, indent=2)

    print(f"Target class index: {y_idx}")
    for k, v in results.items():
        print(f"{k}: insertion_auc={v['insertion_auc']:.4f}, deletion_auc={v['deletion_auc']:.4f}")
    print(f"Saved to: {out_json}")


if __name__ == '__main__':
    main()