from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from utils import device, setup_seed


SEED = 42
IMAGE_DATASETS = {"imagenet", "cub"}
AUDIO_DATASETS = {"esc50", "fma", "audiomnist"}
AP_WIM_ALIASES = {"ap_wim", "wam_mfaba"}
WAVELET_METHODS = {"wam_ig", "wam_sg", "wam_mfaba", "ap_wim"}


class NormalizeLayer(nn.Module):
    def __init__(self, mean: Iterable[float], std: Iterable[float]):
        super().__init__()
        mean_t = torch.tensor(list(mean), dtype=torch.float32).view(1, -1, 1, 1)
        std_t = torch.tensor(list(std), dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def dispatch_attr_method(method: str) -> str:
    return "wam_mfaba" if method == "ap_wim" else method


def method_aliases(method: str) -> list[str]:
    if method in AP_WIM_ALIASES:
        return [method, "ap_wim" if method == "wam_mfaba" else "wam_mfaba"]
    return [method]


def is_wavelet_method(method: str) -> bool:
    return dispatch_attr_method(method) in WAVELET_METHODS


def default_ap_wim_steps(dataset_name: str) -> int:
    return 7 if dataset_name in IMAGE_DATASETS else 3


def clean_tag(tag: str | None) -> str:
    if not tag:
        return ""
    tag = tag.strip()
    if not tag:
        return ""
    tag = re.sub(r"[^A-Za-z0-9_.=-]+", "-", tag)
    return tag.strip("-_")


def wavelet_tag(wavelet: str, J: int, approx_coeffs: bool) -> str:
    return clean_tag(f"wavelet-{wavelet}_J-{J}_approx-{int(bool(approx_coeffs))}")


def maybe_limit_dataset(dataset, max_samples: int | None):
    if max_samples is None or max_samples <= 0:
        return dataset
    return Subset(dataset, range(min(int(max_samples), len(dataset))))


def build_dataset(dataset_name: str, max_samples: int | None = None):
    if dataset_name == "imagenet":
        from datasets import IMAGENETSUBSET1000

        dataset = IMAGENETSUBSET1000()
    elif dataset_name == "esc50":
        from datasets import ESC50

        root = os.environ.get("APWIM_ESC50_ROOT")
        dataset = ESC50(mode="test", num_FOLD=1, root_dir=root) if root else ESC50(mode="test", num_FOLD=1)
    elif dataset_name == "fma":
        from datasets import FMA_SMALL

        root = os.environ.get("APWIM_FMA_ROOT", "data_fma")
        dataset = FMA_SMALL(mode="test", root_dir=root)
    elif dataset_name == "audiomnist":
        from datasets import AudioMNIST

        root = os.environ.get("APWIM_AUDIOMNIST_ROOT", "data/audiomnist")
        dataset = AudioMNIST(mode="test", root_dir=root)
    elif dataset_name == "cub":
        from datasets import CUBDATASET

        root = os.environ.get("APWIM_CUB_TEST_ROOT", "data/CUB/test")
        dataset = CUBDATASET(root=root)
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")
    return maybe_limit_dataset(dataset, max_samples)


def _load_state_dict(path: str | Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def build_model(dataset_name: str, model_name: str, *, softmax: bool = True) -> nn.Module:
    setup_seed(SEED)

    if dataset_name == "imagenet" and model_name == "efficient_b0":
        from torchvision.models import efficientnet_b0

        try:
            from torchvision.models import EfficientNet_B0_Weights

            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            backbone = efficientnet_b0(pretrained=True)
        modules: list[nn.Module] = [
            NormalizeLayer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            backbone,
        ]
        if softmax:
            modules.append(nn.Softmax(dim=1))
        return nn.Sequential(*modules).eval().to(device)

    if dataset_name == "esc50" and model_name == "resnet18":
        from models import build_resnet18_esc50

        model = build_resnet18_esc50()
        ckpt = os.environ.get("APWIM_RESNET18_ESC50_CKPT", "resnet18_esc50_vanilla.pth")
        model.load_state_dict(_load_state_dict(ckpt))
    elif dataset_name == "esc50" and model_name == "resnet50":
        from models import build_resnet50_esc50

        model = build_resnet50_esc50()
        ckpt = os.environ.get(
            "APWIM_RESNET50_ESC50_CKPT",
            "runs_esc50/resnet50_fold1_seed42_1768497163/best.pth",
        )
        model.load_state_dict(_load_state_dict(ckpt))
    elif dataset_name == "esc50" and model_name == "mobilenet_v2":
        from models import build_mobilenetv2_esc50

        model = build_mobilenetv2_esc50()
        ckpt = os.environ.get(
            "APWIM_MOBILENETV2_ESC50_CKPT",
            "runs_esc50/mobilenet_v2_fold1_seed42_1768497160/best.pth",
        )
        model.load_state_dict(_load_state_dict(ckpt))
    elif dataset_name == "fma" and model_name == "resnet18":
        from models import build_resnet18_fma

        model = build_resnet18_fma()
        ckpt = os.environ.get("APWIM_RESNET18_FMA_CKPT", "resnet18_fma_vanilla.pth")
        model.load_state_dict(_load_state_dict(ckpt))
    elif dataset_name == "audiomnist" and model_name == "resnet18":
        from models import build_resnet18_audiomnist

        model = build_resnet18_audiomnist()
        ckpt = os.environ.get("APWIM_RESNET18_AUDIOMNIST_CKPT", "resnet18_audiomnist_vanilla.pth")
        model.load_state_dict(_load_state_dict(ckpt))
    elif dataset_name == "cub" and model_name == "resnet50":
        from models import build_resnet50_cub

        model = build_resnet50_cub()
        ckpt = os.environ.get("APWIM_RESNET50_CUB_CKPT", "resnet50_cub_vanilla.pth")
        model.load_state_dict(_load_state_dict(ckpt))
    else:
        raise NotImplementedError(f"Unsupported model/dataset pair: {model_name}/{dataset_name}")

    if softmax:
        return nn.Sequential(model, nn.Softmax(dim=1)).eval().to(device)
    return model.eval().to(device)


def get_attr_dict(dataset_name: str):
    from attr_methods_baseline import agi, fast_ig, ig, la, mfaba, saliencymap, sg
    from attr_methods_1d import wam_ig as wam_ig_1d
    from attr_methods_1d import wam_mfaba as wam_mfaba_1d
    from attr_methods_1d import wam_sg as wam_sg_1d
    from attr_methods_2d import wam_ig as wam_ig_2d
    from attr_methods_2d import wam_mfaba as wam_mfaba_2d
    from attr_methods_2d import wam_sg as wam_sg_2d

    wam_ig = wam_ig_2d if dataset_name in IMAGE_DATASETS else wam_ig_1d
    wam_sg = wam_sg_2d if dataset_name in IMAGE_DATASETS else wam_sg_1d
    wam_mfaba = wam_mfaba_2d if dataset_name in IMAGE_DATASETS else wam_mfaba_1d
    return {
        "ig": ig,
        "fast_ig": fast_ig,
        "sg": sg,
        "saliencymap": saliencymap,
        "mfaba": mfaba,
        "la": la,
        "agi": agi,
        "wam_ig": wam_ig,
        "wam_sg": wam_sg,
        "wam_mfaba": wam_mfaba,
        "ap_wim": wam_mfaba,
    }


def call_attribution(
    method_name: str,
    method_fn,
    model: nn.Module,
    dataset_name: str,
    batch,
    *,
    wavelet: str = "haar",
    J: int = 3,
    approx_coeffs: bool = False,
    steps: int | None = None,
):
    dispatch_name = dispatch_attr_method(method_name)

    if dataset_name in IMAGE_DATASETS:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        if dispatch_name in {"wam_ig", "wam_sg", "wam_mfaba"}:
            kwargs = {"wavelet": wavelet, "J": J, "approx_coeffs": approx_coeffs}
            if dispatch_name == "wam_mfaba":
                kwargs["steps"] = default_ap_wim_steps(dataset_name) if steps is None else steps
            elif dispatch_name == "wam_ig" and steps is not None:
                kwargs["n_steps"] = steps
            return method_fn(model, x, y, **kwargs)
        return method_fn(model, x, y)

    Xlgmel, y, waveform = batch[:3]
    Xlgmel = Xlgmel.to(device)
    y = y.long().to(device)
    if dispatch_name in {"wam_ig", "wam_sg", "wam_mfaba"}:
        kwargs = {"wavelet": wavelet, "J": J}
        if dispatch_name == "wam_mfaba":
            kwargs["steps"] = default_ap_wim_steps(dataset_name) if steps is None else steps
        elif dispatch_name == "wam_ig" and steps is not None:
            kwargs["n_steps"] = steps
        return method_fn(model, waveform, y, **kwargs)
    if dispatch_name == "mfaba":
        return method_fn(model, Xlgmel, y, data_min=None, data_max=None)
    if dispatch_name in {"la", "agi"}:
        return method_fn(model, Xlgmel, y, clip_min=None, clip_max=None)
    return method_fn(model, Xlgmel, y)


def _parts(*values: str | int | None) -> str:
    cleaned = [clean_tag(str(v)) for v in values if v is not None and str(v) != ""]
    return ("_" + "_".join(cleaned)) if cleaned else ""


def attribution_path(
    attribution_dir: str | Path,
    dataset_name: str,
    model_name: str,
    attr_method: str,
    *,
    steps: int | None = None,
    output_tag: str | None = None,
) -> Path:
    stem = f"{dataset_name}_{model_name}_{attr_method}_attributions"
    stem += _parts(output_tag, steps)
    return Path(attribution_dir) / f"{stem}.npy"


def causal_score_path(
    score_dir: str | Path,
    dataset_name: str,
    model_name: str,
    attr_method: str,
    *,
    steps: int | None = None,
    output_tag: str | None = None,
    substrate: str = "zeros",
) -> Path:
    extra = [output_tag]
    if substrate != "zeros":
        extra.append(f"substrate-{substrate}")
    extra.append(steps)
    stem = f"{dataset_name}_{model_name}_{attr_method}_causal_scores"
    stem += _parts(*extra)
    return Path(score_dir) / f"{stem}.npz"


def road_score_path(
    score_dir: str | Path,
    dataset_name: str,
    model_name: str,
    attr_method: str,
    *,
    steps: int | None = None,
    output_tag: str | None = None,
    baseline: str = "blur",
    target: str = "pred",
    legacy: bool = False,
) -> Path:
    stem = f"{dataset_name}_{model_name}_{attr_method}_road_scores"
    if not legacy:
        extra: list[str | int | None] = [output_tag, f"baseline-{baseline}"]
        if target != "pred":
            extra.append(f"target-{target}")
        extra.append(steps)
        stem += _parts(*extra)
    elif steps is not None:
        stem += f"_{steps}"
    return Path(score_dir) / f"{stem}.npz"


def infid_score_path(
    score_dir: str | Path,
    dataset_name: str,
    model_name: str,
    attr_method: str,
    *,
    output_tag: str | None = None,
) -> Path:
    stem = f"{dataset_name}_{model_name}_{attr_method}_infid_scores"
    stem += _parts(output_tag)
    return Path(score_dir) / f"{stem}.npy"


def resolve_existing_attribution(
    attribution_dir: str | Path,
    dataset_name: str,
    model_name: str,
    attr_method: str,
    *,
    steps: int | None = None,
    output_tag: str | None = None,
) -> Path:
    tried = []
    for method in method_aliases(attr_method):
        path = attribution_path(
            attribution_dir,
            dataset_name,
            model_name,
            method,
            steps=steps,
            output_tag=output_tag,
        )
        tried.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError("Attribution file not found. Tried: " + ", ".join(tried))


def save_npy_atomic(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def save_npz_atomic(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
