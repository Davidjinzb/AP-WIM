from attr_methods_2d import ap_wim as ap_wim_2d
from attr_methods_1d import ap_wim as ap_wim_1d
from datasets import IMAGENETSUBSET1000
import argparse
from torchvision.models import efficientnet_b0
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import device, setup_seed
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os
import torch

setup_seed(42)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="efficient_b0", help="model name")
argparser.add_argument("--attr_method", type=str, default="ap_wim", help="attribution method")
argparser.add_argument("--dataset", type=str, default="imagenet")
argparser.add_argument("--debug", action="store_true")
args = argparser.parse_args()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
sfmx = nn.Softmax(dim=1)

BATCH_SIZE = 16

if __name__ == "__main__":
    if args.dataset == "imagenet":
        dataset = IMAGENETSUBSET1000()
    elif args.dataset == "esc50":
        from datasets import ESC50
        dataset = ESC50(mode="test", num_FOLD=1)
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    if args.model == "efficient_b0" and args.dataset == "imagenet":
        model = efficientnet_b0(pretrained=True)
        model = nn.Sequential(normalize, model, sfmx).eval().to(device)
    elif args.model == "resnet50" and args.dataset == "esc50":
        from models import build_resnet50_esc50
        model = build_resnet50_esc50()
        model.load_state_dict(
            torch.load("runs_esc50/resnet50_fold1_seed42_1768497163/best.pth")["state_dict"]
        )
        model = nn.Sequential(model, nn.Softmax(dim=1)).eval().to(device)
    elif args.model == "mobilenet_v2" and args.dataset == "esc50":
        from models import build_mobilenetv2_esc50
        model = build_mobilenetv2_esc50()
        model.load_state_dict(
            torch.load("runs_esc50/mobilenet_v2_fold1_seed42_1768497160/best.pth")["state_dict"]
        )
        model = nn.Sequential(model, nn.Softmax(dim=1)).eval().to(device)
    else:
        raise NotImplementedError

    out_path = f"attributions/{args.dataset}_{args.model}_{args.attr_method}_attributions.npy"

    if (not os.path.exists(out_path)) or args.debug:
        attr_method = ap_wim_2d if args.dataset == "imagenet" else ap_wim_1d

        attributions = []

        if args.dataset == "imagenet":
            for x, y in tqdm(dataloader):
                x = x.to(device)
                y = y.to(device)
                attribution = attr_method(model, x, y)
                attributions.append(attribution)
        elif args.dataset == "esc50":
            pbar = tqdm(total=len(dataloader))
            for _, (Xlgmel, y, waveform) in enumerate(dataloader):
                y = y.long().to(device)
                attribution = attr_method(
                    model,
                    waveform,  # [B, W]
                    y,
                )
                attributions.append(attribution)
                pbar.update()
            pbar.close()
        else:
            raise NotImplementedError

        attributions = np.concatenate(attributions, axis=0)
        print(attributions.shape, attributions.min(), attributions.max(), attributions.mean())
        np.save(out_path, attributions)
