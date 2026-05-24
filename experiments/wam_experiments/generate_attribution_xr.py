from attr_methods_baseline import ig,fast_ig,sg,saliencymap,mfaba,la,agi
from attr_methods_2d import wam_ig as wam_ig_2d,wam_sg as wam_sg_2d,wam_mfaba as wam_mfaba_2d
from attr_methods_1d import wam_ig as wam_ig_1d,wam_sg as wam_sg_1d,wam_mfaba as wam_mfaba_1d
from datasets import IMAGENETSUBSET1000
import argparse
from torchvision.models import efficientnet_b0
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import device,setup_seed
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os
import torch
setup_seed(42)

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='efficient_b0', help='model name')
argparser.add_argument('--attr_method', type=str, default='wam_mfaba', help='attribution method')
argparser.add_argument('--dataset', type=str, default='imagenet')
argparser.add_argument("--debug",action="store_true")
argparser.add_argument('--steps', type=int, default=5)
args = argparser.parse_args()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
sfmx = nn.Softmax(dim=1)

BATCH_SIZE = 16

if __name__ == '__main__':
    if args.dataset == 'imagenet':
        dataset = IMAGENETSUBSET1000()
    elif args.dataset == 'esc50':
        from datasets import ESC50
        dataset = ESC50(mode='test', num_FOLD=1)
    elif args.dataset == 'fma':
        from datasets import FMA_SMALL
        dataset = FMA_SMALL(mode='test')
    elif args.dataset == "audiomnist":
        from datasets import AudioMNIST
        dataset = AudioMNIST(mode='test')
    elif args.dataset == "cub":
        from datasets import CUBDATASET
        dataset = CUBDATASET(root="data/CUB/test")
    else:
        raise NotImplementedError
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    if args.model == 'efficient_b0' and args.dataset == 'imagenet':
        model = efficientnet_b0(pretrained=True)
        model = nn.Sequential(normalize, model, sfmx).eval().to(device)
    elif args.model == 'resnet18' and args.dataset == "esc50":
        from models import build_resnet18_esc50
        model = build_resnet18_esc50()
        model.load_state_dict(torch.load("resnet18_esc50_vanilla.pth"))
        model = nn.Sequential(
            model,
            nn.Softmax(dim=1)
        ).eval().to(device)
    elif args.model == 'resnet18' and args.dataset == "fma":
        from models import build_resnet18_fma
        model = build_resnet18_fma()
        model.load_state_dict(torch.load("resnet18_fma_vanilla.pth"))
        model = nn.Sequential(
            model,
            nn.Softmax(dim=1)
        ).eval().to(device)
    elif args.model == "resnet18" and args.dataset == "audiomnist":
        from models import build_resnet18_audiomnist
        model = build_resnet18_audiomnist()
        model.load_state_dict(torch.load("resnet18_audiomnist_vanilla.pth"))
        model = nn.Sequential(
            model,
            nn.Softmax(dim=1)
        ).eval().to(device)
    elif args.model == "resnet50" and args.dataset == "cub":
        from models import build_resnet50_cub
        model = build_resnet50_cub()
        model.load_state_dict(torch.load("resnet50_cub_vanilla.pth"))
        model = nn.Sequential(
            model,
            nn.Softmax(dim=1)
        ).eval().to(device)
    elif args.model == "resnet50" and args.dataset == "esc50":
        from models import build_resnet50_esc50
        model = build_resnet50_esc50()
        model.load_state_dict(torch.load("runs_esc50/resnet50_fold1_seed42_1768497163/best.pth")['state_dict'])
        model = nn.Sequential(
            model,
            nn.Softmax(dim=1)
        ).eval().to(device)
    elif args.model == "mobilenet_v2" and args.dataset == "esc50":
        from models import build_mobilenetv2_esc50
        model = build_mobilenetv2_esc50()
        model.load_state_dict(torch.load("runs_esc50/mobilenet_v2_fold1_seed42_1768497160/best.pth")['state_dict'])
        model = nn.Sequential(
            model,
            nn.Softmax(dim=1)
        ).eval().to(device)
    else:
        raise NotImplementedError
    out_path = f"attributions/{args.dataset}_{args.model}_{args.attr_method}_attributions_{args.steps}.npy"
    if not os.path.exists(out_path) or args.debug:
        attr_method = args.attr_method
        attr_dict = {
            "ig": ig,
            "fast_ig": fast_ig,
            "sg": sg,
            "saliencymap": saliencymap,
            "mfaba": mfaba,
            "la": la,
            "agi": agi,
            "wam_ig": wam_ig_2d if args.dataset in ['imagenet', 'cub'] else wam_ig_1d,
            "wam_sg": wam_sg_2d if args.dataset in ['imagenet', 'cub'] else wam_sg_1d,
            "wam_mfaba": wam_mfaba_2d if args.dataset in ['imagenet', 'cub'] else wam_mfaba_1d,
        }
        attr_method = attr_dict[attr_method]
        attributions = list()
        # for x,y in tqdm(dataloader):
        #     x = x.to(device)
        #     y = y.to(device)
        #     attribution = attr_method(model, x, y)
        #     attributions.append(attribution)
        if args.dataset == 'imagenet' or args.dataset == "cub":
            for x,y in tqdm(dataloader):
                x = x.to(device)
                y = y.to(device)
                attribution = attr_method(model, x, y,steps=args.steps)
                attributions.append(attribution)
        elif args.dataset == 'esc50' or args.dataset == 'fma' or args.dataset == "audiomnist":
            pbar = tqdm(total=len(dataloader))
            for idx, (Xlgmel, y, waveform) in enumerate(dataloader):
                Xlgmel = Xlgmel.to(device)
                y = y.long().to(device)
                if attr_method in [wam_ig_1d, wam_sg_1d, wam_mfaba_1d]:
                    attribution = attr_method(
                        model,
                        waveform,          # [B,W] (cpu ok)
                        y,
                        steps=args.steps
                    )
                else:
                    if args.attr_method == "mfaba":
                        attribution = attr_method(model,Xlgmel, y, data_min=None, data_max=None)
                    elif args.attr_method == "la":
                        attribution = attr_method(model,Xlgmel, y, clip_min=None, clip_max=None)
                    elif args.attr_method == "agi":
                        attribution = attr_method(model,Xlgmel, y, clip_min=None, clip_max=None)
                    else:
                        attribution = attr_method(
                            model,
                            Xlgmel,
                            y,
                            steps=args.steps
                        )
                attributions.append(attribution)
                pbar.update()
            pbar.close()
        else:
            raise NotImplementedError
        
        attributions = np.concatenate(attributions, axis=0)
        print(attributions.shape,attributions.min(),attributions.max(),attributions.mean())
        np.save(out_path, attributions)
