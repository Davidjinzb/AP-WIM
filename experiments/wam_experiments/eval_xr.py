from evaluation import CausalMetric,auc_curve
from utils import device
import argparse
from torchvision.models import efficientnet_b0
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import device,setup_seed
from datasets import IMAGENETSUBSET1000
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os
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

BATCH_SIZE = 64

if __name__ == '__main__':
    if args.dataset == 'imagenet':
        dataset = IMAGENETSUBSET1000()
    elif args.dataset == 'cub':
        from datasets import CUBDATASET
        dataset = CUBDATASET(root="data/CUB/test")
    elif args.dataset == 'esc50':
        from datasets import ESC50
        dataset = ESC50(mode='test', num_FOLD=1)
    elif args.dataset == 'fma':
        from datasets import FMA_SMALL
        dataset = FMA_SMALL(mode='test')
    elif args.dataset == "audiomnist":
        from datasets import AudioMNIST
        dataset = AudioMNIST(mode='test')
    else:
        raise NotImplementedError
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    if args.model == 'efficient_b0' and args.dataset == 'imagenet':
        model = efficientnet_b0(pretrained=True)
        model = nn.Sequential(normalize, model, sfmx).eval().to(device)
    elif args.model == 'resnet18' and args.dataset == 'esc50':
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
    out_path = f'scores/{args.dataset}_{args.model}_{args.attr_method}_causal_scores_{args.steps}.npz'
    if not os.path.exists(out_path) or args.debug:
        os.makedirs('scores', exist_ok=True)
        insertion = CausalMetric(model, mode='ins',step=224 if args.dataset == 'imagenet' else 128,substrate_fn=torch.zeros_like, device=device)
        deletion = CausalMetric(model, mode='del',step=224 if args.dataset == 'imagenet' else 128,substrate_fn=torch.zeros_like, device=device)
        attributions = np.load(f'attributions/{args.dataset}_{args.model}_{args.attr_method}_attributions_{args.steps}.npy')
        all_insertion_scores = []
        all_deletion_scores = []
        for idx, (x,*_) in enumerate(tqdm(dataloader)):
            x = x.cpu()
            attr = attributions[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            insertion_scores = insertion.evaluate(x, attr, len(x))
            deletion_scores = deletion.evaluate(x, attr, len(x))
            all_insertion_scores.append(insertion_scores)
            all_deletion_scores.append(deletion_scores)
        all_insertion_scores = np.concatenate(all_insertion_scores, axis=0)
        all_deletion_scores = np.concatenate(all_deletion_scores, axis=0)
        insert_auc = auc_curve(all_insertion_scores.mean(axis=0))
        delete_auc = auc_curve(all_deletion_scores.mean(axis=0))
        print(f'Insertion AUC: {insert_auc:.4f}, Deletion AUC: {delete_auc:.4f}')
        np.savez(out_path,
                insertion_scores=all_insertion_scores,
                deletion_scores=all_deletion_scores)
