from evaluation import CausalMetric,auc_curve
from utils import device
import argparse
from torch.utils.data import DataLoader
from utils import device,setup_seed
import torch
from tqdm import tqdm
import numpy as np
import os
setup_seed(42)
from apwim_experiment_common import (
    build_dataset,
    build_model,
    causal_score_path,
    resolve_existing_attribution,
    save_npz_atomic,
)
from test_road import make_baseline_like


argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='efficient_b0', help='model name')
argparser.add_argument('--attr_method', type=str, default='wam_mfaba', help='attribution method')
argparser.add_argument('--dataset', type=str, default='imagenet')
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--max_samples', type=int, default=None)
argparser.add_argument('--attribution_dir', type=str, default='attributions')
argparser.add_argument('--score_dir', type=str, default='scores')
argparser.add_argument('--output_tag', type=str, default='')
argparser.add_argument('--steps', type=int, default=None)
argparser.add_argument('--substrate', type=str, default='zeros', choices=['zeros', 'noise', 'blur'])
argparser.add_argument("--force",action="store_true")
argparser.add_argument("--debug",action="store_true")
args = argparser.parse_args()

BATCH_SIZE = args.batch_size


def substrate_fn(mode):
    if mode == "zeros":
        return torch.zeros_like

    def _fn(x):
        return make_baseline_like(x, mode)

    return _fn

if __name__ == '__main__':
    dataset = build_dataset(args.dataset, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = build_model(args.dataset, args.model, softmax=True)
    out_path = causal_score_path(
        args.score_dir,
        args.dataset,
        args.model,
        args.attr_method,
        steps=args.steps,
        output_tag=args.output_tag,
        substrate=args.substrate,
    )
    if not os.path.exists(out_path) or args.debug or args.force:
        os.makedirs(args.score_dir, exist_ok=True)
        substrate = substrate_fn(args.substrate)
        insertion = CausalMetric(model, mode='ins',step=224 if args.dataset == 'imagenet' else 128,substrate_fn=substrate, device=device)
        deletion = CausalMetric(model, mode='del',step=224 if args.dataset == 'imagenet' else 128,substrate_fn=substrate, device=device)
        attr_path = resolve_existing_attribution(
            args.attribution_dir,
            args.dataset,
            args.model,
            args.attr_method,
            steps=args.steps,
            output_tag=args.output_tag,
        )
        attributions = np.load(attr_path)
        print(
            f"[EVAL] dataset={args.dataset} model={args.model} method={args.attr_method} "
            f"samples={len(dataset)} substrate={args.substrate}"
        )
        print(f"[EVAL] attribution={attr_path}")
        print(f"[EVAL] output={out_path}")
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
        save_npz_atomic(
            out_path,
            insertion_scores=all_insertion_scores,
            deletion_scores=all_deletion_scores,
            substrate=np.array(args.substrate),
            dataset=np.array(args.dataset),
            model=np.array(args.model),
            attr_method=np.array(args.attr_method),
        )
        print("saved:", out_path)
    else:
        print("exists:", out_path)
