import argparse
from torch.utils.data import DataLoader
from utils import setup_seed
from tqdm import tqdm
import numpy as np
import os
from apwim_experiment_common import (
    attribution_path,
    build_dataset,
    build_model,
    call_attribution,
    default_ap_wim_steps,
    dispatch_attr_method,
    get_attr_dict,
    is_wavelet_method,
    save_npy_atomic,
)
setup_seed(42)

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='efficient_b0', help='model name')
argparser.add_argument('--attr_method', type=str, default='wam_mfaba', help='attribution method')
argparser.add_argument('--dataset', type=str, default='imagenet')
argparser.add_argument('--batch_size', type=int, default=16)
argparser.add_argument('--max_samples', type=int, default=None)
argparser.add_argument('--attribution_dir', type=str, default='attributions')
argparser.add_argument('--output_tag', type=str, default='')
argparser.add_argument('--steps', type=int, default=None)
argparser.add_argument('--wavelet', type=str, default='haar')
argparser.add_argument('--J', type=int, default=3)
argparser.add_argument('--approx_coeffs', action='store_true')
argparser.add_argument("--force",action="store_true")
argparser.add_argument("--debug",action="store_true")
args = argparser.parse_args()

BATCH_SIZE = args.batch_size

if __name__ == '__main__':
    dataset = build_dataset(args.dataset, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = build_model(args.dataset, args.model, softmax=True)
    out_path = attribution_path(
        args.attribution_dir,
        args.dataset,
        args.model,
        args.attr_method,
        steps=args.steps,
        output_tag=args.output_tag,
    )
    if not os.path.exists(out_path) or args.debug or args.force:
        attr_method_name = args.attr_method
        dispatch_method = dispatch_attr_method(attr_method_name)
        steps = args.steps
        if dispatch_method == "wam_mfaba" and steps is None:
            steps = default_ap_wim_steps(args.dataset)
        attr_dict = get_attr_dict(args.dataset)
        if attr_method_name not in attr_dict:
            raise ValueError(f"Unknown attr_method={attr_method_name}. Available: {sorted(attr_dict)}")
        attr_method = attr_dict[attr_method_name]
        print(
            f"[ATTR] dataset={args.dataset} model={args.model} method={attr_method_name} "
            f"dispatch={dispatch_method} samples={len(dataset)} batch_size={BATCH_SIZE}"
        )
        if is_wavelet_method(attr_method_name):
            print(
                f"[ATTR] wavelet={args.wavelet} J={args.J} "
                f"approx_coeffs={int(args.approx_coeffs)} steps={steps}"
            )
        print(f"[ATTR] output={out_path}")
        attributions = list()
        for batch in tqdm(dataloader, desc="generate_attribution"):
            attribution = call_attribution(
                attr_method_name,
                attr_method,
                model,
                args.dataset,
                batch,
                wavelet=args.wavelet,
                J=args.J,
                approx_coeffs=args.approx_coeffs,
                steps=steps,
            )
            attributions.append(attribution)
        
        attributions = np.concatenate(attributions, axis=0)
        print(attributions.shape,attributions.min(),attributions.max(),attributions.mean())
        save_npy_atomic(out_path, attributions)
        print("saved:", out_path)
    else:
        print("exists:", out_path)
