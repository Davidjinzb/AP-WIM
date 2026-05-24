from torch.utils.data import DataLoader
from datasets import ESC50
from models import build_resnet18_esc50
from utils import device
import torch
from attr_methods_1d import wam_ig as wam_ig_1d, wam_sg as wam_sg_1d, wam_mfaba as wam_mfaba_1d
from attr_methods_baseline import ig,fast_ig,sg,saliencymap
import torch.nn as nn
from evaluation import CausalMetric
BATCH_SIZE = 16
if __name__ == '__main__':
    dataset = ESC50(mode='test', num_FOLD=1)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = build_resnet18_esc50()
    model.load_state_dict(torch.load("resnet18_esc50_vanilla.pth"))
    model = model.eval().to(device)
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    insertion = CausalMetric(model, mode='ins',step=128,substrate_fn=torch.zeros_like, device=device)
    deletion = CausalMetric(model, mode='del',step=128,substrate_fn=torch.zeros_like, device=device)
    for idx, (Xlgmel, y, waveform) in enumerate(dataloader):
        print(f'Batch {idx}:')
        print(f'  Xlgmel shape: {Xlgmel.shape}')  # [B, 1, T, n_mels]
        print(f'  y shape: {y.shape}')              # [B]
        print(f'  waveform shape: {waveform.shape}')            # [B, T]
        Xlgmel = Xlgmel.to(device)
        y = y.long().to(device)
        with torch.no_grad():
            outputs = model(Xlgmel.to(device))
            print(f'  Model outputs shape: {outputs.shape}')  # [B, num_classes]
        attr_wam_ig = wam_ig_1d(
            model,
            waveform,          # [B,W] (cpu ok)
            y
        )
        print(f'  WAM-IG Attribution shape: {attr_wam_ig.shape}')  # [B, T, n_mels]
        attr_wam_sg = wam_sg_1d(
            model,
            waveform,          # [B,W] (cpu ok)
            y,
            n_samples=25,
            stdev_spread=0.001,
            seed=42
        )
        print(f'  WAM-SG Attribution shape: {attr_wam_sg.shape}')  # [B, T, n_mels]
        attr_wam_mfaba = wam_mfaba_1d(
            model,
            waveform,          # [B,W] (cpu ok)
            y,
            steps=50
        )
        print(f'  WAM-MFABA Attribution shape: {attr_wam_mfaba.shape}')  # [B, T, n_mels]
        attr_baseline_ig = ig(
            model,
            Xlgmel,
            y
        )
        print(f'  Baseline IG Attribution shape: {attr_baseline_ig.shape}')  # [B, 1, T, n_mels]
        attr_baseline_sg = sg(
            model,
            Xlgmel,
            y
        )
        print(f'  Baseline SG Attribution shape: {attr_baseline_sg.shape}')
        attr_baseline_fast_ig = fast_ig(
            model,
            Xlgmel,
            y
        )
        print(f'  Baseline Fast IG Attribution shape: {attr_baseline_fast_ig.shape}')
        attr_baseline_saliencymap = saliencymap(
            model,
            Xlgmel,
            y
        )
        print(f'  Baseline Saliency Map Attribution shape: {attr_baseline_saliencymap.shape}')
        # insertion_scores = insertion.evaluate(Xlgmel.cpu(), attr.cpu().numpy(), len(Xlgmel))
        # deletion_scores = deletion.evaluate(Xlgmel.cpu(), attr.cpu().numpy(), len(Xlgmel))
        # print(f'  Insertion scores shape: {insertion_scores.shape}')  # [B, T]
        # print(f'  Deletion scores shape: {deletion_scores.shape}')    # [B, T]
        all_results = list()
        for method, attr in zip(
            ['WAM-IG', 'WAM-SG', 'WAM-MFABA',
             'Baseline-IG', 'Baseline-SG', 'Baseline-FastIG', 'Baseline-SaliencyMap'],
            [attr_wam_ig, attr_wam_sg, attr_wam_mfaba,
             attr_baseline_ig, attr_baseline_sg, attr_baseline_fast_ig, attr_baseline_saliencymap]
        ):
            insertion_scores = insertion.evaluate(Xlgmel.cpu(), attr, len(Xlgmel))
            deletion_scores = deletion.evaluate(Xlgmel.cpu(), attr, len(Xlgmel))
            print("  Method: ", method)
            all_results.append({
                'method': method,
                'insertion_scores': insertion_scores.mean(),
                'deletion_scores': deletion_scores.mean()
            })
        for res in all_results:
            print(f"    {res['method']}: Insertion Score = {res['insertion_scores']:.4f}, Deletion Score = {res['deletion_scores']:.4f}")
            
        break  # 只查看第一个批次
    
# # waveform: [B, W] on CPU from dataloader
# # y: [B] on CPU from dataloader

# y_dev = y.to(device, dtype=torch.long)   # y 一定要 long 且在 model device 上

# # 1) SmoothGrad-WAM 1D
# ms_attr_sg, coeff_attr_sg = wam1d_sg(
#     model,
#     waveform,          # [B,W] (cpu ok)
#     y_dev,
#     J=3,
#     n_samples=25,
#     stdev_spread=0.001,
#     seed=42,
#     # 如果你按我建议把 _ensure_x_tensor 加了 normalize 参数：
#     # normalize=False
# )

# # 2) IG-WAM 1D
# ms_attr_ig, coeff_attr_ig = wam1d_ig(
#     model,
#     waveform,
#     y_dev,
#     J=3,
#     n_steps=25,
#     # normalize=False
# )

# # 3) MFABA-WAM 1D
# ms_grad_last, coeff_attr_mfaba = wam1d_mfaba(
#     model,
#     waveform,
#     y_dev,
#     J=3,
#     steps=50,
#     # normalize=False
# )

# print(ms_attr_sg.shape)   # [B, T, n_mels]
# print(ms_attr_ig.shape)   # [B, T, n_mels]
# print(len(coeff_attr_sg)) # J+1
