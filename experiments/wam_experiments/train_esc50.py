# # %%
# from src.dataloader import ESC50
# from torch.utils.data import DataLoader
# import torch
# import torch.nn as nn
# from torchvision import models
# import random
# from tqdm.auto import tqdm
# from torch import optim
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
# setup_seed(42)

# # %%
# def build_resnet18_esc50():
#     model = models.resnet18(pretrained=False)

#     # 输入：1 通道 log-mel
#     model.conv1 = nn.Conv2d(
#         1, 64,
#         kernel_size=7,
#         stride=2,
#         padding=3,
#         bias=False
#     )

#     # 输出：50 类
#     model.fc = nn.Linear(model.fc.in_features, 50)

#     return model
# model = build_resnet18_esc50()
# model = model.to(device)

# # %%
# train_dataset = ESC50(mode='train', num_FOLD=1)
# test_dataset = ESC50(mode='test', num_FOLD=1)

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=8
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=32,
#     shuffle=False,
#     num_workers=8
# )

# # %%
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(
#     model.parameters(),
#     lr=1e-3,
#     weight_decay=0.0   # Vanilla：不加正则
# )


# def train_one_epoch(model, loader, optimizer, criterion):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for x, y, *_ in tqdm(loader, desc="Training", leave=False):
#         x = x.to(device)
#         y = y.to(device)

#         logits = model(x)
#         loss = criterion(logits, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * x.size(0)
#         preds = logits.argmax(dim=1)
#         correct += (preds == y).sum().item()
#         total += y.size(0)

#     return running_loss / total, correct / total

# @torch.no_grad()
# def evaluate(model, loader, criterion):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for x, y, *_ in tqdm(loader, desc="Testing", leave=False):
#         x = x.to(device)
#         y = y.to(device)

#         logits = model(x)
#         loss = criterion(logits, y)

#         running_loss += loss.item() * x.size(0)
#         preds = logits.argmax(dim=1)
#         correct += (preds == y).sum().item()
#         total += y.size(0)

#     return running_loss / total, correct / total


# # %%
# num_epochs = 50
# best_acc = 0.0

# for epoch in range(num_epochs):
#     print(f"\nEpoch [{epoch+1}/{num_epochs}]")

#     train_loss, train_acc = train_one_epoch(
#         model, train_loader, optimizer, criterion
#     )

#     test_loss, test_acc = evaluate(
#         model, test_loader, criterion
#     )

#     print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
#     print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

#     # 保存最优模型
#     if test_acc > best_acc:
#         best_acc = test_acc
#         torch.save(model.state_dict(), "resnet18_esc50_vanilla.pth")
#         print("✅ Saved best model")


# # %%


# # %%


# # %%





# -*- coding: utf-8 -*-
import os
import time
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm.auto import tqdm

from torchvision import models

from datasets import ESC50

# ---------------------------
# Reproducibility
# ---------------------------
def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Model utilities
# ---------------------------
def _replace_first_conv(conv: nn.Conv2d, in_channels: int, pretrained: bool):
    """把 3 通道 stem conv 改成 1 通道，并可选把 pretrained 权重做均值迁移。"""
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )
    if pretrained and conv.weight.shape[1] == 3 and in_channels == 1:
        # [out,3,k,k] -> [out,1,k,k] 取 RGB 平均
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
    return new_conv


def _set_model_in_channels(model: nn.Module, model_name: str, in_channels: int, pretrained: bool):
    """针对不同 torchvision 模型，把第一层改成 in_channels。"""
    name = model_name.lower()

    if name.startswith("resnet") or name.startswith("resnext") or name.startswith("wide_resnet"):
        model.conv1 = _replace_first_conv(model.conv1, in_channels, pretrained)
        return

    if name.startswith("densenet"):
        model.features.conv0 = _replace_first_conv(model.features.conv0, in_channels, pretrained)
        return

    if name.startswith("vgg"):
        model.features[0] = _replace_first_conv(model.features[0], in_channels, pretrained)
        return

    if name.startswith("mobilenet_v2") or name.startswith("mobilenet_v3"):
        # features[0][0] is Conv2d
        model.features[0][0] = _replace_first_conv(model.features[0][0], in_channels, pretrained)
        return

    if name.startswith("efficientnet"):
        model.features[0][0] = _replace_first_conv(model.features[0][0], in_channels, pretrained)
        return

    if name.startswith("shufflenet"):
        # conv1 is Sequential, first element Conv2d
        model.conv1[0] = _replace_first_conv(model.conv1[0], in_channels, pretrained)
        return

    if name.startswith("squeezenet"):
        model.features[0] = _replace_first_conv(model.features[0], in_channels, pretrained)
        return

    if name.startswith("convnext"):
        # features[0][0] is Conv2d
        model.features[0][0] = _replace_first_conv(model.features[0][0], in_channels, pretrained)
        return

    if name.startswith("regnet"):
        # stem is model.stem[0] Conv2d
        model.stem[0] = _replace_first_conv(model.stem[0], in_channels, pretrained)
        return

    if name.startswith("vit"):
        # ViT patch embed conv: model.conv_proj
        model.conv_proj = _replace_first_conv(model.conv_proj, in_channels, pretrained)
        return

    raise ValueError(f"Unsupported model for in_channels adaptation: {model_name}")


def _set_model_num_classes(model: nn.Module, model_name: str, num_classes: int):
    """针对不同 torchvision 模型，把分类头改成 num_classes。"""
    name = model_name.lower()

    if name.startswith("resnet") or name.startswith("resnext") or name.startswith("wide_resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    if name.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return

    if name.startswith("vgg"):
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return

    if name.startswith("mobilenet_v2") or name.startswith("mobilenet_v3"):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return

    if name.startswith("efficientnet"):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return

    if name.startswith("shufflenet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    if name.startswith("squeezenet"):
        # classifier[1] is Conv2d (512 -> num_classes)
        final_conv: nn.Conv2d = model.classifier[1]
        model.classifier[1] = nn.Conv2d(
            final_conv.in_channels,
            num_classes,
            kernel_size=final_conv.kernel_size,
            stride=final_conv.stride,
            padding=final_conv.padding,
            bias=True,
        )
        return

    if name.startswith("convnext"):
        # classifier: [Layerx, Flatten, Linear]
        # torchvision convnext classifier[-1] is Linear
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return

    if name.startswith("regnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    if name.startswith("vit"):
        # heads.head is Linear
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return

    raise ValueError(f"Unsupported model for num_classes adaptation: {model_name}")


def build_model(model_name: str, num_classes: int = 50, in_channels: int = 1, pretrained: bool = False):
    """
    支持的模型（15 个）：
      resnet18, resnet34, resnet50,
      mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,
      efficientnet_b0, efficientnet_b1,
      densenet121,
      shufflenet_v2_x1_0,
      squeezenet1_1,
      convnext_tiny, convnext_small,
      vit_b_16,
      regnet_y_400mf
    """
    name = model_name.lower()
    weights = None
    if pretrained:
        # torchvision 的 weights 枚举名不完全统一，这里用 try-catch 简化
        try:
            weights = getattr(models, f"{name}_weights").DEFAULT  # 不一定存在
        except Exception:
            weights = "DEFAULT"

    # ---- instantiate ----
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    elif name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)

    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)

    elif name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)

    elif name == "squeezenet1_1":
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None)

    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
    elif name == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None)

    elif name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)

    elif name == "regnet_y_400mf":
        model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ---- adapt input/output ----
    _set_model_in_channels(model, model_name, in_channels=in_channels, pretrained=pretrained)
    _set_model_num_classes(model, model_name, num_classes=num_classes)

    return model


# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, amp=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for x, y, *_ in tqdm(loader, desc="Training", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y, *_ in tqdm(loader, desc="Testing", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="model name, e.g. resnet18, efficientnet_b0, ...")
    parser.add_argument("--pretrained", action="store_true", help="use torchvision pretrained weights")
    parser.add_argument("--fold", type=int, default=1, help="ESC50 fold num_FOLD")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="mixed precision")
    parser.add_argument("--outdir", type=str, default="runs_esc50", help="output directory root")
    parser.add_argument("--run-name", type=str, default=None, help="fixed run directory name; default uses timestamp")
    parser.add_argument("--resume", action="store_true", help="resume from last.pth in the run directory")
    args = parser.parse_args()

    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset / loader
    train_dataset = ESC50(mode="train", num_FOLD=args.fold)
    test_dataset = ESC50(mode="test", num_FOLD=args.fold)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # model
    model = build_model(
        model_name=args.model,
        num_classes=50,
        in_channels=1,   # ESC50: 1 通道 log-mel
        pretrained=args.pretrained,
    ).to(device)

    # optim / loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # output
    run_name = args.run_name or f"{args.model}_fold{args.fold}_seed{args.seed}_{int(time.time())}"
    outdir = Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    start_epoch = 0
    best_path = outdir / "best.pth"
    last_path = outdir / "last.pth"

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_acc = float(ckpt.get("best_acc", 0.0))
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[RESUME] {last_path} epoch={start_epoch} best_acc={best_acc:.4f}")

    if start_epoch >= args.epochs:
        print(f"Done. Already reached epochs={args.epochs} | best_acc={best_acc:.4f} | best_ckpt={best_path}")
        return

    # training
    for epoch in range(start_epoch, args.epochs):
        print(f"\n[{run_name}] Epoch [{epoch+1}/{args.epochs}] device={device} amp={args.amp}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, amp=args.amp)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, amp=args.amp)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

        if test_acc > best_acc or not best_path.exists():
            best_acc = test_acc
            torch.save(
                {
                    "model": args.model,
                    "pretrained": args.pretrained,
                    "fold": args.fold,
                    "seed": args.seed,
                    "epoch": epoch + 1,
                    "best_acc": best_acc,
                    "state_dict": model.state_dict(),
                },
                best_path,
            )
            print(f"✅ Saved best model -> {best_path} (acc={best_acc:.4f})")

        torch.save(
            {
                "model": args.model,
                "pretrained": args.pretrained,
                "fold": args.fold,
                "seed": args.seed,
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            last_path,
        )

    print(f"\nDone. Best acc={best_acc:.4f} | best_ckpt={best_path}")


if __name__ == "__main__":
    main()
