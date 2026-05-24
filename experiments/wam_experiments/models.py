from torchvision import models
import torch.nn as nn
def build_resnet18_esc50():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(
        1, 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, 50)

    return model

def build_resnet50_esc50():
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(
        1, 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, 50)

    return model


def build_resnet18_fma():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(
        1, 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, 8)

    return model


# model = build_resnet18_esc50()
# model = model.eval().to(device)
# model.load_state_dict(torch.load("resnet18_esc50_vanilla.pth"))

def build_resnet18_audiomnist():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(
        1, 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, 60)

    return model

def build_mobilenetv2_esc50():
    model = models.mobilenet_v2(pretrained=False)
    model.features[0][0] = nn.Conv2d(
        1, 32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )
    model.classifier[1] = nn.Linear(model.last_channel, 50)
    return model

def build_resnet50_cub():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 200)  # CUB-200-2011 有 200 类

    return model