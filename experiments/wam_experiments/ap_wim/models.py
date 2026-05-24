from torchvision import models
import torch.nn as nn

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