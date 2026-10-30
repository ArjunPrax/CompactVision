"""TorchVision MobileNetV3-Small, with a trained ten-class head."""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def create_model(*, pretrained=False):
    model = mobilenet_v3_small(weights=WEIGHTS if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    return model


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model()
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
