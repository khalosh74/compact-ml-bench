import torch.nn as nn
from torchvision import models

def build_model(arch: str, num_classes: int = 10) -> nn.Module:
    a = arch.lower()
    if a == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if a == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if a == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    raise ValueError(f"Unsupported arch '{arch}'. Use: resnet18, resnet34, mobilenet_v2.")

def params_millions(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6
