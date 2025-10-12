import torch

def pick_device(prefer_cuda: bool = True) -> torch.device:
    """Return cuda if available and preferred, else cpu."""
    return torch.device("cuda") if prefer_cuda and torch.cuda.is_available() else torch.device("cpu")

def cpu_device() -> torch.device:
    return torch.device("cpu")

def device_name(d: torch.device) -> str:
    return "cuda" if d.type == "cuda" else "cpu"
