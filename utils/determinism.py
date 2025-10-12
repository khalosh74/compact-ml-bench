import os
import random
import numpy as np
import torch

def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Avoid TF32 to keep math stable
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")  # avoids 'highest' fast path

def set_performance(seed: int = 42) -> None:
    # Still seed for basic reproducibility, but allow fast kernels
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Allow faster matmul on Ampere+
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")  # or "highest" if you prefer

def set_run_mode_from_env(default_seed: int = 42) -> None:
    mode = os.getenv("ML_MODE", "deterministic").lower()
    if mode in {"det", "deterministic", "repro"}:
        set_deterministic(default_seed)
    else:
        set_performance(default_seed)
