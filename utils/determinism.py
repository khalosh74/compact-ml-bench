# utils/determinism.py
import os, random
import torch
try:
    import numpy as np
except Exception:
    np = None

def _seed_all(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)

def set_deterministic(seed: int = 42):
    # Required for cublas determinism on CUDA>=10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    _seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Disable TF32 to tighten determinism
    torch.set_float32_matmul_precision("highest")
    print(f"[MODE] deterministic (seed={seed})", flush=True)

def set_performance(seed: int = 42):
    _seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    # Allow TF32 on Ampere+/Blackwell for speed
    torch.set_float32_matmul_precision("high")
    print(f"[MODE] perf (seed={seed})", flush=True)

def set_run_mode_from_env(default: str = "perf"):
    mode = os.getenv("ML_MODE", default).lower()
    seed = int(os.getenv("ML_SEED", "42"))
    try:
        if mode in ("det", "deterministic", "repro"):
            set_deterministic(seed)
        else:
            set_performance(seed)
    except Exception as e:
        print(f"[MODE] warning: could not set mode ({e})", flush=True)
