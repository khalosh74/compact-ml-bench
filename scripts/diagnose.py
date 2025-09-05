import json, platform, sys
from pathlib import Path

try:
    import psutil  # optional
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None

def gpu_meta():
    if torch and torch.cuda.is_available():
        return {
            "cuda_available": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_name": torch.cuda.get_device_name(0),
            "torch_cuda": getattr(torch.version, "cuda", None),
        }
    return {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_name": None,
        "torch_cuda": getattr(torch.version, "cuda", None) if torch else None,
    }

info = {
    "python": sys.version.replace("\n"," "),
    "platform": f"{platform.system()}-{platform.release()}",
    "torch_version": getattr(torch, "__version__", None),
    "torchvision_version": None,
    **gpu_meta(),
    "machine": platform.machine(),
    "processor": platform.processor(),
}

try:
    import torchvision
    info["torchvision_version"] = torchvision.__version__
except Exception:
    pass

if psutil:
    vm = psutil.virtual_memory()
    info["mem_total_gb"] = round(vm.total/1024/1024/1024, 2)
    info["mem_available_gb"] = round(vm.available/1024/1024/1024, 2)

print(json.dumps(info, indent=2))
