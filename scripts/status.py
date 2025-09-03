import sys, json, platform
rep = {"python": sys.version.replace("\n"," "), "platform": platform.platform()}
try:
    import torch, torchvision
    rep.update({
        "torch_version": torch.__version__,
        "torchvision_version": getattr(torchvision, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    })
except Exception as e:
    rep["torch_import_error"] = str(e)
print(json.dumps(rep, indent=2))
if "torch_import_error" in rep: raise SystemExit(1)
