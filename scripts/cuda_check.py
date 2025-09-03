import torch, json
print(json.dumps({
  "torch": torch.__version__,
  "runtime_cuda": getattr(torch.version, "cuda", None),
  "cuda_is_available": torch.cuda.is_available(),
  "cuda_device_count": torch.cuda.device_count(),
  "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
}, indent=2))
