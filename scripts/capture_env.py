# scripts/capture_env.py
import os, json, platform, datetime
import torch

def gpu_info():
    if not torch.cuda.is_available():
        return {"available": False, "count": 0}
    i = 0
    info = {
        "available": True,
        "count": torch.cuda.device_count(),
        "devices": []
    }
    for i in range(torch.cuda.device_count()):
        info["devices"].append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": torch.cuda.get_device_capability(i),
        })
    return info

def main():
    d = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "env": {
            "ML_MODE": os.getenv("ML_MODE"),
            "ML_SEED": os.getenv("ML_SEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
        },
        "gpu": gpu_info()
    }
    os.makedirs("outputs", exist_ok=True)
    out = os.path.join("outputs", "env_fingerprint.json")
    with open(out, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[ENV] wrote {out}")

if __name__ == "__main__":
    main()
