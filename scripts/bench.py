import argparse, json, os, time
import torch
from torch import nn
from torchvision import models

def reconstruct(meta):
    name = meta.get("model_name","resnet18").lower()
    num = int(meta.get("num_classes", 10))
    if name=="resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num)
        return m
    if name=="mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num)
        return m
    raise SystemExit(f"Unknown model {name}")

def measure_latency(model, device="cpu", warmups=20, repeats=100):
    model.eval()
    x = torch.randn(1,3,32,32, device=device)
    with torch.inference_mode():
        for _ in range(warmups):
            model(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
        dt = (time.perf_counter()-t0)/repeats
    return dt*1000.0  # ms

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu","gpu"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    dev = "cuda" if (args.device=="gpu" and torch.cuda.is_available()) else "cpu"
    if dev=="cpu":
        torch.set_num_threads(max(1,args.threads))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("meta", {})
    model = reconstruct(meta).to(dev)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    # size/params
    params_m = sum(p.numel() for p in model.parameters())/1e6
    size_mb = os.path.getsize(args.checkpoint)/1e6

    ms = measure_latency(model, device=dev, warmups=args.warmup, repeats=args.repeat)
    out = {
        "device": dev,
        "latency_ms_b1": round(ms,3),
        "params_millions": round(params_m,3),
        "model_size_mb": round(size_mb,3),
        "threads": args.threads if dev=="cpu" else None
    }
    print(json.dumps(out, indent=2))
