import argparse, json, os, time, sys
from pathlib import Path

import torch
import torchvision.models as tvm

def build_model(name: str, num_classes: int):
    name = (name or "resnet18").lower()
    if name == "resnet18":
        m = tvm.resnet18(num_classes=num_classes)
    elif name == "resnet34":
        m = tvm.resnet34(num_classes=num_classes)
    elif name in ("mobilenet_v2","mnv2","mobilenetv2"):
        m = tvm.mobilenet_v2(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {name}")
    return m

def load_checkpoint(path: str):
    # Prefer safe loading (weights_only) if available, fall back to standard torch.load
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)  # PyTorch >=2.4
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    return ckpt

def main():
    p = argparse.ArgumentParser("bench")
    p.add_argument("--checkpoint", required=True, help="Path to .pt/.pth checkpoint (eager PyTorch). Use bench_ts.py for .ts")
    p.add_argument("--device", default="gpu", choices=["gpu","cuda","cpu"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--threads", type=int, default=None, help="CPU only: set torch.set_num_threads")
    p.add_argument("--out", default="outputs/bench_latest.json")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    print("[BENCH] start", flush=True)

    if args.checkpoint.lower().endswith(".ts"):
        print("[BENCH][ERROR] TorchScript artifact detected. Use scripts/bench_ts.py for .ts models.", flush=True)
        sys.exit(2)

    dev = "cuda" if args.device in ("gpu","cuda") else "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        print("[BENCH][WARN] CUDA requested but not available; falling back to CPU.", flush=True)
        dev = "cpu"
    print(f"[BENCH] requested={args.device} -> using device={dev}", flush=True)

    print(f"[BENCH] loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = load_checkpoint(args.checkpoint)

    # Resolve state_dict + meta
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = ckpt.get("meta", {})
    else:
        # some trainers save plain state_dict
        state_dict = ckpt
        meta = {}

    model_name = meta.get("model_name") or ("resnet18" if any(k.startswith("layer") for k in state_dict.keys()) else "mobilenet_v2")
    num_classes = int(meta.get("num_classes", 10))
    if args.verbose:
        print(f"[BENCH] meta={{'model_name': '{model_name}', 'num_classes': {num_classes}}}", flush=True)

    model = build_model(model_name, num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if args.verbose:
        if missing:    print(f"[BENCH][WARN] Missing keys: {len(missing)}", flush=True)
        if unexpected: print(f"[BENCH][WARN] Unexpected keys: {len(unexpected)}", flush=True)

    model.eval()
    if dev == "cuda":
        model.to("cuda")
        torch.backends.cudnn.benchmark = True

    # CIFAR-10 shape
    inp = torch.randn(1,3,32,32, device=("cuda" if dev=="cuda" else "cpu"))

    # Params & file size
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        size_mb = os.path.getsize(args.checkpoint) / (1024*1024)
    except OSError:
        size_mb = None

    # Warmup
    with torch.inference_mode():
        if dev == "cuda":
            for _ in range(args.warmup):
                _ = model(inp)
            torch.cuda.synchronize()
        else:
            if args.threads:
                try: torch.set_num_threads(int(args.threads))
                except Exception: pass
            for _ in range(args.warmup):
                _ = model(inp)

    # Measure latency (ms/sample, batch=1)
    if dev == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.inference_mode():
            for _ in range(args.repeat):
                _ = model(inp)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)  # ms over repeat
        latency_ms_b1 = total_ms / args.repeat
    else:
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.repeat):
                _ = model(inp)
        t1 = time.perf_counter()
        latency_ms_b1 = (t1 - t0) * 1000.0 / args.repeat

    out = {
        "device": "cuda" if dev=="cuda" else "cpu",
        "latency_ms_b1": round(latency_ms_b1, 3),
        "params_millions": round(params_m, 3),
        "model_size_mb": round(size_mb, 3) if size_mb is not None else None,
        "threads": (int(args.threads) if args.threads else None),
    }

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[BENCH] wrote {args.out}", flush=True)
    print(json.dumps(out, indent=2))
    print("[BENCH] done", flush=True)

if __name__ == "__main__":
    main()
