import argparse, json, os, sys, time, traceback
import torch
from torch import nn
from torchvision import models

def log(msg):
    print(f"[BENCH] {msg}", flush=True)

def reconstruct(meta):
    name = meta.get("model_name","resnet18").lower()
    num = int(meta.get("num_classes", 10))
    if name=="resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num); return m
    if name=="mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num); return m
    raise SystemExit(f"Unknown model {name}")

def measure_latency(model, device="cpu", warmups=20, repeats=100):
    model.eval()
    x = torch.randn(1,3,32,32, device=device)
    with torch.inference_mode():
        if device == "cuda":
            # precise on-GPU timing
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            for _ in range(warmups): model(x)
            torch.cuda.synchronize()
            starter.record()
            for _ in range(repeats): model(x)
            ender.record()
            torch.cuda.synchronize()
            dt_ms = starter.elapsed_time(ender) / repeats
            return float(dt_ms)
        else:
            # high-res wall clock for CPU
            for _ in range(warmups): model(x)
            t0 = time.perf_counter()
            for _ in range(repeats): model(x)
            dt = (time.perf_counter()-t0)/repeats
            return dt*1000.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu","gpu"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="outputs/bench_latest.json")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    try:
        if args.verbose: log("start")
        dev = "cuda" if (args.device=="gpu" and torch.cuda.is_available()) else "cpu"
        if args.verbose: log(f"requested={args.device} -> using device={dev}")

        if dev=="cpu":
            torch.set_num_threads(max(1,args.threads))

        if args.verbose: log(f"loading checkpoint: {args.checkpoint}")
        # Safe enough for our own files; warning can be ignored here
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        meta = ckpt.get("meta", {"model_name":"resnet18","num_classes":10})

        model = reconstruct(meta).to(dev)
        model.load_state_dict(ckpt["state_dict"], strict=True)

        params_m = sum(p.numel() for p in model.parameters())/1e6
        size_mb = os.path.getsize(args.checkpoint)/1e6

        if args.verbose: log("measuring latency...")
        ms = measure_latency(model, device=dev, warmups=args.warmup, repeats=args.repeat)

        out = {
            "device": dev,
            "latency_ms_b1": round(ms,3),
            "params_millions": round(params_m,3),
            "model_size_mb": round(size_mb,3),
            "threads": args.threads if dev=="cpu" else None
        }

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        if args.verbose: log(f"wrote {args.out}")

        print(json.dumps(out, indent=2), flush=True)
        if args.verbose: log("done")
    except Exception as e:
        print("[BENCH][ERROR] " + str(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
