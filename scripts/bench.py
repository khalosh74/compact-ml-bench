#!/usr/bin/env python3
import argparse, json, os, statistics as stats, subprocess, threading, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from utils.models import build_model, params_millions

def file_size_mb(p: Path):
    try: return round(Path(p).stat().st_size/(1024*1024), 3)
    except: return None

class EnergySampler(threading.Thread):
    def __init__(self, interval_s=0.1):
        super().__init__(daemon=True); self.interval=interval_s; self.running=False; self.joules=0.0
    def run(self):
        self.running=True
        last = time.perf_counter()
        while self.running:
            try:
                res = subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"], capture_output=True, text=True)
                power_w = float(res.stdout.strip().splitlines()[0])
                now = time.perf_counter(); dt = now-last; last=now
                self.joules += power_w * dt
            except Exception: # noqa
                pass
            time.sleep(self.interval)
    def stop(self): self.running=False

def percentile(xs, p):
    xs = sorted(xs); k = (len(xs)-1)*(p/100.0); f=int(k); c=min(f+1, len(xs)-1)
    return float(xs[f]) if f==c else float(xs[f] + (xs[c]-xs[f])*(k-f))

def run_bench(model, device, batches, warmup, repeat):
    lat = {}; thr={}
    example = torch.randn(max(batches),3,32,32, device=device)
    with torch.inference_mode():
        for _ in range(warmup): model(example[:1])
    for b in batches:
        times=[]
        with torch.inference_mode():
            for _ in range(repeat):
                t0=time.perf_counter_ns(); model(example[:b]); dt=(time.perf_counter_ns()-t0)/1e6
                times.append(dt)
        lat[f"b{b}"] = {
            "mean": stats.mean(times),
            "std": stats.pstdev(times) if len(times)>1 else 0.0,
            "p50": percentile(times,50),
            "p90": percentile(times,90),
            "p99": percentile(times,99),
            "samples": len(times),
        }
        thr[f"b{b}"] = (1000.0/lat[f"b{b}"]["mean"]) * b
    return lat, thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--arch", type=str, default="resnet18")
    ap.add_argument("--device", choices=["cpu","gpu","cuda"], default="gpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=600)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", type=str, default=None)
    args = ap.parse_args()

    dev = torch.device("cuda") if (args.device in ("gpu","cuda") and torch.cuda.is_available()) else torch.device("cpu")
    if dev.type=="cpu": torch.set_num_threads(args.threads)

    # load eager
    m = build_model(args.arch, num_classes=10)
    try: ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError: ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    m.load_state_dict(sd, strict=False)
    m.eval().to(dev)

    batches = [1,8,32,128]
    es=None
    if dev.type=="cuda":
        es = EnergySampler(0.1); es.start()
    lat, thr = run_bench(m, dev, batches, args.warmup, args.repeat)
    energy = es.joules if es else None
    if es: es.stop()

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    variant = args.variant or Path(out).stem.replace("bench_","")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "schema": "bench.v2",
            "model": args.arch,
            "variant": variant,
            "device": "gpu" if dev.type=="cuda" else "cpu",
            "threads": None if dev.type=="cuda" else args.threads,
            "params_millions": round(params_millions(m),3),
            "size_mb": file_size_mb(args.checkpoint),
            "latency_ms": lat,
            "throughput_img_s": thr,
            "macs_g_flops": 0.037 if args.arch=="resnet18" else None,
            "energy_proxy_j": energy,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "batch_sizes": batches
        }, f)
    print(f"[BENCH] wrote {out}")

if __name__ == "__main__":
    main()

