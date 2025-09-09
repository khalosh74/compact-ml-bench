#!/usr/bin/env python3
import argparse, json, statistics as stats, time
from pathlib import Path
import torch

def percentile(xs, p):
    xs = sorted(xs); k = (len(xs)-1)*(p/100.0); f=int(k); c=min(f+1, len(xs)-1)
    return float(xs[f]) if f==c else float(xs[f] + (xs[c]-xs[f])*(k-f))

def size_mb(path: Path) -> float:
    try: return round(path.stat().st_size/(1024*1024), 3)
    except: return None

def bench(module, batch, warmup, repeat):
    x = torch.randn(batch,3,32,32)
    with torch.inference_mode():
        for _ in range(warmup): module(x)
    times=[]
    with torch.inference_mode():
        for _ in range(repeat):
            t0=time.perf_counter_ns(); module(x); dt=(time.perf_counter_ns()-t0)/1e6
            times.append(dt)
    ms_mean = stats.mean(times); ms_std = stats.pstdev(times) if len(times)>1 else 0.0
    return {
        "mean": ms_mean,
        "std": ms_std,
        "p50": percentile(times,50),
        "p90": percentile(times,90),
        "p99": percentile(times,99),
        "samples": len(times),
    }, (1000.0/ms_mean)*batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--repeat", type=int, default=400)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--variant", type=str, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    mod = torch.jit.load(str(args.artifact), map_location="cpu").eval()

    batches = [1,8,32,128]
    lat = {}; thr={}
    for b in batches:
        s, ips = bench(mod, b, args.warmup, args.repeat)
        lat[f"b{b}"] = s; thr[f"b{b}"] = ips

    out = args.out or Path("outputs")/f"bench_{args.variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "schema": "bench.v2",
        "model": Path(args.artifact).stem,
        "variant": args.variant,
        "device": "cpu",
        "threads": args.threads,
        "params_millions": None,
        "size_mb": size_mb(args.artifact),
        "latency_ms": lat,
        "throughput_img_s": thr,
        "macs_g_flops": None,
        "energy_proxy_j": None,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "batch_sizes": batches,
    }, open(out,"w"))
    print(f"[BENCH][INT8] wrote {out}")

if __name__ == "__main__":
    main()
