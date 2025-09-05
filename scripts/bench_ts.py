# scripts/bench_ts.py
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch

from bench import (  # reuse helpers from bench.py
    parse_device, parse_batch_sizes, RepeatingSampler, nvidia_power_w, psutil_rss_mb,
    percentile, _sync
)

def bench_one(module, device, bs, warmup, repeat):
    module.to(device).eval()
    x = torch.randn(bs, 3, 32, 32, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = module(x)
        _sync()
        times = []
        t0 = time.perf_counter()
        for _ in range(repeat):
            t1 = time.perf_counter()
            _ = module(x)
            _sync()
            t2 = time.perf_counter()
            times.append((t2 - t1) * 1000.0)
        t_total = time.perf_counter() - t0
    ms_mean = float(np.mean(times))
    ms_std  = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
    return {
        "ms_mean": ms_mean,
        "ms_std":  ms_std,
        "ms_p50":  percentile(times,50),
        "ms_p90":  percentile(times,90),
        "ms_p99":  percentile(times,99),
        "img_s":   (bs / (ms_mean/1000.0)) if ms_mean>0 else None,
        "repeat":  repeat,
        "warmup":  warmup
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="TorchScript .ts file")
    ap.add_argument("--device", default="gpu", choices=["gpu","cpu","cuda","cpu"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--batch-sizes", default="1,8,32,128")
    ap.add_argument("--out", default="outputs/bench_ts_latest.json")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    dev = parse_device(args.device)
    bs_list = parse_batch_sizes(args.batch_sizes)
    art = Path(args.artifact)
    print(f"[BENCH-TS] device={dev.type}")

    module = torch.jit.load(str(art), map_location=dev)
    module.eval()

    # precision flags
    prec = getattr(torch, "get_float32_matmul_precision", lambda: None)()
    tf32 = None
    cudnn_tf32 = None
    try:
        tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
    except Exception:
        pass

    # memory + energy
    gpu_peak_alloc_mb = gpu_peak_reserved_mb = None
    power_mean_w = energy_j = None

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        p_sam = RepeatingSampler(nvidia_power_w, interval_s=0.1)
        p_sam.start()
    else:
        p_sam = None

    results = {}
    t_region_start = time.perf_counter()
    for bs in bs_list:
        stats = bench_one(module, dev, bs, args.warmup, args.repeat)
        for k,v in stats.items():
            results[f"{k}_b{bs}"] = v
    t_region = time.perf_counter() - t_region_start

    if dev.type == "cuda":
        _sync()
        gpu_peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024*1024)
        gpu_peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024*1024)
        if p_sam:
            p_sam.stop(); p_sam.join(timeout=1.0)
            vals = [v for (_,v) in p_sam.values if v is not None]
            if vals:
                power_mean_w = float(np.mean(vals))
                energy_j = power_mean_w * t_region
    else:
        # optional CPU RSS snapshot (best-effort)
        cpu_rss = psutil_rss_mb()

    params_m = sum(p.numel() for p in module.parameters()) / 1e6
    size_mb  = os.path.getsize(art) / (1024*1024)

    out = {
        "model": "unknown",
        "variant": "torchscript",
        "device": dev.type,
        "acc_top1": None,
        "params_millions": round(params_m,3),
        "size_mb": round(size_mb,3),
        "macs_g_flops": None,  # not computed for TS here
        "b1_ms": results.get("ms_mean_b1"),
        "float32_matmul_precision": prec,
        "allow_tf32": tf32,
        "cudnn_allow_tf32": cudnn_tf32,
        "gpu_peak_alloc_mb": round(gpu_peak_alloc_mb,2) if gpu_peak_alloc_mb is not None else None,
        "gpu_peak_reserved_mb": round(gpu_peak_reserved_mb,2) if gpu_peak_reserved_mb is not None else None,
        "energy_mean_power_w": round(power_mean_w,2) if power_mean_w is not None else None,
        "energy_proxy_j": round(energy_j,3) if energy_j is not None else None,
    }
    out.update({k:(round(v,4) if isinstance(v,(int,float)) and v is not None else v) for k,v in results.items()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[BENCH-TS] wrote {args.out}")
    print(json.dumps({"device": out["device"], "b1_ms": out["b1_ms"], "img_s_b32": out.get("img_s_b32")}, indent=2))

if __name__ == "__main__":
    main()
