# scripts/bench_ts.py
import argparse, json, os, statistics, subprocess, threading, time
from pathlib import Path
import torch

# ---------- small helpers (self-contained; no imports from bench.py) ----------
def percentile(arr, p):
    if not arr: return None
    k = (len(arr)-1) * (p/100.0)
    f = int(k)
    c = min(f+1, len(arr)-1)
    if f == c:
        return float(arr[f])
    return float(arr[f] + (arr[c]-arr[f]) * (k - f))

def matmul_precision():
    try:
        return torch.get_float32_matmul_precision()
    except Exception:
        return None

def device_of(name: str):
    name = (name or "").lower()
    if name in ("cuda", "gpu", "cuda:0") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def nvidia_smi_sampler(stop_evt: threading.Event, device_index: int, samples: list, interval_s=0.1):
    """Sample power (W) via nvidia-smi while stop_evt is not set."""
    exe = "nvidia-smi.exe" if os.name == "nt" else "nvidia-smi"
    if not shutil.which(exe):  # if unavailable, do nothing
        return
    cmd = [exe, f"--query-gpu=power.draw", "--format=csv,noheader,nounits", f"-i={device_index}"]
    while not stop_evt.is_set():
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=0.5)
            w = float(out.decode("utf-8").strip())
            samples.append(w)
        except Exception:
            pass
        time.sleep(interval_s)

def params_millions_from_module(mod) -> float:
    try:
        return sum(p.numel() for p in mod.parameters())/1e6
    except Exception:
        return None

def size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024*1024), 3)
    except Exception:
        return None

def bench_one(module, dev, batch, warmup, repeat):
    """Return (stats_ms_dict, images_per_second) for given batch."""
    module.eval()
    x = torch.randn(batch, 3, 32, 32, device=dev)

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            y = module(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()

    # Timed loop
    times = []
    with torch.inference_mode():
        for _ in range(repeat):
            if dev.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                y = module(x)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                y = module(x)
                t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    times_sorted = sorted(times)
    stats = {
        "mean_ms": float(statistics.fmean(times_sorted)),
        "std_ms": float(statistics.pstdev(times_sorted)) if len(times_sorted) > 1 else 0.0,
        "p50_ms": percentile(times_sorted, 50),
        "p90_ms": percentile(times_sorted, 90),
        "p99_ms": percentile(times_sorted, 99),
    }
    img_per_s = (batch * 1000.0) / stats["mean_ms"] if stats["mean_ms"] else None
    return stats, img_per_s

import shutil  # used in nvidia-smi helper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="Path to TorchScript .ts")
    ap.add_argument("--device", default="cuda", help="cuda|cpu")
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--repeat", type=int, default=400)
    ap.add_argument("--batch-sizes", default="1,8,32,128",
                    help="Comma-separated list: e.g. 1,8,32,128")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-name", default=None,
                    help="Optional model name to tag the row; else try sidecar meta json; else 'unknown'")
    args = ap.parse_args()

    dev = device_of(args.device)
    print(f"[BENCH-TS] device={dev.type}")

    art = Path(args.artifact)
    if not art.exists():
        raise FileNotFoundError(f"{art} does not exist")

    # Load TS
    module = torch.jit.load(str(art), map_location=dev)
    module.to(dev)
    module.eval()

    # Resolve model name
    model_name = args.model_name
    if not model_name:
        sidecar = art.with_suffix(".meta.json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
                model_name = meta.get("model_name") or meta.get("model") or None
            except Exception:
                model_name = None
    if not model_name:
        model_name = "unknown"

    # Params & size
    params_m = params_millions_from_module(module)
    sz_mb = size_mb(art)

    # Percentiles & throughput across batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    # Optional energy proxy (CUDA only)
    power_samples = []
    energy_j = None
    stop_evt = threading.Event()
    sampler = None
    try:
        if dev.type == "cuda" and shutil.which("nvidia-smi.exe" if os.name == "nt" else "nvidia-smi"):
            sampler = threading.Thread(target=nvidia_smi_sampler, args=(stop_evt, 0, power_samples))
            sampler.daemon = True
            sampler.start()

        results = {}
        for b in batch_sizes:
            stats, img_s = bench_one(module, dev, b, args.warmup, args.repeat)
            if b == 1:
                results["b1_ms"]      = round(stats["mean_ms"], 3)
                results["ms_std_b1"]  = round(stats["std_ms"], 3)
                results["ms_p50_b1"]  = round(stats["p50_ms"], 3)
                results["ms_p90_b1"]  = round(stats["p90_ms"], 3)
                results["ms_p99_b1"]  = round(stats["p99_ms"], 3)
                results["img_s_b1"]   = round(img_s, 1) if img_s else None
            elif b == 8:
                results["img_s_b8"]   = round(img_s, 1) if img_s else None
            elif b == 32:
                results["img_s_b32"]  = round(img_s, 1) if img_s else None
            elif b == 128:
                results["img_s_b128"] = round(img_s, 1) if img_s else None

    finally:
        stop_evt.set()
        if sampler is not None:
            sampler.join(timeout=1.0)

    # Energy proxy as mean(power) * elapsed_seconds
    try:
        if power_samples:
            # elapsed approx = repeat * mean_ms / 1000 for batch=1 region + similar for others;
            # use sum of all timed batches as a rough proxy
            # Here, approximate with b1 mean only (dominates our KPI)
            elapsed_s = (results["b1_ms"] / 1000.0) * args.repeat if "b1_ms" in results else 0.0
            mean_w = float(statistics.fmean(power_samples))
            energy_j = round(mean_w * elapsed_s, 3)
    except Exception:
        energy_j = None

    row = {
        "model": model_name,
        "variant": "torchscript",
        "device": dev.type,
        "threads": None if dev.type == "cuda" else os.cpu_count(),
        "params_millions": round(params_m, 3) if params_m is not None else None,
        "size_mb": sz_mb,
        **results,
        "macs_g_flops": None,  # optional (see step 2 below)
        "energy_proxy_j": energy_j,
        "matmul_precision": matmul_precision(),
        "warmup": float(args.warmup),
        "repeat": float(args.repeat),
        "batch_sizes": args.batch_sizes
    }

    out = Path(args.out) if args.out else Path("outputs") / f"bench_{model_name}_ts_{dev.type}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2))
    print(f"[BENCH-TS] wrote {out}\n{json.dumps(row, indent=2)}")

if __name__ == "__main__":
    main()
