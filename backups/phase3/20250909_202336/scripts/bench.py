#!/usr/bin/env python3
# bench.py — Eager (PyTorch) benchmarking with rich metrics and stable JSON schema.

import argparse
import json
import math
import os
import shutil
import statistics as stats
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------- Small utilities -----------------------------

def percentile(values: List[float], q: float) -> float:
    """Simple percentile (q in [0,100]) without numpy."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def parse_device(arg: str) -> torch.device:
    req = arg.strip().lower()
    if req in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            print("[BENCH][WARN] CUDA requested but not available; falling back to CPU.", flush=True)
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def parse_batch_sizes(s: str) -> List[int]:
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception:
        return [1]


def safe_load_checkpoint(path: str):
    """torch.load with weights_only when available (PyTorch >=2.4)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # new API
    except TypeError:
        return torch.load(path, map_location="cpu")  # fallback for older torch


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = (model_name or "").lower()
    if model_name == "resnet34":
        return models.resnet34(num_classes=num_classes)
    if model_name == "mobilenet_v2":
        return models.mobilenet_v2(num_classes=num_classes)
    # default
    return models.resnet18(num_classes=num_classes)


def params_millions(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def size_mb_of(path: Path) -> Optional[float]:
    try:
        return round(path.stat().st_size / (1024 * 1024), 3)
    except Exception:
        return None


def device_of(t: torch.Tensor) -> torch.device:
    return t.device if t.is_cuda else torch.device("cpu")


def _sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()


# ----------------------------- MACs / FLOPs -----------------------------

def macs_g_with_fvcore(model: nn.Module, input_shape=(1, 3, 32, 32)) -> Optional[float]:
    """Return FLOPs (G) using fvcore if available; None otherwise."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None
    try:
        model = model.eval()
        with torch.inference_mode():
            inp = torch.randn(*input_shape)
            flops = FlopCountAnalysis(model, inp).total()
            return round(flops / 1e9, 3)
    except Exception:
        return None


# ----------------------------- Energy proxy (nvidia-smi) -----------------------------

def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


class PowerSampler(threading.Thread):
    """
    Samples GPU power draw via 'nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 100'.
    Use start()/stop() around the timed region. Provides mean power (W) and energy (J).
    """
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.samples: List[float] = []
        self._proc: Optional[subprocess.Popen] = None

    def run(self):
        if not nvidia_smi_available():
            return
        try:
            cmd = ["nvidia-smi",
                   "--query-gpu=power.draw",
                   "--format=csv,noheader,nounits",
                   "-lms", "100"]
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
            )
            start = time.perf_counter()
            while not self._stop_event.is_set():
                line = self._proc.stdout.readline() if self._proc.stdout else ""
                if not line:
                    # Sleep a tad to avoid tight loop if stdout closed
                    time.sleep(0.05)
                    continue
                try:
                    val = float(line.strip())
                    # Filter obvious garbage values
                    if 0.0 <= val < 1500.0:
                        self.samples.append(val)
                except Exception:
                    pass
            self.elapsed = time.perf_counter() - start
        finally:
            try:
                if self._proc:
                    self._proc.terminate()
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()

    def mean_watts(self) -> Optional[float]:
        return float(stats.fmean(self.samples)) if self.samples else None

    def energy_joules(self) -> Optional[float]:
        # Energy ≈ mean power (W) * elapsed_time (s)
        if not self.samples:
            return None
        return self.mean_watts() * getattr(self, "elapsed", 0.0)


# ----------------------------- Timing & Stats -----------------------------

def lat_stats(samples_ms: List[float]) -> Dict[str, float]:
    if not samples_ms:
        return {
            "ms_mean": float("nan"),
            "ms_std": float("nan"),
            "ms_p50": float("nan"),
            "ms_p90": float("nan"),
            "ms_p99": float("nan"),
        }
    mean = stats.fmean(samples_ms)
    std = stats.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return {
        "ms_mean": mean,
        "ms_std": std,
        "ms_p50": percentile(samples_ms, 50),
        "ms_p90": percentile(samples_ms, 90),
        "ms_p99": percentile(samples_ms, 99),
    }


def forward_once(model: nn.Module, x: torch.Tensor, dev: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        return model(x.to(dev, non_blocking=True))


def bench_one(model: nn.Module,
              dev: torch.device,
              bs: int,
              warmup: int,
              repeat: int,
              input_shape: Tuple[int, int, int, int]) -> Tuple[Dict[str, float], float]:
    """
    Returns:
      stats_dict for latency (ms_mean, ms_std, ms_p50, ms_p90, ms_p99),
      images_per_second (float)
    """
    model.eval().to(dev)
    x = torch.randn(bs, *input_shape[1:], device=dev)

    # Warmup
    for _ in range(max(0, warmup)):
        _ = forward_once(model, x, dev)
    _sync(dev)

    # Timed runs
    times_ms: List[float] = []
    t0 = time.perf_counter()
    for _ in range(max(1, repeat)):
        t1 = time.perf_counter()
        _ = forward_once(model, x, dev)
        _sync(dev)
        t2 = time.perf_counter()
        times_ms.append((t2 - t1) * 1000.0)
    total_elapsed = time.perf_counter() - t0

    # Throughput: average across repeats
    img_per_s = (bs * repeat) / total_elapsed if total_elapsed > 0 else float("nan")
    return lat_stats(times_ms), img_per_s


# ----------------------------- Main entry -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Eager (PyTorch) benchmarking with rich metrics.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to .pt/.pth checkpoint (NOT .ts)")
    parser.add_argument("--device", default="cpu", type=str, help="'cpu' or 'gpu/cuda'")
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--repeat", default=100, type=int)
    parser.add_argument("--threads", default=None, type=int, help="CPU threads (pin).")
    parser.add_argument("--batch-sizes", default="1,8,32,128", type=str,
                        help="CSV list of batch sizes for throughput")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "bench_latest.json"), type=str)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("[BENCH] start", flush=True)

    # artifact guard
    if args.checkpoint.lower().endswith(".ts"):
        print("[BENCH][ERROR] TorchScript artifact detected. Use scripts/bench_ts.py for .ts models.", flush=True)
        sys.exit(2)

    dev = parse_device(args.device)
    print(f"[BENCH] requested={args.device} -> using device={dev}", flush=True)

    # Threads pin (CPU)
    if dev.type == "cpu" and args.threads:
        try:
            torch.set_num_threads(int(args.threads))
        except Exception:
            pass
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    ckpt_path = Path(args.checkpoint)
    print(f"[BENCH] loading checkpoint: {ckpt_path}", flush=True)
    ckpt = safe_load_checkpoint(str(ckpt_path))

    # Resolve state_dict + meta
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt)
        meta = ckpt.get("meta", {})
    else:
        state_dict, meta = ckpt, {}

    model_name = meta.get("model_name", "resnet18")
    num_classes = int(meta.get("num_classes", 10))
    seed = meta.get("seed", None)
    if args.verbose:
        print(f"[BENCH] meta={ {'model_name': model_name, 'num_classes': num_classes, 'seed': seed} }")

    # Build model and load weights
    model = build_model(model_name, num_classes)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[BENCH][WARN] load_state_dict(strict=False) failed: {e}", flush=True)

    # Input shape: default CIFAR-10
    input_shape = (1, 3, 32, 32)

    # Precision flags
    try:
        matmul_prec = torch.get_float32_matmul_precision()
    except Exception:
        # Older torch
        matmul_prec = "unknown"

    # MACs / FLOPs
    macs_g = macs_g_with_fvcore(model, input_shape)

    # Energy proxy (CUDA only)
    sampler = PowerSampler() if (dev.type == "cuda" and nvidia_smi_available()) else None

    # Core benchmark
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    # Optional start energy sampling just before timing the first batch
    if sampler:
        sampler.start()
        # tiny delay to get a couple of baseline samples before we begin
        time.sleep(0.1)

    # b1 latency stats & throughput at multiple batch sizes
    stats_b1, img_s_b1 = bench_one(model, dev, bs=1, warmup=args.warmup, repeat=args.repeat, input_shape=input_shape)
    throughput: Dict[int, float] = {1: img_s_b1}
    for bs in batch_sizes:
        if bs == 1:
            continue
        _, img_s = bench_one(model, dev, bs=bs, warmup=args.warmup, repeat=args.repeat, input_shape=input_shape)
        throughput[bs] = img_s

    # Stop energy sampling after all timing is complete
    energy_j = None
    if sampler:
        sampler.stop()
        sampler.join(timeout=1.0)
        energy_j = sampler.energy_joules()

    # Params & checkpoint size
    p_m = params_millions(model)
    size_mb = size_mb_of(ckpt_path)

    # Assemble JSON with the stable schema
    out = {
        "model": model_name,
        "variant": "eager",
        "device": dev.type,
        "threads": int(args.threads) if (dev.type == "cpu" and args.threads) else None,
        "params_millions": round(p_m, 3),
        "size_mb": size_mb,
        "b1_ms": round(stats_b1["ms_mean"], 3),
        "ms_std_b1": round(stats_b1["ms_std"], 3),
        "ms_p50_b1": round(stats_b1["ms_p50"], 3),
        "ms_p90_b1": round(stats_b1["ms_p90"], 3),
        "ms_p99_b1": round(stats_b1["ms_p99"], 3),
        # throughput
        "img_s_b1": round(throughput.get(1, float("nan")), 1) if throughput.get(1) is not None else None,
        "img_s_b8": round(throughput.get(8, float("nan")), 1) if throughput.get(8) is not None else None,
        "img_s_b32": round(throughput.get(32, float("nan")), 1) if throughput.get(32) is not None else None,
        "img_s_b128": round(throughput.get(128, float("nan")), 1) if throughput.get(128) is not None else None,
        # model complexity / energy proxy
        "macs_g_flops": macs_g,
        "energy_proxy_j": round(energy_j, 3) if energy_j is not None else None,
        # flags & run params
        "matmul_precision": matmul_prec,
        "warmup": float(args.warmup),
        "repeat": float(args.repeat),
        "batch_sizes": args.batch_sizes,
    }

    # Write JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[BENCH] wrote {out_path}", flush=True)
    print(json.dumps(out, indent=2))
    print("[BENCH] done", flush=True)


if __name__ == "__main__":
    main()
