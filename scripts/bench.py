# scripts/bench.py
import argparse, json, os, time, io, contextlib, warnings, subprocess, threading
from pathlib import Path

import numpy as np
import torch
from torchvision import models


# -------------------- helpers --------------------
def parse_device(arg: str) -> torch.device:
    if arg.lower() in {"gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_batch_sizes(s: str | None) -> list[int]:
    if not s:
        return [1, 8, 32, 128]
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out or [1, 8, 32, 128]


def safe_load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # PyTorch >= 2.4
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(model_name: str, num_classes: int):
    name = (model_name or "resnet18").lower()
    if name == "resnet18":
        return models.resnet18(num_classes=num_classes)
    if name == "resnet34":
        return models.resnet34(num_classes=num_classes)
    if name in {"mobilenet_v2", "mobilenetv2"}:
        return models.mobilenet_v2(num_classes=num_classes)
    # fallback
    return models.resnet18(num_classes=num_classes)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---- FLOPs/MACs (fvcore); silence stdout/stderr noise ----
def macs_g_with_fvcore(model: torch.nn.Module, sample: torch.Tensor) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.inference_mode():
                fca = FlopCountAnalysis(model, sample)
                flops = fca.total()  # FLOPs
        return flops / 1e9
    except Exception:
        return None


# ---- power sampler (nvidia-smi) ----
class RepeatingSampler(threading.Thread):
    def __init__(self, fn, interval_s: float = 0.1):
        super().__init__(daemon=True)
        self.fn = fn
        self.dt = interval_s
        self.values: list[tuple[float, float]] = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                v = self.fn()
                if v is not None:
                    self.values.append((time.perf_counter(), float(v)))
            except Exception:
                pass
            time.sleep(self.dt)

    def stop(self):
        self._halt.set()


def nvidia_power_w() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "--id=0"],
            stderr=subprocess.DEVNULL, text=True, timeout=0.6
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


# ---- core bench ----
def lat_stats(ms_list: list[float]) -> dict:
    arr = np.asarray(ms_list, dtype=np.float64)
    if arr.size == 0:
        return dict(ms_mean=None, ms_std=None, ms_p50=None, ms_p90=None, ms_p99=None)
    return dict(
        ms_mean=float(arr.mean()),
        ms_std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        ms_p50=float(np.percentile(arr, 50)),
        ms_p90=float(np.percentile(arr, 90)),
        ms_p99=float(np.percentile(arr, 99)),
    )


def bench_one(model: torch.nn.Module, device: torch.device, *, bs: int, warmup: int, repeat: int) -> tuple[dict, float | None]:
    sample = torch.randn(bs, 3, 32, 32, device=device)
    model.to(device).eval()

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(sample)
        _sync()

        times_ms: list[float] = []
        for _ in range(repeat):
            t1 = time.perf_counter()
            _ = model(sample)
            _sync()
            times_ms.append((time.perf_counter() - t1) * 1000.0)

    s = lat_stats(times_ms)
    img_s = (bs / (s["ms_mean"] / 1000.0)) if s["ms_mean"] else None
    return s, img_s


# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser("Compact-ML Bench (eager PyTorch)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (only if --device cpu)")
    parser.add_argument("--batch-sizes", type=str, default="1,8,32,128",
                        help="Comma-separated batch sizes for throughput, e.g. '1,8,32,128'")
    parser.add_argument("--out", type=str, default="outputs/bench_latest.json")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path.lower().endswith(".ts"):
        print("[BENCH][ERROR] TorchScript artifact detected. Use scripts/bench_ts.py for .ts models.", flush=True)
        raise SystemExit(2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[BENCH] start")

    # device & threads
    dev = parse_device(args.device)
    if args.device == "gpu" and dev.type != "cuda":
        print("[BENCH] requested=gpu -> CUDA not available; falling back to CPU", flush=True)
    print(f"[BENCH] requested={args.device} -> using device={dev.type}", flush=True)
    if dev.type == "cpu" and args.threads:
        try:
            torch.set_num_threads(int(args.threads))
        except Exception:
            pass

    # load checkpoint
    try:
        ckpt = safe_load_checkpoint(ckpt_path)
    except FileNotFoundError as e:
        print(f"[BENCH][ERROR] {e}", flush=True)
        raise

    # resolve state_dict/meta
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = ckpt.get("meta", {})
    else:
        state_dict, meta = ckpt, {}

    model_name = str(meta.get("model_name", "resnet18"))
    num_classes = int(meta.get("num_classes", 10))
    if args.verbose:
        print(f"[BENCH] meta={repr(meta)}", flush=True)

    # build model + load weights
    model = build_model(model_name, num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if args.verbose:
        if missing:
            print(f"[BENCH][WARN] missing keys: {len(missing)} (e.g. {missing[:5]})")
        if unexpected:
            print(f"[BENCH][WARN] unexpected keys: {len(unexpected)} (e.g. {unexpected[:5]})")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    except Exception:
        size_mb = None

    bs_list = parse_batch_sizes(args.batch_sizes)

    # MACs once
    sample = torch.randn(1, 3, 32, 32, device=dev)
    macs_g = macs_g_with_fvcore(model, sample)

    # precision flag
    try:
        matmul_prec = torch.get_float32_matmul_precision()
    except Exception:
        matmul_prec = None

    # latency @B=1 + energy proxy
    p_sam = RepeatingSampler(nvidia_power_w, 0.1)
    p_sam.start()
    s_b1, img_s_b1 = bench_one(model, dev, bs=1, warmup=args.warmup, repeat=args.repeat)
    p_sam.stop(); p_sam.join(timeout=1.0)

    energy_j = None
    if s_b1["ms_mean"] and p_sam.values:
        mean_w = sum(v for _, v in p_sam.values) / len(p_sam.values)
        duration_s = (args.repeat * s_b1["ms_mean"] / 1000.0)
        energy_j = mean_w * duration_s

    # throughput @other batch sizes
    img_s_by_batch: dict[int, float | None] = {}
    for B in bs_list:
        sB, img_sB = bench_one(
            model, dev, bs=B,
            warmup=max(5, args.warmup // 4),
            repeat=max(20, args.repeat // 10),
        )
        img_s_by_batch[B] = img_sB

    out = {
        # identity
        "model": model_name,
        "variant": "eager",
        "device": "cuda" if dev.type == "cuda" else "cpu",
        "threads": int(args.threads) if (dev.type == "cpu" and args.threads) else None,

        # static meta
        "params_millions": round(params_m, 3) if params_m is not None else None,
        "size_mb": round(size_mb, 3) if size_mb is not None else None,

        # latency stats @B=1
        "b1_ms": round(s_b1["ms_mean"], 3) if s_b1["ms_mean"] is not None else None,
        "ms_std_b1": round(s_b1["ms_std"], 3) if s_b1["ms_std"] is not None else None,
        "ms_p50_b1": round(s_b1["ms_p50"], 3) if s_b1["ms_p50"] is not None else None,
        "ms_p90_b1": round(s_b1["ms_p90"], 3) if s_b1["ms_p90"] is not None else None,
        "ms_p99_b1": round(s_b1["ms_p99"], 3) if s_b1["ms_p99"] is not None else None,

        # throughput
        "img_s_b1": round(img_s_b1, 1) if img_s_b1 else None,
        "img_s_b8": round(img_s_by_batch.get(8), 1) if img_s_by_batch.get(8) else None,
        "img_s_b32": round(img_s_by_batch.get(32), 1) if img_s_by_batch.get(32) else None,
        "img_s_b128": round(img_s_by_batch.get(128), 1) if img_s_by_batch.get(128) else None,

        # MACs + energy proxy
        "macs_g_flops": round(macs_g, 3) if macs_g else None,
        "energy_proxy_j": round(energy_j, 3) if energy_j else None,

        # env/flags
        "matmul_precision": matmul_prec,
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "batch_sizes": ",".join(str(b) for b in bs_list),
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[BENCH] wrote {out_path}")
    print(json.dumps(out, indent=2))
    print("[BENCH] done")


if __name__ == "__main__":
    main()
