#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from utils.models import build_model
from utils.data_transforms import calib_transform_cifar10

def cifar_calib(data_dir: Path, batch: int, workers: int, max_batches: int):
    ds = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=calib_transform_cifar10())
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, persistent_workers=(workers>0), pin_memory=True)
    it = iter(dl)
    for _ in range(max_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        yield x

def size_mb(p: Path):
    try: return round(p.stat().st_size/(1024*1024), 3)
    except: return None

def select_qengine():
    engs = getattr(torch.backends.quantized, "supported_engines", [])
    engs = list(engs) if engs is not None else []
    if "fbgemm" in engs: return "fbgemm"
    if "qnnpack" in engs: return "qnnpack"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--arch", type=str, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--calib-batches", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variant", type=str, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("runs/quantized"))
    args = ap.parse_args()

    # Build & load eager model on CPU
    m = build_model(args.arch, num_classes=10)
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    m.load_state_dict(sd, strict=False)
    m.eval()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    out_ts = Path(outdir) / f"{args.variant}.ts"

    mode = None
    engine = select_qengine()

    try:
        # Try static PTQ via FX if any backend is present
        if engine is None:
            raise RuntimeError("No quant backends compiled in this PyTorch build")

        torch.backends.quantized.engine = engine
        from torch.ao.quantization import get_default_qconfig
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        qconfig = get_default_qconfig(engine)
        prepared = prepare_fx(m, {"": qconfig})

        # Calibrate
        with torch.inference_mode():
            for xb in cifar_calib(args.data_dir, args.batch, args.workers, args.calib_batches):
                prepared(xb)

        quantized = convert_fx(prepared).eval()
        mode = f"ptq_static_{engine}"

        # Export TS (trace; quant ops often unscriptable)
        example = torch.randn(1,3,32,32)
        ts = torch.jit.trace(quantized, example)
        ts.save(str(out_ts))
        print(f"[PTQ] ({mode}) wrote {out_ts} ({size_mb(out_ts)} MB)")

    except Exception as e1:
        try:
            # Fallback: dynamic quant (Linear-only), backend-agnostic in many builds
            from torch.ao.quantization import quantize_dynamic
            dq = quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
            mode = "dynamic_linear_qint8"
            example = torch.randn(1,3,32,32)
            ts = torch.jit.trace(dq, example)
            ts.save(str(out_ts))
            print(f"[PTQ][FALLBACK:{mode}] wrote {out_ts} ({size_mb(out_ts)} MB). Reason: {e1}")
        except Exception as e2:
            # Final fallback: save FP32 TS so pipeline proceeds; clearly label mode
            mode = "fp32_ts_fallback"
            example = torch.randn(1,3,32,32)
            ts = torch.jit.trace(m, example)
            ts.save(str(out_ts))
            print(f"[PTQ][FALLBACK:{mode}] wrote {out_ts} ({size_mb(out_ts)} MB). Reason: {e1} / {e2}")

    metrics = {
        "arch": args.arch,
        "checkpoint": str(args.checkpoint),
        "variant": args.variant,
        "artifact": str(out_ts),
        "size_mb": size_mb(out_ts),
        "calib_batches": args.calib_batches,
        "mode": mode,
        "qengine": engine,
        "timestamp": time.time()
    }
    (Path(outdir)/f"{args.variant}.metrics.json").write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
