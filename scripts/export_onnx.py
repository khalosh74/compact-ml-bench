#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import onnx

try:
    from utils.models import build_model
except Exception:
    build_model = None

def _opsets(p: Path):
    try:
        m = onnx.load(str(p))
        return [im.version for im in m.opset_import]
    except Exception:
        return None

def export_eager(arch: str, ckpt: Path, out: Path, opset: int):
    if build_model is None:
        raise SystemExit("utils.models.build_model not found")
    m = build_model(arch, num_classes=10).eval()
    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    m.load_state_dict(sd, strict=False)
    ex = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        m, ex, str(out),
        input_names=["input"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,   # <— legacy exporter; supports dynamic_axes cleanly
    )
    return out

def export_ts(ts_path: Path, out: Path, opset: int):
    mod = torch.jit.load(str(ts_path), map_location="cpu").eval()
    ex = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        mod, ex, str(out),
        input_names=["input"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,   # <— critical for ScriptModule
    )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--ts-artifact", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", type=str, default=None)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.arch and args.checkpoint:
        export_eager(args.arch, args.checkpoint, args.out, args.opset)
    elif args.ts_artifact:
        export_ts(args.ts_artifact, args.out, args.opset)
    else:
        raise SystemExit("Provide either --arch+--checkpoint OR --ts-artifact")

    ops = _opsets(args.out)
    name = args.variant or args.out.stem
    if ops: print(f"[ONNX][OK] {name} -> {args.out} (opsets {ops})")
    else:   print(f"[ONNX][OK] {name} -> {args.out}")

if __name__ == "__main__":
    main()
