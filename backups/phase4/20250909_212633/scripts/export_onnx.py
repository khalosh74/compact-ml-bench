#!/usr/bin/env python3
import argparse, json, sys, time
from pathlib import Path
import torch

from utils.models import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--ts-artifact", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", type=str, required=True)
    args = ap.parse_args()

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    model = None
    source = None

    try:
        if args.checkpoint and args.checkpoint.exists():
            model = build_model((args.arch or "resnet18"), num_classes=10)
            try:
                ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                ckpt = torch.load(args.checkpoint, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(sd, strict=False)
            model.eval()
            source = f"eager/{args.arch}"
        elif args.ts_artifact and args.ts_artifact.exists():
            model = torch.jit.load(str(args.ts_artifact), map_location="cpu").eval()
            source = "torchscript"
        else:
            print(f"[ONNX][SKIP] {args.variant}: no valid input (ckpt or TS) provided", file=sys.stderr)
            return 0

        dummy = torch.randn(1,3,32,32)
        t0 = time.time()
        torch.onnx.export(
            model, dummy, f=out,
            input_names=["input"], output_names=["logits"],
            do_constant_folding=True, opset_version=args.opset,
            dynamic_axes=None
        )
        dur = time.time() - t0
        meta = {
            "variant": args.variant,
            "source": source,
            "opset": int(args.opset),
            "path": str(out),
            "duration_s": dur,
            "timestamp": time.time()
        }
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[ONNX][OK] {args.variant} -> {out} (opset {args.opset})")
        return 0

    except Exception as e:
        print(f"[ONNX][SKIP] {args.variant}: {e}", file=sys.stderr)
        return 0

if __name__ == "__main__":
    sys.exit(main())
