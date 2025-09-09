#!/usr/bin/env python3
import argparse, json, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--variant", type=str, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("runs/onnx_quant"))
    ap.add_argument("--calib-batches", type=int, default=64)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    try:
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
        import onnx
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets
        from utils.data_transforms import calib_transform_cifar10

        class CIFARReader(CalibrationDataReader):
            def __init__(self, data_dir, input_name, batches):
                ds = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=calib_transform_cifar10())
                self.dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=2)
                self.batches = batches
                self.input_name = input_name
                self.it = iter(self.dl)
                self.count = 0
            def get_next(self):
                if self.count >= self.batches: return None
                try:
                    x,_ = next(self.it); self.count += 1
                    return { self.input_name: x.numpy().astype(np.float32) }
                except StopIteration:
                    return None

        onnx_model = onnx.load(str(args.onnx))
        input_name = onnx_model.graph.input[0].name
        args.outdir.mkdir(parents=True, exist_ok=True)
        out_path = args.outdir / f"{args.variant}.onnx"

        dr = CIFARReader(args.data_dir, input_name, args.calib_batches)
        quantize_static(
            model_input=str(args.onnx),
            model_output=str(out_path),
            calibration_data_reader=dr,
            activation_type=QuantType.Uint8,
            weight_type=QuantType.QInt8,
            per_channel=True
        )
        (args.outdir/f"{args.variant}.metrics.json").write_text(json.dumps({
            "variant": args.variant, "mode": "ort_static_int8", "calib_batches": args.calib_batches,
            "path": str(out_path), "timestamp": time.time()
        }, indent=2))
        print(f"[ORT-PTQ] wrote {out_path}")
        return 0

    except Exception as e:
        print(f"[ORT-PTQ][SKIP] Quantization unavailable or failed: {e}")
        return 0

if __name__ == "__main__":
    sys.exit(main())
