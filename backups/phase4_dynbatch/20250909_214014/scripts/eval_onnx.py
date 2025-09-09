#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.data_transforms import test_transform_cifar10

def make_loader(data_dir: Path, batch: int, workers: int):
    ds = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform_cifar10())
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, persistent_workers=False, pin_memory=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu","gpu","cuda"], default="cpu")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variant", type=str, required=True)
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"[ORT][SKIP] onnxruntime not available: {e}")
        return 0

    providers = ort.get_available_providers()
    want_gpu = args.device in ("gpu","cuda")
    use_gpu = (want_gpu and "CUDAExecutionProvider" in providers)
    sess = ort.InferenceSession(str(args.onnx), providers=(["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]))
    input_name = sess.get_inputs()[0].name

    dl = make_loader(args.data_dir, args.batch, args.workers)
    correct=0; total=0
    for xb, yb in dl:
        x = xb.numpy().astype(np.float32)
        logits = sess.run(None, {input_name: x})[0]
        pred = logits.argmax(axis=1)
        y = yb.numpy()
        correct += int((pred==y).sum())
        total   += int(y.shape[0])
    acc = 100.0 * (correct/total)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    (Path("outputs")/f"acc_{args.variant}.json").write_text(json.dumps({"variant": args.variant, "acc_top1": float(acc)}))
    print(f"[ORT][EVAL] {args.variant}: acc_top1={acc:.2f}% -> outputs/acc_{args.variant}.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
