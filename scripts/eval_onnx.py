#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

try:
    from utils.data_transforms import test_transform_cifar10
except Exception:
    def test_transform_cifar10():
        import torchvision.transforms as T
        return T.Compose([T.ToTensor()])

def get_static_batch(session) -> int | None:
    try:
        shp = session.get_inputs()[0].shape
        bs = shp[0]
        return int(bs) if isinstance(bs, int) and bs > 0 else None
    except Exception:
        return None

@torch.no_grad()
def evaluate(sess, input_name, device: str):
    # device arg is kept for symmetry; ORT session determines device via providers
    ds = datasets.CIFAR10(root="data", train=False, download=False, transform=test_transform_cifar10())
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=2, pin_memory=False)
    static_bs = get_static_batch(sess)

    total = 0; correct = 0
    for xb, yb in dl:
        x = xb.numpy().astype(np.float32)  # NCHW float32
        y = yb.numpy()
        if static_bs is None:
            # dynamic batch: one shot
            logits = sess.run(None, {input_name: x})[0]
            pred = logits.argmax(axis=1)
            correct += (pred == y).sum()
            total   += y.size
        else:
            # need to chunk to static_bs
            n = x.shape[0]
            start = 0
            while start < n:
                end = min(start + static_bs, n)
                logits = sess.run(None, {input_name: x[start:end]})[0]
                pred = logits.argmax(axis=1)
                correct += (pred == y[start:end]).sum()
                total   += (end - start)
                start = end
    return 100.0 * (correct / total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu","gpu"], default="cpu")
    ap.add_argument("--variant", type=str, required=True)
    args = ap.parse_args()

    if args.device == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider"]
    sess = ort.InferenceSession(str(args.onnx), providers=providers)
    input_name = sess.get_inputs()[0].name

    acc = evaluate(sess, input_name, args.device)
    out = Path("outputs") / f"acc_{args.variant}.json"
    out.write_text(json.dumps({"variant": args.variant, "acc_top1": round(acc, 2)}, indent=2))
    print(f"[EVAL-ONNX] {args.variant}: acc_top1={acc:.2f}% -> {out}")

if __name__ == "__main__":
    main()
