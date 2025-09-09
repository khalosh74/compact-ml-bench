#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.models import build_model
from utils.data_transforms import test_transform_cifar10

def cifar10_loader(data_dir: Path, batch: int, workers: int):
    ds = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform_cifar10())
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)

def load_eager(checkpoint: Path, arch: str, device: torch.device):
    m = build_model(arch, num_classes=10)
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    return m

def load_ts(artifact: Path, device: torch.device):
    m = torch.jit.load(str(artifact), map_location=device)
    m.eval()
    return m

def accuracy(model, loader, device):
    correct = 0; total = 0
    with torch.inference_mode():
        for x,y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).argmax(1).cpu()
            correct += (pred==y).sum().item()
            total += y.numel()
    return 100.0 * correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--device", choices=["cpu","gpu","cuda"], default="gpu")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variant", type=str, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--arch", type=str, help="required with --checkpoint")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dev = torch.device("cuda") if args.device in ("gpu","cuda") and torch.cuda.is_available() else torch.device("cpu")
    if args.checkpoint and args.artifact:
        raise SystemExit("Provide exactly one of --checkpoint OR --artifact")
    if args.checkpoint and not args.arch:
        raise SystemExit("--arch is required when using --checkpoint")

    loader = cifar10_loader(args.data_dir, args.batch, args.workers)
    if args.checkpoint:
        model = load_eager(args.checkpoint, args.arch, dev); model_name = args.arch
    elif args.artifact:
        model = load_ts(args.artifact, dev); model_name = Path(args.artifact).stem
    else:
        raise SystemExit("Missing --checkpoint or --artifact")

    acc = accuracy(model, loader, dev)
    out = args.out or Path("outputs")/f"acc_{args.variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "variant": args.variant,
        "model": model_name,
        "device": "gpu" if dev.type == "cuda" else "cpu",
        "acc_top1": acc,
        "timestamp": time.time(),
    }, open(out,"w"))
    print(f"[EVAL] {args.variant}: acc_top1={acc:.2f}% -> {out}")

if __name__ == "__main__":
    main()
