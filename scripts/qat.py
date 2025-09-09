#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from utils.models import build_model
from utils.data_transforms import train_transform_cifar10, test_transform_cifar10

def make_loaders(data_dir: Path, batch: int, workers: int):
    train = datasets.CIFAR10(root=data_dir, train=True,  download=False, transform=train_transform_cifar10())
    test  = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform_cifar10())
    dl_train = DataLoader(train, batch_size=batch, shuffle=True,  num_workers=workers, persistent_workers=(workers>0), pin_memory=True)
    dl_test  = DataLoader(test,  batch_size=1024, shuffle=False, num_workers=min(4,workers), persistent_workers=False, pin_memory=True)
    return dl_train, dl_test

@torch.no_grad()
def eval_top1(model, loader, device):
    model.eval()
    tot=0; cor=0
    for x,y in loader:
        x=x.to(device,non_blocking=True); y=y.to(device)
        cor += (model(x).argmax(1)==y).sum().item(); tot += y.numel()
    return 100.0*cor/tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--arch", type=str, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)  # smoke default
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--variant", type=str, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("runs/quantized"))
    args = ap.parse_args()

    torch.backends.quantized.engine = "fbgemm"
    # Build & load eager (CUDA for speed during QAT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(args.arch, num_classes=10)
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    m.load_state_dict(sd, strict=False)
    m.train().to(device)

    # FX QAT
    from torch.ao.quantization import get_default_qat_qconfig
    from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
    qconfig = get_default_qat_qconfig("fbgemm")
    qat_model = prepare_qat_fx(m, {"": qconfig}).to(device).train()

    dl_train, dl_test = make_loaders(args.data_dir, args.batch, args.workers)
    opt = torch.optim.Adam(qat_model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best = -1.0
    for epoch in range(1, args.epochs+1):
        for xb,yb in dl_train:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = qat_model(xb)
                loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        acc = eval_top1(qat_model.eval(), dl_test, device)
        print(f"[QAT] epoch={epoch}/{args.epochs} acc_top1={acc:.2f}%"); qat_model.train()
        best = max(best, acc)

    # Convert to INT8 (CPU) and export TS (trace)
    qat_model.cpu().eval()
    int8_model = convert_fx(qat_model).eval()
    example = torch.randn(1,3,32,32)
    ts = torch.jit.trace(int8_model, example)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_ts = outdir / f"{args.variant}.ts"
    ts.save(str(out_ts))

    (outdir/f"{args.variant}.metrics.json").write_text(json.dumps({
        "arch": args.arch, "checkpoint": str(args.checkpoint),
        "variant": args.variant, "acc_top1_eval": best
    }, indent=2))
    print(f"[QAT] wrote {out_ts}")

if __name__ == "__main__":
    main()
