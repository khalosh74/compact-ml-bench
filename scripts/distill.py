#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from utils.models import build_model
from utils.data_transforms import train_transform_cifar10, test_transform_cifar10

def make_loaders(data_dir: Path, batch: int, workers: int):
    train = datasets.CIFAR10(root=data_dir, train=True,  download=False, transform=train_transform_cifar10())
    test  = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform_cifar10())
    dl_train = DataLoader(train, batch_size=batch, shuffle=True,  num_workers=workers, persistent_workers=(workers>0), pin_memory=True)
    dl_test  = DataLoader(test,  batch_size=1024,  shuffle=False, num_workers=min(4,workers), persistent_workers=False, pin_memory=True)
    return dl_train, dl_test

@torch.no_grad()
def eval_top1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total=0; correct=0
    for x,y in loader:
        x=x.to(device,non_blocking=True); y=y.to(device)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total   += y.numel()
    return 100.0*correct/total

def kd_loss_fn(student_logits, teacher_logits, target, alpha: float, T: float):
    ce = F.cross_entropy(student_logits, target)
    # KLDiv with temperature
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(p_s, p_t, reduction="batchmean") * (T*T)
    return (1.0 - alpha) * ce + alpha * kd

def train_kd(args):
    device = torch.device("cuda" if (args.device in ("gpu","cuda") and torch.cuda.is_available()) else "cpu")

    # Teacher
    teacher_arch = (args.teacher_arch or "resnet34").lower()
    student_arch = (args.student_arch or "mobilenet_v2").lower()

    # If teacher ckpt missing, try to use baseline as fallback; otherwise untrained teacher (warn)
    teacher = build_model(teacher_arch, num_classes=10).to(device).eval()
    if args.teacher_ckpt and Path(args.teacher_ckpt).exists():
        try:
            ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        teacher.load_state_dict(sd, strict=False)
        print(f"[KD] Loaded teacher weights from {args.teacher_ckpt}")
    else:
        fallback = Path("runs/baseline/best.pt")
        if fallback.exists() and teacher_arch != "resnet34":
            try:
                ckpt = torch.load(fallback, map_location="cpu", weights_only=True)
            except TypeError:
                ckpt = torch.load(fallback, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            teacher.load_state_dict(sd, strict=False)
            print(f"[KD][WARN] Using baseline/best.pt as teacher weights for arch={teacher_arch}")
        else:
            print("[KD][WARN] No teacher checkpoint provided; teacher may be untrained.")

    # Student
    student = build_model(student_arch, num_classes=10).to(device)

    dl_train, dl_test = make_loaders(args.data_dir, args.batch, args.workers)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    opt = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs*len(dl_train))

    best_acc = -1.0
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        student.train()
        for xb, yb in dl_train:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                with torch.no_grad():
                    logits_t = teacher(xb)
                logits_s = student(xb)
                loss = kd_loss_fn(logits_s, logits_t, yb, alpha=args.alpha, T=args.temperature)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()

        acc = eval_top1(student, dl_test, device)
        if acc > best_acc:
            best_acc = acc
            # save eager student
            torch.save(student.state_dict(), outdir/"best.pt")
        print(f"[KD] epoch={epoch}/{args.epochs} acc_top1={acc:.2f}% best={best_acc:.2f}%")

    metrics = {
        "student_arch": student_arch,
        "teacher_arch": teacher_arch,
        "epochs": args.epochs,
        "acc_top1_eval": best_acc,
        "duration_s": time.time()-t0
    }
    (outdir/"metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[KD] done -> {outdir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--device", choices=["cpu","gpu","cuda"], default="gpu")
    ap.add_argument("--teacher-arch", type=str, default="resnet34")
    ap.add_argument("--teacher-ckpt", type=str, default=None)
    ap.add_argument("--student-arch", type=str, default="mobilenet_v2")
    ap.add_argument("--epochs", type=int, default=2)  # smoke default
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--outdir", type=Path, default=Path("runs/kd_mobilenetv2"))
    args = ap.parse_args()
    train_kd(args)

if __name__ == "__main__":
    main()
