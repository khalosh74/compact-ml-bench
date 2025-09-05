import argparse, os, json, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import csv
from tqdm.auto import tqdm

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_model(name: str, num_classes: int = 10):
    name = name.lower()
    if name == "resnet18":
        return models.resnet18(num_classes=num_classes)
    elif name == "resnet34":
        return models.resnet34(num_classes=num_classes)
    elif name == "mobilenet_v2":
        return models.mobilenet_v2(num_classes=num_classes)
    else:
        return models.resnet18(num_classes=num_classes)

@torch.inference_mode()
def eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(1, total)

def train_one_epoch(model, loader, opt, device, scaler, use_amp, ep, epochs):
    model.train()
    for x, y in tqdm(loader, desc=f"train ep {ep+1}/{epochs}", total=len(loader), dynamic_ncols=True, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        if use_amp:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fractions", type=str, default="0.05,0.1,0.2,0.5,1.0")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--out", type=str, default="outputs/learning_curve.csv")
    p.add_argument("--augment", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if args.augment else []
    tf_train = transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_full = datasets.CIFAR10(root=args.data, train=True, download=True, transform=tf_train)
    test_set   = datasets.CIFAR10(root=args.data, train=False, download=True, transform=tf_test)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    fractions = [float(x) for x in args.fractions.split(",")]
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    rows = []
    for frac in tqdm(fractions, desc="fractions", dynamic_ncols=True):
        n = int(len(train_full) * frac)
        gen = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(train_full), generator=gen)[:n]
        train_subset = Subset(train_full, idx.tolist())
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        model = build_model(args.model, num_classes=10).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        for ep in tqdm(range(args.epochs), desc="epochs", dynamic_ncols=True, leave=False):
            train_one_epoch(model, train_loader, opt, device, scaler, use_amp, ep, args.epochs)

        final_train_acc = eval_acc(model, train_loader, device)
        final_test_acc  = eval_acc(model, test_loader, device)
        gap = final_train_acc - final_test_acc
        rows.append({
            "fraction": frac,
            "train_acc": round(final_train_acc, 2),
            "test_acc": round(final_test_acc, 2),
            "gap_pp": round(gap, 2),
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "augment": bool(args.augment)
        })

    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[LC] wrote {args.out}")
if __name__ == "__main__":
    main()
