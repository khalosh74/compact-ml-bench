import argparse, csv, json, math, os, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm

def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def logu(a,b):
    import math, random
    return math.exp(random.uniform(math.log(a), math.log(b)))

def build_model(name:str, num_classes:int=10):
    name = name.lower()
    if name == "resnet18":   return models.resnet18(num_classes=num_classes)
    if name == "resnet34":   return models.resnet34(num_classes=num_classes)
    if name == "mobilenet_v2": return models.mobilenet_v2(num_classes=num_classes)
    return models.resnet18(num_classes=num_classes)

def make_loaders(data_dir, batch_size, workers, seed, fraction=1.0, augment=True):
    mean,std = (0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
    tf_train = transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(mean,std)])
    tf_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf_train)
    test_set   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf_test)

    val_size = 5000
    train_size = len(full_train) - val_size
    g = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(full_train, [train_size, val_size], generator=g)

    if fraction < 1.0:
        n = max(1, int(len(train_split) * fraction))
        idx = torch.randperm(len(train_split), generator=g)[:n]
        train_split = Subset(train_split, idx.tolist())

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_split,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_set,    batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader

@torch.inference_mode()
def eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for x,y in loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total   += y.numel()
    return 100.0 * correct / max(1,total)

def train_epochs(model, train_loader, epochs, device, lr, wd, label_smoothing, use_amp=True):
    import torch.nn.functional as F
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and (device=='cuda'))
    model.train()
    for ep in tqdm(range(epochs), desc="epochs", dynamic_ncols=True, leave=False):
        for x,y in tqdm(train_loader, desc=f"train ep {ep+1}/{epochs}", total=len(train_loader),
                        dynamic_ncols=True, leave=False):
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp and (device=='cuda')):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()

def save_checkpoint(model, out_dir, model_name="resnet18", num_classes=10):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {"state_dict": model.state_dict(), "meta": {"model_name": model_name, "num_classes": num_classes}}
    torch.save(ckpt, Path(out_dir)/"best.pt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--train-fraction", type=float, default=0.5)
    ap.add_argument("--model", type=str, default="resnet18")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--out", type=str, default="outputs/sweep_baseline.csv")
    ap.add_argument("--train-finalists", action="store_true")
    ap.add_argument("--final-epochs", type=int, default=40)
    ap.add_argument("--promote-topk", type=int, default=3)
    ap.add_argument("--gap-threshold", type=float, default=3.0)
    ap.add_argument("--runs-dir", type=str, default="runs")
    args = ap.parse_args()

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    rows = []
    if args.budget > 0:
        for i in tqdm(range(args.budget), desc="sweep configs", dynamic_ncols=True):
            lr  = logu(0.01, 0.2)
            wd  = logu(1e-5, 1e-3)
            ls  = random.uniform(0.0, 0.1)
            aug = random.choice([True, False])

            train_loader, val_loader, _ = make_loaders(args.data, args.batch_size, args.num_workers,
                                                       args.seed+i, args.train_fraction, aug)
            model = build_model(args.model, num_classes=10).to(device)

            t0 = time.perf_counter()
            train_epochs(model, train_loader, args.epochs, device, lr, wd, ls, use_amp=True)
            dt = time.perf_counter() - t0

            train_acc = eval_acc(model, train_loader, device)
            val_acc   = eval_acc(model, val_loader, device)
            gap = train_acc - val_acc

            row = {
                "trial": i+1, "model": args.model,
                "lr": round(lr,6), "weight_decay": round(wd,8),
                "label_smoothing": round(ls,4), "augment": aug,
                "epochs": args.epochs, "train_fraction": args.train_fraction,
                "train_acc": round(train_acc,2), "val_acc": round(val_acc,2),
                "gap_pp": round(gap,2), "time_sec": round(dt,1), "device": device
            }
            rows.append(row)

        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    if args.train_finalists:
        import pandas as pd
        if not rows:
            rows = pd.read_csv(args.out).to_dict("records")
        rows_sorted = sorted(rows, key=lambda r: (-r["val_acc"], r["gap_pp"]))
        finalists = [r for r in rows_sorted if r["gap_pp"] <= args.gap_threshold][:args.promote_topk]

        for j, r in tqdm(list(enumerate(finalists, start=1)), desc="promote finalists", dynamic_ncols=True):
            train_loader, val_loader, test_loader = make_loaders(
                args.data, args.batch_size, args.num_workers, args.seed+100+j,
                fraction=1.0, augment=bool(r["augment"])
            )
            model = build_model(args.model, num_classes=10).to(device)
            train_epochs(model, train_loader, args.final_epochs, device,
                         lr=float(r["lr"]), wd=float(r["weight_decay"]),
                         label_smoothing=float(r["label_smoothing"]), use_amp=True)
            train_acc = eval_acc(model, train_loader, device)
            val_acc   = eval_acc(model, val_loader, device)
            test_acc  = eval_acc(model, test_loader, device)
            run_name  = f"{args.model}_tuned_{j}"
            out_dir   = Path(args.runs_dir)/run_name
            save_checkpoint(model, out_dir, model_name=args.model, num_classes=10)
            print(f"[FINAL] {run_name}: test={test_acc:.2f}% (val={val_acc:.2f}%, gap={train_acc-val_acc:.2f} pp) -> {out_dir/'best.pt'}")

if __name__ == "__main__":
    main()
