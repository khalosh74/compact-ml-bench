import argparse, json, os, time
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm.auto import tqdm

def top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item()*100.0

def get_model(name, num_classes):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise SystemExit(f"Unknown model: {name}")

def fmt(secs: float) -> str:
    secs = int(secs)
    m, s = divmod(secs, 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18")
    p.add_argument("--data", default="data/")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", default="runs/baseline")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    t_train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])
    t_test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = torchvision.datasets.CIFAR10(root=args.data, train=True, download=False, transform=t_train_tf)
    test_set  = torchvision.datasets.CIFAR10(root=args.data, train=False, download=False, transform=t_test_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    model = get_model(args.model, num_classes=10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.out, exist_ok=True)
    best_acc, best_path = 0.0, os.path.join(args.out, "best.pt")

    global_t0 = time.time()
    epoch_times = []

    for epoch in range(1, args.epochs+1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            avg_loss = running_loss / (pbar.n * x.size(0) if pbar.n>0 else 1)
            pbar.set_postfix(loss=f"{avg_loss:.3f}", lr=f"{sched.get_last_lr()[0]:.4f}")

        sched.step()
        epoch_time = time.time() - ep_t0
        epoch_times.append(epoch_time)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining = avg_epoch * (args.epochs - epoch)

        # Eval
        model.eval()
        correct, total, ev_loss = 0, 0, 0.0
        with torch.inference_mode():
            for x,y in test_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                ev_loss += loss_fn(logits, y).item()*x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.numel()
        acc = 100.0*correct/total
        print(f"[EPOCH {epoch:02d}] acc={acc:.2f}% | epoch_time={fmt(epoch_time)} | ETA_total={fmt(remaining)}")

        if acc >= best_acc:
            best_acc = acc
            torch.save({
                "state_dict": model.state_dict(),
                "meta": {"model_name": args.model, "num_classes": 10, "seed": args.seed}
            }, best_path)

    params = sum(p.numel() for p in model.parameters())/1e6
    model_size_mb = os.path.getsize(best_path)/1e6 if os.path.exists(best_path) else None
    metrics = {
        "best_acc_top1": best_acc,
        "params_millions": round(params,3),
        "model_size_mb": round(model_size_mb,3) if model_size_mb else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "elapsed_sec": round(time.time()-global_t0,1),
        "avg_epoch_sec": round(sum(epoch_times)/len(epoch_times),1)
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[TRAIN] Done:", metrics)
    print("[TRAIN] Best checkpoint ->", best_path)

if __name__ == "__main__":
    main()
