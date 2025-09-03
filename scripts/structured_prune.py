import argparse, json, os, time, platform, subprocess
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm.auto import tqdm

try:
    import torch_pruning as tp
except Exception as e:
    raise SystemExit("torch-pruning is required. Install with: pip install torch-pruning")

def make_model(name, num_classes=10):
    n = name.lower()
    if n=="resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if n=="mobilenet_v2":
        m = models.mobilenet_v2(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes); return m
    raise SystemExit(f"Unknown model {name}")

def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {"model_name":"resnet18","num_classes":10})
    model = make_model(meta.get("model_name","resnet18"), num_classes=int(meta.get("num_classes",10)))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model, meta

def find_classification_head(model, num_classes):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last = model.classifier[-1]
        if isinstance(last, nn.Linear) and getattr(last, "out_features", None) == num_classes:
            return last
    for m in model.modules():
        if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == num_classes:
            return m
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ratio", type=float, default=0.3, help="global channel pruning ratio 0..1")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="runs/structured_pruned")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
    t_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(), transforms.Normalize(mean,std)])
    t_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    pin = torch.cuda.is_available(); pw = bool(args.num_workers)
    train_set = torchvision.datasets.CIFAR10(root=args.data, train=True,  download=False, transform=t_train)
    test_set  = torchvision.datasets.CIFAR10(root=args.data, train=False, download=False, transform=t_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, persistent_workers=pw)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin, persistent_workers=pw)

    model, meta = load_ckpt(args.checkpoint)
    numc = int(meta.get("num_classes", 10))
    model.to(device).eval()

    example_inputs = torch.randn(1,3,32,32, device=device)
    importance = tp.importance.MagnitudeImportance(p=2)
    ignored = []
    head = find_classification_head(model, numc)
    if head is not None:
        ignored.append(head)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        global_pruning=True,
        pruning_ratio=args.ratio,
        ignored_layers=ignored
    )
    pruner.step()

    with torch.inference_mode():
        logits = model(example_inputs)
    out_dim = int(logits.shape[1])
    print(f"[SP] sanity: logits.shape={tuple(logits.shape)} (expect out_dim={numc})")
    if out_dim != numc:
        raise SystemExit(f"[SP][ERROR] classifier out_dim changed to {out_dim}. Head must be ignored.")

    # Fine-tune
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_acc=0.0; os.makedirs(args.out, exist_ok=True)
    best_path=os.path.join(args.out,"best.pt")
    for ep in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"SP Epoch {ep}/{args.epochs}", leave=False)
        for x,y in pbar:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()

        # eval
        model.eval(); correct=0; total=0
        with torch.inference_mode():
            for x,y in test_loader:
                x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                pred = model(x).argmax(1); correct += (pred==y).sum().item(); total += y.numel()
        acc = 100.0*correct/total
        print(f"[SP {ep:02d}] acc={acc:.2f}%")
        if acc>=best_acc:
            best_acc=acc
            torch.save({"state_dict": model.state_dict(),
                        "meta": {"model_name": meta.get("model_name","resnet18"),
                                 "num_classes": numc, "structured_ratio": args.ratio}}, best_path)

    # Save TorchScript (bench-ready, architecture preserved)
    model.eval().to("cpu")
    ts = torch.jit.trace(model, torch.randn(1,3,32,32))
    ts_path = os.path.join(args.out, "structured.ts")
    torch.jit.save(ts, ts_path)

    params_m = sum(p.numel() for p in model.parameters())/1e6
    size_mb = os.path.getsize(best_path)/1e6 if os.path.exists(best_path) else None
    git_rev = subprocess.getoutput("git rev-parse --short HEAD") or "n/a"
    metrics = {"best_acc_top1": round(best_acc,2), "params_millions": round(params_m,3),
               "model_size_mb": round(size_mb,3) if size_mb else None,
               "epochs": args.epochs, "device": device, "torch": torch.__version__,
               "platform": platform.platform(), "git_commit": git_rev, "structured_ratio": args.ratio,
               "artifact_ts": ts_path}
    with open(os.path.join(args.out,"metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
    print("[SP] Done:", metrics); print("[SP] Best checkpoint ->", best_path); print("[SP] TS ->", ts_path)

if __name__ == "__main__":
    main()
