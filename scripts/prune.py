import argparse, json, os, time
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm.auto import tqdm
import torch.nn.utils.prune as prune

def reconstruct(meta):
    name = meta.get("model_name","resnet18").lower()
    num = int(meta.get("num_classes", 10))
    if name=="resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, num); return m
    if name=="mobilenet_v2":
        m = models.mobilenet_v2(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num); return m
    raise SystemExit(f"Unknown model {name}")

def count_zeros(model):
    total = 0; zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += torch.count_nonzero(p==0).item()
    return 100.0*zeros/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--amount", type=float, default=0.5, help="global sparsity 0..1")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="runs/pruned")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    mean=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
    t_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(), transforms.Normalize(mean,std)])
    t_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    train_set = torchvision.datasets.CIFAR10(root=args.data, train=True,  download=False, transform=t_train)
    test_set  = torchvision.datasets.CIFAR10(root=args.data, train=False, download=False, transform=t_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    test_loader  = DataLoader(test_set,  batch_size=256,           shuffle=False, num_workers=args.num_workers)

    # Load baseline
    os.makedirs(args.out, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")  # our own file; warning ok
    meta = ckpt.get("meta", {"model_name":"resnet18","num_classes":10})
    model = reconstruct(meta)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # Global unstructured L1 pruning over conv/linear weights
    params_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params_to_prune.append((module, "weight"))
    prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=args.amount)

    # Make pruning permanent (remove reparam buffers)
    for m, _ in params_to_prune:
        prune.remove(m, "weight")

    sparsity = count_zeros(model)
    print(f"[PRUNE] target={args.amount*100:.1f}% | observed={sparsity:.2f}% zeros")

    # Fine-tune
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_path = os.path.join(args.out, "best.pt")
    t0 = time.time()

    for epoch in range(1, args.epochs+1):
        ep0 = time.time()
        model.train()
        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch}/{args.epochs}", leave=False)
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{sched.get_last_lr()[0]:.4f}")
        sched.step()

        # Eval
        model.eval(); correct=0; total=0
        with torch.inference_mode():
            for x,y in test_loader:
                x,y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.numel()
        acc = 100.0*correct/total
        dt = time.time()-ep0
        print(f"[FT {epoch:02d}] acc={acc:.2f}% | epoch_time={dt:.1f}s")

        if acc >= best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(),
                        "meta": {**meta, "sparsity_percent": sparsity}}, best_path)

    # Write metrics
    params_m = sum(p.numel() for p in model.parameters())/1e6
    size_mb = os.path.getsize(best_path)/1e6 if os.path.exists(best_path) else None
    metrics = {
        "post_prune_observed_sparsity_percent": round(sparsity,2),
        "best_acc_top1": round(best_acc,2),
        "params_millions": round(params_m,3),
        "model_size_mb": round(size_mb,3) if size_mb else None,
        "epochs": args.epochs,
        "elapsed_sec": round(time.time()-t0,1)
    }
    with open(os.path.join(args.out,"metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[PRUNE] Done:", json.dumps(metrics, indent=2))
    print("[PRUNE] Best checkpoint ->", best_path)

if __name__ == "__main__":
    main()
