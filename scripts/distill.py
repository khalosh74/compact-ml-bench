import argparse, json, os, time, platform, subprocess
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm.auto import tqdm

def make_model(name, num_classes=10):
    n = name.lower()
    if n=="resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if n=="mobilenet_v2":
        m = models.mobilenet_v2(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes); return m
    raise SystemExit(f"Unknown model: {name}")

def load_ckpt_model(ckpt_path, force_name=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    meta = ckpt.get("meta", {})
    name = force_name or meta.get("model_name","resnet18")
    numc = int(meta.get("num_classes", 10))
    m = make_model(name, num_classes=numc)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    return m, meta

def kd_loss_fn(student_logits, teacher_logits, hard_targets, T=4.0, alpha=0.5):
    kl = nn.KLDivLoss(reduction="batchmean")
    s = torch.log_softmax(student_logits / T, dim=1)
    t = torch.softmax(teacher_logits / T, dim=1)
    soft = kl(s, t) * (T*T)
    hard = nn.functional.cross_entropy(student_logits, hard_targets)
    return alpha*hard + (1.0-alpha)*soft

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--student", default="mobilenet_v2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--data", default="data/")
    ap.add_argument("--out", default="runs/distilled")
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
    t_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(), transforms.Normalize(mean,std)])
    t_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])

    train_set = torchvision.datasets.CIFAR10(root=args.data, train=True,  download=False, transform=t_train)
    test_set  = torchvision.datasets.CIFAR10(root=args.data, train=False, download=False, transform=t_test)
    pin = torch.cuda.is_available(); pw = bool(args.num_workers)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, persistent_workers=pw)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin, persistent_workers=pw)

    teacher, tmeta = load_ckpt_model(args.teacher_ckpt)
    teacher.to(device).eval()

    student = make_model(args.student, num_classes=10).to(device)
    opt = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    os.makedirs(args.out, exist_ok=True)
    best_acc, best_path = 0.0, os.path.join(args.out, "best.pt")

    for ep in range(1, args.epochs+1):
        student.train()
        pbar = tqdm(train_loader, desc=f"KD Epoch {ep}/{args.epochs}", leave=False)
        for x,y in pbar:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.inference_mode():
                t_logits = teacher(x)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                s_logits = student(x)
                loss = kd_loss_fn(s_logits, t_logits, y, T=args.T, alpha=args.alpha)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{sched.get_last_lr()[0]:.4f}")
        sched.step()

        # eval
        student.eval(); correct=0; total=0
        with torch.inference_mode():
            for x,y in test_loader:
                x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                pred = student(x).argmax(1)
                correct += (pred==y).sum().item(); total += y.numel()
        acc = 100.0*correct/total
        print(f"[KD {ep:02d}] acc={acc:.2f}%")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"state_dict": student.state_dict(),
                        "meta": {"model_name": args.student, "num_classes": 10, "seed": args.seed,
                                 "teacher_meta": tmeta}}, best_path)

    params_m = sum(p.numel() for p in student.parameters())/1e6
    size_mb = os.path.getsize(best_path)/1e6 if os.path.exists(best_path) else None
    git_rev = subprocess.getoutput("git rev-parse --short HEAD") or "n/a"
    metrics = {"best_acc_top1": round(best_acc,2), "params_millions": round(params_m,3),
               "model_size_mb": round(size_mb,3) if size_mb else None,
               "epochs": args.epochs, "device": device, "torch": torch.__version__,
               "platform": platform.platform(), "git_commit": git_rev}
    with open(os.path.join(args.out,"metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
    print("[KD] Done:", metrics); print("[KD] Best checkpoint ->", best_path)

if __name__ == "__main__":
    main()

