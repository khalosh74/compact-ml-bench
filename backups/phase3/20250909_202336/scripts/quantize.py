import argparse, json, os
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig, QConfigMapping

def reconstruct(model_name, num_classes=10):
    n=model_name.lower()
    if n=="resnet18":
        m=models.resnet18(weights=None); m.fc=torch.nn.Linear(m.fc.in_features,num_classes); return m
    if n=="mobilenet_v2":
        m=models.mobilenet_v2(weights=None); m.classifier[-1]=torch.nn.Linear(m.classifier[-1].in_features,num_classes); return m
    raise SystemExit(f"Unknown model {model_name}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--data", default="data/")
    ap.add_argument("--calib-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="runs/quantized")
    args=ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt=torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    meta=ckpt.get("meta", {})
    model_name=(args.model_name or meta.get("model_name") or "resnet18")
    numc=int(meta.get("num_classes",10))

    fp32=reconstruct(model_name, num_classes=numc)
    fp32.load_state_dict(ckpt["state_dict"], strict=True)
    fp32.eval()

    mean=(0.4914,0.4822,0.4465); std=(0.2470,0.2435,0.2616)
    t=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    calib=datasets.CIFAR10(root=args.data, train=True, download=False, transform=t)
    calib_loader=DataLoader(calib, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=False)
    test=datasets.CIFAR10(root=args.data, train=False, download=False, transform=t)
    test_loader=DataLoader(test, batch_size=512, shuffle=False,
                           num_workers=args.num_workers, pin_memory=False)

    torch.backends.quantized.engine="fbgemm"
    qmap=QConfigMapping().set_global(get_default_qconfig("fbgemm"))
    example_inputs=(torch.randn(1,3,32,32),)
    prepared=prepare_fx(fp32, qmap, example_inputs=example_inputs)

    with torch.inference_mode():
        for i,(x,_) in enumerate(calib_loader):
            prepared(x)
            if i+1>=args.calib_batches: break

    quantized=convert_fx(prepared)

    # Accuracy check (in-process)
    correct=0; total=0
    with torch.inference_mode():
        for x,y in test_loader:
            pred=quantized(x).argmax(1)
            correct+=(pred==y).sum().item(); total+=y.numel()
    acc=100.0*correct/total

    # Save TorchScript artifact (robust across envs)
    ts_path=os.path.join(args.out,"quantized_fx.ts")
    try:
        scripted=torch.jit.script(quantized)
    except Exception:
        scripted=torch.jit.trace(quantized, torch.randn(1,3,32,32))
    torch.jit.save(scripted, ts_path)

    size_mb=os.path.getsize(ts_path)/1e6
    metrics={"acc_top1": round(acc,2), "artifact": ts_path, "size_mb": round(size_mb,3), "model": model_name}
    with open(os.path.join(args.out,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    print("[PTQ] Done:", metrics)
if __name__=="__main__":
    main()

