import argparse, os
import torchvision
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--out", default="data/")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.dataset.lower()!="cifar10":
        raise SystemExit("Only cifar10 supported in setup stub.")
    torchvision.datasets.CIFAR10(root=args.out, train=True, download=True)
    torchvision.datasets.CIFAR10(root=args.out, train=False, download=True)
    print("[DATA] CIFAR-10 downloaded to", args.out)
if __name__=="__main__":
    main()
