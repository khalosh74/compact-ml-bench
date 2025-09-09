#!/usr/bin/env python3
from pathlib import Path
from torchvision import datasets, transforms
from PIL import Image

root = Path("demo_assets/cifar10_samples"); root.mkdir(parents=True, exist_ok=True)
ds = datasets.CIFAR10(root="data", train=False, download=False,
                      transform=transforms.ToTensor())
labels = ds.classes
# pick ~2 per class (first occurrences)
picked = {c:0 for c in range(10)}
i=0
while not all(v>=2 for v in picked.values()) and i < len(ds):
    img, y = ds[i]
    if picked[y] < 2:
        # de-normalize to 0-255
        im = Image.fromarray((img.permute(1,2,0).numpy()*255).astype("uint8"))
        p = root / f"{labels[y]}_{picked[y]}.jpg"
        im.save(p, quality=95)
        picked[y] += 1
    i += 1
print("Wrote samples to", root.resolve())
