"""
Copies one example tile per EuroSAT class into deployment/static/samples so the
live demo has one-click sample images (visitors rarely have satellite tiles handy).

Run once from the project root (downloads EuroSAT if not already present):
    python scripts/export_samples.py
"""
import os
import random
import shutil

from torchvision.datasets import EuroSAT

OUT = "deployment/static/samples"
os.makedirs(OUT, exist_ok=True)

ds = EuroSAT(root="./data/eurosat", download=True)
by_class = {}
for path, target in ds.samples:
    by_class.setdefault(target, []).append(path)

rng = random.Random(7)
for target, paths in sorted(by_class.items()):
    name = ds.classes[target]
    shutil.copy(rng.choice(paths), os.path.join(OUT, f"{name}.jpg"))
    print("saved", name)
print(f"Done. {len(by_class)} samples in {OUT}")
