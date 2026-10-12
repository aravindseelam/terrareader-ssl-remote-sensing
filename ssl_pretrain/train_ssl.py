"""
Self-supervised (SimCLR) pretraining on unlabeled EuroSAT imagery.

Run from the project root:
    python -m ssl_pretrain.train_ssl --epochs 100 --batch-size 256

Laptop-friendly defaults: ResNet-18 backbone, batch size 256, image size 64.
If you're on CPU only, drop batch size to ~64-128 and expect it to be slow
but functional -- reduce --epochs for a quick smoke test first (e.g. --epochs 2).
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_pretrain.dataset import SSLDataset
from ssl_pretrain.simclr_model import SimCLRModel, nt_xent_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data/eurosat")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SSLDataset(root=args.data_root, image_size=args.image_size, download=True)
    print(f"Loaded {len(dataset)} unlabeled images for SSL pretraining.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    model = SimCLRModel(arch=args.arch, small_image=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, (view1, view2) in enumerate(progress):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % args.log_every == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | avg loss {avg_loss:.4f} | {elapsed:.1f}s")

        # Save checkpoint every epoch (cheap for ResNet-18, and lets you stop anytime)
        ckpt_path = os.path.join(args.checkpoint_dir, "ssl_backbone_latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "arch": args.arch,
                "feat_dim": model.feat_dim,
            },
            ckpt_path,
        )

    # Final checkpoint, named explicitly for downstream fine-tuning scripts
    final_path = os.path.join(args.checkpoint_dir, "ssl_backbone_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": args.arch,
            "feat_dim": model.feat_dim,
        },
        final_path,
    )
    print(f"Saved final SSL-pretrained backbone to {final_path}")


if __name__ == "__main__":
    main()
