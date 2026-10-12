"""
Fine-tune (or linear-probe) an SSL-pretrained backbone on a chosen fraction of
labeled EuroSAT data. This is the script that produces the "how well does the
SSL representation transfer with only X% of labels" result.

Examples (run from project root):

  # Linear probe: freeze backbone, train linear head, using only 5% of labels
  python -m finetune.train_finetune --mode linear_probe --label-fraction 0.05 \
      --ssl-checkpoint ./checkpoints/ssl_backbone_final.pt

  # Full fine-tune with 10% of labels
  python -m finetune.train_finetune --mode finetune --label-fraction 0.10 \
      --ssl-checkpoint ./checkpoints/ssl_backbone_final.pt

  # Baseline with no SSL pretraining at all (random init), for comparison
  python -m finetune.train_finetune --mode finetune --label-fraction 0.10 \
      --ssl-checkpoint none
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ssl_pretrain.augmentations import get_eval_transform
from ssl_pretrain.dataset import LabeledSubsetDataset, CLASS_NAMES, make_splits, subsample_labels
from finetune.classifier import RemoteSensingClassifier, load_pretrained_backbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data/eurosat")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--mode", type=str, default="finetune", choices=["linear_probe", "finetune"])
    parser.add_argument("--label-fraction", type=float, default=1.0, help="fraction of TRAIN labels to use (0-1]")
    parser.add_argument("--ssl-checkpoint", type=str, default="./checkpoints/ssl_backbone_final.pt",
                         help="path to SSL checkpoint, or 'none' for random-init baseline")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./checkpoints/classifier_final.pt")
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, f1


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    transform = get_eval_transform(args.image_size)
    train_idx, val_idx, test_idx = make_splits(root=args.data_root, seed=args.seed, download=True)
    train_idx = subsample_labels(train_idx, args.label_fraction, seed=args.seed)
    print(f"Training on {len(train_idx)} labeled examples "
          f"({args.label_fraction*100:.1f}% of the training split).")

    train_ds = LabeledSubsetDataset(args.data_root, transform, indices=train_idx, download=True)
    val_ds = LabeledSubsetDataset(args.data_root, transform, indices=val_idx, download=True)
    test_ds = LabeledSubsetDataset(args.data_root, transform, indices=test_idx, download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    model = RemoteSensingClassifier(arch=args.arch, num_classes=len(CLASS_NAMES), small_image=True)

    if args.ssl_checkpoint.lower() != "none":
        model = load_pretrained_backbone(model, args.ssl_checkpoint, device)
        print(f"Loaded SSL-pretrained backbone from {args.ssl_checkpoint}")
    else:
        print("Training from random initialization (no SSL pretraining) -- baseline run.")

    if args.mode == "linear_probe":
        model.freeze_backbone()
        print("Backbone frozen: linear probe mode.")
    else:
        model.unfreeze_backbone()
        print("Backbone trainable: full fine-tuning mode.")

    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss {running_loss/len(train_loader):.4f} "
              f"| val acc {val_acc:.4f} | val macro-F1 {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arch": args.arch,
                    "num_classes": len(CLASS_NAMES),
                    "class_names": CLASS_NAMES,
                    "image_size": args.image_size,
                },
                args.output,
            )

    # Final test-set evaluation using the best checkpoint on val
    ckpt = torch.load(args.output, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"\nFINAL TEST RESULTS | accuracy {test_acc:.4f} | macro-F1 {test_f1:.4f}")
    print(f"Best model saved to {args.output}")


if __name__ == "__main__":
    main()
