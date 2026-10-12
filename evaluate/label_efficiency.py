"""
Runs the core experiment of the project: trains a classifier at several
labeled-data fractions (1%, 5%, 10%, 25%, 50%, 100%), once starting from the
SSL-pretrained backbone and once from random initialization, and plots test
accuracy vs. label fraction for both. This curve is the evidence that SSL
pretraining helps in the low-label regime.

Run from project root (after SSL pretraining has produced a checkpoint):
    python -m evaluate.label_efficiency --ssl-checkpoint ./checkpoints/ssl_backbone_final.pt

This calls finetune.train_finetune as a subprocess for each run so results
are cleanly isolated, then parses the printed test accuracy from each run.
"""

import argparse
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving PNGs without a display
import matplotlib.pyplot as plt


LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def run_single(mode, fraction, ssl_checkpoint, epochs, arch, image_size, output_dir):
    output_path = f"{output_dir}/clf_{mode}_{int(fraction*100)}pct.pt"
    cmd = [
        sys.executable, "-m", "finetune.train_finetune",
        "--mode", "finetune",
        "--label-fraction", str(fraction),
        "--ssl-checkpoint", ssl_checkpoint if mode == "ssl" else "none",
        "--epochs", str(epochs),
        "--arch", arch,
        "--image-size", str(image_size),
        "--output", output_path,
    ]
    print(f"\n=== Running: mode={mode}, label_fraction={fraction} ===")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:])  # tail of stdout, to keep logs readable
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError(f"Run failed for mode={mode}, fraction={fraction}")

    matches = re.findall(r"FINAL TEST RESULTS \| accuracy ([0-9.]+)", result.stdout)
    if not matches:
        raise RuntimeError(f"Could not parse test accuracy from output for {mode}/{fraction}")
    return float(matches[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-checkpoint", type=str, default="./checkpoints/ssl_backbone_final.pt")
    parser.add_argument("--epochs", type=int, default=15, help="epochs per run (kept short since this loops many runs)")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--plot-path", type=str, default="./label_efficiency_curve.png")
    args = parser.parse_args()

    ssl_results, random_results = [], []

    for fraction in LABEL_FRACTIONS:
        ssl_acc = run_single("ssl", fraction, args.ssl_checkpoint, args.epochs, args.arch, args.image_size, args.output_dir)
        random_acc = run_single("random", fraction, args.ssl_checkpoint, args.epochs, args.arch, args.image_size, args.output_dir)
        ssl_results.append(ssl_acc)
        random_results.append(random_acc)
        print(f"Fraction {fraction:.2f} -> SSL-pretrained acc={ssl_acc:.4f}, random-init acc={random_acc:.4f}")

    plt.figure(figsize=(7, 5))
    x = [f * 100 for f in LABEL_FRACTIONS]
    plt.plot(x, ssl_results, marker="o", label="SSL-pretrained backbone")
    plt.plot(x, random_results, marker="s", label="Random-init baseline")
    plt.xlabel("% of labeled training data used")
    plt.ylabel("Test accuracy")
    plt.title("Label efficiency: SSL pretraining vs. random init (EuroSAT)")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=150)
    print(f"\nSaved label-efficiency plot to {args.plot_path}")

    print("\nSummary:")
    print(f"{'Label %':>10} | {'SSL Acc':>8} | {'Random Acc':>10}")
    for frac, s, r in zip(LABEL_FRACTIONS, ssl_results, random_results):
        print(f"{frac*100:>9.1f}% | {s:>8.4f} | {r:>10.4f}")


if __name__ == "__main__":
    main()
