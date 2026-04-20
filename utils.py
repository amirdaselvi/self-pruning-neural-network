"""
utils.py
--------
Data loading, plotting, and results saving utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ── Data ───────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 256):
    """Download CIFAR-10 and return (train_loader, test_loader)."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_gate_distributions(results: list, save_path: str):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]

    for ax, res, color in zip(axes, results, colors):
        gates = res["gate_values"]
        ax.hist(gates, bins=80, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.axvline(x=0.01, color="black", linestyle="--", linewidth=1.2, label="Prune threshold")
        ax.set_title(
            f"λ = {res['lambda']}\n"
            f"Acc: {res['accuracy']:.1f}%  |  Sparsity: {res['sparsity']:.1f}%",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Gate Value", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize=9)

    fig.suptitle("Gate Value Distributions Across λ Values",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_tradeoff(results: list, save_path: str):
    labels    = [f"λ={r['lambda']}" for r in results]
    accs      = [r["accuracy"]  for r in results]
    sparsities= [r["sparsity"]  for r in results]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.bar(x - width/2, accs,       width, label="Test Accuracy (%)",  color="#3498db", alpha=0.85)
    ax2.bar(x + width/2, sparsities, width, label="Sparsity Level (%)", color="#e74c3c", alpha=0.85)

    ax1.set_xlabel("Lambda (λ)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)",  color="#3498db", fontsize=12)
    ax2.set_ylabel("Sparsity Level (%)", color="#e74c3c", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.title("Accuracy vs Sparsity Trade-off", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Results Table (FIXED HERE) ─────────────────────────────────────────────────

def save_table(results: list, save_path: str):
    rows = "\n".join(
        f"| `{r['lambda']:.5f}` | {r['accuracy']:.2f}% | {r['sparsity']:.2f}% |"
        for r in results
    )
    best = max(results, key=lambda r: r["accuracy"])

    md = f"""# Results

## Lambda vs Accuracy vs Sparsity

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
|:----------:|:-------------:|:------------------:|
{rows}

**Best accuracy:** λ = `{best['lambda']}` → {best['accuracy']:.2f}% accuracy, {best['sparsity']:.2f}% sparsity

## Observations

- Higher λ → more gates driven to 0 → higher sparsity, lower accuracy.
- Lower λ → network keeps more connections → better accuracy, less pruning.
- The gate distribution plots (`plots.png`) confirm a bimodal pattern:
  a large spike near 0 (pruned weights) and a cluster near 1 (active weights).
"""

    # ✅ FIX: UTF-8 encoding added
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"  Saved → {save_path}")


# ── Master Save ────────────────────────────────────────────────────────────────

def save_results(results: list, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    print("\nSaving results...")
    plot_gate_distributions(results, os.path.join(out_dir, "plots.png"))
    plot_tradeoff(results,           os.path.join(out_dir, "tradeoff.png"))
    save_table(results,              os.path.join(out_dir, "table.md"))