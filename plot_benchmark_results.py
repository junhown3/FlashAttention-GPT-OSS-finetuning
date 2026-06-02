"""Generate PNG charts for BENCHMARK_README.md from the reported T4 results."""

import matplotlib.pyplot as plt
import numpy as np

MODES = ["forward_only", "forward_backward", "backward_only"]
PYTORCH_MS = [38.26, 93.22, 55.51]
TRITON_MS = [1.59, 10.78, 9.57]
PYTORCH_GB = [2.04, 3.54, 3.54]
TRITON_GB = [0.024, 0.044, 0.044]
SPEEDUP = [24.08, 8.65, 5.80]


def plot_grouped_bars(labels, series, ylabel, title, filename, colors):
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, (name, values) in enumerate(series.items()):
        offset = (idx - 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    print(f"Wrote {filename}")


def main():
    plot_grouped_bars(
        MODES,
        {"PyTorch Naive": PYTORCH_MS, "Triton Atomic": TRITON_MS},
        "Average latency (ms)",
        "Attention benchmark latency (Tesla T4, fp16, Hq=16, Hkv=16, L=4096)",
        "benchmark_results_latency.png",
        ["#4C72B0", "#55A868"],
    )
    plot_grouped_bars(
        MODES,
        {"PyTorch Naive": PYTORCH_GB, "Triton Atomic": TRITON_GB},
        "Peak GPU memory (GB)",
        "Attention benchmark peak memory (Tesla T4, fp16, Hq=16, Hkv=16, L=4096)",
        "benchmark_results_memory.png",
        ["#4C72B0", "#55A868"],
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(MODES, SPEEDUP, color="#C44E52")
    ax.set_ylabel("Speedup (x)")
    ax.set_title("Triton Atomic speedup vs PyTorch Naive")
    ax.grid(axis="y", alpha=0.3)
    for i, value in enumerate(SPEEDUP):
        ax.text(i, value + 0.4, f"{value:.2f}x", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig("benchmark_results_speedup.png", dpi=160)
    plt.close(fig)
    print("Wrote benchmark_results_speedup.png")


if __name__ == "__main__":
    main()
