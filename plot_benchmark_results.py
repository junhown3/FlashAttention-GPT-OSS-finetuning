"""Generate PNG charts for README.md from the reported T4 benchmark results."""

import matplotlib.pyplot as plt
import numpy as np

MODES = ["forward_only", "forward_backward", "backward_only"]
MODE_LABELS = ["Forward", "Forward +\nBackward", "Backward"]
PYTORCH_MS = [38.2578, 93.2606, 55.3403]
TRITON_MS = [1.3437, 4.0357, 3.1075]
PYTORCH_GB = [2.0374, 3.5432, 3.5432]
TRITON_GB = [0.0239, 0.0435, 0.0435]
SPEEDUP = [28.47, 23.11, 17.81]
MEMORY_SAVING = [85.15, 81.53, 81.53]

PYTORCH_COLOR = "#E76F51"
TRITON_COLOR = "#2A9D8F"
SPEEDUP_COLOR = "#457B9D"

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": "#333333",
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "grid.color": "#DDDDDD",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
}


def _style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_bars(ax, bars, values, fmt, y_offset=0.02, log_scale=False):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if log_scale:
            y = height * (1 + y_offset)
        else:
            y = height + y_offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )


def plot_grouped_bars(labels, series, ylabel, title, filename, colors, log_scale=False):
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars_by_series = []
    for idx, (name, values) in enumerate(series.items()):
        offset = (idx - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=colors[idx], edgecolor="white", linewidth=0.8)
        bars_by_series.append((bars, values))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    if log_scale:
        ax.set_yscale("log")
    _style_axes(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD")
    for bars, values in bars_by_series:
        fmt = "{:.2f}" if max(values) < 10 else "{:.1f}"
        _annotate_bars(ax, bars, values, fmt, y_offset=0.08 if log_scale else 0.02, log_scale=log_scale)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {filename}")


def plot_speedup(labels, values, memory_saving, filename):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(labels, values, color=SPEEDUP_COLOR, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Triton Atomic vs PyTorch Naive", pad=10)
    _style_axes(ax)
    for bar, speedup, mem in zip(bars, values, memory_saving):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{speedup:.1f}× speedup\n{mem:.0f}× less memory",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    ax.set_ylim(0, max(values) * 1.35)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {filename}")


def _format_ms(value):
    return f"{value:.1f} ms" if value >= 10 else f"{value:.2f} ms"


def _format_gb(value):
    return f"{value:.2f} GB" if value >= 0.1 else f"{value:.3f} GB"


def _plot_combined_grouped(ax, x, width, pytorch_values, triton_values, title, ylabel, log_scale=False):
    pytorch_bars = ax.bar(
        x - width / 2,
        pytorch_values,
        width,
        label="PyTorch Naive",
        color=PYTORCH_COLOR,
        edgecolor="white",
    )
    triton_bars = ax.bar(
        x + width / 2,
        triton_values,
        width,
        label="Triton Atomic",
        color=TRITON_COLOR,
        edgecolor="white",
    )
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(MODE_LABELS)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=8)

    for bars, values, fmt in (
        (pytorch_bars, pytorch_values, _format_ms if "ms" in ylabel else _format_gb),
        (triton_bars, triton_values, _format_ms if "ms" in ylabel else _format_gb),
    ):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            y = height * 1.15 if log_scale else height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                fmt(value),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )


def plot_combined_dashboard(filename):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    x = np.arange(len(MODE_LABELS))
    width = 0.36

    _plot_combined_grouped(
        axes[0],
        x,
        width,
        PYTORCH_MS,
        TRITON_MS,
        "Latency",
        "Avg ms (log scale)",
        log_scale=True,
    )
    _plot_combined_grouped(
        axes[1],
        x,
        width,
        PYTORCH_GB,
        TRITON_GB,
        "Peak memory",
        "Peak GB (log scale)",
        log_scale=True,
    )

    ax = axes[2]
    bars = ax.bar(MODE_LABELS, SPEEDUP, color=SPEEDUP_COLOR, edgecolor="white")
    ax.set_title("Speedup (×)", pad=8)
    ax.set_ylabel("× faster")
    _style_axes(ax)
    for bar, value in zip(bars, SPEEDUP):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.1f}×",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, max(SPEEDUP) * 1.2)

    fig.suptitle("Attention benchmark · Tesla T4 · fp16 · L=4096", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {filename}")


def main():
    plt.rcParams.update(STYLE)

    subtitle = "Tesla T4 · fp16 · Hq=16 · L=4096"

    plot_grouped_bars(
        MODE_LABELS,
        {"PyTorch Naive": PYTORCH_MS, "Triton Atomic": TRITON_MS},
        "Average latency (ms)",
        f"Latency · {subtitle}",
        "benchmark_results_latency.png",
        [PYTORCH_COLOR, TRITON_COLOR],
        log_scale=True,
    )
    plot_grouped_bars(
        MODE_LABELS,
        {"PyTorch Naive": PYTORCH_GB, "Triton Atomic": TRITON_GB},
        "Peak GPU memory (GB)",
        f"Peak memory · {subtitle}",
        "benchmark_results_memory.png",
        [PYTORCH_COLOR, TRITON_COLOR],
    )
    plot_speedup(MODE_LABELS, SPEEDUP, MEMORY_SAVING, "benchmark_results_speedup.png")
    plot_combined_dashboard("benchmark_results_combined.png")


if __name__ == "__main__":
    main()
