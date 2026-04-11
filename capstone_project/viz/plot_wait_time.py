# plot_wait_time.py
# ─────────────────────────────────────────────────────────────────────────────
# Wait time visualization.
#
# WHAT THIS PLOTS:
#   The NCCL AllReduce wait time per rank — how long each rank sat idle
#   waiting for the slowest rank before AllReduce could start.
#
#   A rank with LOW wait time is the bottleneck (it arrived late, no waiting).
#   A rank with HIGH wait time is a victim (it arrived early and waited).
#
# THREE PANELS:
#   Panel 1 — Mean wait time per rank with ± std error bars
#   Panel 2 — Min / Mean / Max grouped bars per rank
#   Panel 3 — Smoothed wait time trend per rank across all steps
#
# DEPENDENCIES:
#   analysis/wait_analysis.py  ← produces wait_df input
#   numpy, pandas, matplotlib
#
# USAGE:
#   from viz.plot_wait_time import plot_wait_time_summary
#   plot_wait_time_summary(wait_df, output_path="wait_time.png")
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ── Color palette — one color per rank ───────────────────────────────────────
RANK_COLORS = ["#E24B4A", "#BA7517", "#1D9E75", "#378ADD"]


def plot_wait_time_summary(
    wait_df:     pd.DataFrame,
    output_path: str = "wait_time_summary.png",
    dpi:         int = 150,
) -> None:
    """
    Three-panel wait time summary plot.

    PANEL 1 — Mean ± Std:
        Shows the average wait time per rank with error bars.
        Simple example:
            Rank 0: mean=40ms, std=5ms   → consistent victim
            Rank 2: mean= 2ms, std=8ms   → usually the bottleneck

    PANEL 2 — Min / Mean / Max:
        Shows the range of wait times per rank.
        A tall max with a low mean means the rank occasionally has very
        long waits (spike events) but is usually fine.

    PANEL 3 — Smoothed trend:
        Shows how wait time evolves across training steps.
        A rising trend = the bottleneck is getting worse over time
        (e.g. due to memory pressure or thermal throttling).

    Parameters:
        wait_df     → output of load_all_gpu_compute_wait_time()
                      columns: rank, step, bucket_within_step, pure_wait_ms
        output_path → file path to save the PNG
        dpi         → image resolution (150 is good for reports)

    Returns:
        None — saves the figure to output_path and prints the path.
    """

    # Filter out step=-1 (pre-training DDP init AllReduce calls)
    df = wait_df[wait_df["step"] != -1].copy()
    if df.empty:
        print("[WARNING] plot_wait_time_summary: No training data available.")
        return

    # Aggregate: sum all bucket wait times within each (step, rank)
    # This gives total wait time per step per rank
    grouped = (
        df.groupby(["step", "rank"])["pure_wait_ms"]
        .sum()
        .reset_index()
    )

    ranks        = sorted(grouped["rank"].unique())
    target_steps = sorted(grouped["step"].unique())
    n_ranks      = len(ranks)
    colors       = RANK_COLORS

    # Build per-rank time series (one value per step, 0 if step missing)
    rank_series = {}
    for rank in ranks:
        rdf = grouped[grouped["rank"] == rank].set_index("step")
        rank_series[rank] = [
            float(rdf.loc[s, "pure_wait_ms"]) if s in rdf.index else 0.0
            for s in target_steps
        ]

    means = {r: np.mean(v) for r, v in rank_series.items()}
    stds  = {r: np.std(v)  for r, v in rank_series.items()}
    mins  = {r: np.min(v)  for r, v in rank_series.items()}
    maxs  = {r: np.max(v)  for r, v in rank_series.items()}

    def smooth(arr: list, window: int = 5) -> np.ndarray:
        """Moving average smoothing."""
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor("#FAFAF8")

    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])   # Panel 1: mean bar
    ax2 = fig.add_subplot(gs[1])   # Panel 2: min/mean/max
    ax3 = fig.add_subplot(gs[2])   # Panel 3: trend line

    rank_labels = [f"Rank {r}" for r in ranks]
    x     = np.arange(n_ranks)
    bar_w = 0.55

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#F5F4EF")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Panel 1: Mean ± Std ───────────────────────────────────────────────────
    mean_vals = [means[r] for r in ranks]
    std_vals  = [stds[r]  for r in ranks]

    bars = ax1.bar(
        x, mean_vals,
        width  = bar_w,
        color  = [colors[i % len(colors)] for i in range(n_ranks)],
        alpha  = 0.82,
        zorder = 3,
    )
    ax1.errorbar(
        x, mean_vals, yerr=std_vals,
        fmt="none", color="#444", capsize=5, linewidth=1.2, zorder=4,
    )
    for bar, val in zip(bars, mean_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(std_vals) * 0.05 + 0.01,
            f"{val:.2f}ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(rank_labels, fontsize=9)
    ax1.set_ylabel("Wait time (ms)", fontsize=9)
    ax1.set_title(
        "Mean wait time per rank\n± std deviation",
        fontsize=10, fontweight="bold",
    )

    # ── Panel 2: Min / Mean / Max grouped bars ────────────────────────────────
    sub_w      = bar_w / 3
    offsets_p2 = [-sub_w, 0, sub_w]
    labels_mmm = ["Min", "Mean", "Max"]
    alphas     = [0.45, 0.80, 1.0]

    for i, (vals_dict, alpha, lbl) in enumerate(zip(
        [mins, means, maxs], alphas, labels_mmm
    )):
        ax2.bar(
            x + offsets_p2[i],
            [vals_dict[r] for r in ranks],
            width  = sub_w * 0.88,
            color  = [colors[j % len(colors)] for j in range(n_ranks)],
            alpha  = alpha,
            label  = lbl,
            zorder = 3,
        )

    # Custom legend with square markers
    legend_handles = [
        Line2D(
            [0], [0],
            marker          = "s",
            color           = "w",
            markerfacecolor = "#888",
            markersize      = 8,
            markeredgecolor = "none",
            alpha           = a,
            label           = l,
        )
        for a, l in zip(alphas, labels_mmm)
    ]
    ax2.legend(
        handles    = legend_handles,
        fontsize   = 8,
        loc        = "upper right",
        framealpha = 0.5,
        edgecolor  = "none",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(rank_labels, fontsize=9)
    ax2.set_ylabel("Wait time (ms)", fontsize=9)
    ax2.set_title(
        "Min / Mean / Max per rank\ntall max + low mean = spike events",
        fontsize=10, fontweight="bold",
    )

    # ── Panel 3: Smoothed trend ───────────────────────────────────────────────
    window = max(3, len(target_steps) // 10)
    for i, rank in enumerate(ranks):
        smoothed = smooth(rank_series[rank], window=window)
        ax3.plot(
            target_steps, smoothed,
            color     = colors[i % len(colors)],
            linewidth = 2,
            label     = f"Rank {rank}",
            alpha     = 0.9,
        )

    ax3.set_xlabel("Training Step", fontsize=9)
    ax3.set_ylabel("Wait time (ms)", fontsize=9)
    ax3.set_title(
        "Smoothed wait trend per rank\nrising = getting worse over time",
        fontsize=10, fontweight="bold",
    )
    ax3.legend(fontsize=8, loc="upper right", framealpha=0.5, edgecolor="none")

    # ── Super title ───────────────────────────────────────────────────────────
    fig.suptitle(
        "NCCL AllReduce Wait Time Summary\n"
        "Low wait = bottleneck (slow rank)     High wait = victim (waiting for slow rank)",
        fontsize=11, fontweight="bold", y=1.03,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[plot_wait_time_summary] Saved: {output_path}")
    plt.close()