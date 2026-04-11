# plot_sm_timeline.py
# ─────────────────────────────────────────────────────────────────────────────
# SM utilization timeline visualization.
#
# WHAT THIS PLOTS:
#   A stacked bar chart showing GPU SM utilization over time for one training
#   step, with one panel per GPU rank.
#
#   Each bar represents one time bucket (~0.18ms wide for A100 ResNet-50).
#   The bar is split by colour into stream types:
#       yellow  = opt_step   (optimizer kernels)
#       teal    = forward    (forward pass kernels)
#       purple  = backward   (backward pass kernels)
#       red     = nccl_active (AllReduce communication kernels)
#       green   = compute    (kernels whose phase is unknown)
#       grey    = unknown
#
#   Empty space above the bars = idle SMs (GPU not fully utilized).
#   Grey background shading    = H2D DMA transfer active (not on SMs).
#   Pink background shading    = NCCL AllReduce active.
#   Amber dashed vertical line = step boundary (this rank moved to next step).
#
# DEPENDENCIES:
#   analysis/sm_analysis.py  ← produces df_sm_timeline input
#   numpy, pandas, matplotlib
#
# USAGE:
#   from viz.plot_sm_timeline import plot_sm_timeline
#   plot_sm_timeline(df_sm_timeline, step=2, output_path="sm_step2.png")
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Visual constants ──────────────────────────────────────────────────────────

# Order in which stream types are stacked (bottom to top)
STREAM_ORDER = [
    "opt_step",    # optimizer kernels — bottom of the stack
    "forward",
    "backward",
    "nccl_active",
    "compute",     # forward/backward when phase is unknown
    "unknown",
]

# Bar colour per stream type
STREAM_COLOR = {
    "opt_step":    "#fbbf24",  # yellow
    "forward":     "#22d3b8",  # teal
    "backward":    "#818cf8",  # purple
    "nccl_active": "#f87171",  # red
    "nccl_spin":   "#f87171",  # red (hatched pattern)
    "compute":     "#a3e635",  # lime green
    "unknown":     "#6b7280",  # grey
}

H2D_COLOR   = "#94a3b8"   # grey  — H2D background shading
IDLE_COLOR  = "#1a2030"   # dark  — idle SM background
BG_COLOR    = "#0d1117"   # very dark — full figure background
GRID_COLOR  = "#1f2937"   # dark grey — horizontal grid lines
LABEL_COLOR = "#8899bb"   # blue-grey — axis labels and tick marks
GUIDE_COLOR = "#ffffff"   # white — 80% reference line


# ==============================================================================
# Public function
# ==============================================================================

def plot_sm_timeline(
    df_sm_timeline: pd.DataFrame,
    step:           int,
    n_buckets:      int   = 350,
    figsize:        tuple = (22, 12),
    output_path:    str   = "sm_timeline.png",
    dpi:            int   = 150,
    show:           bool  = False,
) -> None:
    """
    Draw a stacked bar SM utilization timeline for one training step.

    One panel is drawn per GPU rank, all sharing the same x-axis (time).

    HOW TO READ THE PLOT:
        X-axis  → time (ms), relative to step start for the fastest rank
        Y-axis  → SM utilization % (0% to 100%)
        Colors  → which phase the GPU was running (forward, backward, etc.)
        Empty   → idle SMs (no kernels running, GPU is under-utilized)

        If all panels look similar → ranks are well balanced
        If one panel has more empty space → that rank is wasting GPU time

        The 80% dashed line is a reference — anything consistently below it
        suggests the GPU is under-utilized for that phase.

    Simple example interpretation:
        Rank 0 backward: SM=65%, lots of NCCL overlap  → DDP working well
        Rank 2 backward: SM=30%, lots of empty space   → rank 2 is slow
        → rank 2 finishes backward later → everyone waits → wasted time

    Parameters:
        df_sm_timeline → output of build_sm_timeline_df()
                         columns: step, rank, bucket_idx, t_abs_ms,
                                  stream_type, sm_pct, h2d_active
        step           → which training step to plot
        n_buckets      → must match the value used in build_sm_timeline_df()
        figsize        → matplotlib figure size in inches (width, height)
        output_path    → path to save the PNG file
        dpi            → image resolution (150 is good for reports)
        show           → if True, calls plt.show() — set False on servers

    Returns:
        None — saves the figure and prints the output path.
    """

    # Filter to the requested step only
    df_step = df_sm_timeline[df_sm_timeline["step"] == step].copy()
    if df_step.empty:
        print(f"[Warning] plot_sm_timeline: no data for step {step}")
        return

    ranks  = sorted(df_step["rank"].unique().tolist())
    n_gpus = len(ranks)

    # ── Create figure with one subplot per rank ───────────────────────────────
    fig, axes = plt.subplots(
        nrows     = n_gpus,
        ncols     = 1,
        figsize   = figsize,
        sharex    = True,
        facecolor = BG_COLOR,
    )
    fig.subplots_adjust(hspace=0.08)

    if n_gpus == 1:
        axes = [axes]

    # ── Draw each rank panel ──────────────────────────────────────────────────
    for ax, rank in zip(axes, ranks):
        ax.set_facecolor(BG_COLOR)
        ax.set_ylim(0, 100)

        df_rank = df_step[df_step["rank"] == rank]
        if df_rank.empty:
            _style_ax(ax, rank, n_buckets)
            continue

        # Pivot: rows=bucket_idx, cols=stream_type, values=sm_pct (summed)
        # This gives us one column per stream type per bucket
        df_pivot = df_rank.pivot_table(
            index   = "bucket_idx",
            columns = "stream_type",
            values  = "sm_pct",
            aggfunc = "sum",
        ).reindex(range(n_buckets), fill_value=0.0).fillna(0.0)

        x      = df_pivot.index.values   # 0 to n_buckets-1
        bottom = np.zeros(len(x))

        # ── Draw stacked bars, one stream_type per layer ──────────────────────
        for stream in STREAM_ORDER:
            if stream not in df_pivot.columns:
                continue
            heights = df_pivot[stream].values
            if stream == "nccl_spin":
                # Hatched pattern for NCCL spin-waiting
                ax.bar(
                    x, heights, bottom=bottom,
                    color=STREAM_COLOR[stream],
                    alpha=0.45, width=1.0, linewidth=0, hatch="////",
                )
            else:
                ax.bar(
                    x, heights, bottom=bottom,
                    color=STREAM_COLOR.get(stream, "#6b7280"),
                    alpha=0.90, width=1.0, linewidth=0,
                )
            bottom += heights

        # ── H2D background shading ────────────────────────────────────────────
        # Grey shading behind bars wherever a DMA transfer was active
        # H2D uses the DMA engine (not SMs) so it can overlap with compute
        h2d_mask = (
            df_rank.groupby("bucket_idx")["h2d_active"]
            .any()
            .reindex(range(n_buckets), fill_value=False)
        )
        if h2d_mask.any():
            for seg_start, seg_end in _find_consecutive_segments(
                h2d_mask[h2d_mask].index.tolist()
            ):
                ax.axvspan(
                    seg_start - 0.5, seg_end + 0.5,
                    alpha=0.12, color=H2D_COLOR, zorder=0,
                )

        # ── NCCL background shading ───────────────────────────────────────────
        # Pink shading wherever AllReduce was running
        nccl_mask = (
            df_rank.groupby("bucket_idx")
            .apply(lambda g: (g["stream_type"] == "nccl_active").any())
            .reindex(range(n_buckets), fill_value=False)
        )
        if nccl_mask.any():
            for seg_start, seg_end in _find_consecutive_segments(
                nccl_mask[nccl_mask].index.tolist()
            ):
                ax.axvspan(
                    seg_start - 0.5, seg_end + 0.5,
                    alpha=0.08, color="#f87171", zorder=0,
                )

        # ── Step boundary line ────────────────────────────────────────────────
        # Amber dashed vertical line where this rank moved into the next step
        # (faster ranks can move ahead while slower ranks are still in step N)
        next_step_rows = df_rank[df_rank["step"] == step + 1]
        if not next_step_rows.empty:
            boundary_bi = next_step_rows["bucket_idx"].min()
            ax.axvline(
                x=boundary_bi - 0.5,
                color="#f59e0b", linewidth=1.2, linestyle="--",
                alpha=0.5, zorder=3,
            )

        _style_ax(ax, rank, n_buckets)

    # ── X-axis tick labels: bucket_idx → milliseconds ────────────────────────
    # Use rank 0's t_abs_ms mapping to label x-axis in real time units
    df_rank0   = df_step[df_step["rank"] == ranks[0]]
    t_map      = (
        df_rank0.groupby("bucket_idx")["t_abs_ms"]
        .first()
        .reindex(range(n_buckets))
    )
    tick_pos    = np.linspace(0, n_buckets - 1, 7, dtype=int)
    tick_labels = [
        f"{t_map.get(bi, bi * (t_map.max() / n_buckets)):.0f}ms"
        for bi in tick_pos
    ]
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_labels, color=LABEL_COLOR, fontsize=9)
    axes[-1].tick_params(axis="x", colors=LABEL_COLOR)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []

    for stream in STREAM_ORDER:
        if stream not in df_step["stream_type"].unique():
            continue
        if stream == "nccl_spin":
            handle = mpatches.Patch(
                facecolor=STREAM_COLOR[stream], hatch="////",
                alpha=0.45, label="nccl spin (polling)",
            )
        else:
            handle = mpatches.Patch(
                color=STREAM_COLOR.get(stream, "#6b7280"),
                label=stream,
            )
        legend_handles.append(handle)

    legend_handles.append(mpatches.Patch(
        facecolor=H2D_COLOR, alpha=0.25,
        label="H2D  (DMA — no SM usage)",
    ))
    legend_handles.append(mpatches.Patch(
        facecolor=IDLE_COLOR,
        label="idle SM",
    ))

    # Step boundary legend entry (only if at least one rank crossed into next step)
    has_boundary = any(
        not df_step[(df_step["rank"] == r) & (df_step["step"] == step + 1)].empty
        for r in ranks
    )
    if has_boundary:
        legend_handles.append(
            plt.Line2D(
                [0], [0],
                color="#f59e0b", linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"step {step} → step {step + 1} boundary",
            )
        )

    fig.legend(
        handles    = legend_handles,
        loc        = "upper right",
        fontsize   = 9,
        framealpha = 0.2,
        labelcolor = "white",
        facecolor  = "#1f2937",
        edgecolor  = "#374151",
        ncol       = 2,
    )

    # ── Title and save ────────────────────────────────────────────────────────
    fig.suptitle(
        f"SM Utilization Timeline  —  Step {step}",
        color="white", fontsize=13, y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"[plot_sm_timeline] Saved: {output_path}")

    if show:
        plt.show()

    plt.close()


# ==============================================================================
# Private helpers
# ==============================================================================

def _style_ax(ax: plt.Axes, rank: int, n_buckets: int) -> None:
    """
    Apply consistent dark-theme styling to one GPU panel.

    Applied to every rank panel:
        - 80% reference line (dotted white) — target SM utilization
        - Y-axis ticks at 0%, 80%, 100% only
        - GPU label on left side
        - Horizontal grid lines (very faint)
        - Remove all spines (border lines around the plot)
        - Hide x-axis ticks except on the bottom panel (shared via sharex)
    """
    # 80% reference line — target for good GPU utilization
    ax.axhline(
        y=80, color=GUIDE_COLOR, linewidth=0.6, linestyle=":", alpha=0.25,
    )

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 80, 100])
    ax.set_yticklabels(["0%", "80%", "100%"], color=LABEL_COLOR, fontsize=8)
    ax.tick_params(axis="y", colors=LABEL_COLOR, length=0)
    ax.set_xlim(-1, n_buckets)

    ax.set_ylabel(
        f"GPU {rank}",
        color=LABEL_COLOR, fontsize=11,
        rotation=0, labelpad=40, va="center",
    )

    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="x", bottom=False, labelbottom=False)


def _find_consecutive_segments(indices: List[int]) -> List[tuple]:
    """
    Convert a list of integers into a list of (start, end) consecutive ranges.

    Simple example:
        [18, 19, 20, 45, 46, 47, 50] → [(18, 20), (45, 47), (50, 50)]

    Used to draw axvspan background shading efficiently — one span per
    consecutive run instead of one span per bucket (much faster rendering).
    """
    if not indices:
        return []

    segments  = []
    seg_start = indices[0]
    seg_prev  = indices[0]

    for idx in indices[1:]:
        if idx == seg_prev + 1:
            seg_prev = idx
        else:
            segments.append((seg_start, seg_prev))
            seg_start = idx
            seg_prev  = idx

    segments.append((seg_start, seg_prev))
    return segments