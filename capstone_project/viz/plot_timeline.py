# plot_timeline.py
# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU detailed event timeline visualization.
#
# WHAT THIS PLOTS:
#   A horizontal Gantt-chart style timeline showing CPU and GPU events for
#   every rank side by side. Each row represents one type of activity
#   (forward, backward, NCCL, H2D, etc.). Each rank gets its own lane
#   within each row, stacked vertically.
#
#   This is a "Level 2" diagnostic plot — more detailed than the SM timeline.
#   Use it to visually inspect timing alignment across ranks for specific steps.
#
# HOW TO READ THE PLOT:
#   Y-axis  → activity type (one row per phase)
#   X-axis  → time (ms), relative to the global step start
#   Bars    → event duration, colored by rank (or by step if color_by="step")
#   Faint bars  → CPU launch time (how long the CPU spent issuing commands)
#   Solid bars  → actual GPU execution time
#   Black vline → CPU step end boundary
#   Blue vline  → GPU step end boundary
#
#   If all ranks' GPU bars align horizontally → perfect load balance
#   If one rank's GPU bar ends later → that rank is the bottleneck
#
# ROWS IN THE PLOT:
#   data_wait   → CPU time waiting for DataLoader to produce next batch
#   h2d         → Host-to-Device memory transfer (DMA)
#   gpu_compute → all GPU kernels (wrapper view)
#   NCCL        → AllReduce communication kernels
#   zero_grad   → optimizer zero_grad
#   forward     → forward pass (CPU launch + GPU execution)
#   loss        → loss computation
#   backward    → backward pass (CPU launch + GPU execution)
#   opt_step    → optimizer step
#
# DEPENDENCIES:
#   loaders/data_loader.py  ← GpuDataset, get_all_step_intervals
#   pandas, matplotlib, collections.Counter
#
# USAGE:
#   from viz.plot_timeline import plot_timeline_custom_axis
#   plot_timeline_custom_axis(
#       gpu_data_map           = gpu_data_map,
#       df_rank_order_per_step = df_rank_order,
#       all_steps_map          = all_steps_map,
#       steps_to_plot          = [2, 3, 4],
#       out_png                = "timeline.png",
#       color_by               = "rank",   # or "step"
#       offsets                = offsets,
#   )
# ─────────────────────────────────────────────────────────────────────────────

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch


# ── Color palette ─────────────────────────────────────────────────────────────
RANK_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]

# ==============================================================================
# Section 1: Helper utilities
# ==============================================================================

def get_merged_intervals(df: pd.DataFrame) -> list:
    """
    Merge overlapping time intervals from a DataFrame into non-overlapping spans.

    Input DataFrame must have 'start' and 'end' columns (nanoseconds).

    Simple example:
        Intervals: [(100, 300), (200, 400), (500, 600)]
        → Merged:  [(100, 400), (500, 600)]
        The first two overlap at 200-300, so they are merged.

    Used by aggregate_gpu_kernels_by_nvtx() to collapse many small kernel
    executions within one NVTX phase into one continuous span.

    Returns list of [start, end] pairs sorted by start time.
    """
    if df is None or df.empty:
        return []

    start_col = next(
        (c for c in ["start", "kernel_start", "gpu_start"] if c in df.columns),
        None,
    )
    end_col = next(
        (c for c in ["end", "kernel_end", "gpu_end"] if c in df.columns),
        None,
    )
    if start_col is None or end_col is None:
        return []

    intervals = []
    for _, row in df.sort_values(start_col).iterrows():
        st, en = row[start_col], row[end_col]
        if not intervals:
            intervals.append([st, en])
        else:
            if st <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], en)
            else:
                intervals.append([st, en])

    return intervals


def aggregate_gpu_kernels_by_nvtx(df_gpu_kernels: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse individual GPU kernel rows into one merged span per NVTX phase.

    df_gpu_duration has one row per individual kernel execution (thousands of rows).
    For plotting purposes we need one row per phase per step — the union of all
    kernel times within that phase.

    Simple example:
        Forward pass has 800 kernels: [t=300~305ns, t=305~312ns, t=312~400ns, ...]
        After merging: one row → gpu_forward_duration: start=300ns, end=9000ns

    Parameters:
        df_gpu_kernels → GpuDataset.df_gpu_duration
                         columns: name, gpu_start, gpu_end, dur_ns,
                                  correlation_id, data_batch_idx

    Returns DataFrame with columns: name, gpu_start, gpu_end, dur_ns, data_batch_idx
    One row per (name, data_batch_idx) combination.
    """
    def merge_intervals_for_group(group):
        df_k = pd.DataFrame({
            "start": group["gpu_start"].values,
            "end":   group["gpu_end"].values,
        })
        intervals = get_merged_intervals(df_k)
        if not intervals:
            return None
        return pd.Series({
            "gpu_start": min(st for st, en in intervals),
            "gpu_end":   max(en for st, en in intervals),
            "dur_ns":    sum(en - st for st, en in intervals),
        })

    result = (
        df_gpu_kernels
        .groupby(["name", "data_batch_idx"])
        .apply(merge_intervals_for_group, include_groups=False)
        .reset_index()
        .dropna()
    )

    result["gpu_start"] = result["gpu_start"].astype(int)
    result["gpu_end"]   = result["gpu_end"].astype(int)
    result["dur_ns"]    = result["dur_ns"].astype(int)

    return result[["name", "gpu_start", "gpu_end", "dur_ns", "data_batch_idx"]]


def filter_by_step_ranges(
    source_df:          pd.DataFrame,
    step_df_sel:        pd.DataFrame,
    global_start:       int,
    use_data_batch_idx: bool = False,
    rank:               int  = 0,
    offsets:            Optional[Dict] = None,
    ) -> pd.DataFrame:
    """
    Filter source_df to only the requested steps and compute relative timestamps.

    After filtering, adds two new columns:
        rel_start_ms → start time relative to global_start, in milliseconds
        dur_ms       → event duration in milliseconds

    Simple example:
        global_start = 21_300_000_000 ns  (rank 0 reference)
        rank 2 offset = +179_868_000 ns   (rank 2 clock is 180ms behind)
        adjusted_global_start = 21_300_000_000 - 179_868_000 = 21_120_132_000 ns

        rank 2 forward start = 21_200_000_000 ns (rank 2 clock)
        rel_start_ms = (21_200_000_000 - 21_120_132_000) / 1e6 = 79.868ms
        → this forward pass started ~80ms after the global step reference

    Parameters:
        source_df          → df_nvtx, df_nccl, df_memcpy, or df_gpu_duration
        step_df_sel        → selected steps from all_steps_map[rank]
        global_start       → earliest start across all ranks (rank 0 clock, ns)
        use_data_batch_idx → if True, filter by data_batch_idx column
        rank               → GPU rank (used to look up clock offset)
        offsets            → dict from calculate_clock_offsets()

    Returns filtered DataFrame with rel_start_ms and dur_ms columns added.
    Returns empty DataFrame if no matching events found.
    """
    adjusted_global_start = (
        global_start - offsets[rank]
        if offsets is not None and rank in offsets
        else global_start
    )

    rows = []
    for _, srow in step_df_sel.iterrows():
        s       = int(srow["step"])
        s_start = int(srow["start"])
        s_end   = int(srow["end"])

        if use_data_batch_idx and "data_batch_idx" in source_df.columns:
            d = source_df[source_df["data_batch_idx"] == s].copy()
        elif "data_batch_idx" in source_df.columns:
            d = source_df[source_df["data_batch_idx"] == s].copy()
        else:
            # Fallback: filter by timestamp range
            start_col = next(
                (c for c in ["start", "kernel_start", "gpu_start"]
                 if c in source_df.columns), None,
            )
            if start_col is None:
                continue
            d = source_df[
                (source_df[start_col] >= s_start) &
                (source_df[start_col] <  s_end)
            ].copy()

        if d.empty:
            continue
        d["step"] = s
        rows.append(d)

    if not rows:
        return pd.DataFrame()

    filtered_df = pd.concat(rows, ignore_index=True)

    start_col = next(
        (c for c in ["start", "kernel_start", "gpu_start"]
         if c in filtered_df.columns), None,
    )
    end_col = next(
        (c for c in ["end", "kernel_end", "gpu_end"]
         if c in filtered_df.columns), None,
    )

    if start_col is None or end_col is None:
        return pd.DataFrame()

    filtered_df = filtered_df.drop_duplicates(subset=["name", start_col, end_col])
    filtered_df["rel_start_ms"] = (filtered_df[start_col] - adjusted_global_start) / 1e6
    filtered_df["dur_ms"]       = (filtered_df[end_col]   - filtered_df[start_col]) / 1e6

    return filtered_df


def draw_bar(
    ax:           plt.Axes,
    rel_start_ms: float,
    dur_ms:       float,
    y_pos:        float,
    lane_height:  float,
    rank_color:   str,
    alpha:        float = 0.9,
    ) -> None:
    """
    Draw one horizontal bar in the timeline using broken_barh.

    Simple example:
        draw_bar(ax, rel_start_ms=50.0, dur_ms=120.0,
                 y_pos=2.1, lane_height=0.2, rank_color="tab:blue")
        → draws a blue bar from x=50ms to x=170ms at y=2.1~2.3
    """
    ax.broken_barh(
        [(rel_start_ms, dur_ms)],
        (y_pos, lane_height),
        facecolors = rank_color,
        alpha      = alpha,
        linewidth  = 0.5,
        edgecolor  = "black",
    )


# ==============================================================================
# Section 2: Main plotting function
# ==============================================================================

def plot_timeline_custom_axis(
    gpu_data_map:           dict,
    df_rank_order_per_step: pd.DataFrame,
    all_steps_map:          dict,
    steps_to_plot:          List[int],
    out_png:                str            = "timeline.png",
    show:                   bool           = False,
    color_by:               str            = "rank",
    offsets:                Optional[Dict] = None,
    ) -> None:
    """
    Draw a multi-GPU Gantt-chart timeline showing CPU and GPU events per phase.

    Each horizontal row = one activity type (forward, backward, NCCL, etc.)
    Each rank gets its own sub-lane within each row.

    TWO COLORING MODES:
        color_by="rank"  → each rank has its own color (default)
                           good for comparing the SAME phase across ranks
        color_by="step"  → each step has its own color
                           good for seeing step boundaries and transitions

    TWO BAR STYLES PER ROW:
        Faint bar  (alpha=0.3) → CPU launch time
                                  how long the CPU spent queuing GPU commands
                                  ends with a black vertical tick mark
        Solid bar  (alpha=1.0) → actual GPU execution time
                                  drawn over the faint CPU bar in the same row

        This overlay shows the CPU-GPU launch gap:
            the gap between the faint bar end and the solid bar start
            = time the GPU was waiting in the command queue

    BOUNDARY LINES:
        Black dashed  → CPU step start (earliest across all ranks)
        Black solid   → CPU step end   (latest across all ranks)
        Blue solid    → GPU step end   (latest GPU kernel end)
        Step label    → shown between CPU start and end at the top

    Parameters:
        gpu_data_map           → dict of {rank: GpuDataset}
        df_rank_order_per_step → output of compute_rank_order_per_step()
        all_steps_map          → output of get_all_step_intervals()
        steps_to_plot          → list of step numbers to include, e.g. [2, 3, 4]
        out_png                → output file path
        show                   → if True, calls plt.show() (False on servers)
        color_by               → "rank" or "step"
        offsets                → clock offset dict from calculate_clock_offsets()
    """

    # ── Global time range ─────────────────────────────────────────────────────
    target_stats = df_rank_order_per_step[
        df_rank_order_per_step["step"].isin(steps_to_plot)
    ]
    if target_stats.empty:
        print(f"[Error] plot_timeline_custom_axis: no data for steps {steps_to_plot}")
        return

    global_start         = int(target_stats["earliest_start"].min())
    global_end           = int(target_stats["bwd_latest_end"].max())
    timeline_duration_ms = (global_end - global_start) / 1e6

    print(f"  global_start : {global_start}")
    print(f"  global_end   : {global_end}")
    print(f"  duration     : {timeline_duration_ms:.2f} ms")

    # ── Y-axis layout ─────────────────────────────────────────────────────────
    # Each row = one activity type
    # Rows are stacked: base_names first (overview), then per-phase detail
    base_names   = ["data_wait", "h2d", "gpu_compute", "NCCL"]
    fixed_nvtx   = ["zero_grad", "forward", "loss", "backward", "opt_step"]
    full_y_names = list(dict.fromkeys(base_names + fixed_nvtx))
    y_map        = {name: i for i, name in enumerate(full_y_names)}

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig_height   = len(full_y_names) * 1.5 + 2
    fig, ax      = plt.subplots(figsize=(24, fig_height))

    colors       = RANK_COLORS
    step_color_map = {
        step: colors[i % len(colors)]
        for i, step in enumerate(sorted(steps_to_plot))
    }

    def get_color(rank: int, step_num: int) -> str:
        if color_by == "step":
            return step_color_map.get(step_num, "tab:gray")
        return colors[rank % len(colors)]

    # ── Lane geometry — each rank gets a sub-lane within each row ─────────────
    lane_height  = 0.8 / len(gpu_data_map)
    sorted_ranks = sorted(gpu_data_map.keys())

    # NVTX phase → y_map key
    phase_to_ymap = {
        "cpu_data_wait_launch":       "data_wait",
        "cpu_h2d_launch":             "h2d",
        "cpu_zero_grad_launch":       "zero_grad",
        "cpu_forward_launch":         "forward",
        "cpu_loss_launch":            "loss",
        "cpu_backward_launch":        "backward",
        "cpu_opt_step_launch":        "opt_step",
        "cpu_nccl_allreduce_launch":  "NCCL",
        "cpu_train_compute_wrapper":  "gpu_compute",
    }

    # GPU duration name → y_map key
    gpu_to_ymap = {
        "gpu_forward_duration":         "forward",
        "gpu_backward_duration":        "backward",
        "gpu_loss_duration":            "loss",
        "gpu_opt_step_duration":        "opt_step",
        "gpu_zero_grad_duration":       "zero_grad",
        "gpu_train_compute_duration":   "gpu_compute",
        "gpu_nccl_allreduce_duration":  "NCCL",
    }

    target_phases = list(phase_to_ymap.keys())

    # ── Draw events for each rank ─────────────────────────────────────────────
    for rank in sorted_ranks:
        dataset     = gpu_data_map[rank]
        rank_offset = rank * lane_height

        # Aggregate individual GPU kernels into per-phase spans
        df_gpu_agg = aggregate_gpu_kernels_by_nvtx(dataset.df_gpu_duration)

        if rank not in all_steps_map:
            continue

        step_df_sel = all_steps_map[rank][
            all_steps_map[rank]["step"].isin(steps_to_plot)
        ]
        if step_df_sel.empty:
            continue

        # Filter each DataFrame to selected steps + compute rel_start_ms
        df_nvtx = filter_by_step_ranges(
            dataset.df_nvtx, step_df_sel, global_start,
            use_data_batch_idx=True, rank=rank, offsets=offsets,
        )
        df_nccl = (
            filter_by_step_ranges(
                dataset.df_nccl, step_df_sel, global_start,
                rank=rank, offsets=offsets,
            )
            if not dataset.df_nccl.empty
            else pd.DataFrame()
        )
        df_memcpy = (
            filter_by_step_ranges(
                dataset.df_memcpy, step_df_sel, global_start,
                use_data_batch_idx=True, rank=rank, offsets=offsets,
            )
            if not dataset.df_memcpy.empty
            else pd.DataFrame()
        )
        df_gpu_duration = (
            filter_by_step_ranges(
                df_gpu_agg, step_df_sel, global_start,
                rank=rank, offsets=offsets,
            )
            if not df_gpu_agg.empty
            else pd.DataFrame()
        )

        # ── [A] gpu_compute row — all GPU duration events ─────────────────────
        if not df_gpu_duration.empty and "gpu_compute" in y_map:
            y_base = y_map["gpu_compute"]
            for _, row in df_gpu_duration.iterrows():
                step_num  = int(row.get("data_batch_idx", row["step"]))
                bar_color = get_color(rank, step_num)
                draw_bar(ax, row["rel_start_ms"], row["dur_ms"],
                         y_base + rank_offset, lane_height, bar_color, alpha=0.9)
                if row["dur_ms"] > 10:
                    ax.text(
                        row["rel_start_ms"] + row["dur_ms"] / 2,
                        y_base + rank_offset + lane_height / 2,
                        f"S{step_num}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white", clip_on=True,
                    )
                if color_by == "step":
                    ax.text(
                        row["rel_start_ms"] + 1,
                        y_base + rank_offset + lane_height / 2,
                        f"R{rank}", ha="left", va="center",
                        fontsize=6, fontweight="bold", color="white", clip_on=True,
                    )

        # ── [B] h2d row — DMA memory copy events ──────────────────────────────
        if "h2d" in y_map and not df_memcpy.empty:
            y_base    = y_map["h2d"]
            current_y = y_base + rank_offset
            for _, row in df_memcpy.iterrows():
                step_num  = int(row.get("data_batch_idx", row["step"]))
                bar_color = get_color(rank, step_num)
                draw_bar(ax, row["rel_start_ms"], row["dur_ms"],
                         current_y, lane_height, bar_color, alpha=1.0)
                if row["dur_ms"] > 10:
                    ax.text(
                        row["rel_start_ms"] + row["dur_ms"] / 2,
                        current_y + lane_height / 2,
                        f"S{step_num}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white", clip_on=True,
                    )

        # ── [C] NCCL row — AllReduce GPU kernels ──────────────────────────────
        if not df_nccl.empty and "NCCL" in y_map:
            y_base    = y_map["NCCL"]
            current_y = y_base + rank_offset
            for _, row in df_nccl.iterrows():
                step_num  = int(row.get("data_batch_idx", row["step"]))
                bar_color = get_color(rank, step_num)
                draw_bar(ax, row["rel_start_ms"], row["dur_ms"],
                         current_y, lane_height, bar_color, alpha=0.9)
                if row["dur_ms"] > 10:
                    ax.text(
                        row["rel_start_ms"] + row["dur_ms"] / 2,
                        current_y + lane_height / 2,
                        f"S{step_num}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white", clip_on=True,
                    )

        # ── [D] Per-phase rows — CPU (faint) + GPU (solid) overlay ───────────
        if not df_nvtx.empty:
            for target_name in target_phases:
                cpu_rows = df_nvtx[df_nvtx["name"] == target_name]
                if cpu_rows.empty:
                    continue

                ymap_key = phase_to_ymap.get(target_name)
                if ymap_key is None or ymap_key not in y_map:
                    continue

                current_y = y_map[ymap_key] + rank_offset

                # CPU events — drawn faint with a black tick at the end
                for _, cpu_row in cpu_rows.iterrows():
                    step_num  = int(cpu_row["step"])
                    bar_color = get_color(rank, step_num)

                    ax.broken_barh(
                        [(cpu_row["rel_start_ms"], cpu_row["dur_ms"])],
                        (current_y, lane_height),
                        facecolors=bar_color, alpha=0.3, linewidth=0,
                    )
                    # Black tick at CPU end = "CPU finished issuing commands here"
                    ax.vlines(
                        x        = cpu_row["rel_start_ms"] + cpu_row["dur_ms"],
                        ymin     = current_y,
                        ymax     = current_y + lane_height,
                        colors   = "black",
                        linewidth= 1.5,
                        alpha    = 0.8,
                    )
                    ax.text(
                        cpu_row["rel_start_ms"] + cpu_row["dur_ms"] / 2,
                        current_y + lane_height / 2,
                        f"S{step_num}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="black", clip_on=True,
                    )

                # GPU events — drawn solid over the CPU bars
                if not df_gpu_duration.empty:
                    gpu_name = next(
                        (k for k, v in gpu_to_ymap.items() if v == ymap_key), None,
                    )
                    if gpu_name:
                        gpu_rows = df_gpu_duration[df_gpu_duration["name"] == gpu_name]
                        for _, gpu_row in gpu_rows.iterrows():
                            step_num  = int(gpu_row.get("data_batch_idx", gpu_row["step"]))
                            bar_color = get_color(rank, step_num)
                            draw_bar(ax, gpu_row["rel_start_ms"], gpu_row["dur_ms"],
                                     current_y, lane_height, bar_color, alpha=1.0)
                            ax.text(
                                gpu_row["rel_start_ms"] + gpu_row["dur_ms"] / 2,
                                current_y + lane_height / 2,
                                f"S{step_num}", ha="center", va="center",
                                fontsize=7, fontweight="bold", color="white", clip_on=True,
                            )

    # ── Step boundary lines ───────────────────────────────────────────────────
    for _, srow in target_stats.iterrows():
        x_cpu_start = (srow["earliest_start"] - global_start) / 1e6
        x_cpu_end   = (srow["latest_end"]     - global_start) / 1e6
        x_gpu_end   = (srow["bwd_latest_end"] - global_start) / 1e6

        ax.axvline(x_cpu_start, color="k",    ls="--",    alpha=0.5)
        ax.axvline(x_cpu_end,   color="k",    ls="solid", alpha=0.5)
        ax.axvline(x_gpu_end,   color="blue", ls="solid", alpha=0.3, linewidth=1.5)

        ax.text(
            (x_cpu_start + x_cpu_end) / 2,
            len(full_y_names),
            f"Step {int(srow['step'])}",
            ha="center", va="bottom", fontweight="bold",
        )

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax.set_xlim(0, timeline_duration_ms)
    ax.set_ylim(0, len(full_y_names))
    ax.set_yticks([i + 0.5 for i in range(len(full_y_names))])
    ax.set_yticklabels(full_y_names, fontsize=11, fontweight="bold")

    for i in range(len(full_y_names) + 1):
        ax.axhline(y=i, color="black", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Time (ms) relative to global step start", fontsize=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.grid(False, axis="y")

    # ── Title — shows GPU model and step list ─────────────────────────────────
    all_gpu_names = [
        next(iter(ds.gpu_info.values()))
        for ds in gpu_data_map.values()
        if ds.gpu_info
    ]
    gpu_counts    = Counter(all_gpu_names)
    gpu_title_str = (
        ", ".join(f"{name} ({cnt} GPUs)" for name, cnt in gpu_counts.items())
        if gpu_counts
        else "Unknown GPU"
    )
    ax.set_title(
        f"[{gpu_title_str}]  Multi-GPU Detailed Timeline  |  Steps: {steps_to_plot}",
        fontsize=14, fontweight="bold",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(
            facecolor = colors[r % len(colors)],
            label     = (
                f"Rank {r} "
                f"({next(iter(gpu_data_map[r].gpu_info.values()), 'GPU')})"
            ),
        )
        for r in sorted_ranks
    ]
    legend_elements += [
        Patch(facecolor="none", edgecolor="black",
              label="CPU step start (dashed) / end (solid)"),
        Patch(facecolor="none", edgecolor="blue",
              label="GPU backward end — bwd_latest_end (blue)"),
    ]
    ax.legend(
        handles          = legend_elements,
        loc              = "upper right",
        title            = "GPU Ranks",
        bbox_to_anchor   = (1.05, 1),
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[plot_timeline_custom_axis] Saved: {out_png}")

    if show:
        plt.show()

    plt.close()