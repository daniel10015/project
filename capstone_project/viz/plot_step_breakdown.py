# plot_step_breakdown.py
# ─────────────────────────────────────────────────────────────────────────────
# Per-step time breakdown visualization for DDP training.
#
# WHAT THIS FILE CONTAINS:
#
#   Section 1: Interval math utilities
#       get_merged_intervals()     ← merge overlapping time intervals
#       get_total_duration_ms()    ← total duration of merged intervals
#       get_intersection_ms()      ← overlap time between two interval lists
#
#   Section 2: Data computation
#       aggregate_per_step_breakdown()  ← compute breakdown DataFrame from raw data
#
#   Section 3: Visualization
#       plot_step_breakdown()      ← produce 3 plots from the breakdown DataFrame
#
# WHAT plot_step_breakdown() PRODUCES (3 PNG files):
#
#   1. *_stacked.png
#      Stacked bar chart — one bar per (rank, step).
#      Bar segments = exclusive and overlapping time categories.
#      Pink dot above each bar = pure_wait as % of NCCL time (straggler metric).
#
#   2. *_grouped_log.png
#      Grouped bar chart on log scale — same data as stacked but side by side.
#      Useful for comparing small values (overlap times) that are invisible
#      on the linear stacked chart.
#
#   3. *_overlap_ratio.png
#      Line chart showing overlap and idle ratios as % of step duration.
#      Tells you how efficiently the GPU is being used per step.
#
# TERMINOLOGY:
#   excl (exclusive) → time spent on this operation with NO other operation
#                       running concurrently. Pure serial time.
#   overlap          → time window where TWO operations ran simultaneously.
#                       e.g. compute + nccl overlap = DDP gradient overlap ✅
#   pure_wait_ms     → time this rank spent idle waiting for the slowest rank
#                       before AllReduce could start. Straggler cost.
#   unaccounted_ms   → step_duration minus all explained categories.
#                       includes: zero_grad, loss CPU, kernel launch latency,
#                       synchronization points, unmeasured GPU time.
#
# PIPELINE:
#   gpu_data_map + wait_df + offsets + df_rank_order_per_step
#       ↓
#   aggregate_per_step_breakdown()  → df_breakdown
#       ↓
#   plot_step_breakdown(df_breakdown)  → 3 PNG files
#
# USAGE:
#   from viz.plot_step_breakdown import (
#       aggregate_per_step_breakdown,
#       plot_step_breakdown,
#   )
#
#   df_breakdown = aggregate_per_step_breakdown(
#       gpu_data_map           = gpu_data_map,
#       steps_to_plot          = [2, 3, 4],
#       nccl_wait_df           = wait_df,
#       offsets                = offsets,
#       df_rank_order_per_step = df_rank_order_per_step,
#   )
#
#   plot_step_breakdown(
#       df_breakdown  = df_breakdown,
#       output_prefix = "plots/resnet_50_bs256_img224_mb50/breakdown/step_breakdown",
#   )
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D

from viz.plot_timeline import (
        get_merged_intervals as _get_merged,
        filter_by_step_ranges,
        aggregate_gpu_kernels_by_nvtx,
    )

from loaders.data_loader import build_step_df_from_nvtx

# ==============================================================================
# Section 1: Interval math utilities
# ==============================================================================

def get_merged_intervals(df: pd.DataFrame) -> list:
    """
    Merge overlapping time intervals from a DataFrame into non-overlapping spans.

    Automatically detects the start/end column names from:
        start columns: 'start', 'kernel_start', 'gpu_start'
        end   columns: 'end',   'kernel_end',   'gpu_end'

    Simple example:
        Intervals: [(100, 300), (200, 400), (500, 600)]
        → Merged:  [(100, 400), (500, 600)]
        The first two overlap (200-300), so they are merged into one span.

    Why this matters:
        If you sum raw durations without merging, overlapping kernels
        get double-counted. Merging gives the true wall-clock time.

    Returns list of [start, end] pairs sorted by start time (nanoseconds).
    Returns empty list if df is None, empty, or has no valid columns.
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


def get_total_duration_ms(intervals: list) -> float:
    """
    Return the total wall-clock duration of a list of merged intervals in ms.

    Simple example:
        intervals = [(0, 100_000_000), (200_000_000, 400_000_000)]
        → total = (100_000_000 + 200_000_000) / 1e6 = 300.0 ms

    Always call get_merged_intervals() first to avoid double-counting overlaps.
    """
    return sum(en - st for st, en in intervals) / 1e6


def get_intersection_ms(ints_a: list, ints_b: list) -> float:
    """
    Return the total overlapping (concurrent) time between two interval lists in ms.

    Uses a two-pointer sweep — O(n + m), works on pre-sorted merged intervals.

    Simple example:
        ints_a = [(0, 500ms), (600ms, 800ms)]   ← GPU compute
        ints_b = [(300ms, 700ms)]                ← NCCL AllReduce
        Overlap regions: (300~500ms) and (600~700ms) = 200ms + 100ms = 300ms
        → 300ms of compute + NCCL concurrent execution (DDP gradient overlap)

    Why this matters:
        Overlap between compute and NCCL is GOOD — it means DDP is successfully
        overlapping gradient AllReduce with backward computation.
        Overlap between prev_gpu and current step events is also good —
        it means GPU pipeline is being utilized across step boundaries.
    """
    i = j = 0
    overlap_ns = 0
    while i < len(ints_a) and j < len(ints_b):
        s_a, e_a = ints_a[i]
        s_b, e_b = ints_b[j]
        overlap_start = max(s_a, s_b)
        overlap_end   = min(e_a, e_b)
        if overlap_start < overlap_end:
            overlap_ns += overlap_end - overlap_start
        if e_a < e_b:
            i += 1
        else:
            j += 1
    return overlap_ns / 1e6


# ==============================================================================
# Section 2: Data computation
# ==============================================================================

def aggregate_per_step_breakdown(
    gpu_data_map:           dict,
    steps_to_plot:          List[int],
    nccl_wait_df:           pd.DataFrame,
    offsets:                Dict[int, int],
    df_rank_order_per_step: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    For each (rank, step) pair, compute a detailed time breakdown showing
    exactly how each millisecond of the step was spent.

    BREAKDOWN CATEGORIES:
        exclusive times (only this operation was running):
            data_wait_excl_ms    → CPU waiting for DataLoader batch
            h2d_excl_ms          → Host-to-Device DMA transfer (no overlap)
            prev_gpu_excl_ms     → Previous step's GPU kernels still running
                                   at the start of the current step (pipeline stall)
            compute_excl_ms      → Forward + backward GPU compute (no overlap)
            nccl_excl_ms         → AllReduce communication (no overlap with compute)

        overlap times (two operations ran concurrently):
            overlap_prev_data_wait_ms → prev step GPU + current step data loading
            overlap_prev_h2d_ms       → prev step GPU + current step H2D transfer
            overlap_prev_compute_ms   → prev step GPU + current step compute 
            overlap_h2d_compute_ms    → H2D transfer + compute (non-blocking H2D )
            overlap_compute_nccl_ms   → Compute + AllReduce (DDP gradient overlap )

        other:
            gpu_idle_ms          → step time where NO GPU kernels were running
            unaccounted_ms       → unexplained time (zero_grad, kernel launch, sync)
            pure_wait_ms         → straggler wait inside NCCL (from wait_df)

    HOW EXCLUSIVE TIME IS COMPUTED (example):
        Simple scalar subtraction is WRONG when overlaps involve 3+ operations:

        Example:
            prev_gpu: [0~100ms]
            h2d:      [50~80ms]   → overlap with prev_gpu = 30ms
            compute:  [70~150ms]  → overlap with prev_gpu = 30ms
            [70~80ms] overlaps with BOTH h2d and compute

            Wrong:  100 - 30 - 30 = 40ms  ← 10ms double-subtracted
            Right:  union([50~80], [70~100]) = [50~100] = 50ms
                    100 - 50 = 50ms  ← correct

        This function uses interval union then intersection to avoid
        the double-subtraction problem.

    Parameters:
        gpu_data_map           → {rank: GpuDataset}
        steps_to_plot          → step from 1 to 9 (9 steps)
        nccl_wait_df           → output of load_all_gpu_compute_wait_time()
        offsets                → clock offsets from calculate_clock_offsets()
        df_rank_order_per_step → output of compute_rank_order_per_step()

    Returns DataFrame with one row per (rank, step) and columns for every
    time category listed above.
    """
    

    target_stats = df_rank_order_per_step[
        df_rank_order_per_step["step"].isin(steps_to_plot)
    ]
    global_start  = int(target_stats["earliest_start"].min())
    all_result    = []
    sorted_steps  = sorted(steps_to_plot)

    for rank, gpu_data in gpu_data_map.items():

        df_nvtx         = gpu_data.df_nvtx
        df_nccl         = gpu_data.df_nccl
        df_memcpy       = gpu_data.df_memcpy
        df_gpu_duration = aggregate_gpu_kernels_by_nvtx(gpu_data.df_gpu_duration)

        step_df     = build_step_df_from_nvtx(df_nvtx)
        step_df_sel = step_df[step_df["step"].isin(steps_to_plot)].sort_values("start")

        if step_df_sel.empty:
            continue

        def safe_filter(df, use_data_batch_idx=False):
            if df is None or df.empty:
                return pd.DataFrame()
            res = filter_by_step_ranges(
                df, step_df_sel, global_start,
                use_data_batch_idx=use_data_batch_idx,
                rank=rank, offsets=offsets,
            )
            return res if res is not None and not res.empty else pd.DataFrame()

        df_nvtx_f         = safe_filter(df_nvtx,         use_data_batch_idx=True)
        df_nccl_f         = safe_filter(df_nccl)
        df_memcpy_f       = safe_filter(df_memcpy,        use_data_batch_idx=True)
        df_gpu_duration_f = safe_filter(df_gpu_duration)

        for step_idx, step in enumerate(sorted_steps):

            step_nvtx  = df_nvtx_f[df_nvtx_f["step"] == step]
            step_nccl  = df_nccl_f[df_nccl_f["step"] == step]
            step_memcp = df_memcpy_f[df_memcpy_f["step"] == step]
            step_gpu   = df_gpu_duration_f[df_gpu_duration_f["step"] == step]

            # ── Step duration ─────────────────────────────────────────────────
            cpu_batch = step_nvtx[
                step_nvtx["name"].str.contains(r"cpu_batch_\d+_duration", regex=True)
            ]
            gpu_batch = step_gpu[
                step_gpu["name"].str.contains(r"gpu_batch_\d+_duration", regex=True)
            ]
            if not cpu_batch.empty and not gpu_batch.empty:
                step_start    = cpu_batch["start"].min()
                step_end      = gpu_batch["gpu_end"].max()
                step_duration = (step_end - step_start) / 1e6
            elif not cpu_batch.empty:
                step_duration = (cpu_batch["end"].max() - cpu_batch["start"].min()) / 1e6
            else:
                step_duration = 0.0

            # ── Filter by operation type ──────────────────────────────────────
            df_data_wait = step_nvtx[step_nvtx["name"] == "cpu_data_wait_launch"]
            df_h2d       = step_memcp[step_memcp["name"] == "gpu_h2d_duration"]
            df_compute   = step_gpu[step_gpu["name"].str.contains(
                r"gpu_forward_duration|gpu_backward_duration|"
                r"gpu_loss_duration|gpu_opt_step_duration",
                regex=True,
            )]
            df_nccl_step = step_nccl

            # ── Previous step GPU spillover ───────────────────────────────────
            # Some ranks may still be running GPU kernels from step N-1 when
            # step N's CPU data_wait starts. This is a pipeline stall when
            # it overlaps with current step compute, or harmless pipelining
            # when it only overlaps with data_wait and H2D.
            if step_idx == 0:
                df_prev_gpu = pd.DataFrame()
            else:
                prev_step = sorted_steps[step_idx - 1]
                df_prev_all = df_gpu_duration_f[df_gpu_duration_f["step"] == prev_step]
                df_prev_compute = df_prev_all[df_prev_all["name"].str.contains(
                    r"gpu_forward_duration|gpu_backward_duration|"
                    r"gpu_loss_duration|gpu_opt_step_duration",
                    regex=True,
                )]
                curr_cpu_start = (
                    df_data_wait["start"].min() if not df_data_wait.empty else None
                )
                if curr_cpu_start is None or df_prev_compute.empty:
                    df_prev_gpu = pd.DataFrame()
                else:
                    prev_last_gpu_end = df_prev_compute["gpu_end"].max()
                    df_prev_gpu = df_prev_compute[
                        df_prev_compute["gpu_end"] >= curr_cpu_start
                    ].copy()
                    df_prev_gpu["clipped_gpu_start"] = df_prev_gpu["gpu_start"].clip(
                        lower=curr_cpu_start
                    )
                    df_prev_gpu["clipped_gpu_end"] = df_prev_gpu["gpu_end"].clip(
                        upper=prev_last_gpu_end
                    )

            # ── Merge intervals ───────────────────────────────────────────────
            data_wait_ints = get_merged_intervals(df_data_wait) if not df_data_wait.empty else []
            h2d_ints       = get_merged_intervals(
                df_h2d.rename(columns={"kernel_start": "start", "kernel_end": "end"})
            ) if not df_h2d.empty else []
            nccl_ints      = get_merged_intervals(
                df_nccl_step.rename(columns={"kernel_start": "start", "kernel_end": "end"})
            ) if not df_nccl_step.empty else []
            compute_ints   = get_merged_intervals(
                df_compute.rename(columns={"gpu_start": "start", "gpu_end": "end"})
            ) if not df_compute.empty else []
            prev_gpu_ints  = get_merged_intervals(
                df_prev_gpu.rename(columns={
                    "clipped_gpu_start": "start",
                    "clipped_gpu_end":   "end",
                })
            ) if not df_prev_gpu.empty else []

            # ── Total durations ───────────────────────────────────────────────
            data_wait_total   = get_total_duration_ms(data_wait_ints)
            h2d_total         = get_total_duration_ms(h2d_ints)
            compute_total     = get_total_duration_ms(compute_ints)
            nccl_total        = get_total_duration_ms(nccl_ints)
            prev_gpu_total    = get_total_duration_ms(prev_gpu_ints)

            # ── GPU SM idle ───────────────────────────────────────────────────
            # SM is busy during: prev_gpu, compute, nccl
            # H2D uses DMA engine — does NOT occupy SMs
            all_sm_busy = (
                [(s, e) for s, e in prev_gpu_ints] +
                [(s, e) for s, e in compute_ints]  +
                [(s, e) for s, e in nccl_ints]
            )
            if all_sm_busy:
                busy_ints  = get_merged_intervals(
                    pd.DataFrame(all_sm_busy, columns=["start", "end"])
                )
            else:
                busy_ints = []
            gpu_busy_ms    = get_total_duration_ms(busy_ints)
            gpu_idle_ms    = max(0.0, step_duration - gpu_busy_ms)
            gpu_idle_ratio = gpu_idle_ms / step_duration if step_duration > 0 else 0.0

            # ── Overlaps ──────────────────────────────────────────────────────
            # ① prev_gpu ↔ data_wait: prev step GPU + current step data loading 
            # ② prev_gpu ↔ h2d:       prev step GPU + current step H2D (DMA) 
            # ③ prev_gpu ↔ compute:   prev step GPU + current step compute 
            # ④ h2d ↔ compute:        H2D + compute (non-blocking=True) 
            # ⑤ compute ↔ nccl:       backward + AllReduce (DDP gradient overlap) 

            overlap_prev_data_wait = get_intersection_ms(prev_gpu_ints, data_wait_ints)
            overlap_prev_h2d       = get_intersection_ms(prev_gpu_ints, h2d_ints)
            overlap_prev_compute   = get_intersection_ms(prev_gpu_ints, compute_ints)
            overlap_h2d_compute    = get_intersection_ms(h2d_ints,      compute_ints)
            overlap_compute_nccl   = get_intersection_ms(compute_ints,  nccl_ints)

            # ── Exclusive times (interval-based to avoid double subtraction) ──
            def _excl(target_ints, other_lists):
                """Subtract the union of all other intervals from target."""
                flat_others = [item for lst in other_lists for item in lst]
                if not flat_others or not target_ints:
                    return get_total_duration_ms(target_ints)
                others_merged = get_merged_intervals(
                    pd.DataFrame(flat_others, columns=["start", "end"])
                )
                overlap = get_intersection_ms(target_ints, others_merged)
                return max(0.0, get_total_duration_ms(target_ints) - overlap)

            prev_gpu_excl  = _excl(prev_gpu_ints,  [data_wait_ints, h2d_ints, compute_ints])
            compute_excl   = _excl(compute_ints,   [nccl_ints, prev_gpu_ints, h2d_ints])
            nccl_excl      = _excl(nccl_ints,      [compute_ints])
            h2d_excl       = _excl(h2d_ints,       [prev_gpu_ints, compute_ints])
            data_wait_excl = _excl(data_wait_ints, [prev_gpu_ints])

            # ── Pure wait (straggler cost inside NCCL) ────────────────────────
            pure_wait = nccl_wait_df[
                (nccl_wait_df["rank"] == rank) &
                (nccl_wait_df["step"] == step)
            ]["pure_wait_ms"].sum()

            # ── Unaccounted time ──────────────────────────────────────────────
            explained = (
                data_wait_excl + h2d_excl + prev_gpu_excl + compute_excl +
                nccl_excl + overlap_prev_data_wait + overlap_prev_h2d +
                overlap_prev_compute + overlap_h2d_compute + overlap_compute_nccl +
                gpu_idle_ms
            )
            unaccounted_ms = max(0.0, step_duration - explained)

            all_result.append({
                "rank":                      rank,
                "step":                      step,
                "step_duration_ms":          step_duration,
                "data_wait_total_ms":        data_wait_total,
                "h2d_total_ms":              h2d_total,
                "compute_total_ms":          compute_total,
                "nccl_total_ms":             nccl_total,
                "prev_gpu_total_ms":         prev_gpu_total,
                "data_wait_excl_ms":         data_wait_excl,
                "h2d_excl_ms":               h2d_excl,
                "prev_gpu_excl_ms":          prev_gpu_excl,
                "compute_excl_ms":           compute_excl,
                "nccl_excl_ms":              nccl_excl,
                "overlap_prev_data_wait_ms": overlap_prev_data_wait,
                "overlap_prev_h2d_ms":       overlap_prev_h2d,
                "overlap_prev_compute_ms":   overlap_prev_compute,
                "overlap_h2d_compute_ms":    overlap_h2d_compute,
                "overlap_compute_nccl_ms":   overlap_compute_nccl,
                "gpu_busy_ms":               gpu_busy_ms,
                "gpu_idle_ms":               gpu_idle_ms,
                "gpu_idle_ratio":            gpu_idle_ratio,
                "pure_wait_ms":              pure_wait,
                "unaccounted_ms":            unaccounted_ms,
            })

    return pd.DataFrame(all_result)


# ==============================================================================
# Section 3: Visualization
# ==============================================================================

# ── Column definitions ────────────────────────────────────────────────────────

# Order of segments in the stacked bar (bottom to top)
PLOT_COLS = [
    "data_wait_excl_ms",
    "h2d_excl_ms",
    "prev_gpu_excl_ms",
    "compute_excl_ms",
    "nccl_excl_ms",
    "overlap_prev_data_wait_ms",
    "overlap_prev_h2d_ms",
    "overlap_prev_compute_ms",
    "overlap_h2d_compute_ms",
    "overlap_compute_nccl_ms",
    "gpu_idle_ms",
    "unaccounted_ms",
]

COL_LABELS = {
    "data_wait_excl_ms":         "Data Wait (excl)",
    "h2d_excl_ms":               "H2D (excl)",
    "prev_gpu_excl_ms":          "Prev GPU (excl)",
    "compute_excl_ms":           "GPU Compute (excl)",
    "nccl_excl_ms":              "NCCL (excl)",
    "overlap_prev_data_wait_ms": "Overlap (Prev+DataWait)",
    "overlap_prev_h2d_ms":       "Overlap (Prev+H2D)",
    "overlap_prev_compute_ms":   "Overlap (Prev+Compute)",
    "overlap_h2d_compute_ms":    "Overlap (H2D+Compute)",
    "overlap_compute_nccl_ms":   "Overlap (Compute+NCCL)",
    "gpu_idle_ms":               "GPU Idle",
    "unaccounted_ms":            "Unaccounted",
}

COL_COLORS = [
    "#1f77b4",  # data_wait_excl        — blue
    "#ff7f0e",  # h2d_excl              — orange
    "#9467bd",  # prev_gpu_excl         — purple
    "#2ca02c",  # compute_excl          — dark green
    "#d62728",  # nccl_excl             — red
    "#aec7e8",  # overlap_prev_data_wait— light blue
    "#ffbb78",  # overlap_prev_h2d      — light orange
    "#c5b0d5",  # overlap_prev_compute  — light purple
    "#f7b6d2",  # overlap_h2d_compute   — light pink
    "#98df8a",  # overlap_compute_nccl  — light green
    "#000000",  # gpu_idle              — black
    "#7f7f7f",  # unaccounted           — grey
]


def plot_step_breakdown(
    df_breakdown:  pd.DataFrame,
    output_prefix: str = "step_breakdown",
    dpi:           int = 200,
) -> None:
    """
    Produce three PNG plots from a step breakdown DataFrame.

    INPUT:
        df_breakdown → output of aggregate_per_step_breakdown()
        output_prefix → path prefix for output files (without extension)
        dpi → image resolution

    OUTPUT FILES:
        {output_prefix}_stacked.png
            Stacked bar chart. Each bar = one (rank, step) combination.
            Segments = time categories stacked bottom to top.
            Pink dot above bar = pure_wait % of NCCL time (straggler cost).

        
        {output_prefix}_overlap_ratio.png
            Line chart showing each overlap category as % of step_duration.
            GPU idle ratio shown as dashed black line.

    HOW TO READ THE STACKED CHART:
        A tall 'GPU Compute (excl)' segment = most time is pure compute 
        A tall 'GPU Idle' segment = GPU sitting idle, wasted time 
        A tall 'NCCL (excl)' segment = AllReduce not overlapping backward 
        A large 'Overlap (Compute+NCCL)' = DDP gradient overlap working 
        Pink dot W:XX% = XX% of NCCL time was wasted waiting for slow rank

    Simple example:
        Step 2, Rank 0:
            compute_excl = 180ms         ← pure compute
            overlap_compute_nccl = 50ms  ← good DDP overlap
            nccl_excl = 20ms             ← small NCCL stall
            gpu_idle = 10ms              ← small idle gap
            W:5%                         ← only 5% of NCCL was wait → balanced

    """
    df = df_breakdown.copy()
    df["label"] = "R" + df["rank"].astype(str) + "_S" + df["step"].astype(str)
    df = df.sort_values(["step", "rank"]).set_index("label")

    valid_cols   = [c for c in PLOT_COLS   if c in df.columns]
    valid_colors = [COL_COLORS[i] for i, c in enumerate(PLOT_COLS) if c in df.columns]
    df_plot      = df[valid_cols].rename(columns=COL_LABELS)
    step_totals  = df["step_duration_ms"]

    # ── Plot 1: Stacked bar chart ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, len(df_plot) * 0.8), 7))

    df_plot.plot(
        kind="bar", stacked=True,
        color=valid_colors, edgecolor="black", linewidth=0.5, ax=ax,
    )

    # Percentage labels inside each segment (only if ≥ 3% of step)
    for n, label in enumerate(df_plot.index):
        total   = step_totals.loc[label]
        cum_val = 0
        for col in df_plot.columns:
            val = df_plot.loc[label, col]
            if val > 0:
                pct     = (val / total) * 100
                y_pos   = cum_val + val / 2
                cum_val += val
                if pct >= 3.0:
                    ax.text(
                        n, y_pos, f"{pct:.1f}%",
                        ha="center", va="center",
                        color="white", fontweight="bold", fontsize=8,
                        path_effects=[
                            path_effects.withStroke(linewidth=2, foreground="black")
                        ],
                    )

    # Pink dot above each bar = pure_wait as % of NCCL total
    # Shows "what fraction of NCCL time was straggler wait"
    if "pure_wait_ms" in df.columns and "nccl_total_ms" in df.columns:
        for n, label in enumerate(df_plot.index):
            pure_wait  = df.loc[label, "pure_wait_ms"]
            nccl_total = df.loc[label, "nccl_total_ms"]
            total      = step_totals.loc[label]
            if nccl_total > 0 and pure_wait > 0:
                ratio = (pure_wait / nccl_total) * 100
                y_pos = total + total * 0.02
                ax.scatter(n, y_pos, s=80, color="#e377c2", zorder=5,
                           marker="o", edgecolors="black", linewidths=0.5)
                ax.text(n, y_pos + total * 0.02, f"W:{ratio:.0f}%",
                        ha="center", va="bottom",
                        fontsize=7, color="#e377c2", fontweight="bold")

    # Step boundary lines
    steps   = df["step"].unique()
    ranks   = df["rank"].unique()
    n_ranks = len(ranks)
    for i in range(1, len(steps)):
        ax.axvline(x=i * n_ranks - 0.5, color="black",
                   linewidth=1.5, linestyle="--", alpha=0.5)
        ax.text(
            (i - 0.5) * n_ranks - 0.5, ax.get_ylim()[1] * 0.95,
            f"Step {steps[i-1]}", ha="center", va="top",
            fontsize=9, fontweight="bold",
        )

    ax.set_title(
        "Per-Step Time Breakdown\n"
        "excl = exclusive time,  overlap = concurrent interval,  "
        "W% = pure_wait / nccl_total",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Rank_Step", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_xticklabels(df_plot.index, rotation=45, ha="right", fontsize=9)

    handles, labels_leg = ax.get_legend_handles_labels()
    dot_handle = Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor="#e377c2", markeredgecolor="black",
        markersize=8, label="Pure Wait % (of NCCL)",
    )
    ax.legend(
        handles=handles[::-1] + [dot_handle],
        labels=labels_leg[::-1] + ["Pure Wait % (of NCCL)"],
        title="Operations", bbox_to_anchor=(1.05, 1),
        loc="upper left", fontsize=9,
    )

    plt.tight_layout()
    out1 = f"{output_prefix}_stacked.png"
    plt.savefig(out1, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot_step_breakdown] Saved: {out1}")

    # ── Plot 2: Grouped bar chart (log scale) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, len(df_plot) * 0.8), 6))

    df_plot.plot(
        kind="bar", stacked=False, logy=True,
        color=valid_colors, edgecolor="black", linewidth=0.5, ax=ax,
    )

    ax.set_title(
        "Step Execution Time Components (Log Scale)\n"
        "Log scale reveals small overlap values invisible on the linear chart",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Rank_Step", fontsize=11)
    ax.set_ylabel("Time (ms) [Log Scale]", fontsize=11)
    ax.set_xticklabels(df_plot.index, rotation=45, ha="right", fontsize=9)
    ax.yaxis.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(title="Operations", bbox_to_anchor=(1.05, 1),
              loc="upper left", fontsize=9)

    plt.tight_layout()
    out2 = f"{output_prefix}_grouped_log.png"
    plt.savefig(out2, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot_step_breakdown] Saved: {out2}")

    # ── Plot 3: Overlap and idle ratios as % of step_duration ────────────────
    fig, ax = plt.subplots(figsize=(max(12, len(df_plot) * 0.8), 5))

    overlap_items = [
        ("overlap_prev_data_wait_ms", "Overlap (Prev+DataWait)", "#aec7e8"),
        ("overlap_prev_h2d_ms",       "Overlap (Prev+H2D)",      "#ffbb78"),
        ("overlap_prev_compute_ms",   "Overlap (Prev+Compute)",  "#c5b0d5"),
        ("overlap_h2d_compute_ms",    "Overlap (H2D+Compute)",   "#f7b6d2"),
        ("overlap_compute_nccl_ms",   "Overlap (Compute+NCCL)",  "#98df8a"),
    ]

    for col, label, color in overlap_items:
        if col in df.columns:
            ratio = df[col] / step_totals * 100
            ax.plot(df.index, ratio, marker="o", label=label,
                    color=color, linewidth=2)

    if "gpu_idle_ratio" in df.columns:
        ax.plot(df.index, df["gpu_idle_ratio"] * 100,
                marker="s", label="GPU Idle", color="#000000",
                linewidth=2, linestyle="--")

    ax.set_title(
        "Overlap & Idle Ratios per Rank/Step  (% of step_duration)\n"
        "Higher compute+NCCL overlap = better DDP efficiency",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Rank_Step", fontsize=11)
    ax.set_ylabel("% of step_duration", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=9)
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="100%")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out3 = f"{output_prefix}_overlap_ratio.png"
    plt.savefig(out3, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot_step_breakdown] Saved: {out3}")