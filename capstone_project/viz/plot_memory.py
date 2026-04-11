# plot_memory.py
# ─────────────────────────────────────────────────────────────────────────────
# GPU memory usage visualization for DDP training.
#
# WHAT THIS PLOTS:
#   Three types of memory plots from the CSV files produced by MemoryLogger:
#
#   1. plot_memory_per_steps()
#      Line chart: allocated + NCCL buffer per step per rank (3 panels)
#
#   2. plot_phase_memory_by_rank()
#      Bar chart: memory by training phase for one representative step
#
#   3. plot_memory_breakdown_per_phase()
#      Stacked bar: memory components per phase per rank per target step
#
# INPUT:
#   CSV files from MemoryLogger (mem_log_*rank*.csv in the mem/ directory)
#   Loaded via load_mem_csvs()
#
# USAGE:
#   from viz.plot_memory import load_mem_csvs, plot_memory_per_steps, \
#                               plot_phase_memory_by_rank, \
#                               plot_memory_breakdown_per_phase
#
#   rank_dfs, combined_df = load_mem_csvs("mem/")
#   plot_memory_per_steps(rank_dfs, out_path="plots/memory/memory_trend.png")
#   plot_phase_memory_by_rank(combined_df, out_path="plots/memory/phase.png")
#   plot_memory_breakdown_per_phase(rank_dfs, [2, 3], out_dir="plots/memory/")
# ─────────────────────────────────────────────────────────────────────────────

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RANK_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6",
    "#e67e22", "#1abc9c", "#e91e63", "#607d8b",
]


# ==============================================================================
# Section 1: Data loading
# ==============================================================================

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_single_csv(csv_path: str) -> pd.DataFrame:
    """
    Load one mem_log CSV and normalise all columns to MB and ms.

    Added columns:
        t_ms_norm           elapsed time in ms
        allocated_plot      allocated memory in MB
        reserved_plot       reserved memory in MB
        slack_plot          reserved - allocated (clamped >= 0)
        nccl_buffer_plot    NCCL buffer in MB
        bytes_sent_plot     bytes sent in MB
        allocated_with_nccl allocated + NCCL buffer
    """
    df = pd.read_csv(csv_path)

    t_col = _pick_col(df, ["elapsed_ms", "t_ms", "t_us", "t_ns"])
    if t_col is None:
        raise RuntimeError(f"CSV missing time column: {csv_path}")

    if t_col == "t_ns":
        df["t_ms_norm"] = df["t_ns"] / 1e6
    elif t_col == "t_us":
        df["t_ms_norm"] = df["t_us"] / 1e3
    else:
        df["t_ms_norm"] = df[t_col]

    alloc_col      = _pick_col(df, ["allocated_MB",     "allocated_B"])
    reserv_col     = _pick_col(df, ["reserved_MB",      "reserved_B"])
    max_alloc_col  = _pick_col(df, ["max_allocated_MB", "max_allocated_B"])
    max_reserv_col = _pick_col(df, ["max_reserved_MB",  "max_reserved_B"])

    if alloc_col is None or reserv_col is None:
        raise RuntimeError(f"CSV missing allocated/reserved columns: {csv_path}")

    def to_mb(series, colname):
        return series / (1024 ** 2) if colname.endswith("_B") else series

    df["allocated_plot"]  = to_mb(df[alloc_col],  alloc_col)
    df["reserved_plot"]   = to_mb(df[reserv_col], reserv_col)
    df["max_alloc_plot"]  = to_mb(df[max_alloc_col],  max_alloc_col)  if max_alloc_col  else np.nan
    df["max_reserv_plot"] = to_mb(df[max_reserv_col], max_reserv_col) if max_reserv_col else np.nan
    df["slack_plot"]      = (df["reserved_plot"] - df["allocated_plot"]).clip(lower=0)

    nccl_col       = _pick_col(df, ["nccl_buffer_MB",    "nccl_buffer_B"])
    sent_col       = _pick_col(df, ["bytes_sent_MB",     "bytes_sent_B"])
    recv_col       = _pick_col(df, ["bytes_recv_MB",     "bytes_recv_B"])
    comm_delta_col = _pick_col(df, ["comm_mem_delta_MB", "comm_mem_delta_B"])

    df["nccl_buffer_plot"]    = to_mb(df[nccl_col],       nccl_col)       if nccl_col       else np.nan
    df["bytes_sent_plot"]     = to_mb(df[sent_col],       sent_col)       if sent_col       else np.nan
    df["bytes_recv_plot"]     = to_mb(df[recv_col],       recv_col)       if recv_col       else np.nan
    df["comm_mem_delta_plot"] = to_mb(df[comm_delta_col], comm_delta_col) if comm_delta_col else np.nan
    df["comm_duration_plot"]  = df["comm_duration_ms"] if "comm_duration_ms" in df.columns else np.nan

    df["allocated_with_nccl"] = df["allocated_plot"] + df["nccl_buffer_plot"].fillna(0)

    # ── Component columns (param, grad, opt_state, activation) ───────────────
    # These are recorded directly by MemoryLogger if the training script
    # tracks them via model parameter introspection.
    # Pass-through: keep as-is if present, otherwise leave absent (don't add nan cols)
    for col in ["param_MB", "grad_MB", "opt_state_MB", "activation_MB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Also expose allocated_MB as a plain float for get_val() lookups
    if "allocated_MB" not in df.columns and "allocated_plot" in df.columns:
        df["allocated_MB"] = df["allocated_plot"]

    return df.sort_values("t_ms_norm").reset_index(drop=True)


def load_mem_csvs(mem_dir: str) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Load all mem_log CSV files from mem_dir.

    Simple example:
        mem/ contains mem_log_bs32_img224_mb50_rank0.csv ... rank3.csv
        load_mem_csvs("mem/")
        -> rank_dfs    = {0: df0, 1: df1, 2: df2, 3: df3}
        -> combined_df = all 4 DataFrames concatenated

    Returns:
        rank_dfs    -> {rank: DataFrame}
        combined_df -> all ranks concatenated
    """
    all_files = sorted(glob.glob(os.path.join(mem_dir, "mem_log_*rank*.csv")))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(mem_dir, "*rank*.csv")))
    if not all_files:
        raise FileNotFoundError(
            f"No mem_log CSV files found in '{mem_dir}'\n"
            f"Expected: mem_log_*rank*.csv"
        )

    rank_dfs = {}
    for filepath in all_files:
        print(f"  Loading: {os.path.basename(filepath)}")
        df   = _load_single_csv(filepath)
        rank = int(df["rank"].iloc[0]) if "rank" in df.columns else len(rank_dfs)
        rank_dfs[rank] = df
        print(f"    rank {rank} | {len(df)} records | "
              f"steps {df['step'].min()}~{df['step'].max()}")

    combined_df = pd.concat(rank_dfs.values(), ignore_index=True)
    print(f"\n  Total: {len(combined_df)} records | "
          f"{combined_df['step'].nunique()} steps | "
          f"{combined_df['rank'].nunique()} ranks")

    return rank_dfs, combined_df


# ==============================================================================
# Section 2: plot_memory_per_steps
# ==============================================================================

def plot_memory_per_steps(
    rank_dfs: Dict[int, pd.DataFrame],
    out_path: str = "memory_trend.png",
    dpi:      int = 150,
) -> None:
    """
    Three-panel memory trend plot across all training steps.

    Panel 1 - Allocated + NCCL buffer per rank (active usage)
    Panel 2 - Reserved memory + slack per rank (cache held by PyTorch)
    Panel 3 - Imbalance band (max - min across ranks, dark = alert)

    Simple example:
        If rank 2 uses 200MB more than others in Panel 1 AND
        Panel 3 shows wide dark-red band -> rank 2 is a memory bottleneck.
    """
    step_peak   = {r: df.groupby("step")["allocated_with_nccl"].max()
                   for r, df in rank_dfs.items()}
    reserv_peak = {r: df.groupby("step")["reserved_plot"].max()
                   for r, df in rank_dfs.items()}
    all_steps   = sorted(set().union(*[p.index.tolist() for p in step_peak.values()]))

    fig, axes = plt.subplots(3, 1, figsize=(16, 15))
    fig.suptitle("GPU Memory Usage — Multi-Rank DDP Training",
                 fontsize=13, fontweight="bold")

    # Panel 1
    ax = axes[0]
    for rank, peak in step_peak.items():
        ax.plot(peak.index, peak.values,
                color=RANK_COLORS[rank % len(RANK_COLORS)],
                linewidth=2, label=f"Rank {rank}")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Allocated + NCCL buffer per rank\n"
                 "active GPU memory usage per step")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(all_steps[::5])
    ax.grid(axis="y", alpha=0.3)
    all_peaks = pd.DataFrame(step_peak)
    ax.set_ylim(all_peaks.min().min() * 0.99, all_peaks.max().max() * 1.01)

    # Panel 2
    ax = axes[1]
    for rank in step_peak.keys():
        color  = RANK_COLORS[rank % len(RANK_COLORS)]
        peak   = step_peak[rank]
        reserv = reserv_peak[rank]
        ax.plot(reserv.index, reserv.values,
                color=color, linewidth=2, linestyle="--",
                label=f"Rank {rank} reserved")
        ax.fill_between(peak.index, peak.values,
                        reserv.reindex(peak.index).values,
                        color=color, alpha=0.15,
                        label=f"Rank {rank} slack")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Reserved memory + slack\n"
                 "slack = reserved - (allocated + NCCL)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_xticks(all_steps[::5])
    ax.grid(axis="y", alpha=0.3)

    # Panel 3
    ax        = axes[2]
    peak_df   = pd.DataFrame(step_peak).reindex(all_steps)
    mean      = peak_df.mean(axis=1)
    vmin      = peak_df.min(axis=1)
    vmax      = peak_df.max(axis=1)
    diff      = vmax - vmin
    threshold = diff.mean() + diff.std()

    ax.fill_between(all_steps, vmin.values, vmax.values,
                    color="#e74c3c", alpha=0.15, label="rank variance")
    ax.fill_between(all_steps, vmin.values, vmax.values,
                    where=diff.values > threshold,
                    color="#e74c3c", alpha=0.5,
                    label=f"imbalance (>{threshold:.1f} MB)")
    ax.plot(all_steps, mean.values,
            color="black", linewidth=2.5, label="mean all ranks")
    ax.plot(all_steps, vmin.values,
            color="#e74c3c", linewidth=1, linestyle="--", alpha=0.7, label="min rank")
    ax.plot(all_steps, vmax.values,
            color="#e74c3c", linewidth=1, linestyle="--", alpha=0.7, label="max rank")

    margin = max(diff.max() * 0.5, 1.0)
    ax.set_ylim(mean.min() - margin, mean.max() + margin)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory imbalance band across ranks\n"
                 "dark red = steps where one rank uses significantly more memory")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(all_steps[::5])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot_memory_per_steps] Saved: {out_path}")


# ==============================================================================
# Section 3: plot_phase_memory_by_rank
# ==============================================================================

def plot_phase_memory_by_rank(
    combined_df: pd.DataFrame,
    out_path:    str = "phase_memory.png",
    dpi:         int = 150,
) -> None:
    """
    Grouped bar chart — 6 training phases x N ranks.

    Each phase bar shows the SIZE of one memory component at that phase:

        Idle       → param_MB at 'batch_start'
                     = model parameter memory (constant throughout training)

        Forward    → activation_MB at 'after_forward'
                     = intermediate tensors kept alive for backward pass

        Backward   → grad_MB at 'after_backward'
                     = gradient buffers accumulated during backward

        AllReduce  → bytes_sent_MB max in step
                     = NCCL communication volume (not in-GPU memory)

        Optimizer  → opt_state_MB at 'after_opt_step'
                     = Adam/SGD moment buffers (1st + 2nd moments per param)

        Cleanup    → (allocated_MB - param_MB - opt_state_MB) at 'step_end'
                     = residual memory that should be ~0 (no leak)

    Simple example:
        param=94MB  activation=800MB  grad=600MB  nccl=94MB
        opt=282MB   cleanup=0MB
        Total peak = 94 + 800 + 600 + 282 = 1776MB at backward end

    Uses second-to-last step as reference (more stable than last step).
    """
    # Find the best reference step — needs non-zero param_MB values.
    # MemoryLogger only records detailed component breakdown every N steps,
    # so most steps have param_MB=0. We pick the last step that has real data.
    candidate_steps = sorted(combined_df["step"].unique())

    target_step = None
    for step in reversed(candidate_steps):
        step_df_check = combined_df[combined_df["step"] == step]
        if (step_df_check["param_MB"] != 0).any():
            target_step = step
            break

    # Fallback: if no step has param_MB data, use second-to-last step
    if target_step is None:
        target_step = candidate_steps[-2] if len(candidate_steps) > 1 else candidate_steps[0]
        print(f"  [Warning] No step has param_MB data. Using step {target_step} (allocated_MB only).")
    else:
        print(f"  [plot_phase_memory_by_rank] reference step = {target_step} (last step with component data)")

    step_df = combined_df[combined_df["step"] == target_step]
    ranks   = sorted(step_df["rank"].unique())

    phases       = [
        "Idle\n(Params)",
        "Forward\n(Activations)",
        "Backward\n(Gradients)",
        "AllReduce\n(NCCL Vol.)",
        "Optimizer\n(States)",
        "Cleanup\n(Residual)",
    ]
    phase_colors = ["#95a5a6", "#2ecc71", "#e74c3c", "#9b59b6", "#e67e22", "#bdc3c7"]

    def get_val(rdf, tag, col, default=0.0):
        rows = rdf[rdf["tag"] == tag]
        if rows.empty or col not in rows.columns:
            return default
        v = rows[col].iloc[0]
        return float(v) if not pd.isna(v) else default

    rank_data = {}
    for rank in ranks:
        rdf = step_df[step_df["rank"] == rank]

        # ① Idle: param size is constant — use batch_start
        idle_mb = get_val(rdf, "batch_start", "param_MB")

        # ② Forward: activation tensors kept for backward
        forward_mb = get_val(rdf, "after_forward", "activation_MB")

        # ③ Backward: gradient buffers accumulated
        backward_mb = get_val(rdf, "after_backward", "grad_MB")

        # ④ AllReduce: NCCL communication volume (bytes sent)
        nccl_mb = 0.0
        if "bytes_sent_MB" in rdf.columns:
            nccl_mb = float(rdf["bytes_sent_MB"].max())
        if pd.isna(nccl_mb):
            nccl_mb = 0.0

        # ⑤ Optimizer: Adam/SGD moment buffers
        opt_mb = get_val(rdf, "after_opt_step", "opt_state_MB")

        # ⑥ Cleanup: residual after step (should be ~0 if no leak)
        end_alloc = get_val(rdf, "step_end", "allocated_MB")
        end_param = get_val(rdf, "step_end", "param_MB")
        end_opt   = get_val(rdf, "step_end", "opt_state_MB")
        cleanup_mb = max(0.0, end_alloc - end_param - end_opt)

        rank_data[rank] = [
            idle_mb, forward_mb, backward_mb,
            nccl_mb, opt_mb, cleanup_mb,
        ]

    # ── Draw grouped bars ─────────────────────────────────────────────────────
    fig, ax   = plt.subplots(figsize=(16, 8))
    x         = np.arange(len(phases))
    num_ranks = len(ranks)
    bar_width = 0.8 / num_ranks

    for i, rank in enumerate(ranks):
        pos = x + (i - num_ranks / 2 + 0.5) * bar_width
        for j in range(len(phases)):
            val       = rank_data[rank][j]
            phase_col = phase_colors[j]

            ax.bar(pos[j], val, width=bar_width,
                   color=phase_col, edgecolor="black",
                   linewidth=0.8, alpha=0.88)

            if val > 0:
                # Value label above bar
                ax.text(pos[j], val + val * 0.015 + 3,
                        f"{val:.0f}M",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#2c3e50")
                # Rank label inside bar
                label_y = val * 0.5 if val > 50 else val + 5
                ax.text(pos[j], label_y, f"R{rank}",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white",
                        bbox=dict(facecolor="black", alpha=0.3,
                                  pad=1.0, edgecolor="none"))

    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=12, fontweight="bold")
    ax.set_ylabel("Memory Component Size (MB)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Memory Bottleneck by Phase & Rank  (Step {target_step})\n"
        f"Each bar = one memory component directly from MemoryLogger CSV",
        fontsize=14, fontweight="bold", pad=20,
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=phase_colors[j], alpha=0.88,
                       label=phases[j].replace("\n", " "))
        for j in range(len(phases))
    ]
    ax.legend(handles=legend_handles, fontsize=9,
              loc="upper right", ncol=2,
              title="Memory Component", title_fontsize=9)

    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot_phase_memory_by_rank] Saved: {out_path}")


# ==============================================================================
# Section 4: plot_memory_breakdown_per_phase
# ==============================================================================

def plot_memory_breakdown_per_phase(
    rank_dfs:     Dict[int, pd.DataFrame],
    target_steps: List[int],
    out_dir:      str = ".",
    tag:          str = "",
    dpi:          int = 150,
) -> None:
    """
    For each requested step, draw a stacked bar chart showing allocated memory
    and NCCL buffer at each phase checkpoint, one subplot per rank.

    WHY THIS DESIGN:
        MemoryLogger records total allocated_MB and nccl_buffer_MB at each tag.
        It does NOT break down memory by component (param/grad/activation/opt)
        because that requires model introspection not available via CUPTI.

        Instead this chart shows:
            Allocated (non-NCCL)  = the blue bar   = all GPU tensors combined
            NCCL buffer           = the red bar     = communication buffer on top

        The total bar height = actual GPU memory in use at that checkpoint.

    HOW TO READ IT:
        Tall bar at "after_forward"  → large activation memory footprint
        Tall bar at "after_backward" → gradients accumulated
        Red bar appears              → NCCL AllReduce buffer allocated
        Drop from "after_backward" to "after_opt_step" → gradients freed

    Simple example (step 2, rank 0):
        batch_start:    alloc=5000MB, nccl=0     → total 5000MB
        after_forward:  alloc=5800MB, nccl=0     → +800MB activations
        after_backward: alloc=6400MB, nccl=0     → +600MB gradients
        after_opt_step: alloc=5000MB, nccl=94MB  → gradients freed, NCCL active
        step_end:       alloc=5000MB, nccl=0     → back to baseline

    One subplot per rank. One PNG per target_step.

    Parameters:
        rank_dfs      → dict from load_mem_csvs()
        target_steps  → list of step numbers to plot
        out_dir       → directory to save PNGs
        tag           → experiment tag appended to filename
        dpi           → image resolution
    """
    suffix = f"_{tag}" if tag else ""

    for step in target_steps:
        if not any(step in df["step"].values for df in rank_dfs.values()):
            print(f"  [Skip] Step {step} not found in any rank.")
            continue

        num_ranks = len(rank_dfs)
        fig, axes = plt.subplots(num_ranks, 1,
                                 figsize=(12, 4 * num_ranks), squeeze=False)
        fig.suptitle(
            f"GPU Memory at Each Phase Checkpoint  (Step {step})\n"
            f"Blue = allocated memory   Red = NCCL buffer",
            fontsize=13, fontweight="bold", y=1.02,
        )

        for i, (rank, df) in enumerate(sorted(rank_dfs.items())):
            ax      = axes[i, 0]
            df_step = df[df["step"] == step].copy()

            if df_step.empty:
                ax.set_title(f"Rank {rank} — no data for step {step}")
                continue

            # Use only the known phase tags that exist in this step
            available = [t for t in PHASE_TAGS if t in df_step["tag"].values]
            if not available:
                ax.set_title(f"Rank {rank} — no phase tags found")
                continue

            df_plot = (
                df_step[df_step["tag"].isin(available)]
                .drop_duplicates("tag")
                .set_index("tag")
                .reindex(available)
            )

            alloc_vals = df_plot["allocated_plot"].fillna(0).values
            nccl_vals  = df_plot["nccl_buffer_plot"].fillna(0).values

            labels = [PHASE_LABELS[PHASE_TAGS.index(t)] for t in available]
            x      = np.arange(len(available))

            # Stacked bar: base = allocated, top = NCCL buffer
            bars1 = ax.bar(x, alloc_vals, color="#3498db", alpha=0.85,
                           width=0.6, label="Allocated memory")
            bars2 = ax.bar(x, nccl_vals, bottom=alloc_vals,
                           color="#e74c3c", alpha=0.85,
                           width=0.6, label="NCCL buffer")

            # Annotate total height
            for xi, (a, n) in enumerate(zip(alloc_vals, nccl_vals)):
                total = a + n
                if total > 0:
                    ax.text(xi, total + total * 0.01 + 2,
                            f"{total:.0f}",
                            ha="center", va="bottom",
                            fontsize=8, fontweight="bold", color="#2c3e50")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Memory (MB)", fontsize=9)
            ax.set_title(f"Rank {rank}", fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if i == 0:
                ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"memory_breakdown_step{step}{suffix}.png")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[plot_memory_breakdown_per_phase] Saved: {out_path}")