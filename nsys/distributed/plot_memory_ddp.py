import sys
import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D


def pick_col(df, candidates):

    for c in candidates:
        
        if c in df.columns:
            return c
            
    return None


def load_data_from_csv(csv_path):

    df = pd.read_csv(csv_path)

    # ---- pick time col
    t_col = pick_col(df, ["elapsed_ms", "t_ms", "t_us", "t_ns"])

    if t_col is None:
        raise RuntimeError("CSV is missing one of: t_ms / t_us / t_ns")


    # normalize time to ms
    if t_col == "t_ns":
        df["t_ms_norm"] = df["t_ns"] / 1e6
    elif t_col == "t_us":
        df["t_ms_norm"] = df["t_us"] / 1e3
    else:
        df["t_ms_norm"] = df[t_col]
    
    # ---- pick allocated/reserved columns (prefer MB)
    alloc_col = pick_col(df, ["allocated_MB", "allocated_B"])
    reserv_col = pick_col(df, ["reserved_MB", "reserved_B"])
    max_alloc_col = pick_col(df, ["max_allocated_MB", "max_allocated_B"])
    max_reserv_col = pick_col(df, ["max_reserved_MB", "max_reserved_B"])

 
    if alloc_col is None or reserv_col is None:
        raise RuntimeError(
            "CSV is missing allocated/reserved columns. "
            "(Check: allocated_MB/reserved_MB or allocated_B/reserved_B)"
        )

    # ---- convert to MB if needed
    def to_mb(series, colname):
        if colname.endswith("_B"):
            return series / (1024 ** 2)
        return series

    df["allocated_plot"]  = to_mb(df[alloc_col],  alloc_col)
    df["reserved_plot"]   = to_mb(df[reserv_col], reserv_col)

    df["max_alloc_plot"]  = to_mb(df[max_alloc_col],  max_alloc_col) \
                            if max_alloc_col  is not None else np.nan
    df["max_reserv_plot"] = to_mb(df[max_reserv_col], max_reserv_col) \
                            if max_reserv_col is not None else np.nan

    df["slack_plot"] = (df["reserved_plot"] - df["allocated_plot"]).clip(lower=0)


    nccl_col       = pick_col(df, ["nccl_buffer_MB",    "nccl_buffer_B"])
    sent_col       = pick_col(df, ["bytes_sent_MB",     "bytes_sent_B"])
    recv_col       = pick_col(df, ["bytes_recv_MB",     "bytes_recv_B"])
    comm_delta_col = pick_col(df, ["comm_mem_delta_MB", "comm_mem_delta_B"])

    df["nccl_buffer_plot"]    = to_mb(df[nccl_col],       nccl_col) \
                                if nccl_col       is not None else np.nan
    df["bytes_sent_plot"]     = to_mb(df[sent_col],       sent_col) \
                                if sent_col       is not None else np.nan
    df["bytes_recv_plot"]     = to_mb(df[recv_col],       recv_col) \
                                if recv_col       is not None else np.nan
    df["comm_mem_delta_plot"] = to_mb(df[comm_delta_col], comm_delta_col) \
                                if comm_delta_col is not None else np.nan

    df["comm_duration_plot"]  = df["comm_duration_ms"] \
                                if "comm_duration_ms" in df.columns else np.nan

    df["allocated_with_nccl"] = df["allocated_plot"] + df["nccl_buffer_plot"].fillna(0)
    
    df = df.sort_values("t_ms_norm").reset_index(drop=True)

    return df


def plot_memory_per_steps(rank_dfs: dict, out_path="memory.png"):

    RANK_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6",
                   "#e67e22", "#1abc9c", "#e91e63", "#607d8b"]

    step_peak = {}
    reserv_peak = {}
    for rank, df in rank_dfs.items():
        step_peak[rank]   = df.groupby("step")["allocated_with_nccl"].max()
        reserv_peak[rank] = df.groupby("step")["reserved_plot"].max()

    all_steps = sorted(
        set().union(*[p.index.tolist() for p in step_peak.values()])
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 15))  
    fig.suptitle("Memory Usage - Multi-GPU",
                 fontsize=13, fontweight="bold")

    # ── 서브플롯 1: allocated+nccl 만 (rank별 겹치기) ──
    ax = axes[0]
    for rank, peak in step_peak.items():
        color = RANK_COLORS[rank % len(RANK_COLORS)]
        ax.plot(peak.index, peak.values,
                color=color, linewidth=2,
                label=f"rank {rank}")

    ax.set_ylabel("Memory (MB)")
    ax.set_title("Allocated + NCCL buffer (per rank)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(all_steps[::5])   
    ax.grid(axis="y", alpha=0.3)

    all_peaks = pd.DataFrame(step_peak)
    ymin = all_peaks.min().min() * 0.99
    ymax = all_peaks.max().max() * 1.01
    ax.set_ylim(ymin, ymax)


    ax = axes[1]
    for rank in step_peak.keys():
        color  = RANK_COLORS[rank % len(RANK_COLORS)]
        peak   = step_peak[rank]
        reserv = reserv_peak[rank]

        ax.plot(reserv.index, reserv.values,
                color=color, linewidth=2,
                linestyle="--", label=f"rank {rank} reserved")
        ax.fill_between(peak.index,
                        peak.values,
                        reserv.reindex(peak.index).values,
                        color=color, alpha=0.15,
                        label=f"rank {rank} slack")

    ax.set_ylabel("Memory (MB)")
    ax.set_title("Reserved + Slack (reserved - alloc+nccl)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_xticks(all_steps[::5])
    ax.grid(axis="y", alpha=0.3)


    ax = axes[2]
    peak_df   = pd.DataFrame(step_peak).reindex(all_steps)
    mean      = peak_df.mean(axis=1)
    vmin      = peak_df.min(axis=1)
    vmax      = peak_df.max(axis=1)
    diff      = vmax - vmin
    threshold = diff.mean() + diff.std()

    ax.fill_between(all_steps, vmin.values, vmax.values,
                    color="#e74c3c", alpha=0.15,
                    label="rank variance")
    ax.fill_between(all_steps, vmin.values, vmax.values,
                    where=diff.values > threshold,
                    color="#e74c3c", alpha=0.5,
                    label=f"imbalance (>{threshold:.1f} MB)")
    ax.plot(all_steps, mean.values,
            color="black", linewidth=2.5,
            label="mean (all ranks)")
    ax.plot(all_steps, vmin.values,
            color="#e74c3c", linewidth=1,
            linestyle="--", alpha=0.7, label="min rank")
    ax.plot(all_steps, vmax.values,
            color="#e74c3c", linewidth=1,
            linestyle="--", alpha=0.7, label="max rank")

    margin = max(diff.max() * 0.5, 1.0)
    ax.set_ylim(mean.min() - margin, mean.max() + margin)

    ax.set_xlabel("Step")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Option 3 - Imbalance band (parallelism bottleneck)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(all_steps[::5])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":


    csv_dir = "."

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--dir" and i + 1 < len(args):
            csv_dir = args[i + 1]
        elif arg.isdigit():
            target_steps.append(int(arg))

    # Detect all CSV files in the target directory
    all_files = sorted(glob.glob(os.path.join(csv_dir, "mem_log_rank*.csv")))

    if not all_files:
        print(f"[Error] No mem_log_rank*.csv files found in '{csv_dir}'")
        sys.exit(1)


    # Group files by common prefix (strip rank number)
    # e.g. profile_bs128_rank0, profile_bs128_rank1 -> group "profile_bs128"
    from collections import defaultdict
    import re

    groups = defaultdict(list)

    for f in all_files:
        basename = os.path.basename(f)
        # Remove rank suffix to generate group key
        group_key = re.sub(r'_?rank\d+', '', basename).replace('.csv', '')
        groups[group_key].append(f)

    # Print available groups
    group_keys = sorted(groups.keys())
    print("\nAvailable experiment groups:")
    for i, key in enumerate(group_keys):
        files = groups[key]
        print(f"  [{i}] {key}  ({len(files)} ranks)")
        for f in sorted(files):
            print(f"       - {os.path.basename(f)}")

    # Prompt user to select a group
    print()
    choice = input("Select group number: ").strip()
    try:
        idx = int(choice)
        selected_key = group_keys[idx]
    except (ValueError, IndexError):
        print("[Error] Invalid input.")
        sys.exit(1)

    csv_files = sorted(groups[selected_key])

    print(f"\n--- Configuration ---")
    print(f"Selected experiment: {selected_key}")
    print(f"Files ({len(csv_files)}):")

    for f in csv_files:
        print(f"  - {f}")

    print("---------------------\n")



    rank_dfs = {}   

    for i, filepath in enumerate(csv_files):
        print(f"[로드 {i}] {filepath}")
        df   = load_data_from_csv(filepath)
        rank = df["rank"].iloc[0]   
        rank_dfs[rank] = df
        print(f"  → rank {rank} | {len(df)}개 레코드")

    combined_df = pd.concat(rank_dfs.values(), ignore_index=True)


    print(f"\n총 {len(combined_df)}개 레코드 로드 완료")
    print(f"스텝 수: {combined_df['step'].nunique()}")
    print(f"GPU 수:  {combined_df['rank'].nunique()}")

    plot_memory_per_steps(
        rank_dfs,
        out_path=f"{selected_key}_memory.png"
    )
    for rank, df in rank_dfs.items():
        peak = df.groupby("step")["allocated_with_nccl"].max()
        print(f"rank {rank}: min={peak.min():.1f} max={peak.max():.1f}")


