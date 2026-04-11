# clock_offset.py
# ─────────────────────────────────────────────────────────────────────────────
# Cross-rank GPU clock offset calculation.
#
# PROBLEM THIS FILE SOLVES:
#   When training runs across multiple nodes (machines), each machine has its
#   own hardware clock. These clocks are never perfectly synchronized — one
#   node's clock might read 17ms ahead of another's even though they are
#   running the same training step at the same real-world time.
#
#   If you plot timelines from different ranks without correcting for this,
#   it looks like rank 2 started its step 17ms before rank 0, even though
#   they actually started at the same moment.
#
# HOW IT IS FIXED:
#   AllReduce is a collective operation — ALL ranks must participate at the
#   same time. If rank 0 shows AllReduce ending at t=100ms and rank 2 shows
#   the same AllReduce ending at t=83ms, then rank 2's clock is 17ms behind
#   rank 0's clock. We compute this difference and store it as an offset.
#
#   offset[rank] = rank0_clock_time - rankN_clock_time
#
#   To convert rank N's timestamp to rank 0's clock:
#       corrected_time = rankN_time + offset[rankN]
#
# PIPELINE POSITION:
#   data_loader.py  →  clock_offset.py  →  offsets dict
#                                           ↓
#                                       wait_analysis.py  (needs corrected times)
#                                       compute_rank_order_per_step()
#                                       build_sm_timeline_df()
#
# FUNCTIONS IN THIS FILE:
#   get_allreduce_df()       ← filter df_nccl to AllReduce rows only
#   calculate_clock_offsets() ← main function, returns {rank: offset_ns}
#   debug_clock_offsets()    ← prints raw numbers for manual inspection
#
# DEPENDENCIES:
#   numpy, pandas only — no imports from loaders/ or analysis/
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict

import numpy as np
import pandas as pd


# ==============================================================================
# Section 1: get_allreduce_df
# ==============================================================================

def get_allreduce_df(df_nccl: pd.DataFrame) -> pd.DataFrame:
    """
    Filter df_nccl to keep only AllReduce kernel rows.

    AllReduce is the only NCCL operation that ALL ranks participate in
    simultaneously — making it the ideal reference point for clock alignment.
    Other NCCL operations (Broadcast, AllGather) happen at different times
    across ranks and cannot be used for clock comparison.

    Simple example:
        df_nccl has these kernel names:
            'ncclDevKernel_AllGather_RING_LL'       ← skip
            'ncclDevKernel_Broadcast_RING_LL'        ← skip
            'ncclDevKernel_AllReduce_Sum_f32_RING_LL' ← keep 
            'ncclDevKernel_AllReduce_Sum_f32_TREE_LL' ← keep 

        → returns only the two AllReduce rows

    Parameters:
        df_nccl → GpuDataset.df_nccl

    Returns:
        Filtered DataFrame with only AllReduce rows.
        Returns empty DataFrame if df_nccl is empty.
    """
    if df_nccl.empty:
        return pd.DataFrame()

    return df_nccl[df_nccl["name"].str.contains("AllReduce", case=False)]


# ==============================================================================
# Section 2: calculate_clock_offsets
# ==============================================================================

def calculate_clock_offsets(
    gpu_data_map: dict,
    n_kernels:    int = 20,
    ) -> Dict[int, int]:
    """
    Calculate the clock offset between each rank and rank 0 (the reference).

    WHY THE MEDIAN:
        AllReduce end times should be identical across ranks (since all ranks
        must finish before any can proceed). Any difference is pure clock skew.
        We take the median across n_kernels AllReduce operations to get a
        stable estimate that is robust to occasional outliers.

        Simple example:
            Per-bucket end time differences (rank 0 - rank 2) across 5 buckets:
                [17.1ms, 17.0ms, 16.9ms, 17.2ms, 31.5ms]  ← last is outlier
            median = 17.1ms  ← robust to the outlier
            mean   = 19.9ms  ← pulled up by the outlier, wrong

    HOW THE OFFSET IS USED:
        After getting offsets = {0: 0, 1: -5ms, 2: +17ms, 3: +16ms}:

        To compare timestamps across ranks, add the offset:
            rank2_corrected = rank2_timestamp + offset[2]
            rank2_corrected = rank2_timestamp + 17ms

        This shifts rank 2's clock forward to match rank 0.

    Simple example:
        AllReduce bucket 0:
            Rank 0 ends at t = 1000ms  (reference)
            Rank 1 ends at t =  983ms  → offset[1] = 1000 - 983 = +17ms
            Rank 2 ends at t = 1005ms  → offset[2] = 1000 - 1005 = -5ms

        After correction:
            Rank 1 corrected: 983ms + 17ms = 1000ms  matches rank 0
            Rank 2 corrected: 1005ms + (-5ms) = 1000ms matches rank 0

    Parameters:
        gpu_data_map → dict of {rank: GpuDataset}
        n_kernels    → how many AllReduce operations to use for estimation
                       more = more stable, but only useful if training ran
                       long enough to have that many AllReduce calls
                       default = 20

    Returns:
        Dict[int, int] mapping rank → offset in nanoseconds
        Rank 0 always has offset = 0 (it is the reference clock).

        Example:
            {0: 0, 1: 17165371, 2: -5000000, 3: 16800000}
            units: nanoseconds
    """
    # Step 1: collect AllReduce end times for each rank, sorted by time
    rank_ends = {}

    for rank, dataset in gpu_data_map.items():

        allreduce_df = get_allreduce_df(dataset.df_nccl)

        if allreduce_df.empty:
            rank_ends[rank] = np.array([])
            continue

        rank_ends[rank] = (
            allreduce_df
            .sort_values("kernel_start")["kernel_end"]
            .values
        )

    # Step 2: rank 0 is the reference — compute offsets for all other ranks
    base_ends = rank_ends.get(0, np.array([]))
    
    n         = min(len(base_ends), n_kernels)

    offsets    = {0: 0}

    print("\n=== Starting Clock Offset Calculation ===")

    for rank in sorted(gpu_data_map.keys()):
        if rank == 0:
            continue

        comp_ends = rank_ends.get(rank, np.array([]))
        if len(comp_ends) == 0:
            print(f"  Rank {rank}: No AllReduce data — offset set to 0")
            offsets[rank] = 0
            continue

        # Compare the same-index AllReduce bucket across ranks
        # Both arrays are sorted by kernel_start, so index i in rank 0
        # corresponds to the same physical AllReduce as index i in rank N

        m               = min(n, len(comp_ends))
        per_bucket_diff = base_ends[:m] - comp_ends[:m]

        offset          = int(np.median(per_bucket_diff))
        offsets[rank]   = offset

    print("\n=== Final Offsets (relative to Rank 0) ===")
    for rank, offset in sorted(offsets.items()):
        print(f"  Rank {rank}: {offset / 1e6:+.3f} ms")

    return offsets


# ==============================================================================
# Section 3: debug_clock_offsets
# ==============================================================================

def debug_clock_offsets(
    gpu_data_map: dict,
    n_kernels:    int = 20,
    ) -> None:
    """
    Print the raw AllReduce start/end times across all ranks side by side
    for manual inspection.

    USE THIS WHEN:
        - calculate_clock_offsets() gives unexpected results
        - You want to verify that the offset is really clock skew and not
          actual load imbalance between ranks
        - The offset std is very high (ranks are inconsistent)

    HOW TO INTERPRET THE OUTPUT:
        If offset is TRUE clock skew:
            All per-bucket diffs should be nearly identical.
            Example: [17.1ms, 17.0ms, 17.2ms, 17.0ms, 16.9ms]
            → std is tiny → pure clock difference → safe to correct

        If offset is LOAD IMBALANCE:
            Per-bucket diffs will vary a lot.
            Example: [5ms, 23ms, 8ms, 31ms, 12ms]
            → std is large → ranks are genuinely running at different speeds
            → clock correction will not fully fix the misalignment

    Parameters:
        gpu_data_map → dict of {rank: GpuDataset}
        n_kernels    → number of AllReduce buckets to print

    Returns:
        None — prints to stdout only.
    """
    print("\n=== Debug: Per-Bucket AllReduce Start/End Times ===")

    rank_allreduce = {}
    for rank, dataset in gpu_data_map.items():
        allreduce_df = get_allreduce_df(dataset.df_nccl)
        if allreduce_df.empty:
            print(f"  Rank {rank}: NO AllReduce DATA")
            continue
        rank_allreduce[rank] = (
            allreduce_df
            .sort_values("kernel_start")
            .head(n_kernels)
            .reset_index(drop=True)
        )

    if not rank_allreduce:
        print("  No AllReduce data found in any rank.")
        return

    ranks = sorted(rank_allreduce.keys())
    n     = min(len(rank_allreduce[r]) for r in ranks)

    # ── Print header ──────────────────────────────────────────────────────────
    header = f"{'bucket':>6} | "
    header += " | ".join(
        f"R{r}_start_ms    R{r}_end_ms" for r in ranks
    )
    print(header)
    print("-" * len(header))

    # ── Print per-bucket rows ─────────────────────────────────────────────────
    for i in range(n):
        row = f"{i:>6} | "
        parts = []
        for r in ranks:
            start_ms = rank_allreduce[r].iloc[i]["kernel_start"] / 1e6
            end_ms   = rank_allreduce[r].iloc[i]["kernel_end"]   / 1e6
            parts.append(f"{start_ms:12.3f}  {end_ms:12.3f}")
        print(row + " | ".join(parts))

    # ── Print end time differences vs rank 0 ─────────────────────────────────
    print("\n=== Per-Bucket End Time Diff (vs Rank 0) ===")
    base = rank_allreduce[0]
    for r in ranks:
        if r == 0:
            continue
        diffs = base["kernel_end"].values[:n] - rank_allreduce[r]["kernel_end"].values[:n]
        print(f"\n  Rank 0 vs Rank {r}:")
        print(f"    mean  = {diffs.mean() / 1e6:+.3f} ms")
        print(f"    std   = {diffs.std()  / 1e6:.3f} ms  ← small = true clock skew, large = load imbalance")
        print(f"    min   = {diffs.min()  / 1e6:+.3f} ms")
        print(f"    max   = {diffs.max()  / 1e6:+.3f} ms")

    # ── Print start time differences vs rank 0 ────────────────────────────────
    print("\n=== Per-Bucket Start Time Diff (vs Rank 0) ===")
    for r in ranks:
        if r == 0:
            continue
        diffs = base["kernel_start"].values[:n] - rank_allreduce[r]["kernel_start"].values[:n]
        print(f"\n  Rank 0 vs Rank {r}:")
        print(f"    mean  = {diffs.mean() / 1e6:+.3f} ms")
        print(f"    std   = {diffs.std()  / 1e6:.3f} ms")
        print(f"    min   = {diffs.min()  / 1e6:+.3f} ms")
        print(f"    max   = {diffs.max()  / 1e6:+.3f} ms")