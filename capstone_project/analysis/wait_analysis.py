# wait_analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# NCCL AllReduce wait time analysis.
#
# PROBLEM THIS FILE SOLVES:
#   In DDP training, AllReduce is a collective barrier — every rank must wait
#   until ALL ranks have finished their backward pass before AllReduce can
#   start. The rank that finishes backward the LATEST causes all other ranks
#   to sit idle, wasting GPU compute time.
#
#   This file measures exactly how long each rank waited, per AllReduce
#   bucket, per training step.
#
# HOW WAIT TIME IS MEASURED:
#   Within one training step, there are multiple AllReduce buckets
#   (one per gradient bucket in DDP). For each bucket:
#
#       wait_time = max(allreduce_start across all ranks)
#                   - this_rank's allreduce_start
#
#   Simple example (step 0, bucket 0):
#       Rank 0 AllReduce starts at t=100ms  → wait = 140 - 100 = 40ms
#       Rank 1 AllReduce starts at t= 95ms  → wait = 140 -  95 = 45ms
#       Rank 2 AllReduce starts at t=140ms  → wait = 140 - 140 =  0ms ← bottleneck
#       Rank 3 AllReduce starts at t= 98ms  → wait = 140 -  98 = 42ms
#
#   Rank 2 has zero wait — it finished backward last and triggered AllReduce
#   immediately. All other ranks had already finished and were waiting for it.
#
# PIPELINE POSITION:
#   clock_offset.py → offsets dict
#                          ↓
#   data_loader.py  → gpu_data_map (df_nvtx + df_nccl per rank)
#                          ↓
#   wait_analysis.py → wait_df
#                          ↓
#   plot_wait_time_summary()  (visualization)
#
# FUNCTIONS IN THIS FILE:
#   get_allreduce_start_times_per_bucket()  ← per-rank, per-bucket AllReduce start
#   load_all_gpu_compute_wait_time()        ← main function, across all ranks
#
# DEPENDENCIES:
#   pandas only — no imports from loaders/ or other analysis/ files
#   Requires offsets from clock_offset.calculate_clock_offsets()
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict
import pandas as pd

# ==============================================================================
# Section 1: get_allreduce_start_times_per_bucket
# ==============================================================================

def get_allreduce_start_times_per_bucket(
    nvtx_df: pd.DataFrame,
    nccl_df: pd.DataFrame,
    ) -> pd.DataFrame:

    """
    For one rank, pair each NVTX AllReduce CPU launch marker with its
    corresponding GPU AllReduce kernel start time.

    WHY WE NEED BOTH NVTX AND NCCL:
        NVTX tells us WHICH STEP (data_batch_idx) each AllReduce belongs to,
        because NVTX markers are on the CPU clock and aligned with step ranges.

        NCCL (df_nccl) tells us the actual GPU kernel start time — this is
        what we use for wait time calculation because it is on the GPU clock
        which is comparable across ranks after offset correction.

        We match them by ORDER (bucket 0 in NVTX → bucket 0 in NCCL) because
        both are sorted by time and correspond 1:1.

    Simple example:
        NVTX cpu_nccl_allreduce_launch rows (sorted by start):
            index  start_ms   step
            0      210ms      0     ← step 0, bucket 0
            1      820ms      0     ← step 0, bucket 1
            2      1260ms     1     ← step 1, bucket 0

        NCCL AllReduce kernel rows (sorted by kernel_start):
            index  kernel_start_ms
            0      300ms            ← GPU actually started at 300ms
            1      910ms            ← GPU started at 910ms
            2      1350ms           ← GPU started at 1350ms

        Matched result:
            bucket_idx  step  allreduce_start_ns
            0           0     300_000_000
            1           0     910_000_000
            2           1     1_350_000_000

    Parameters:
        nvtx_df  → GpuDataset.df_nvtx with a 'step' column already assigned
                   (produced inside load_all_gpu_compute_wait_time)
        nccl_df  → GpuDataset.df_nccl

    Returns DataFrame with columns:
        bucket_idx         → int, sequential AllReduce index (0, 1, 2, ...)
        step               → int, which training step this bucket belongs to
        allreduce_start_ns → int (ns), GPU clock time when AllReduce kernel started
        allreduce_start_ms → float (ms), same value in milliseconds for readability

    Example output for ONE rank (2 steps × 7 buckets each = 14 rows):
 
        bucket_idx  step  allreduce_start_ns  allreduce_start_ms
                 0     0       21987_000_000           21987.000   ← step 0, bucket 0 (first AllReduce)
                 1     0       22075_000_000           22075.000   ← step 0, bucket 1
                 2     0       22080_000_000           22080.000   ← step 0, bucket 2
                 3     0       22087_000_000           22087.000   ← step 0, bucket 3
                 4     0       22095_000_000           22095.000   ← step 0, bucket 4
                 5     0       22277_000_000           22277.000   ← step 0, bucket 5
                 6     0       22376_000_000           22376.000   ← step 0, bucket 6 (last AllReduce)
                 7     1       24122_000_000           24122.000   ← step 1, bucket 0
                 8     1       24210_000_000           24210.000
                 9     1       24215_000_000           24215.000
                10     1       24222_000_000           24222.000
                11     1       24230_000_000           24230.000
                12     1       24412_000_000           24412.000
                13     1       24511_000_000           24511.000   ← step 1, bucket 6 (last AllReduce)

    """

    # Filter NVTX to only AllReduce CPU launch markers, sorted by time
    nvtx_nccl_df = nvtx_df[
        nvtx_df["name"] == "cpu_nccl_allreduce_launch"
    ].sort_values("start").reset_index(drop=True)

    # Filter NCCL kernels to only AllReduce GPU kernels, sorted by time
    nccl_allreduce_df = nccl_df[
        nccl_df["name"].str.contains("AllReduce", case=False)
    ].sort_values("kernel_start").reset_index(drop=True)

    n_nvtx   = len(nvtx_nccl_df)
    n_kernel = len(nccl_allreduce_df)

    # The counts should match — one CPU launch per GPU kernel
    # A mismatch usually means some kernels happened outside the profiled range
    if n_nvtx != n_kernel:
        print(f"  [WARNING] AllReduce count mismatch: "
              f"NVTX={n_nvtx} CPU launches, NCCL={n_kernel} GPU kernels")
        print(f"  → Using min({n_nvtx}, {n_kernel}) = {min(n_nvtx, n_kernel)} pairs")

    
    results = []

    for i in range(min(n_nvtx, n_kernel)):
        allreduce_start_ns = nccl_allreduce_df.iloc[i]["kernel_start"]
        cur_step           = nvtx_nccl_df.iloc[i].get("step", -1)

        results.append({
            "bucket_idx":         i,
            "step":               cur_step,
            "allreduce_start_ns": allreduce_start_ns,
            "allreduce_start_ms": allreduce_start_ns / 1e6,
        })

    return pd.DataFrame(results)


# ==============================================================================
# Section 2: load_all_gpu_compute_wait_time
# ==============================================================================

def load_all_gpu_compute_wait_time(
    gpu_data_map: dict,
    offsets:      Dict[int, int],
    ) -> pd.DataFrame:
    """
    Compute AllReduce wait time for every rank, every bucket, every step.

    STEP-BY-STEP PIPELINE:
        For each rank:
            1. Extract step ranges from NVTX (cpu_batch_N_wrapper rows)
            2. Assign a 'step' number to each NVTX row by timestamp
            3. Call get_allreduce_start_times_per_bucket() to get GPU AllReduce
               start times with step labels
            4. Apply clock offset to put all ranks on rank 0's clock

        Then across all ranks:
            5. Concatenate all rank results into one DataFrame
            6. Assign bucket_within_step (0, 1, 2, ... within each step)
            7. For each (step, bucket) pair, find the maximum AllReduce start
               time across all ranks → this is when the slowest rank started
            8. wait_time = max_start - this_rank_start
               → the rank with wait=0 is the bottleneck (it started last)
               → all other ranks waited for it

    Simple example (step 0, 4 ranks, bucket 0):
        After offset correction:
            Rank 0: allreduce_start = 100ms
            Rank 1: allreduce_start =  95ms
            Rank 2: allreduce_start = 140ms  ← started last = bottleneck
            Rank 3: allreduce_start =  98ms

        max_allreduce_start = 140ms

        pure_wait_ms:
            Rank 0: 140 - 100 = 40ms  ← wasted 40ms waiting for rank 2
            Rank 1: 140 -  95 = 45ms  ← wasted 45ms
            Rank 2: 140 - 140 =  0ms  ← no wait, it was the bottleneck
            Rank 3: 140 -  98 = 42ms  ← wasted 42ms

    WHAT 'bucket_within_step' MEANS:
        DDP splits gradients into buckets (e.g. 7 buckets for ResNet-50).
        Each bucket triggers its own AllReduce independently.
        bucket_within_step=0 is the first AllReduce in the step,
        bucket_within_step=6 is the last (largest gradients).

        Simple example (step 0 with 7 buckets):
            bucket_within_step  description
            0                   first AllReduce (starts during backward)
            1                   second AllReduce
            ...
            6                   last AllReduce (triggers optimizer step)

    Parameters:
        gpu_data_map → dict of {rank: GpuDataset}
                       each GpuDataset must have df_nvtx and df_nccl loaded
        offsets      → dict of {rank: offset_ns} from calculate_clock_offsets()
                       used to align GPU timestamps across nodes

    Returns DataFrame with one row per (rank, bucket) combination:
        rank                        → int, GPU rank
        bucket_idx                  → int, sequential AllReduce index
        step                        → int, training step number
        allreduce_start_ns          → int (ns), raw GPU clock AllReduce start
        allreduce_start_ms          → float (ms), same value in ms
        allreduce_start_corrected_ns → int (ns), after clock offset correction
        bucket_within_step          → int, AllReduce index within this step (0-based)
        pure_wait_ms                → float (ms), how long this rank waited
                                      0.0 = this rank was the bottleneck
                                      >0  = this rank was waiting for another rank

    Example output (4 ranks, step 0, 7 buckets each = 28 rows for step 0):

        rank  bucket_idx  step  bucket_within_step  allreduce_start_ms  pure_wait_ms
        0     0           0     0                   100.0               40.0
        1     0           0     0                    95.0               45.0
        2     0           0     0                   140.0                0.0  ← bottleneck
        3     0           0     0                    98.0               42.0
        0     1           0     1                   150.0               35.0
        ...

    To find the bottleneck rank per step:
        wait_df[wait_df["pure_wait_ms"] == 0].groupby("step")["rank"].value_counts()
    """

    all_results = []

    for rank, gpu_data in gpu_data_map.items():

        # ── Step 1: extract step ranges from NVTX ────────────────────────────
        # cpu_batch_N_wrapper rows give us the CPU start/end for each step
        # We need this to assign a step number to each NVTX AllReduce marker
        step_ranges = gpu_data.df_nvtx[
            gpu_data.df_nvtx["name"].str.contains(
                r"cpu_batch_\d+_duration", regex=True
            )
        ][["data_batch_idx", "start", "end"]].drop_duplicates("data_batch_idx")

        # ── Step 2: assign step number to each NVTX row by timestamp ─────────
        # For each NVTX row, find which step range its start time falls into
        nvtx_with_step = gpu_data.df_nvtx.copy()

        def assign_step(row_start):
            match = step_ranges[
                (step_ranges["start"] <= row_start) &
                (step_ranges["end"]   >= row_start)
            ]
            return int(match.iloc[0]["data_batch_idx"]) if not match.empty else -1

        nvtx_with_step["step"] = nvtx_with_step["start"].apply(assign_step)

        # ── Step 3: get AllReduce GPU start times with step labels ────────────
        rank_df = get_allreduce_start_times_per_bucket(
            nvtx_df = nvtx_with_step,
            nccl_df = gpu_data.df_nccl,
        )

        if rank_df.empty:
            print(f"  [WARNING] Rank {rank}: No AllReduce data found — skipping.")
            continue

        rank_df["rank"] = rank

        # ── Step 4: apply clock offset to align with rank 0's clock ──────────
        # Without this, cross-node timestamps cannot be compared
        # e.g. rank 2 might be 180ms behind rank 0 due to clock skew
        rank_df["allreduce_start_corrected_ns"] = (
            rank_df["allreduce_start_ns"] + offsets.get(rank, 0)
        )

        all_results.append(rank_df)

    if not all_results:
        return pd.DataFrame()

    # ── Step 5: combine all ranks ─────────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.sort_values(["step", "rank", "bucket_idx"])

    # ── Step 6: assign bucket_within_step ────────────────────────────────────
    # Within each (step, rank) pair, number the AllReduce buckets 0, 1, 2, ...
    # This lets us compare the same bucket across ranks (bucket 0 in rank 0
    # vs bucket 0 in rank 1 are the same gradient bucket fired at the same time)
    combined["bucket_within_step"] = (
        combined.groupby(["step", "rank"]).cumcount()
    )

    # ── Step 7: find the latest AllReduce start per (step, bucket) ───────────
    # This is when the SLOWEST rank started AllReduce for that bucket
    # All ranks were ready BEFORE this time; the one with this time is the bottleneck
    max_allreduce_start = combined.groupby(
        ["step", "bucket_within_step"]
    )["allreduce_start_corrected_ns"].transform("max")

    # ── Step 8: compute wait time ─────────────────────────────────────────────
    # wait = max_start - this_rank_start
    # clip(lower=0) handles tiny floating point negatives from clock jitter
    combined["pure_wait_ms"] = (
        (max_allreduce_start - combined["allreduce_start_corrected_ns"])
        .clip(lower=0) / 1e6
    )

    return combined