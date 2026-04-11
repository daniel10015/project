
# sm_analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# SM (Streaming Multiprocessor) utilization analysis.
#
# This file answers one question:
#   "For each GPU kernel, what percentage of the GPU's SMs were actually used?"
#
# PIPELINE POSITION:
#   data_loader.py  →  sm_analysis.py  →  GpuDataset.df_all_kernels_with_sm
#                                          ↓
#                                       build_sm_timeline_df()  (plot)
#
# FUNCTIONS IN THIS FILE (call order from outside):
#   _calc_kernel_sm_pct()            ← Step 1: compute SM% per kernel
#   _get_all_kernels_with_sm_for_rank() ← Step 2: merge all kernel sources
#   compute_sm_utilization()         ← Step 3: time-weighted SM% per step
#   collect_gpu_sm_utilization()     ← Step 4: aggregate across all ranks
#
# DEPENDENCIES:
#   loaders/schemas.py  ← GpuDataset (type hint only)
#   numpy, pandas, math
# ─────────────────────────────────────────────────────────────────────────────

import math
from typing import Dict
 
import numpy as np
import pandas as pd

# ==============================================================================
# Section 1: _calc_kernel_sm_pct
# ==============================================================================

def _calc_kernel_sm_pct(
    df_kernel_occ: pd.DataFrame,
    gpu_specs:     dict,
    ) -> pd.DataFrame:
    """
    Compute SM utilization % for every individual kernel execution.
 
    Uses the same formula as compute_sm_utilization(), applied per-kernel
    instead of per-step so that _get_all_kernels_with_sm_for_rank() can
    attach an SM% to each row in df_all_kernels_with_sm.
 
    HOW SM% IS CALCULATED:
        Every SM (Streaming Multiprocessor) on the GPU can run a limited
        number of thread blocks at the same time. Four hardware limits
        determine how many blocks fit on one SM:
 
            limit_by_threads   = sm_max_threads   // threads_per_block
            limit_by_blocks    = sm_max_blocks     (fixed hardware cap)
            limit_by_registers = sm_max_registers  // registers_per_block
            limit_by_shared    = sm_max_shared_mem // shared_per_block
 
        blocks_per_sm = min of all four limits  (the tightest constraint wins)
 
        active_sm     = min(ceil(total_blocks / blocks_per_sm), sm_count)
        sm_pct_kernel = active_sm / sm_count * 100
 
    Simple example:
        GPU has 108 SMs (A100).
        Kernel launches 256 blocks, each with 128 threads.
        threads_per_block = 128
        limit_by_threads  = 2048 // 128 = 16  (16 blocks fit per SM by thread count)
        limit_by_blocks   = 32               (hardware cap)
        registers / shared memory limits are higher → not the bottleneck
        blocks_per_sm = min(16, 32) = 16
        active_sm     = min(ceil(256 / 16), 108) = min(16, 108) = 16
        sm_pct_kernel = 16 / 108 * 100 = 14.8%
        → only 16 out of 108 SMs are busy — low utilization
 
    Parameters:
        df_kernel_occ  → output of load_kernel_occupancy_stats(), with
                         data_batch_idx already assigned via merge_asof
        gpu_specs      → GPU hardware spec dict (sm_count, sm_max_threads, etc.)
 
    Returns DataFrame with columns:
        correlation_id  → int, links back to the original kernel row
        data_batch_idx  → int, which training step this kernel belongs to
        sm_pct_kernel   → float, SM utilization % for this kernel (0.0 ~ 100.0)
        duration_ns     → int, actual GPU execution time in nanoseconds
    """
    if df_kernel_occ.empty:
        return pd.DataFrame(
            columns=["correlation_id",          
                     "data_batch_idx",
                     "sm_pct_kernel", 
                     "duration_ns"]
        )
 

    sm_count      = gpu_specs["sm_count"]
    max_threads   = gpu_specs["sm_max_threads"]
    max_blocks    = gpu_specs["sm_max_blocks"]
    max_registers = gpu_specs["sm_max_registers"]
    max_shared    = gpu_specs["sm_max_shared_mem"]
 
    df = df_kernel_occ.copy()
 
    def _sm_util_row(row) -> float:
        """Compute SM% for one kernel row."""
 
        tpb            = max(1, int(row["threads_per_block"]))
        total_blocks   = max(1, int(row["total_blocks"]))
        reg_per_thread = int(row["registers_per_thread"])
        static_shared  = int(row["static_shared_mem"])
        dynamic_shared = int(row["dynamic_shared_mem"])
        shared_per_block = static_shared + dynamic_shared
 
        # Start with the two always-present limits
        limits = [max_threads // tpb, max_blocks]
 
        # Register limit — only applies when registers > 0
        if reg_per_thread > 0:
            reg_per_block = reg_per_thread * tpb
            limits.append(max_registers // max(1, reg_per_block))
 
        # Shared memory limit — only applies when shared memory > 0
        if shared_per_block > 0:
            limits.append(max_shared // shared_per_block)
 
        blocks_per_sm = max(1, min(limits))
 
        # How many SMs are actually active?
        active_sm = min(math.ceil(total_blocks / blocks_per_sm), sm_count)
 
        return round((active_sm / sm_count) * 100, 2)
 
    df["sm_pct_kernel"] = df.apply(_sm_util_row, axis=1)
 
    return df[["correlation_id", 
               "data_batch_idx",
               "sm_pct_kernel", 
               "duration_ns"]].copy()


def _get_all_kernels_with_sm_for_rank(
    df_gpu_duration:              pd.DataFrame,
    df_memcpy:                    pd.DataFrame,
    df_nccl:                      pd.DataFrame,
    df_kernel_occupancy_per_step: pd.DataFrame,
    gpu_specs:                    dict,
    ) -> pd.DataFrame:
    """
    Merge all kernel sources for one rank into a single DataFrame with
    SM utilization % attached to each row.
 
    WHY THREE SOURCES:
        GPU kernels come from three different DataFrames, each capturing a
        different type of GPU activity:
 
        Source 1 — df_gpu_duration  (compute, optimizer, loss kernels)
            These are the main training kernels: forward, backward, opt_step.
            NCCL kernels that overlapped with backward were REMOVED from here
            earlier in join_cpu_api_in_nvtx_with_gpu_kernel_timings() to avoid
            double-counting. We add them back from Source 3.
 
        Source 2 — df_memcpy  (H2D memory copies)
            DMA transfers run on a separate copy engine, NOT on SMs.
            → sm_pct_kernel = 0.0 for all H2D rows.
            Only rows with data_batch_idx assigned (training steps) are kept.
            Pre-training small transfers (NaN data_batch_idx) are dropped.
 
        Source 3 — df_nccl  (AllReduce GPU kernels)
            NCCL kernels were loaded separately in load_nccl_kernels().
            They are added here so the SM timeline shows when NCCL was
            running and how much SM capacity it was consuming.
 
    Simple example:
        After merging all three sources for step 2:
 
        kernel_start_ns  kernel_end_ns  duration_ns  stream_type   sm_pct_kernel
        12_195_000_000   12_209_000_000  14_000_000   h2d           0.0
        12_215_000_000   12_215_450_000     450_000   forward       76.3
        12_215_450_000   12_215_920_000     470_000   forward       81.5
        12_518_000_000   12_518_540_000     540_000   backward      42.1
        12_518_000_000   12_552_000_000  34_000_000   nccl_active   19.3
 
    Parameters:
        df_gpu_duration              → GpuDataset.df_gpu_duration
        df_memcpy                    → GpuDataset.df_memcpy
        df_nccl                      → GpuDataset.df_nccl
        df_kernel_occupancy_per_step → GpuDataset.df_kernel_occupancy_per_step
        gpu_specs                    → GPU hardware spec dict
 
    Returns DataFrame with columns:
        kernel_start_ns  → int (ns), GPU clock timestamp when kernel started
        kernel_end_ns    → int (ns), GPU clock timestamp when kernel ended
        duration_ns      → int (ns), actual GPU execution time
        correlation_id   → int, unique kernel ID for debugging
        stream_type      → str: 'forward'|'backward'|'nccl_active'|
                                'opt_step'|'h2d'|'compute'|'unknown'
        data_batch_idx   → int, which training step this kernel belongs to
        sm_pct_kernel    → float, SM utilization % (0.0 for h2d)
    Sorted by kernel_start_ns ascending.
    """
    rows = []
 
    # ── Pre-compute SM% lookup table ─────────────────────────────────────────
    # _calc_kernel_sm_pct() runs once here and produces a small lookup table
    # indexed by correlation_id. We then join it into each source DataFrame
    # instead of recomputing SM% for every source separately.
    df_sm = pd.DataFrame()
    if not df_kernel_occupancy_per_step.empty:
        df_sm = _calc_kernel_sm_pct(df_kernel_occupancy_per_step, gpu_specs)
 
    # ── Source 1: df_gpu_duration (compute / optimizer / loss) ───────────────
    df_kern = df_gpu_duration.copy()
 
    # Remove wrapper rows — these are aggregate spans covering the entire step,
    # not individual kernel executions. Including them would inflate SM%.
    # Simple example:
    #   'gpu_batch_0_duration' spans t=100~900ms (the whole step)
    #   'gpu_forward_duration' spans t=100~400ms (just the forward pass)
    #   Including both would double-count the forward period.
    WRAPPER_PAT = r"gpu_batch_\d+_duration|gpu_train_compute_duration"
    df_kern = df_kern[
        ~df_kern["name"].str.contains(WRAPPER_PAT, regex=True)
    ].copy()
 
    # Assign stream_type if not already present
    if "stream_type" not in df_kern.columns:
        def _infer(name: str) -> str:
            n = name.lower()
            if "nccl"      in n: return "nccl_active"
            if "opt_step"  in n: return "opt_step"
            if "zero_grad" in n: return "opt_step"
            if "backward"  in n: return "backward"
            if "forward"   in n: return "forward"
            if "loss"      in n: return "forward"
            return "unknown"
        df_kern["stream_type"] = df_kern["name"].apply(_infer)
 
    # Join SM% from the pre-computed lookup table
    if not df_sm.empty:
        df_kern = pd.merge(
            df_kern,
            df_sm[["correlation_id", "sm_pct_kernel", "duration_ns"]],
            on="correlation_id",
            how="left",
        )
        df_kern["sm_pct_kernel"] = df_kern["sm_pct_kernel"].fillna(0.0)
        df_kern["duration_ns"]   = df_kern["duration_ns"].fillna(
            df_kern["gpu_end"] - df_kern["gpu_start"]
        ).astype(int)
    else:
        # No occupancy data → SM% unknown, use gpu span as duration
        df_kern["sm_pct_kernel"] = 0.0
        df_kern["duration_ns"]   = (
            df_kern["gpu_end"] - df_kern["gpu_start"]
        ).astype(int)
 
    rows.append(
        df_kern[[
            "gpu_start", "gpu_end", "duration_ns",
            "correlation_id", "stream_type",
            "data_batch_idx", "sm_pct_kernel",
        ]].rename(columns={"gpu_start": "kernel_start_ns",
                            "gpu_end":   "kernel_end_ns"})
    )
 
    # ── Source 2: df_memcpy (H2D memory copies) ──────────────────────────────
    df_mem = df_memcpy.copy()
 
    # Drop pre-training transfers (NaN = happened before step 0)
    df_mem = df_mem[df_mem["data_batch_idx"].notna()].copy()
 
    if not df_mem.empty:
        if "stream_type" not in df_mem.columns:
            df_mem["stream_type"] = "h2d"
 
        # H2D runs on DMA engine, NOT on SMs → SM% is always 0
        df_mem["sm_pct_kernel"] = 0.0
        df_mem["duration_ns"]   = df_mem["dur_ns"].astype(int)
 
        rows.append(
            df_mem[[
                "kernel_start", "kernel_end", "duration_ns",
                "correlation_id", "stream_type",
                "data_batch_idx", "sm_pct_kernel",
            ]].rename(columns={"kernel_start": "kernel_start_ns",
                                "kernel_end":   "kernel_end_ns"})
        )
 
    # ── Source 3: df_nccl (AllReduce kernels) ────────────────────────────────
    if not df_nccl.empty:
        df_nc = df_nccl.copy()
 
        # Drop pre-training NCCL (DDP init — before step 0)
        df_nc = df_nc[df_nc["data_batch_idx"].notna()].copy()
 
        if not df_nc.empty:
            if "stream_type" not in df_nc.columns:
                df_nc["stream_type"] = "nccl_active"
 
            # Join SM% for NCCL kernels
            if not df_sm.empty:
                df_nc = pd.merge(
                    df_nc,
                    df_sm[["correlation_id", "sm_pct_kernel", "duration_ns"]],
                    on="correlation_id",
                    how="left",
                )
                df_nc["sm_pct_kernel"] = df_nc["sm_pct_kernel"].fillna(0.0)
                df_nc["duration_ns"]   = df_nc["duration_ns"].fillna(
                    df_nc["kernel_end"] - df_nc["kernel_start"]
                ).astype(int)
            else:
                df_nc["sm_pct_kernel"] = 0.0
                df_nc["duration_ns"]   = (
                    df_nc["kernel_end"] - df_nc["kernel_start"]
                ).astype(int)
 
            rows.append(
                df_nc[[
                    "kernel_start", "kernel_end", "duration_ns",
                    "correlation_id", "stream_type",
                    "data_batch_idx", "sm_pct_kernel",
                ]].rename(columns={"kernel_start": "kernel_start_ns",
                                    "kernel_end":   "kernel_end_ns"})
            )
 
    # ── Merge all sources ─────────────────────────────────────────────────────
    if not rows:
        return pd.DataFrame(columns=[
            "kernel_start_ns", "kernel_end_ns", "duration_ns",
            "correlation_id", "stream_type", "data_batch_idx", "sm_pct_kernel",
        ])
 
    result = (
        pd.concat(rows, ignore_index=True)
        .assign(data_batch_idx=lambda df: df["data_batch_idx"].astype(int))
        .sort_values("kernel_start_ns")
        .reset_index(drop=True)
    )
 
    return result



# ==============================================================================
# Section 3: compute_sm_utilization
# ==============================================================================
 
def compute_sm_utilization(
    df_kernel_occupancy_per_step: pd.DataFrame,
    gpu_specs:                    dict,
    ) -> pd.DataFrame:
    """
    Compute time-weighted SM utilization % per training step.
 
    This gives one number per step that summarises how hard the GPU worked
    during that step — weighted by how long each kernel ran.
 
    WHY TIME-WEIGHTED:
        A kernel that ran for 10ms should contribute more to the step's SM%
        than a kernel that ran for 0.1ms, even if both had the same SM%.
 
        time-weighted SM% = Σ(sm_util_i × duration_i) / Σ(duration_i)
 
    Simple example:
        Step 2 has 3 kernels:
            Kernel A: sm_util=80%, duration=5ms   → contribution = 400
            Kernel B: sm_util=60%, duration=3ms   → contribution = 180
            Kernel C: sm_util=20%, duration=2ms   → contribution =  40
        Total duration = 10ms
        weighted_sm_util = (400 + 180 + 40) / 10 = 62%
 
        If you used a simple average instead:
            mean_sm_util = (80 + 60 + 20) / 3 = 53.3%
        That undercounts kernel A which ran the longest.
 
    Parameters:
        df_kernel_occupancy_per_step → output of load_kernel_occupancy_stats()
                                       with data_batch_idx assigned
        gpu_specs                    → GPU hardware spec dict
 
    Returns DataFrame with columns per step:
        step                      → int, training step number
        weighted_sm_utilization   → float (%), time-weighted SM util
        weighted_sm_occupancy     → float (%), time-weighted SM occupancy
        mean_sm_utilization       → float (%), simple average (reference only)
        mean_sm_occupancy         → float (%), simple average (reference only)
        kernel_count              → int, number of kernels in this step
 
    Also sets DataFrame.attrs:
        median_weighted_sm_util   → float, median across all steps
        mean_weighted_sm_util     → float, mean across all steps
        median_weighted_sm_occ    → float
        mean_weighted_sm_occ      → float
    """
    if df_kernel_occupancy_per_step.empty:
        raise ValueError("df_kernel_occupancy_per_step is empty.")
 
    sm_count         = gpu_specs["sm_count"]
    sm_max_threads   = gpu_specs["sm_max_threads"]
    sm_max_blocks    = gpu_specs["sm_max_blocks"]
    sm_max_registers = gpu_specs["sm_max_registers"]
    sm_max_shared    = gpu_specs["sm_max_shared_mem"]
 
    df = df_kernel_occupancy_per_step.copy()
 
    # shared_per_block=0 means no shared memory limit → treat as infinite
    df["shared_per_block"] = (
        df["static_shared_mem"] + df["dynamic_shared_mem"]
    ).replace(0, float("inf"))
 
    df["registers_per_block"] = df["registers_per_thread"] * df["threads_per_block"]
 
    # The four hardware limits — tightest one wins
    df["limit_by_threads"]   = sm_max_threads // df["threads_per_block"]
    df["limit_by_blocks"]    = sm_max_blocks
    df["limit_by_registers"] = sm_max_registers // df["registers_per_block"].replace(0, 1)
    df["limit_by_shared"]    = (
        sm_max_shared / df["shared_per_block"]
    ).apply(lambda x: int(x) if x != float("inf") else sm_max_blocks)
 
    df["blocks_per_sm"] = df[[
        "limit_by_threads", "limit_by_blocks",
        "limit_by_registers", "limit_by_shared",
    ]].min(axis=1).clip(lower=1)
 
    df["active_sm"] = np.minimum(
        np.ceil(df["total_blocks"] / df["blocks_per_sm"]).astype(int),
        sm_count,
    )
 
    # SM utilization = fraction of SMs that are active
    df["sm_utilization"] = (df["active_sm"] / sm_count * 100).round(2)
 
    # SM occupancy = how full each active SM is with threads
    df["sm_occupancy"] = (
        (df["blocks_per_sm"] * df["threads_per_block"]) / sm_max_threads * 100
    ).clip(upper=100).round(2)
 
    # Time-weighted aggregation per step
    records = []
    for step, group in df.groupby("data_batch_idx"):
        total_duration = group["duration_ns"].sum()
        if total_duration == 0:
            print(f"[Warning] step {step}: total duration = 0, skipped.")
            continue
 
        weighted_sm_util = (
            (group["sm_utilization"] * group["duration_ns"]).sum() / total_duration
        )
        weighted_sm_occ = (
            (group["sm_occupancy"] * group["duration_ns"]).sum() / total_duration
        )
 
        records.append({
            "step":                   int(step),
            "weighted_sm_utilization": round(weighted_sm_util, 2),
            "weighted_sm_occupancy":   round(weighted_sm_occ,  2),
            "mean_sm_utilization":     round(group["sm_utilization"].mean(), 2),
            "mean_sm_occupancy":       round(group["sm_occupancy"].mean(),   2),
            "kernel_count":            len(group),
        })
 
    df_result = (
        pd.DataFrame(records)
        .sort_values("step")
        .reset_index(drop=True)
    )
 
    # Store summary stats as DataFrame attributes
    wsm = df_result["weighted_sm_utilization"]
    wso = df_result["weighted_sm_occupancy"]
    df_result.attrs["median_weighted_sm_util"] = float(wsm.median())
    df_result.attrs["mean_weighted_sm_util"]   = float(wsm.mean())
    df_result.attrs["median_weighted_sm_occ"]  = float(wso.median())
    df_result.attrs["mean_weighted_sm_occ"]    = float(wso.mean())
 
    return df_result


# ==============================================================================
# Section 4: collect_gpu_sm_utilization
# ==============================================================================
 
def collect_gpu_sm_utilization(
    gpu_data_map: dict,
    gpu_specs:    dict,
    ) -> pd.DataFrame:
    """
    Compute SM utilization summary for every GPU rank and return one row
    per rank.
 
    This is the top-level function you call from run_analysis() in main.py
    to get a cross-rank SM utilization comparison table.
 
    Simple example:
        4 ranks, 50 steps each.
        Returns:
            rank  median_weighted_sm_util  mean_weighted_sm_util  n_steps
            0     72.3                     71.8                   50
            1     71.9                     71.5                   50
            2     73.1                     72.6                   50
            3     70.8                     70.2                   50
 
        If one rank consistently has lower SM%, it may be the bottleneck
        that is slowing down the whole training job.
 
    Parameters:
        gpu_data_map → dict returned by load_data() in main.py
        gpu_specs    → GPU hardware spec dict
 
    Returns DataFrame with columns:
        rank                    → int
        median_weighted_sm_util → float (%), median over all steps
        mean_weighted_sm_util   → float (%)
        median_weighted_sm_occ  → float (%)
        mean_weighted_sm_occ    → float (%)
        n_steps                 → int, number of steps processed
    """
    records = []
 
    for rank, dataset in gpu_data_map.items():
        try:
            if dataset.df_kernel_occupancy_per_step.empty:
                print(f"  [Warning] Rank {rank}: df_kernel_occupancy_per_step "
                      f"is empty — skipping.")
                continue
 
            df_sm_result = compute_sm_utilization(
                df_kernel_occupancy_per_step = dataset.df_kernel_occupancy_per_step,
                gpu_specs                    = gpu_specs,
            )
 
            records.append({
                "rank":                    rank,
                "median_weighted_sm_util": df_sm_result.attrs["median_weighted_sm_util"],
                "mean_weighted_sm_util":   df_sm_result.attrs["mean_weighted_sm_util"],
                "median_weighted_sm_occ":  df_sm_result.attrs["median_weighted_sm_occ"],
                "mean_weighted_sm_occ":    df_sm_result.attrs["mean_weighted_sm_occ"],
                "n_steps":                 len(df_sm_result),
            })
 
            print(
                f"  Rank {rank}: "
                f"SM util = {df_sm_result.attrs['median_weighted_sm_util']:.1f}%  "
                f"SM occ  = {df_sm_result.attrs['median_weighted_sm_occ']:.1f}%  "
                f"({len(df_sm_result)} steps)"
            )
 
        except Exception as e:
            print(f"  [Warning] Rank {rank}: SM utilization failed — {e}")
 
    if not records:
        raise RuntimeError("SM utilization calculation failed for all ranks.")
 
    return (
        pd.DataFrame(records)
        .sort_values("rank")
        .reset_index(drop=True)
    )

# ==============================================================================
# Section 5: build_sm_timeline_df
# ==============================================================================
 
def build_sm_timeline_df(
    gpu_data_map: dict,
    offsets:      Dict[int, int],
    steps:        list,
    n_buckets:    int = 350,
    ) -> pd.DataFrame:
    """
    Convert df_all_kernels_with_sm into a bucketed SM timeline DataFrame
    ready for plotting.
 
    WHY BUCKETING:
        df_all_kernels_with_sm has one row per individual GPU kernel execution.
        A single training step can have 2000+ kernels, each lasting ~0.1ms.
        Plotting each kernel individually would produce an unreadable graph.
 
        Instead, we divide the step duration into n_buckets equal time slices
        and compute the SM% contribution of each stream_type within each bucket.
 
        Simple example (n_buckets=5, step duration=100ms):
            bucket 0: t=0~20ms   → forward=72%, h2d active
            bucket 1: t=20~40ms  → forward=81%
            bucket 2: t=40~60ms  → backward=65%, nccl_active=19%
            bucket 3: t=60~80ms  → backward=58%
            bucket 4: t=80~100ms → opt_step=31%
 
    HOW SM% IS COMPUTED PER BUCKET:
        For each kernel that overlaps with the bucket:
            overlap_ns   = min(kernel_end, bucket_end) - max(kernel_start, bucket_start)
            contribution = sm_pct_kernel × (overlap_ns / bucket_ns)
 
        All contributions from the same stream_type are summed.
 
        Simple example (bucket_ns = 1ms):
            Kernel A: sm_pct=80%, overlap=0.6ms → contribution = 80 × 0.6 = 48%
            Kernel B: sm_pct=70%, overlap=0.4ms → contribution = 70 × 0.4 = 28%
            (both are stream_type='forward')
            Total forward SM% for this bucket = 48 + 28 = 76%
 
    HOW BUCKET_NS IS DETERMINED:
        We find the longest step duration across all ranks and steps, then:
            bucket_ns = max_duration / n_buckets
 
        Using the slowest rank ensures all ranks fit within n_buckets.
        Faster ranks will have empty buckets at the end (filled by next step).
 
    MULTIPLE ROWS PER BUCKET:
        If two stream_types are active in the same bucket (e.g. backward and
        nccl_active overlap), the bucket gets two rows — one per stream_type.
        The plotting function stacks them into a stacked bar chart.
 
    H2D HANDLING:
        H2D runs on DMA engine → sm_pct = 0.0 always.
        H2D does NOT get its own row in the output (sm_pct < 0.01 filter).
        Instead, h2d_active=True is set on any bucket where H2D is running.
        The plotting function uses this flag to draw a background shading,
        indicating "data transfer was happening here" without affecting the SM bar.
 
    Parameters:
        gpu_data_map → dict of {rank: GpuDataset}
                       each GpuDataset.df_all_kernels_with_sm must be populated
                       (run load_single_gpu() with sm_analysis imported first)
        offsets      → dict from calculate_clock_offsets(), aligns GPU clocks
        steps        → list of step numbers to include, e.g. [2, 3, 4]
        n_buckets    → number of time buckets per step (default 350)
                       higher = more detail but slower to plot
 
    Returns DataFrame with columns:
        step        → int, training step number
        rank        → int, GPU rank
        bucket_idx  → int, 0 to n_buckets-1
        t_abs_ms    → float (ms), time of bucket start relative to step start
        stream_type → str: forward|backward|nccl_active|opt_step|h2d|compute|unknown
        sm_pct      → float, SM% contribution of this stream_type in this bucket
        h2d_active  → bool, True if H2D was running during this bucket
                       (used for background shading in plot, not SM bar height)
 
    Example output (4 ranks, step 2, n_buckets=350):
 
        step  rank  bucket_idx  t_abs_ms  stream_type   sm_pct  h2d_active
           2     0           0     0.000      forward    72.40        True
           2     0           1     0.184      forward    81.20       False
           2     0          45    82.800      backward   65.30       False
           2     0          45    82.800   nccl_active   19.10       False  ← same bucket
           2     0         280   515.200     opt_step    31.80       False
           2     1           0     0.000      forward    70.10        True
           ...
 
    Total rows ≈ n_buckets × n_ranks × n_steps × avg_stream_types_per_bucket
    Typical: 350 × 4 × 3 steps × 1.5 = ~6300 rows
    """
 
    # ── Step 1: apply clock offsets to all kernel timestamps ─────────────────
    # GPU timestamps from different nodes need to be shifted to rank 0's clock
    # before we can compare the same bucket_idx across ranks
    all_kernels: Dict[int, pd.DataFrame] = {}
 
    for rank, ds in gpu_data_map.items():
        if ds.df_all_kernels_with_sm.empty:
            print(f"  [Warning] Rank {rank}: df_all_kernels_with_sm is empty — skipping.")
            continue
 
        offset = offsets.get(rank, 0)
        df     = ds.df_all_kernels_with_sm.copy()
        df["kernel_start_ns"] = df["kernel_start_ns"] + offset
        df["kernel_end_ns"]   = df["kernel_end_ns"]   + offset
        all_kernels[rank]     = df
 
    if not all_kernels:
        print("[Warning] build_sm_timeline_df: no kernel data available.")
        return pd.DataFrame()
 
    # ── Step 2: determine bucket_ns from the slowest rank ────────────────────
    # Find the longest step duration across all rank × step combinations
    # This ensures n_buckets covers the full duration of the slowest rank
    max_dur_ns = 0
 
    for rank, df in all_kernels.items():
        for step in steps:
            step_df = df[df["data_batch_idx"] == step]
            if step_df.empty:
                continue
            dur = step_df["kernel_end_ns"].max() - step_df["kernel_start_ns"].min()
            if dur > max_dur_ns:
                max_dur_ns = dur
 
    if max_dur_ns <= 0:
        print("[Warning] build_sm_timeline_df: could not compute step duration.")
        return pd.DataFrame()
 
    bucket_ns = max_dur_ns / n_buckets
 
    print(
        f"\n[build_sm_timeline_df]"
        f"  max_step_duration = {max_dur_ns / 1e6:.1f}ms"
        f"  bucket_ns = {bucket_ns / 1e6:.3f}ms"
        f"  n_buckets = {n_buckets}"
    )
 
    # ── Step 3: bucket each rank × step ──────────────────────────────────────
    output_rows = []
 
    for rank, df in all_kernels.items():
        for step in steps:
 
            step_df = df[df["data_batch_idx"] == step].copy()
            if step_df.empty:
                continue
 
            # Step start = earliest kernel in this step for this rank
            step_start_ns = step_df["kernel_start_ns"].min()
 
            # Include next step kernels — faster ranks may finish their step
            # before n_buckets runs out, so the remaining buckets show next step
            next_step    = step + 1
            next_step_df = df[df["data_batch_idx"] == next_step].copy()
 
            combined_df = (
                pd.concat([step_df, next_step_df], ignore_index=True)
                if not next_step_df.empty
                else step_df
            )
 
            # ── Iterate over each bucket ──────────────────────────────────────
            for bi in range(n_buckets):
                b_start = step_start_ns + bi * bucket_ns
                b_end   = b_start + bucket_ns
 
                # Find kernels that overlap with this bucket
                # Overlap condition: kernel_start < b_end AND kernel_end > b_start
                mask = (
                    (combined_df["kernel_start_ns"] < b_end) &
                    (combined_df["kernel_end_ns"]   > b_start)
                )
                bucket_kernels = combined_df[mask]
 
                if bucket_kernels.empty:
                    continue
 
                # ── Compute SM% contribution per stream_type ──────────────────
                stream_sm:      Dict[str, float] = {}
                h2d_overlap_ns: float            = 0.0
 
                for _, k in bucket_kernels.iterrows():
                    ks         = k["kernel_start_ns"]
                    ke         = k["kernel_end_ns"]
                    overlap_ns = min(ke, b_end) - max(ks, b_start)
 
                    if overlap_ns <= 0:
                        continue
 
                    st           = k["stream_type"]
                    sm_pct       = k["sm_pct_kernel"]
                    contribution = sm_pct * (overlap_ns / bucket_ns)
 
                    stream_sm[st] = stream_sm.get(st, 0.0) + contribution
 
                    # Track H2D separately for background shading
                    # H2D always has sm_pct=0 so it never appears in stream_sm sums
                    if st == "h2d":
                        h2d_overlap_ns += overlap_ns
 
                # Decide which step label to show for this bucket
                # If the bucket midpoint is past this step's last kernel,
                # label it as the next step
                b_mid        = (b_start + b_end) / 2
                display_step = (
                    next_step
                    if (
                        not next_step_df.empty
                        and b_mid > step_df["kernel_end_ns"].max()
                    )
                    else step
                )
 
                h2d_active  = h2d_overlap_ns > 0
                has_non_h2d = False
 
                # ── Emit one row per stream_type in this bucket ───────────────
                for st, sm in stream_sm.items():
                    if st == "h2d":
                        continue        # H2D shown via h2d_active flag only
                    if sm < 0.01:
                        continue        # Remove negligible contributions
 
                    has_non_h2d = True
                    output_rows.append({
                        "step":        display_step,
                        "rank":        rank,
                        "bucket_idx":  bi,
                        "t_abs_ms":    round((b_start - step_start_ns) / 1e6, 3),
                        "stream_type": st,
                        "sm_pct":      round(sm, 2),
                        "h2d_active":  h2d_active,
                    })
 
                # H2D-only bucket — emit one row with sm_pct=0 so the
                # plotting function can still draw the background shading
                if h2d_active and not has_non_h2d:
                    output_rows.append({
                        "step":        display_step,
                        "rank":        rank,
                        "bucket_idx":  bi,
                        "t_abs_ms":    round((b_start - step_start_ns) / 1e6, 3),
                        "stream_type": "h2d",
                        "sm_pct":      0.0,
                        "h2d_active":  True,
                    })
 
    if not output_rows:
        print("[Warning] build_sm_timeline_df: no output rows generated.")
        return pd.DataFrame()
 
    df_result = pd.DataFrame(output_rows)
 
    # Sort: step → rank → bucket_idx → sm_pct descending
    # Descending sm_pct so the plotting function stacks highest SM% first
    df_result = df_result.sort_values(
        ["step", "rank", "bucket_idx", "sm_pct"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
 
    print(f"\n[build_sm_timeline_df] Done")
    print(f"  total rows   : {len(df_result)}")
    print(f"  ranks        : {sorted(df_result['rank'].unique().tolist())}")
    print(f"  steps        : {sorted(df_result['step'].unique().tolist())}")
    print(f"  stream_types : {sorted(df_result['stream_type'].unique().tolist())}")
    print(
        f"  sm_pct range : "
        f"{df_result['sm_pct'].min():.1f}% ~ "
        f"{df_result['sm_pct'].max():.1f}%"
    )
 
    return df_result
 