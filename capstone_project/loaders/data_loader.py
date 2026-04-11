# data_loader.py
# ─────────────────────────────────────────────────────────────────────────────
# All functions that read data from SQLite and transform it into DataFrames.
# Also contains the high-level pipeline functions:
#   process_full_analysis()  → orchestrates one SQLite file
#   load_single_gpu()        → loads one rank's full GpuDataset
#   load_all_gpus()          → loads all ranks
#   get_all_step_intervals() → extracts step start/end times
#   compute_rank_order_per_step() → compares step timing across ranks
#
# DEPENDENCIES:
#   db_helpers.py  ← try_read_df(), list_tables(), get_all_gpus()
#   schemas.py     ← GpuDataset, KernelSchema, NvtxSchema, etc.
#
# STILL DEPENDS ON ORIGINAL FILE (will be removed as you split further):
#   plot_sm_timeline.py ← _get_all_kernels_with_sm_for_rank()
#   (This function handles SM utilization and will be extracted next.)
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .db_helpers import try_read_df

from .schema import (
    GpuDataset,
    KernelSchema, NvtxSchema, MemcpySchema, StringIdsSchema, RuntimeSchema,
    find_kernel_schema, find_nvtx_schema, find_memcpy_schema,
    find_stringids_schema, find_runtime_schema,
)
from analysis.sm_analysis import _get_all_kernels_with_sm_for_rank
# ==============================================================================
# Section 0: GPU hardware info
# ==============================================================================

def get_all_gpus(con: sqlite3.Connection) -> Dict[int, str]:

    """
    Return a dict mapping GPU id → GPU name from the TARGET_INFO_GPU table.


    Simple example:
        Table rows:  id=0, name='NVIDIA A100'
                     id=1, name='NVIDIA A100'
        → returns {0: 'NVIDIA A100', 1: 'NVIDIA A100'}

    Returns an empty dict {} if the table does not exist or query fails.
    """
    gpu_map = {}

    try:
        df = try_read_df(con, "SELECT id, name FROM TARGET_INFO_GPU")

        if not df.empty:

            for _, row in df.iterrows():
                gpu_map[int(row["id"])] = str(row["name"])

    except Exception as e:
        print(f"  [Error] Failed to read GPU Info Table: {e}")

    return gpu_map
    '''
    ex) {0: 'NVIDIA A100', 1: 'NVIDIA A100'}
    '''


# # ── Still imported from original monolithic file ──────────────────────────────
# # Remove this import once _get_all_kernels_with_sm_for_rank is extracted
# # into its own file (e.g. sm_analysis.py).
# try:
#     from plot_sm_timeline import _get_all_kernels_with_sm_for_rank
#     _SM_AVAILABLE = True
# except ImportError:
#     _SM_AVAILABLE = False
#     print("[data_loader] Warning: _get_all_kernels_with_sm_for_rank not available. "
#           "df_all_kernels_with_sm will be empty.")


# ==============================================================================
# Section 3-A: Low-level data loaders
# ==============================================================================

def load_nvtx_events(con: sqlite3.Connection,
                     sch: NvtxSchema) -> pd.DataFrame:
    """
    Load all NVTX range events from the database and rename them to
    standardised names used throughout this project.

    Simple example:
        Raw NVTX table row:  text='forward', start=210, end=350
        After rename:        name='cpu_forward_launch', start=210, end=350
        After batch rename:  name='cpu_batch_0_duration'  (from 'Batch_0')

    Returns a DataFrame with columns: name, start, end, dur_ns

    
    name                          start   end     dur_ns
    cpu_batch_0_duration          100     500     400
    cpu_data_wait_launch          100     110     10
    cpu_h2d_launch                110     115     5
    cpu_train_compute_duration    200     490     290
    cpu_zero_grad_launch          200     210     10
    cpu_forward_launch            210     350     140
    cpu_loss_launch               350     360     10
    cpu_backward_launch           360     480     120
    cpu_opt_step_launch           480     490     10
    cpu_nccl_allreduce_launch     460     475     15
    cpu_batch_1_duration          500     900     400
    cpu_data_wait_launch          500     510     10
    """

    q = f"""
    SELECT
        {sch.name_col}  AS name,
        {sch.start_col} AS start,
        {sch.end_col}   AS end
    FROM {sch.table}
    WHERE {sch.end_col} > {sch.start_col}
    """
    
    df = try_read_df(con, q)
    df = df.dropna(subset=["name"])
    df["dur_ns"] = df["end"] - df["start"]

    # Rename short marker names to longer descriptive names
    rename_map = {
        "data_wait":      "cpu_data_wait_launch",
        "h2d":            "cpu_h2d_launch",
        "gpu_compute":    "cpu_train_compute_duration",
        "zero_grad":      "cpu_zero_grad_launch",
        "forward":        "cpu_forward_launch",
        "loss":           "cpu_loss_launch",
        "backward":       "cpu_backward_launch",
        "opt_step":       "cpu_opt_step_launch",
        "NCCL_AllReduce": "cpu_nccl_allreduce_launch",
    }

    df["name"] = df["name"].apply(lambda x: rename_map.get(x, x))

    # Rename 'Batch_N' → 'cpu_batch_N_duration'
    df["name"] = df["name"].str.replace(
        r"^Batch_(\d+)$", r"cpu_batch_\1_duration", regex=True
    )

    return df


def load_nccl_kernels(con: sqlite3.Connection,  
                      k_sch: KernelSchema,
                      s_sch: StringIdsSchema,
                      like_pattern: str = "%nccl%") -> pd.DataFrame:

    """
    Load GPU kernel events whose names match 'like_pattern' (default: NCCL).

    Joins the kernel table with StringIds to get human-readable kernel names.

    Simple example:
        Kernel table has a row with nameId=504.
        StringIds has: id=504, value='ncclAllReduceRingLLKernel'.
        → returned row has name='ncclAllReduceRingLLKernel'

    Returns DataFrame with: name, kernel_start, kernel_end, dur_ns,
                             correlation_id, stream_id, stream_type

    name                      kernel_start  kernel_end   dur_ns correlation_id  stream_id  stream_type
    ncclAllReduceRingLLKernel   420            480        60                         
    ncclAllReduceRingLLKernel   850            910        60                 
    ncclAllReduceRingLLKernel   1280           1340       60                 
    """

    q = f"""
    SELECT
        s.{s_sch.value_col}    AS name,
        k.{k_sch.start_col}   AS kernel_start,
        k.{k_sch.end_col}     AS kernel_end,
        k.{k_sch.corr_id_col} AS correlation_id,
        k.{k_sch.stream_id_col} AS stream_id
    FROM {k_sch.table} k
    JOIN {s_sch.table} s
      ON k.{k_sch.name_id_col} = s.{s_sch.id_col}
    WHERE k.{k_sch.end_col} > k.{k_sch.start_col}
      AND s.{s_sch.value_col} LIKE '{like_pattern}'
    """

    df = try_read_df(con, q)

    df["dur_ns"]     = df["kernel_end"] - df["kernel_start"]

    df["stream_type"] = "nccl_active"

    if "stream_id" in df.columns:
        df["stream_id"] = df["stream_id"].fillna(-1).astype(int)
    else:
        df["stream_id"] = -1

    return df[["name", "kernel_start", "kernel_end",
               "dur_ns", "correlation_id", "stream_id", "stream_type"]]


def load_memcpy_events(con: sqlite3.Connection,
                       sch: MemcpySchema,
                       want_kinds: Optional[List[int]] = None,
                       name: str = "gpu_h2d_duration") -> pd.DataFrame:
    """
    Load Host-to-Device (H2D) memory copy events.

    'want_kinds' filters by memcpy direction:
        kind=1 → H2D (CPU → GPU)  
        kind=2 → D2H (GPU → CPU)

    Simple example:
        One row: start=150ns, end=160ns (10ns copy), kind=1
        → label it 'gpu_h2d_duration', stream_type='h2d'

    Returns DataFrame with: name, kernel_start, kernel_end, dur_ns,
                            correlation_id, kind, stream_id, stream_type
                            
    name              kernel_start  kernel_end  dur_ns  correlation_id  kind  stream_id  stream_type
    gpu_h2d_duration  150           160         10000   4001            1     19         h2d
    gpu_h2d_duration  160           168         8000    4002            1     19         h2d
    gpu_h2d_duration  480           490         10000   4004            1     19         h2d          
    """

    stream_select = f", {sch.stream_id_col} AS stream_id" if sch.stream_id_col else ""

    q = f"""
    SELECT
        {sch.start_col}   AS kernel_start,
        {sch.end_col}     AS kernel_end,
        {sch.corr_id_col} AS correlation_id,
        {sch.kind_col}    AS kind{stream_select}
    FROM {sch.table}
    WHERE {sch.end_col} > {sch.start_col}
    """
    df = try_read_df(con, q)

    if df.empty:
        return pd.DataFrame(columns=[
            "name", "kernel_start", "kernel_end", "dur_ns",
            "correlation_id", "stream_id", "stream_type",
        ])

    if want_kinds is not None and sch.kind_col and "kind" in df.columns:
        df = df[df["kind"].isin(want_kinds)].copy()

    df["name"]        = name
    df["dur_ns"]      = df["kernel_end"] - df["kernel_start"]
    df["stream_type"] = "h2d"

    if "stream_id" in df.columns:
        df["stream_id"] = df["stream_id"].fillna(-1).astype(int)

    else:
        df["stream_id"] = -1

    return df[["name", 
               "kernel_start", 
               "kernel_end", 
               "dur_ns",
               "correlation_id", 
               "kind",
               "stream_id", 
               "stream_type"]].copy()


def load_kernel_occupancy_stats(con: sqlite3.Connection,

                                k_sch: KernelSchema) -> pd.DataFrame:
    """
    Load kernel occupancy data needed to compute SM utilization.

    For each kernel execution, we record:
        threads_per_block    = blockX × blockY × blockZ
        total_blocks         = gridX  × gridY  × gridZ
        registers_per_thread = how many registers each thread uses
        static/dynamic_shared_mem = shared memory usage (bytes)
        duration_ns          = how long the kernel ran

    Simple example:
        A matrix multiply kernel:
            blockX=128, blockY=1, blockZ=1  → threads_per_block = 128
            gridX=256,  gridY=1, gridZ=1   → total_blocks = 256
            → 256 blocks × 128 threads = 32768 threads running in parallel

    Returns DataFrame with: correlation_id, k_start, threads_per_block,
                             total_blocks, registers_per_thread,
                             static_shared_mem, dynamic_shared_mem, duration_ns
    """

    q = f"""
        SELECT
            {k_sch.corr_id_col}        AS correlation_id,
            {k_sch.start_col}          AS k_start,
            {k_sch.end_col}            AS k_end,
            {k_sch.block_x_col}        AS blockX,
            {k_sch.block_y_col}        AS blockY,
            {k_sch.block_z_col}        AS blockZ,
            {k_sch.grid_x_col}         AS gridX,
            {k_sch.grid_y_col}         AS gridY,
            {k_sch.grid_z_col}         AS gridZ,
            {k_sch.registers_col}      AS registersPerThread,
            {k_sch.static_shared_col}  AS staticSharedMemory,
            {k_sch.dynamic_shared_col} AS dynamicSharedMemory
        FROM {k_sch.table}
        WHERE {k_sch.end_col} > {k_sch.start_col}
    """
    df = pd.read_sql_query(q, con)

    if df.empty:
        raise ValueError("CUPTI_ACTIVITY_KIND_KERNEL table is empty.")

    df["threads_per_block"]    = df["blockX"] * df["blockY"] * df["blockZ"]
    df["total_blocks"]         = df["gridX"]  * df["gridY"]  * df["gridZ"]
    df["registers_per_thread"] = df["registersPerThread"]
    df["static_shared_mem"]    = df["staticSharedMemory"]
    df["dynamic_shared_mem"]   = df["dynamicSharedMemory"]
    df["duration_ns"]          = df["k_end"] - df["k_start"]

    invalid = df["threads_per_block"] <= 0  # find corrupted/non-compute rows

    if invalid.any():
        print(f"[Warning] threads_per_block <= 0 in {invalid.sum()} kernels — removed.") # tell the user how many were removed
        df = df[~invalid] # remove them before SM% math runs

    return df[[
        "correlation_id", 
        "k_start",
        "threads_per_block", 
        "total_blocks",
        "registers_per_thread", 
        "static_shared_mem", 
        "dynamic_shared_mem",
        "duration_ns",
    ]]

def load_gpu_kernel_timings(con: sqlite3.Connection,
                             k_sch: KernelSchema,
                             s_sch: StringIdsSchema = None) -> pd.DataFrame:
    """
    Load ALL GPU kernel executions with their timestamps and kernel names.

    If a StringIdsSchema is provided, the kernel name is joined in.
    Otherwise, only the correlation_id and timestamps are returned.

    Simple example:
        correlation_id=1001, kernel_start=300ns, kernel_end=400ns
        kernel_name='volta_sgemm_128x32_tn'
        → This forward-pass matrix-multiply kernel ran for 100ns starting at 300ns.

    Returns DataFrame with: correlation_id, kernel_start, kernel_end,
                             stream_id, (kernel_name if s_sch given)

    correlation_id   kernel_start   kernel_end       kernel_name
      1001            300           400             volta_sgemm_128x32_tn
      1002            400           500             volta_sgemm_128x32_tn
      1003            500           520             vectorized_elementwise
      2001            550           650             volta_sgemm_128x32_nt
      2002            650           750             volta_sgemm_128x32_nt
      3001            760           820             ncclAllReduceRingLL

    """
    if s_sch:
        q = f"""
        SELECT
            k.{k_sch.corr_id_col}   AS correlation_id,
            k.{k_sch.start_col}     AS kernel_start,
            k.{k_sch.end_col}       AS kernel_end,
            k.{k_sch.stream_id_col} AS stream_id,
            s.{s_sch.value_col}     AS kernel_name
        FROM {k_sch.table} k
        LEFT JOIN {s_sch.table} s
            ON k.{k_sch.name_id_col} = s.{s_sch.id_col}
        WHERE k.{k_sch.end_col} > k.{k_sch.start_col}
        """

    else:
        q = f"""
        SELECT
            {k_sch.corr_id_col}   AS correlation_id,
            {k_sch.start_col}     AS kernel_start,
            {k_sch.end_col}       AS kernel_end,
            {k_sch.stream_id_col} AS stream_id
        FROM {k_sch.table}
        WHERE {k_sch.end_col} > {k_sch.start_col}
        """

    return try_read_df(con, q)

def load_runtime_events_in_nvtx(con: sqlite3.Connection,
                                 nvtx_df: pd.DataFrame,
                                 r_sch: RuntimeSchema) -> pd.DataFrame:
    """
    Load all CPU Runtime API calls that fall within the time range of the NVTX data.

    These are the CUDA calls (cudaLaunchKernel, cudaMemcpyAsync, …) the CPU
    made to schedule GPU work.  Each call has a correlation_id that links to
    the GPU kernel that actually ran.

    Simple example:
        NVTX spans from t=100 to t=900.
        Runtime API call: correlation_id=1001, cpu_api_start=210, cpu_api_end=211


    Returns DataFrame with: correlation_id, cpu_api_start, cpu_api_end

     correlation_id   cpu_api_start   cpu_api_end
      1001                 210             210.5      
      1002                 211             211.5       
      1003                 212             212.5       
      ...                  ...             ...
      2001                 361             361.5      
      2002                 362             362.5       
      3001                 461             461.5       
    """
    if nvtx_df.empty:
        return pd.DataFrame()

    
    min_start = nvtx_df["start"].min()
    max_end   = nvtx_df["end"].max()

    q = f"""
    SELECT
        {r_sch.corr_id_col} AS correlation_id,
        {r_sch.start_col}   AS cpu_api_start,
        {r_sch.end_col}     AS cpu_api_end
    FROM {r_sch.table}
    WHERE {r_sch.start_col} >= {min_start}
      AND {r_sch.end_col}   <= {max_end}
    """

    df = try_read_df(con, q)

    return df.sort_values("cpu_api_start") if not df.empty else df

# ==============================================================================
# Section 3-B: CPU → GPU correlation helpers
# ==============================================================================

def map_nvtx_to_runtime(df_nvtx: pd.DataFrame,
                         df_runtime_api: pd.DataFrame) -> pd.DataFrame:
    """

    For each NVTX range, 
    collect the correlation_ids of CPU runtime calls that happened INSIDE that range.

    Simple example:
        NVTX range 'cpu_forward_launch': start=210, end=350
        Runtime API calls inside this range:
            correlation_id=1001, cpu_api_start=211
            correlation_id=1002, cpu_api_start=220
            correlation_id=1003, cpu_api_start=300
        → 3 rows added:
            (cpu_launch_in_nvtx='cpu_forward_launch', correlation_id=1001)
            (cpu_launch_in_nvtx='cpu_forward_launch', correlation_id=1002)
            (cpu_launch_in_nvtx='cpu_forward_launch', correlation_id=1003)

    Returns DataFrame with: cpu_launch_in_nvtx, cpu_launch_start,
                             cpu_launch_end, correlation_id

    cpu_launch_in_nvtx       cpu_launch_start  cpu_launch_end  correlation_id
      cpu_batch_0_wrapper           100         500                1001
      cpu_batch_0_wrapper           100         500                1002
      cpu_batch_0_wrapper           100         500                1003
      cpu_batch_0_wrapper           100         500                2001
      cpu_batch_0_duration          100         500                2002
      cpu_batch_0_duration          100         500                3001
      cpu_data_wait_launch          100         110                NaN
      cpu_h2d_launch                110         115                NaN
      cpu_train_compute_wrapper     200         490                1001
      cpu_train_compute_wrapper     200         490                1002
      cpu_train_compute_wrapper     200         490                1003
      cpu_train_compute_wrapper     200         490                2001
      cpu_train_compute_wrapper     200         490                2002
      cpu_train_compute_wrapper     200         490                3001
      cpu_forward_launch            210         350                1001
      cpu_forward_launch            210         350                1002
      cpu_forward_launch            210         350                1003
      cpu_backward_launch           360         480                2001
      cpu_backward_launch           360         480                2002
      cpu_nccl_allreduce_launch     460         475                3001
    """
    if df_nvtx.empty or df_runtime_api.empty:
        return pd.DataFrame()

    nvtx_sorted = df_nvtx.sort_values("start").reset_index(drop=True)

    runtime_sorted   = df_runtime_api.sort_values("cpu_api_start").reset_index(drop=True)

    runtime_starts = runtime_sorted["cpu_api_start"].values
    runtime_corrs  = runtime_sorted["correlation_id"].values

    rows = []

    for _, row in nvtx_sorted.iterrows():
        cpu_launch_start    = row["start"]
        cpu_launch_end      = row["end"]
        cpu_launch_in_nvtx  = row["name"]

        
        mask          = (runtime_starts >= cpu_launch_start) & (runtime_starts < cpu_launch_end)
        '''
        runtime_starts = [150, 211, 220, 300, 362, 461]   ← all runtime call times
        
        cpu_launch_start = 210,  cpu_launch_end = 350 ← come from nvtx range

        mask = [False, True, True, True, False, False]
                    ↑     ↑     ↑
                    211   220   300  all fall inside 210~350

        matched_corrs = [1001, 1002, 1003]   ← correlation IDs of those calls
        '''

        matched_corrs = runtime_corrs[mask]

        for cid in matched_corrs:
            rows.append({
                "cpu_launch_in_nvtx": cpu_launch_in_nvtx,
                "cpu_launch_start":   cpu_launch_start,
                "cpu_launch_end":     cpu_launch_end,
                "correlation_id":     cid,
            })

    return pd.DataFrame(rows)


def join_cpu_api_in_nvtx_with_gpu_kernel_timings(
        df_map_nvtx_to_cpu_api: pd.DataFrame,
        df_kernels_all:          pd.DataFrame,
        df_step_ranges_in_nvtx:  pd.DataFrame = None) -> pd.DataFrame:
    """
    Join CPU-launch correlation IDs with actual GPU kernel timestamps.

    After this join, each row tells you:
        "The CPU called 'forward' at t=210ns (CPU clock).
         The GPU ran kernel X from t=310ns to t=550ns (GPU clock)."

    Simple example:
        df_map_nvtx_to_cpu_api:
            cpu_launch_in_nvtx='cpu_forward_launch', correlation_id=1001
        df_kernels_all:
            correlation_id=1001, kernel_start=310, kernel_end=400
        → joined row:
            name='gpu_forward_duration', gpu_start=310, gpu_end=400

    Also assigns data_batch_idx (which training step this belongs to)
    using df_step_ranges_in_nvtx.

    Returns DataFrame with: name, gpu_start, gpu_end, dur_ns,
                             correlation_id, stream_id, stream_type,
                             (data_batch_idx if step ranges provided)
    """
    if df_map_nvtx_to_cpu_api.empty or df_kernels_all.empty:
        return pd.DataFrame()

    print(f"  Merging {len(df_map_nvtx_to_cpu_api)} CPU-API mappings "
          f"with {len(df_kernels_all)} GPU kernels...")

    # Map CPU NVTX names → GPU duration names
    nvtx_to_gpu_name = {
        "cpu_forward_launch":         "gpu_forward_duration",
        "cpu_backward_launch":        "gpu_backward_duration",
        "cpu_loss_launch":            "gpu_loss_duration",
        "cpu_opt_step_launch":        "gpu_opt_step_duration",
        "cpu_zero_grad_launch":       "gpu_zero_grad_duration",
        "cpu_train_compute_duration": "gpu_train_compute_duration",
        "cpu_nccl_allreduce_launch":  "gpu_nccl_allreduce_duration",
    }

    def to_gpu_name(cpu_name: str) -> str:
        m = re.match(r"cpu_batch_(\d+)_duration", cpu_name)
        if m:
            return f"gpu_batch_{m.group(1)}_duration"
        return nvtx_to_gpu_name.get(cpu_name, f"gpu_{cpu_name}")

    # Step 1: join on correlation_id
    merged = pd.merge(df_map_nvtx_to_cpu_api, df_kernels_all,
                       on="correlation_id", how="inner")

    # Step 2: remove NCCL kernels that are inside the 'backward' NVTX range
    # (DDP overlaps AllReduce with backward; NCCL is tracked separately)
    if "kernel_name" in merged.columns:
        nccl_in_bwd = (
            merged["cpu_launch_in_nvtx"].str.contains("cpu_backward_launch", case=False) &
            merged["kernel_name"].str.contains("nccl", case=False)
        )
        nccl_in_ar = (
            merged["cpu_launch_in_nvtx"].str.contains("cpu_nccl_allreduce_launch", case=False) &
            merged["kernel_name"].str.contains("nccl", case=False)
        )
        merged = merged[~(nccl_in_bwd | nccl_in_ar)]

    if merged.empty:
        return pd.DataFrame()

    # Step 3: keep needed columns
    cols = ["cpu_launch_in_nvtx", "cpu_launch_start", "cpu_launch_end",
            "correlation_id", "kernel_start", "kernel_end", "kernel_name"]
    if "stream_id" in merged.columns:
        cols.append("stream_id")
    result = merged[cols].copy()

    result["dur_ns"] = result["kernel_end"] - result["kernel_start"]
    result["name"]   = result["cpu_launch_in_nvtx"].apply(to_gpu_name)

    # Step 4: stream_type
    if "kernel_name" in result.columns:

        from .db_helpers import _stream_type_from_nvtx_name
        result["stream_type"] = result["name"].apply(_stream_type_from_nvtx_name)
    else:
        result["stream_type"] = "unknown"

    # Step 5: rename kernel_start/end → gpu_start/end
    result = result.rename(columns={"kernel_start": "gpu_start",
                                     "kernel_end":   "gpu_end"})
    result = result.dropna(subset=["gpu_start", "gpu_end"])
    result["gpu_start"] = result["gpu_start"].astype(int)
    result["gpu_end"]   = result["gpu_end"].astype(int)
    result["dur_ns"]    = result["dur_ns"].astype(int)
    if "stream_id" in result.columns:
        result["stream_id"] = result["stream_id"].fillna(-1).astype(int)
    else:
        result["stream_id"] = -1

    # Step 6: assign data_batch_idx via merge_asof
    if df_step_ranges_in_nvtx is not None and not df_step_ranges_in_nvtx.empty:
        result = result.sort_values("cpu_launch_start")
        step_ranges = df_step_ranges_in_nvtx.sort_values("start")
        result = pd.merge_asof(
            result,
            step_ranges[["start", "step"]].rename(
                columns={"start": "cpu_launch_start"}),
            on="cpu_launch_start",
            direction="backward",
        )
        result = result.rename(columns={"step": "data_batch_idx"})
        return result[[
            "name", "gpu_start", "gpu_end", "dur_ns",
            "correlation_id", "data_batch_idx", "stream_id", "stream_type",
        ]].reset_index(drop=True)

    return result[[
        "name", "gpu_start", "gpu_end", "dur_ns",
        "correlation_id", "stream_id", "stream_type",
    ]].reset_index(drop=True)


def join_kernel_occupancy_with_steps(df_kernel_occupancy: pd.DataFrame,
                                      df_gpu_duration: pd.DataFrame) -> pd.DataFrame:
    """
    Assign data_batch_idx (training step) to each kernel occupancy row
    by joining on correlation_id.

    Simple example:
        df_kernel_occupancy has: correlation_id=1001, threads_per_block=128
        df_gpu_duration has:     correlation_id=1001, data_batch_idx=2
        → joined row: correlation_id=1001, threads_per_block=128, data_batch_idx=2

    Returns DataFrame with: correlation_id, data_batch_idx, threads_per_block,
                             total_blocks, registers_per_thread,
                             static_shared_mem, dynamic_shared_mem, duration_ns
    """
    if df_kernel_occupancy.empty:
        raise ValueError("df_kernel_occupancy is empty.")
    if df_gpu_duration.empty:
        raise ValueError("df_gpu_duration is empty.")

    WRAPPER_PATTERN = r"gpu_batch_\d+_duration|gpu_train_compute_duration"
    df_step_map = df_gpu_duration[
        ~df_gpu_duration["name"].str.contains(WRAPPER_PATTERN, regex=True)
    ][["correlation_id", "data_batch_idx"]].drop_duplicates(subset=["correlation_id"])

    df = pd.merge(df_kernel_occupancy, df_step_map,
                   on="correlation_id", how="inner")

    if df.empty:
        raise ValueError("No matching correlation_ids between occupancy and gpu_duration.")

    missing = df["data_batch_idx"].isna().sum()
    if missing > 0:
        print(f"[Warning] {missing} kernels with no data_batch_idx removed.")
        df = df.dropna(subset=["data_batch_idx"])

    df["data_batch_idx"] = df["data_batch_idx"].astype(int)

    return df[[
        "correlation_id", "data_batch_idx",
        "threads_per_block", "total_blocks",
        "registers_per_thread", "static_shared_mem",
        "dynamic_shared_mem", "duration_ns",
    ]].sort_values(["data_batch_idx", "correlation_id"]).reset_index(drop=True)


# ==============================================================================
# Section 3-C: Step extraction helpers
# ==============================================================================

def build_step_df_from_nvtx(nvtx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract training-step start/end times from NVTX markers.

    Looks for patterns like: 'Batch_0', 'step 1', 'Iter-2', 'Iteration:3',
    'Global Step 4' — all of these map to a step number.

    Simple example:
        NVTX has rows:
            name='cpu_batch_0_duration', start=100, end=500
            name='cpu_batch_1_duration', start=500, end=900
        → returns:
            step  start  end
            0     100    500
            1     500    900

    Returns DataFrame with columns: step, start, end
    """
    if nvtx_df.empty:
        return pd.DataFrame()

    pattern = r"(?i)(?:Batch|step|Iter|Iteration|Global Step)[_\s:\-]*(\d+)"
    df = nvtx_df.copy()
    df["step"] = pd.to_numeric(
        df["name"].str.extract(pattern, expand=False), errors="coerce"
    )
    valid = df.dropna(subset=["step"])
    if valid.empty:
        return pd.DataFrame(columns=["step", "start", "end"])

    step_df = (
        valid.groupby("step")
        .agg(start=("start", "min"), end=("end", "max"))
        .reset_index()
    )
    step_df["step"] = step_df["step"].astype(int)
    return step_df.sort_values("start")[["step", "start", "end"]]


# ==============================================================================
# Section 4: process_full_analysis  (orchestrates one SQLite file)
# ==============================================================================

def process_full_analysis(
        con: sqlite3.Connection,
        steps: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full extraction pipeline for one SQLite file.

    Steps performed:
        1. Load NVTX events (CPU markers)
        2. Load NCCL kernels
        3. Load Memcpy events (H2D)
        4. Extract step ranges from NVTX
        5. Load Runtime API calls within NVTX ranges
        6. Load all GPU kernel timings
        7. Map NVTX ranges → correlation IDs → GPU kernel timestamps
        8. Assign data_batch_idx (training step) to each event

    Simple example:
        Input:  one .sqlite file from Nsight Systems
        Output: four DataFrames:
            df_nvtx        → CPU markers with step labels
            df_nccl        → NCCL kernel events with step labels
            df_memcpy      → H2D copy events with step labels
            df_gpu_duration → True GPU execution times per NVTX phase

    Returns: (df_nvtx, df_nccl, df_memcpy, df_gpu_duration)
    """
    nvtx_sch = find_nvtx_schema(con)
    k_sch    = find_kernel_schema(con)
    r_sch    = find_runtime_schema(con)
    m_sch    = find_memcpy_schema(con)
    s_sch    = find_stringids_schema(con)

    # ── 1. NVTX ──────────────────────────────────────────────────────────────
    df_nvtx = load_nvtx_events(con, nvtx_sch) if nvtx_sch else pd.DataFrame()

    # ── 2. NCCL kernels ───────────────────────────────────────────────────────
    df_nccl = pd.DataFrame()
    if k_sch and s_sch:
        df_nccl = load_nccl_kernels(con, k_sch, s_sch)

    # ── 3. Memcpy ─────────────────────────────────────────────────────────────
    df_memcpy = pd.DataFrame()
    if m_sch:
        df_memcpy = load_memcpy_events(con, m_sch,
                                        want_kinds=[1],
                                        name="gpu_h2d_duration")

    # ── 4-8. True GPU span calculation ───────────────────────────────────────
    df_gpu_duration = pd.DataFrame()

    if nvtx_sch and r_sch and k_sch and not df_nvtx.empty:
        try:
            # Step 4: extract CPU-side step ranges
            df_step_ranges_in_nvtx = build_step_df_from_nvtx(df_nvtx)

            # Step 5: Runtime API calls within NVTX time range
            df_runtime_api_in_nvtx = load_runtime_events_in_nvtx(
                con, df_nvtx, r_sch
            )

            # Step 6: All GPU kernel timings (with names)
            df_kernels_all = load_gpu_kernel_timings(con, k_sch, s_sch=s_sch)

            # Step 7: Map NVTX → correlation_id → GPU kernel
            df_map_nvtx_to_cpu_api = map_nvtx_to_runtime(
                df_nvtx, df_runtime_api_in_nvtx
            )

            # Step 8: Join → df_gpu_duration
            df_gpu_duration = join_cpu_api_in_nvtx_with_gpu_kernel_timings(
                df_map_nvtx_to_cpu_api,
                df_kernels_all,
                df_step_ranges_in_nvtx=df_step_ranges_in_nvtx,
            )

            if df_gpu_duration.empty:
                raise ValueError("df_gpu_duration is empty after join.")

            # ── Assign data_batch_idx to NVTX events ─────────────────────────
            def assign_data_batch_idx(event_start):
                for _, row in df_step_ranges_in_nvtx.iterrows():
                    if row["start"] <= event_start < row["end"]:
                        return int(row["step"])
                return -1

            if not df_nvtx.empty:
                df_nvtx["data_batch_idx"] = df_nvtx["start"].apply(
                    assign_data_batch_idx
                )

            # ── Assign data_batch_idx to NCCL kernels ────────────────────────
            if not df_nccl.empty and not df_nvtx.empty:
                nvtx_nccl = df_nvtx[
                    df_nvtx["name"] == "cpu_nccl_allreduce_launch"
                ].sort_values("start")

                nccl_corr_mapping = df_map_nvtx_to_cpu_api[
                    df_map_nvtx_to_cpu_api["cpu_launch_in_nvtx"]
                    == "cpu_nccl_allreduce_launch"
                ][["cpu_launch_in_nvtx", "cpu_launch_start", "correlation_id"]]

                nccl_corr_mapping = nccl_corr_mapping.merge(
                    nvtx_nccl[["start", "data_batch_idx"]],
                    left_on="cpu_launch_start", right_on="start", how="left",
                )
                df_nccl = df_nccl.merge(
                    nccl_corr_mapping[["correlation_id", "data_batch_idx"]],
                    on="correlation_id", how="left",
                )

            # ── Assign data_batch_idx to Memcpy ──────────────────────────────
            if not df_memcpy.empty:
                nvtx_h2d = df_nvtx[
                    df_nvtx["name"] == "cpu_h2d_launch"
                ].sort_values("start").reset_index(drop=True)

                h2d_corr_mapping = df_map_nvtx_to_cpu_api[
                    df_map_nvtx_to_cpu_api["cpu_launch_in_nvtx"] == "cpu_h2d_launch"
                ][["cpu_launch_start", "correlation_id"]].merge(
                    nvtx_h2d[["start", "data_batch_idx"]],
                    left_on="cpu_launch_start", right_on="start", how="left",
                )
                df_memcpy = df_memcpy.merge(
                    h2d_corr_mapping[["correlation_id", "data_batch_idx"]],
                    on="correlation_id", how="left",
                )

        except Exception as e:
            print(f"  [Warning] process_full_analysis: {e}")
            import traceback
            traceback.print_exc()

    return df_nvtx, df_nccl, df_memcpy, df_gpu_duration


# ==============================================================================
# Section 5: load_single_gpu  (loads one rank's full GpuDataset)
# ==============================================================================

def load_single_gpu(filepath: str,
                     rank: int,
                     steps: List[int],
                     gpu_specs: dict) -> Optional[GpuDataset]:
    """
    Load all profiling data for one GPU rank from its SQLite file.

    Simple example:
        filepath = 'experiment_rank0.sqlite'
        rank     = 0
        steps    = [2, 3, 4]
        gpu_specs = {'sm_count': 108, ...}

        → returns GpuDataset(rank=0, df_nvtx=..., df_nccl=..., ...)

    Returns None if the file does not exist or loading fails.
    """
    if not os.path.exists(filepath):
        print(f"[Skip] File not found: {filepath}")
        return None

    print(f"\nLoading Rank {rank}: {filepath} ...")

    con = sqlite3.connect(filepath)
    try:
        gpu_info = get_all_gpus(con)

        df_nvtx, df_nccl, df_memcpy, df_gpu_duration = process_full_analysis(
            con, steps
        )


        df_kernel_occupancy_per_step = pd.DataFrame()

        try:
            k_sch = find_kernel_schema(con)
            if k_sch:
                df_kernel_occupancy_per_step = load_kernel_occupancy_stats(con, k_sch)

                # Use GPU batch durations to assign step numbers to each kernel
                gpu_step_ranges = df_gpu_duration[
                    df_gpu_duration["name"].str.contains(
                        r"gpu_batch_\d+_duration", regex=True
                    )
                ][["data_batch_idx", "gpu_start", "gpu_end"]].sort_values("gpu_start")

                df_kernel_occupancy_per_step = pd.merge_asof(
                    df_kernel_occupancy_per_step.sort_values("k_start"),
                    gpu_step_ranges[["gpu_start", "data_batch_idx"]].rename(
                        columns={"gpu_start": "k_start"}
                    ),
                    on="k_start",
                    direction="backward",
                )
                df_kernel_occupancy_per_step = df_kernel_occupancy_per_step[
                    df_kernel_occupancy_per_step["data_batch_idx"].notna()
                ].copy()

                df_kernel_occupancy_per_step["data_batch_idx"] = (
                    df_kernel_occupancy_per_step["data_batch_idx"].astype(int)
                )
        except Exception as e:
            print(f"  [Warning] Rank {rank}: kernel occupancy failed: {e}")

        # ── SM utilization per kernel ─────────────────────────────────────────
        # Still imported from original file. Will be moved to sm_analysis.py later.
        df_all_kernels_with_sm = pd.DataFrame()
        try:
            df_all_kernels_with_sm = _get_all_kernels_with_sm_for_rank(
                df_gpu_duration              = df_gpu_duration,
                df_memcpy                    = df_memcpy,
                df_nccl                      = df_nccl,
                df_kernel_occupancy_per_step = df_kernel_occupancy_per_step,
                gpu_specs                    = gpu_specs,                )
            print(
                f"  [Rank {rank}] all_kernels_with_sm: "
                f"{len(df_all_kernels_with_sm)} rows / "
                f"stream_types: "
                f"{sorted(df_all_kernels_with_sm['stream_type'].unique().tolist())}"
            )
        except Exception as e:
            print(f"  [Warning] Rank {rank}: SM utilization failed: {e}")

        return GpuDataset(
            rank                         = rank,
            filename                     = filepath,
            df_nvtx                      = df_nvtx,
            df_nccl                      = df_nccl,
            df_memcpy                    = df_memcpy,
            df_gpu_duration              = df_gpu_duration,
            gpu_info                     = gpu_info,
            df_kernel_occupancy_per_step = df_kernel_occupancy_per_step,
            df_all_kernels_with_sm       = df_all_kernels_with_sm,
        )

    except Exception as e:
        print(f"  [Error] Failed to load {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        con.close()


# ==============================================================================
# Section 6: Multi-GPU loaders & step-timing analysis
# ==============================================================================

def load_all_gpus(file_list: List[str],
                   steps_to_analyze: List[int],
                   gpu_specs: dict) -> Dict[int, GpuDataset]:
    """
    Load all GPU SQLite files into a gpu_data_map dict.

    Simple example:
        file_list = ['exp_rank0.sqlite', 'exp_rank1.sqlite']
        → gpu_data_map = {0: GpuDataset(...), 1: GpuDataset(...)}
    """
    gpu_data_map = {}
    for idx, filepath in enumerate(sorted(file_list)):
        rank    = get_rank_from_filename(filepath, idx)
        dataset = load_single_gpu(filepath, rank, steps_to_analyze, gpu_specs)
        if dataset:
            gpu_data_map[rank] = dataset

    print(f"\n[Done] Loaded data for {len(gpu_data_map)} GPU(s).")
    return gpu_data_map


def get_all_step_intervals(gpu_data_map: Dict[int, GpuDataset]
                            ) -> Dict[int, pd.DataFrame]:
    """
    For each GPU rank, extract start/end timestamps of each training step
    from its NVTX markers.

    Simple example:
        Rank 0 NVTX has: cpu_batch_0_duration (t=100~500), cpu_batch_1_duration (t=500~900)
        Rank 1 NVTX has: cpu_batch_0_duration (t=102~502), cpu_batch_1_duration (t=502~902)

        Returns:
            {
                0: DataFrame(step=[0,1], start=[100,500], end=[500,900]),
                1: DataFrame(step=[0,1], start=[102,502], end=[502,902]),
            }

    all_steps_map example:
    {
      0: step  start      end
         0     13800000   14200000   ← Cpu time(ns)
         1     14200000   14600000
         2     14600000   15000000

      1: step  start      end
         0     13800100   14200100   
         1     14200100   14600100
         2     14600100   15000100

      2: step  start      end       
         0     13697000   14097000
         1     14097000   14497000
         2     14497000   14897000

      3: step  start      end       
         0     13684000   14084000
         1     14084000   14484000
         2     14484000   14884000
    }
    """
    all_steps_map = {}

    print("--- Extracting Step Intervals from NVTX ---")

    for rank, dataset in gpu_data_map.items():

        step_df = build_step_df_from_nvtx(dataset.df_nvtx)

        if not step_df.empty:
            all_steps_map[rank] = step_df
        else:
            print(f"[Warning] Rank {rank}: No step markers found in NVTX.")

    return all_steps_map


def compute_rank_order_per_step(
        all_steps_map: Dict[int, pd.DataFrame],
        gpu_data_map:  Dict[int, GpuDataset],
        offsets:       Dict[int, int]
        ) -> pd.DataFrame:

    """
    Compare step timing across all GPU ranks and find, for each step
    who started first, who finished last, and who was the bottleneck.
 
    Returns a DataFrame with one row per step and columns:
        step               → int, training step number
 
        earliest_start     → int (ns), earliest CPU step start across all ranks
        latest_end         → int (ns), latest CPU step end across all ranks
        cpu_fastest_rank   → int, rank whose CPU started first
        cpu_slowest_rank   → int, rank whose CPU ended last
 
        fwd_earliest_start → int (ns), earliest GPU forward start across all ranks
        fwd_fastest_rank   → int, rank whose GPU forward started first
 
        bwd_latest_end     → int (ns), latest GPU backward end across all ranks
        bwd_slowest_rank   → int, rank that finished backward last (THE BOTTLENECK)
 
    Example output (timestamps in ms for readability):
 
    step  earliest_start  latest_end  cpu_fastest  cpu_slowest  fwd_earliest  fwd_fastest  bwd_latest  bwd_slowest
                     (ms)        (ms)         rank         rank           (ms)         rank        (ms)         rank
       0       21322.8     24032.2            1            0        21881.8            1     23850.0            2
       1       24032.0     24104.4            3            0        24038.8            2     24085.0            2
       2       24104.4     24362.0            1            2        24299.4            1     24343.0            2
 
    If bwd_slowest_rank is consistently the same rank across many steps,
    that rank is your training bottleneck.
    """
    #-----------------------
    # CPU timing
    #---------------------
    combined_list = []
    
    for rank, df in all_steps_map.items():
        temp_df          = df.copy()
        temp_df["rank"]  = rank
        offset           = offsets.get(rank, 0)
        temp_df["start"] = temp_df["start"] + offset
        temp_df["end"]   = temp_df["end"]   + offset

        #after adding offset
        combined_list.append(temp_df)


    if not combined_list:
        return pd.DataFrame()

    #combine all start time and end time from all ranks
    big_df    = pd.concat(combined_list, ignore_index=True)

    df_grouped = big_df.groupby("step")
    '''
    df_grouped:
      step 0 group (index 0, 2, 4, 6):
        index  step  start      end        rank
        0      0     13800000   14200000   0
        2      0     13800100   14200100   1
        4      0     13697000   14097000   2
        6      0     13684000   14084000   3
    
      step 1 group (index 1, 3, 5, 7):
        index  step  start      end        rank
        1      1     14200000   14600000   0
        3      1     14200100   14600100   1
        5      1     14097000   14497000   2
        7      1     14084000   14484000   3
    '''

    df_rank_order = df_grouped.agg(
        earliest_start=("start", "min"),
        latest_end    =("end",   "max"),
    )
    '''
    df_rank_order_per_step exampe:
    step  earliest_start  latest_end
    0     13799800        14200100
    1     14199800        14600100
    '''

    min_idx = df_grouped["start"].idxmin()
    max_idx = df_grouped["end"].idxmax()

    fastest = big_df.loc[min_idx, ["step", "rank"]].rename(
        columns={"rank": "cpu_fastest_rank"}
    )
    slowest = big_df.loc[max_idx, ["step", "rank"]].rename(
        columns={"rank": "cpu_slowest_rank"}
    )

    df_rank_order = df_rank_order.merge(fastest, on="step", how="left")
    df_rank_order = df_rank_order.merge(slowest, on="step", how="left")

    #-----------------------
    # GPU timing
    #---------------------

    fwd_start_map      = {}   # step → earliest forward start (any rank)
    fwd_start_rank_map = {}   # step → rank that started forward first

    bwd_end_map        = {}   # step → latest backward end (any rank)  
    bwd_end_rank_map   = {}   # step → rank that finished backward last ← bottleneck

    for rank, dataset in gpu_data_map.items():
        if dataset.df_gpu_duration.empty:
            continue

        df = dataset.df_gpu_duration
        offset = offsets.get(rank, 0)

        for name_pattern, start_map, start_rank_map, end_map, end_rank_map in [
            ("gpu_forward_duration",  fwd_start_map, fwd_start_rank_map, None, None),
            ("gpu_backward_duration", None, None, bwd_end_map, bwd_end_rank_map),
        ]:
            rows = df[df["name"] == name_pattern]
            for _, row in rows.iterrows():
                step  = int(row["data_batch_idx"])
                start = int(row["gpu_start"]) + offset
                end   = int(row["gpu_end"])   + offset

                if start_map is not None:
                    if start < start_map.get(step, start + 1):
                        start_map[step]      = start
                        start_rank_map[step] = rank

                if end_map is not None:
                    if end > end_map.get(step, end - 1):
                        end_map[step]      = end
                        end_rank_map[step] = rank

    df_rank_order["fwd_earliest_start"]  = df_rank_order["step"].map(fwd_start_map)
    df_rank_order["fwd_fastest_rank"]    = df_rank_order["step"].map(fwd_start_rank_map)
    df_rank_order["bwd_latest_end"]      = df_rank_order["step"].map(bwd_end_map)
    df_rank_order["bwd_slowest_rank"]    = df_rank_order["step"].map(bwd_end_rank_map)

    return df_rank_order.sort_values("step")