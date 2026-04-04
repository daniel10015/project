import sys
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from matplotlib.patches import Patch
import glob
# ==============================================================================
# 1. Generic SQLite Helpers
# ==============================================================================


def list_tables(con: sqlite3.Connection) -> List[str]:  

    # Return a list of all table names in this SQLite database.
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]

def table_columns(con: sqlite3.Connection, table: str) -> List[str]:

    # Return a list of column names for the given table.
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [r[1] for r in rows]

def try_read_df(con: sqlite3.Connection, query: str) -> pd.DataFrame:
    
    # Run the SQL query and return the result as a pandas DataFrame.
    return pd.read_sql_query(query, con)



def get_all_gpus(con: sqlite3.Connection) -> Dict[int, str]:

    """Fetches all GPUs and their names from the TARGET_INFO_GPU table."""

    gpu_map = {}
    try:
        q = "SELECT id, name FROM TARGET_INFO_GPU"
        df = try_read_df(con, q)
        
        if not df.empty:
            for _, row in df.iterrows():
                gpu_id = int(row['id'])
                gpu_name = str(row['name'])
                gpu_map[gpu_id] = gpu_name
                
    except Exception as e:
        print(f"  [Error] Failed to read GPU Info Table: {e}")
        
    return gpu_map

# ==============================================================================
# 2. Schemas & Finders (Runtime, NVTX, Kernel, etc.)
# ==============================================================================

@dataclass
class GpuDataset:
    rank: int
    filename: str
    nvtx_df: pd.DataFrame      # CPU-side NVTX markers (ranges)
    nccl_df: pd.DataFrame      # NCCL communication kernels
    memcpy_df: pd.DataFrame    # H2D, D2H memory copies
    true_gpu_df: pd.DataFrame  # Real GPU kernel span time mapped to NVTX
    gpu_info: Dict[int, str]

def get_rank_from_filename(filename: str, index: int) -> int:

    """
    Find patterns like 'rank0', 'rank1' from the filename.
    If not found, use the list order (index) as the rank.
    """
    match = re.search(r"rank_?(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return index

def load_single_gpu(filepath: str, rank: int, steps: List[int]) -> Optional[GpuDataset]:

    """
    Load detailed analysis data from a single SQLite file and return it as a GpuDataset object.
    """

    if not os.path.exists(filepath):
        print(f"[Skip] File not found: {filepath}")
        return None

    print(f"Loading Rank {rank}: {filepath} ...")
    
    con = sqlite3.connect(filepath)
    try:
        gpu_info = get_all_gpus(con)

        nvtx_df, nccl_df, memcpy_df, true_gpu_df = process_full_analysis(con, steps)
        
        return GpuDataset(
            rank=rank,
            filename=filepath,
            nvtx_df=nvtx_df,
            nccl_df=nccl_df,
            memcpy_df=memcpy_df,
            true_gpu_df=true_gpu_df,
            gpu_info=gpu_info
        )

    except Exception as e:
        print(f"  [Error] Failed to load {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        con.close()

def load_all_gpus(file_list: List[str], steps_to_analyze: List[int]) -> Dict[int, GpuDataset]:

    """
    Given a list of files, load data for all GPUs.
    """
    
    gpu_data_map = {}
    
    file_list.sort()

    for idx, filepath in enumerate(file_list):
        rank = get_rank_from_filename(filepath, idx)
        dataset = load_single_gpu(filepath, rank, steps_to_analyze)
        
        if dataset:
            gpu_data_map[rank] = dataset
            
    print(f"\n[Done] Loaded data for {len(gpu_data_map)} GPUs.")

    return gpu_data_map

def get_all_step_intervals(gpu_data_map: Dict[int, 'GpuDataset']) -> Dict[int, pd.DataFrame]:

    """
    Iterate over all loaded GPU datasets (GpuDataset),
    and for each GPU, extract the start/end timestamps of training steps
    recorded in NVTX markers (Step/Batch/Iteration).
    """
    all_steps_map = {}
    print("--- Extracting Step Intervals from NVTX ---")

    for rank, dataset in gpu_data_map.items():
        
        step_df = build_step_df_from_nvtx(dataset.nvtx_df)
        
        if not step_df.empty:
            all_steps_map[rank] = step_df
        else:
            print(f"[Warning] Rank {rank}: No step markers (Batch/Step/Iter) found in NVTX.")

    return all_steps_map

def get_global_earliest_steps(all_steps_map: Dict[int, pd.DataFrame],
                              gpu_data_map: Dict[int, GpuDataset]) -> pd.DataFrame:

    """
    Compare step timestamps across all ranks and compute, for each step:
    - Global Start (earliest start among all ranks)
    - Global End   (latest end among all ranks)
    - Fastest Rank (rank that started earliest)
    - Slowest Rank (rank that ended latest)
    """
    
    combined_list = []

    for rank, df in all_steps_map.items():
        temp_df = df.copy()
        temp_df["rank"] = rank
        combined_list.append(temp_df)
    
    if not combined_list:
        return pd.DataFrame()
        
    big_df = pd.concat(combined_list, ignore_index=True)
    
    grouped = big_df.groupby("step")
    
    stats_df = grouped.agg(
        earliest_start=("start", "min"),
        latest_end=("end", "max")
    )
    
    min_idx_series = grouped["start"].idxmin()
    max_idx_series = grouped["end"].idxmax()
    
    fastest_ranks = big_df.loc[min_idx_series, ["step", "rank"]].rename(columns={"rank": "fastest_rank"})
    slowest_ranks = big_df.loc[max_idx_series, ["step", "rank"]].rename(columns={"rank": "slowest_rank"})
    
    final_df = stats_df.merge(fastest_ranks, on="step", how="left")
    final_df = final_df.merge(slowest_ranks, on="step", how="left")
    
    gpu_end_map   = {}
    gpu_start_map = {}

    for rank, dataset in gpu_data_map.items():
        if dataset.true_gpu_df.empty:
            continue

        gpu_batch_rows = dataset.true_gpu_df[
            dataset.true_gpu_df["name"].str.contains(r"\[GPU\] Batch", regex=True)
        ]

        for _, row in gpu_batch_rows.iterrows():
            step  = int(row["cpu_step"])
            end   = int(row["end"])
            start = int(row["start"])

            if step not in gpu_end_map:
                gpu_end_map[step] = end
            else:
                gpu_end_map[step] = max(gpu_end_map[step], end)

            if step not in gpu_start_map:
                gpu_start_map[step] = start
            else:
                gpu_start_map[step] = min(gpu_start_map[step], start)

    final_df["gpu_latest_end"]     = final_df["step"].map(gpu_end_map)
    final_df["gpu_earliest_start"] = final_df["step"].map(gpu_start_map)

    return final_df.sort_values("step")


# --- Runtime Schema (CPU API calls) ---
@dataclass
class RuntimeSchema:
    table: str
    cbid_col: str
    start_col: str
    end_col: str
    corr_id_col: str

def find_runtime_schema(con: sqlite3.Connection) -> Optional[RuntimeSchema]:
    tables = list_tables(con)
    candidates = [t for t in tables if t == "CUPTI_ACTIVITY_KIND_RUNTIME"]
    if not candidates:
        candidates = [t for t in tables if "RUNTIME" in t.upper() and "CUPTI" in t.upper()]
    if not candidates: return None

    for t in candidates:
        cols = set(table_columns(con, t))
        cid = next((c for c in ["correlationId", "correlation_id"] if c in cols), None)
        sc = next((c for c in ["start", "startNs", "timestamp_start"] if c in cols), None)
        ec = next((c for c in ["end", "endNs", "timestamp_end"] if c in cols), None)
        if cid and sc and ec:
            return RuntimeSchema(table=t, cbid_col="cbid", start_col=sc, end_col=ec, corr_id_col=cid)
    return None

# --- Kernel Schema (GPU execution) ---
@dataclass
class KernelSchema:
    table: str
    start_col: str
    end_col: str
    name_id_col: str
    corr_id_col: str = "correlationId"

def find_kernel_schema(con: sqlite3.Connection) -> Optional[KernelSchema]:
    tables = list_tables(con)
    kernel_tables = [t for t in tables if t.upper() == "CUPTI_ACTIVITY_KIND_KERNEL"]
    if not kernel_tables:
        kernel_tables = [t for t in tables if "kernel" in t.lower() and "cupti" in t.lower()]
    if not kernel_tables: return None

    possible_start   = ["start", "startNs", "timestamp_start", "start_time"]
    possible_end     = ["end", "endNs", "timestamp_end", "end_time"]
    possible_name_id = ["shortName", "shortNameId", "nameId", "demangledName"]
    possible_corr    = ["correlationId", "correlation_id"]

    for t in kernel_tables:
        cols = set(table_columns(con, t))
        sc  = next((c for c in possible_start   if c in cols), None)
        ec  = next((c for c in possible_end     if c in cols), None)
        nid = next((c for c in possible_name_id if c in cols), None)
        cid = next((c for c in possible_corr    if c in cols), None)
        
        if sc and ec and nid:
            return KernelSchema(table=t, start_col=sc, end_col=ec, name_id_col=nid, corr_id_col=cid or "correlationId")
    return None

# --- NVTX Schema ---
@dataclass
class NvtxSchema:
    table: str
    name_col: str
    start_col: str
    end_col: str

def find_nvtx_schema(con: sqlite3.Connection) -> Optional[NvtxSchema]:
    """Locate the correct NVTX events table with column validation."""
    tables = list_tables(con)
    priority_candidates = ["NVTX_EVENTS", "NVTX_PUSHPOP_RANGES"]
    
    possible_start = ["start", "startNs", "timestamp_start"]
    possible_end   = ["end", "endNs", "timestamp_end"]
    possible_name  = ["text", "message", "name"]

    for t_name in priority_candidates:
        actual_name = next((t for t in tables if t.upper() == t_name), None)
        if actual_name:
            cols = set(table_columns(con, actual_name))
            sc = next((c for c in possible_start if c in cols), None)
            ec = next((c for c in possible_end   if c in cols), None)
            nc = next((c for c in possible_name  if c in cols), None)
            if sc and ec and nc:
                return NvtxSchema(actual_name, nc, sc, ec)

    fallback = [t for t in tables if "NVTX" in t.upper()]
    for t in fallback:
        cols = set(table_columns(con, t))
        sc = next((c for c in possible_start if c in cols), None)
        ec = next((c for c in possible_end   if c in cols), None)
        nc = next((c for c in possible_name  if c in cols), None)
        if sc and ec and nc:
            return NvtxSchema(t, nc, sc, ec)
            
    return None

# --- MEMCPY Schema ----
@dataclass
class MemcpySchema:
    table: str
    start_col: str
    end_col: str
    kind_col: Optional[str] = None
    bytes_col: Optional[str] = None

def find_memcpy_schema(con: sqlite3.Connection) -> Optional[MemcpySchema]:

    tables     = list_tables(con)
    candidates = [t for t in tables if "memcpy" in t.lower()]

    if not candidates: return None

    possible_start = ["start", "startNs", "timestamp_start"]
    possible_end   = ["end", "endNs", "timestamp_end"]
    possible_kind  = ["copyKind", "kind", "memcpyKind"]
    possible_bytes = ["bytes", "byteCount", "size"]

    for t in candidates:
        cols = set(table_columns(con, t))
        sc = next((c for c in possible_start if c in cols), None)
        ec = next((c for c in possible_end   if c in cols), None)
        if not sc or not ec: continue

        kc = next((c for c in possible_kind  if c in cols), None)
        bc = next((c for c in possible_bytes if c in cols), None)
    
        return MemcpySchema(
            table=t,
            start_col=sc,
            end_col=ec,
            kind_col=kc,
            bytes_col=bc
        )
        
    return None

# --- String IDs Schema ---
@dataclass
class StringIdsSchema:
    table: str
    id_col: str
    value_col: str

def find_stringids_schema(con: sqlite3.Connection) -> Optional[StringIdsSchema]:
    tables     = list_tables(con)
    candidates = [t for t in tables if "string" in t.lower() and "id" in t.lower()]

    for t in candidates:
        cols = set(table_columns(con, t))
        if "id" in cols and "value" in cols:
            return StringIdsSchema(t, "id", "value")
    return None

# ==============================================================================
# 3. Data Loading Functions
# ==============================================================================
def load_nvtx_events(con: sqlite3.Connection, 
                     sch: NvtxSchema) -> pd.DataFrame:
    """Loads all NVTX events."""
    q = f"""
    SELECT
        {sch.name_col} AS name,
        {sch.start_col} AS start,
        {sch.end_col} AS end
    FROM {sch.table}
    WHERE {sch.end_col} > {sch.start_col}
    """
    df = try_read_df(con, q)
    df = df.dropna(subset=["name"])
    df["dur_ns"] = df["end"] - df["start"]
    return df

def load_nccl_kernels(con: sqlite3.Connection,
                      k_sch: KernelSchema,
                      s_sch: StringIdsSchema,
                      like_pattern: str = "%nccl%") -> pd.DataFrame:
    """
    Loads kernel events that match a specific pattern (e.g., '%nccl%').
    Joins the Kernel table with the StringIds table to get human-readable names.
    """
    q = f"""
    SELECT
        s.{s_sch.value_col} AS name,
        k.{k_sch.start_col} AS start,
        k.{k_sch.end_col} AS end
    FROM {k_sch.table} k
    JOIN {s_sch.table} s
      ON k.{k_sch.name_id_col} = s.{s_sch.id_col}
    WHERE k.{k_sch.end_col} > k.{k_sch.start_col}
      AND s.{s_sch.value_col} LIKE '{like_pattern}'
    """

    df = try_read_df(con, q)
    df["dur_ns"] = df["end"] - df["start"]
    return df

def load_memcpy_events(con: sqlite3.Connection,
                       sch: MemcpySchema,
                       want_kinds: Optional[List[int]] = None,
                       name: str = "gpu_memcpy_h2d") -> pd.DataFrame:

    q = f"""
    SELECT
        {sch.start_col} AS start,
        {sch.end_col}   AS end
    FROM {sch.table}
    WHERE {sch.end_col} > {sch.start_col}
    """
    
    df = try_read_df(con, q)
    if df.empty:
        return pd.DataFrame(columns=["name", "start", "end", "dur_ns"])

    if want_kinds is not None and sch.kind_col and "kind" in df.columns:
        df = df[df["kind"].isin(want_kinds)].copy()

    df["name"]   = name
    df["dur_ns"] = df["end"] - df["start"]

    return df[["name", "start", "end", "dur_ns"]].copy()

# ==============================================================================
# 4. [CORE LOGIC] True GPU Span Calculation
# ==============================================================================
def load_runtime_events_in_nvtx(
    con: sqlite3.Connection, 
    nvtx_df: pd.DataFrame, 
    r_sch: RuntimeSchema
    ) -> pd.DataFrame:
    
    """Loads Runtime API calls within NVTX ranges (CPU launch time)."""
    if nvtx_df.empty: 
        return pd.DataFrame()
    min_start = nvtx_df["start"].min()
    max_end   = nvtx_df["end"].max()

    q = f"""
    SELECT 
        {r_sch.corr_id_col} AS correlationId, 
        {r_sch.start_col} AS start, 
        {r_sch.end_col} AS end
    FROM {r_sch.table}
    WHERE {r_sch.start_col} >= {min_start} AND {r_sch.end_col} <= {max_end}
    """

    df = try_read_df(con, q)

    return df.sort_values("start") if not df.empty else df


def load_all_kernels_for_span(
    con: sqlite3.Connection, 
    k_sch: KernelSchema,
    s_sch: StringIdsSchema = None 
    ) -> pd.DataFrame:
    
    """Loads ALL kernels to calculate spans (needs correlationId). GPU kernel time."""
    
    if s_sch:
        q = f"""
        SELECT 
            k.{k_sch.corr_id_col} AS correlationId, 
            k.{k_sch.start_col}   AS k_start, 
            k.{k_sch.end_col}     AS k_end,
            s.{s_sch.value_col}   AS name
        FROM {k_sch.table} k
        LEFT JOIN {s_sch.table} s
            ON k.{k_sch.name_id_col} = s.{s_sch.id_col}
        WHERE k.{k_sch.end_col} > k.{k_sch.start_col}
        """
    else:
        q = f"""
        SELECT 
            {k_sch.corr_id_col} AS correlationId, 
            {k_sch.start_col}   AS k_start, 
            {k_sch.end_col}     AS k_end
        FROM {k_sch.table} 
        WHERE {k_sch.end_col} > {k_sch.start_col}
        """
    
    return try_read_df(con, q)


def map_nvtx_to_runtime(
    nvtx_df: pd.DataFrame, 
    runtime_df: pd.DataFrame, 
    ):

    """Maps NVTX ranges to Runtime API correlation IDs."""
    if nvtx_df.empty or runtime_df.empty: 
        return pd.DataFrame()
    
    nvtx_sorted = nvtx_df.sort_values("start").reset_index(drop=True)
    rt_sorted   = runtime_df.sort_values("start").reset_index(drop=True)
    
    rt_starts = rt_sorted["start"].values
    rt_corrs  = rt_sorted["correlationId"].values
    
    mapped_rows = []

    for _, row in nvtx_sorted.iterrows():
        nvtx_start = row["start"]
        nvtx_end   = row["end"]
        nvtx_name  = row["name"]

        mask = (rt_starts >= nvtx_start) & (rt_starts < nvtx_end)
        matched_corrs = rt_corrs[mask]
        
        for cid in matched_corrs:
            mapped_rows.append({
                "nvtx_name":     nvtx_name,
                "nvtx_start":    nvtx_start,
                "nvtx_end":      nvtx_end,
                "correlationId": cid
            })
            
    return pd.DataFrame(mapped_rows)


def compute_true_gpu_spans(
    mapping_df: pd.DataFrame, 
    kernel_df: pd.DataFrame,
    step_df: pd.DataFrame = None
    ) -> pd.DataFrame:
    
    """Calculates [Min Kernel Start, Max Kernel End] for each NVTX."""

    if mapping_df.empty or kernel_df.empty: 
        return pd.DataFrame()

    print(f"Merging {len(mapping_df)} CPU-Mappings with {len(kernel_df)} GPU-Kernels...")

    merged = pd.merge(mapping_df, kernel_df, on="correlationId", how="inner")

    nccl_in_backward_mask = (
        merged["nvtx_name"].str.contains("backward", case=False) &
        merged["name"].str.contains("nccl", case=False)
    )
    merged = merged[~nccl_in_backward_mask]

    if merged.empty: 
        return pd.DataFrame()

    result = merged.groupby(["nvtx_name", "nvtx_start", "nvtx_end"], as_index=False).agg(
        true_start=("k_start", "min"),
        true_end=("k_end", "max")
    )
    
    result["dur_ns"] = result["true_end"] - result["true_start"]
    result = result.rename(columns={"true_start": "start", "true_end": "end"})
    result["name"] = result["nvtx_name"].apply(lambda x: f"[GPU] {x}")

    if step_df is not None and not step_df.empty:
        def assign_cpu_step(nvtx_start):
            match = step_df[
                (step_df["start"] <= nvtx_start) &
                (step_df["end"]   >= nvtx_start)
            ]
            if not match.empty:
                return int(match.iloc[0]["step"])
            return -1

        result["cpu_step"] = result["nvtx_start"].apply(assign_cpu_step)

        gpu_batch_rows = result[
            result["nvtx_name"].str.contains(r"Batch", regex=True)
        ]

        gpu_step_df = pd.DataFrame({
            "step":  gpu_batch_rows["cpu_step"].values,
            "start": gpu_batch_rows["start"].values,
            "end":   gpu_batch_rows["end"].values
        }).sort_values("step").reset_index(drop=True)

        def assign_gpu_step(gpu_start):
            for _, row in gpu_step_df.iterrows():
                if row["start"] <= gpu_start < row["end"]:
                    return int(row["step"])
            return -1

        result["gpu_step"] = result["start"].apply(assign_gpu_step)
        result = result.sort_values(["cpu_step", "start"])

        return result[["name", "start", "end", "dur_ns", "cpu_step", "gpu_step"]]

    return result[["name", "start", "end", "dur_ns"]]

# ==============================================================================
# 5. Clock Offset Calculation
# ==============================================================================

def calculate_clock_offsets(
    gpu_data_map: Dict[int, GpuDataset],
    n_kernels: int = 20
) -> Dict[int, int]:
    """
    Calculates clock offsets using the median end time of the first n_kernels
    AllReduce kernels for each rank.

    Principle:
      AllReduce completes physically simultaneously across all ranks.
      -> Difference between median AllReduce end times = clock offset.
    """

    print("\n=== Starting Clock Offset Calculation ===")

    def get_allreduce_df(nccl_df: pd.DataFrame) -> pd.DataFrame:
        if nccl_df.empty:
            return pd.DataFrame()
        return nccl_df[nccl_df["name"].str.contains("AllReduce", case=False)]

    median_allreduce_ends = {}

    for rank, dataset in gpu_data_map.items():

        allreduce_df = get_allreduce_df(dataset.nccl_df)

        if allreduce_df.empty:
            median_allreduce_ends[rank] = 0
            continue

        first_n_end_times         = allreduce_df["end"].nsmallest(n_kernels)
        median_allreduce_end_time = int(first_n_end_times.median())
        median_allreduce_ends[rank] = median_allreduce_end_time

    base_time = median_allreduce_ends[0]

    offsets = {}

    for rank, median_allreduce_end_time in median_allreduce_ends.items():

        offset = base_time - median_allreduce_end_time
        offsets[rank] = offset

        print(f"\n  Rank {rank} calculation:")
        print(f"    median_allreduce_ends[0]      = {base_time} ns  (Rank 0 reference)")
        print(f"    median_allreduce_ends[{rank}]      = {median_allreduce_end_time} ns  (Rank {rank} median)")
        print(f"    offset = {base_time} - {median_allreduce_end_time}")
        print(f"           = {offset} ns")
        print(f"           = {offset/1e6:+.3f} ms")

    print("\n=== Final Offsets ===")
    for rank, offset in offsets.items():
        print(f"  Rank {rank}: {offset/1e6:+.3f} ms")

    return offsets

# ==============================================================================
# 6. Wait Time Analysis
# ==============================================================================

def compute_pure_wait_time(
    nvtx_df: pd.DataFrame,
    nccl_df: pd.DataFrame,
) -> pd.DataFrame:

    nvtx_nccl_df = nvtx_df[
        nvtx_df["name"] == "NCCL_AllReduce"
    ].sort_values("start").reset_index(drop=True)

    nccl_allreduce_df = nccl_df[
        nccl_df["name"].str.contains("AllReduce", case=False)
    ].sort_values("start").reset_index(drop=True)

    n_nvtx   = len(nvtx_nccl_df)
    n_kernel = len(nccl_allreduce_df)
    print(f"NVTX ranges:       {n_nvtx}")
    print(f"AllReduce kernels: {n_kernel}")

    if n_nvtx != n_kernel:
        print(f"[WARNING] Count mismatch: NVTX={n_nvtx}, kernels={n_kernel}")

    results   = []
    prev_t4   = 0
    prev_step = -1

    for i in range(min(n_nvtx, n_kernel)):
        t1        = nvtx_nccl_df.iloc[i]["start"]
        t3        = nccl_allreduce_df.iloc[i]["start"]
        t4_kernel = nccl_allreduce_df.iloc[i]["end"]
        cur_step  = nvtx_nccl_df.iloc[i].get("step", -1)

        if cur_step != prev_step:
            prev_t4   = 0
            prev_step = cur_step

        earliest_possible = max(t1, prev_t4)
        cause_a_ns        = max(0, prev_t4 - t1)
        pure_wait_ns      = max(0, t3 - earliest_possible)

        results.append({
            "bucket_idx":   i,
            "step":         cur_step,
            "t1_ns":        t1,
            "t1_ms":        t1        / 1e6,
            "t3_ms":        t3        / 1e6,
            "t4_ms":        t4_kernel / 1e6,
            "prev_t4_ms":   prev_t4   / 1e6,
            "cause_a_ms":   cause_a_ns   / 1e6,
            "pure_wait_ms": pure_wait_ns / 1e6,
            "comm_ms":      (t4_kernel - t3) / 1e6,
            "total_ms":     (t4_kernel - t1) / 1e6,
        })

        prev_t4 = t4_kernel

    return pd.DataFrame(results)



def load_all_gpu_compute_wait_time(
    gpu_data_map: Dict[int, GpuDataset]
) -> pd.DataFrame:
    """
    Loads wait time data for all ranks, applies clock-offset correction,
    and computes pure_wait_ms per bucket per step.

    pure_wait_ms = time a rank spent waiting for the slowest rank
                   (i.e. max corrected t1 across ranks - this rank's corrected t1)
    - Low value  -> this rank arrived late -> straggler
    - High value -> this rank arrived early and waited -> victim
    """

    offsets = calculate_clock_offsets(gpu_data_map)

    all_results = []

    for rank, gpu_data in gpu_data_map.items():

        # Extract step ranges from GPU batch rows
        step_ranges = gpu_data.true_gpu_df[
            gpu_data.true_gpu_df["name"].str.contains(
                r"\[GPU\].*Batch", regex=True
            )
        ][["cpu_step", "start", "end"]].drop_duplicates("cpu_step")

        # Assign step to each NVTX event
        nvtx_with_step = gpu_data.nvtx_df.copy()

        def assign_step(row_start):
            match = step_ranges[
                (step_ranges["start"] <= row_start) &
                (step_ranges["end"]   >= row_start)
            ]
            return int(match.iloc[0]["cpu_step"]) if not match.empty else -1

        nvtx_with_step["step"] = nvtx_with_step["start"].apply(assign_step)

        rank_wait_df = compute_pure_wait_time(
            nvtx_df = nvtx_with_step,
            nccl_df = gpu_data.nccl_df,
        )

        if rank_wait_df.empty:
            print(f"[WARNING] Rank {rank}: No wait time data found")
            continue

        rank_wait_df["rank"] = rank

        # Apply clock-offset correction to t1
        rank_wait_df["t1_corrected_ns"] = (
            rank_wait_df["t1_ns"] + offsets.get(rank, 0)
        )

        all_results.append(rank_wait_df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Assign bucket-within-step index (same index = same physical AllReduce)
    combined = combined.sort_values(["step", "rank", "bucket_idx"])
    combined["bucket_within_step"] = (
        combined.groupby(["step", "rank"]).cumcount()
    )

    # Compute pure_wait_ms using clock-corrected t1
    # max(t1_corrected) across ranks = the last rank to arrive
    max_t1_corrected = combined.groupby(
        ["step", "bucket_within_step"]
    )["t1_corrected_ns"].transform("max")

    combined["pure_wait_ms"] = (
        (max_t1_corrected - combined["t1_corrected_ns"])
        .clip(lower=0) / 1e6
    )

    return combined


# ==============================================================================
# 7. Plotting
# ==============================================================================

def plot_wait_time_per_step(
    wait_df: pd.DataFrame,
    output_path: str = "level1_wait_time.png"
) -> None:
    """
    Level 1 visualization: wait time per rank across all steps.

    Interpretation:
      Low  wait -> rank arrived late -> straggler (bottleneck cause)
      High wait -> rank arrived early and waited -> victim
    """
    df = wait_df[wait_df["step"] != -1].copy()
    if df.empty:
        print("[WARNING] No data available")
        return

    # Sum pure_wait per step per rank
    grouped = (df.groupby(["step", "rank"])["pure_wait_ms"]
                 .sum()
                 .reset_index())

    target_steps = sorted(grouped["step"].unique())
    ranks        = sorted(grouped["rank"].unique())
    n_ranks      = len(ranks)
    x            = np.arange(len(target_steps))
    bar_width    = 0.7 / n_ranks
    colors       = ["#D85A30", "#BA7517", "#1D9E75", "#378ADD"]

    fig, ax = plt.subplots(figsize=(max(10, len(target_steps) * 0.8), 5))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#F5F4EF")

    for ri, rank in enumerate(ranks):
        rank_data = grouped[grouped["rank"] == rank]
        waits = [
            rank_data[rank_data["step"] == s]["pure_wait_ms"].values[0]
            if not rank_data[rank_data["step"] == s].empty else 0
            for s in target_steps
        ]
        offset = (ri - n_ranks / 2 + 0.5) * bar_width
        ax.bar(x + offset, waits, bar_width * 0.85,
               label=f"Rank {rank}",
               color=colors[ri % len(colors)],
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in target_steps],
                       fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Wait time (ms)", fontsize=10)
    ax.set_title(
        "Level 1 — Wait time per rank (all steps)\n"
        "Low = slow rank (straggler)   High = waiting rank (victim)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.close()


def draw_bar_with_cpu_boundary(
    ax,
    rel_start_ms: float,
    dur_ms: float,
    cpu_step: int,
    gpu_step: int,
    cpu_boundaries_ms: List[float],
    y_pos: float,
    lane_height: float,
    rank_color: str,
    alpha: float = 0.9
):
    bar_start = rel_start_ms
    bar_end   = rel_start_ms + dur_ms

    if cpu_step == gpu_step or gpu_step == -1:
        ax.broken_barh([(bar_start, dur_ms)], (y_pos, lane_height),
                       facecolors=rank_color, alpha=alpha,
                       linewidth=0.5, edgecolor='black')
        return

    crossed_boundary = None
    for boundary_ms in cpu_boundaries_ms:
        if bar_start < boundary_ms < bar_end:
            crossed_boundary = boundary_ms
            break

    if crossed_boundary is None:
        ax.broken_barh([(bar_start, dur_ms)], (y_pos, lane_height),
                       facecolors=rank_color, alpha=alpha,
                       linewidth=0.5, edgecolor='black')
        return

    left_dur = crossed_boundary - bar_start
    ax.broken_barh([(bar_start, left_dur)], (y_pos, lane_height),
                   facecolors=rank_color, alpha=alpha,
                   linewidth=0.5, edgecolor='black')

    right_dur = bar_end - crossed_boundary
    ax.broken_barh([(crossed_boundary, right_dur)], (y_pos, lane_height),
                   facecolors=rank_color, alpha=alpha * 0.5,
                   hatch='////', linewidth=0.5, edgecolor='red')

    ax.vlines(x=crossed_boundary, ymin=y_pos, ymax=y_pos + lane_height,
              colors='red', linewidth=2, linestyle='--', alpha=0.8)


def plot_timeline_custom_axis(
    gpu_data_map, 
    final_df, 
    all_steps_map,
    steps_to_plot, 
    out_png="timeline.png", 
    show=True,
    color_by="rank",
    offsets: Optional[Dict] = None
    ):

    target_stats = final_df[final_df["step"].isin(steps_to_plot)]
    if target_stats.empty:
        print(f"[Error] No data for steps {steps_to_plot}")
        return

    global_start  = int(target_stats["earliest_start"].min())
    global_end    = int(target_stats["gpu_latest_end"].max())
    global_end_ms = (global_end - global_start) / 1e6

    print(f"  -> global_start (CPU): {global_start}")
    print(f"  -> global_end   (GPU): {global_end}")
    print(f"  -> Total timeline:     {global_end_ms:.2f} ms")

    cpu_boundaries_ms = []
    for _, srow in target_stats.iterrows():
        boundary = (srow["latest_end"] - global_start) / 1e6
        cpu_boundaries_ms.append(boundary)

    print(f"  -> CPU step boundaries (ms): {cpu_boundaries_ms}")

    base_names   = ["data_wait", "h2d", "gpu_compute", "NCCL"]
    fixed_nvtx   = ["zero_grad","forward", "loss", "backward", "opt_step"]
    full_y_names = list(dict.fromkeys(base_names + fixed_nvtx))
    y_map        = {name: i for i, name in enumerate(full_y_names)}

    fig_height = len(full_y_names) * 1.5 + 2
    plt.figure(figsize=(24, fig_height))

    ax = plt.gca()

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    step_color_map = {}
    for i, step in enumerate(sorted(steps_to_plot)):
        step_color_map[step] = colors[i % len(colors)]

    def get_color(rank, step_num):
        if color_by == "step":
            return step_color_map.get(step_num, 'tab:gray')
        else:
            return colors[rank % len(colors)]

    lane_height  = 0.8 / len(gpu_data_map)
    sorted_ranks = sorted(gpu_data_map.keys())

    for rank in sorted_ranks:

        dataset    = gpu_data_map[rank]
        rank_color = colors[rank % len(colors)]

        if rank not in all_steps_map:
            continue

        step_df_sel = all_steps_map[rank][all_steps_map[rank]["step"].isin(steps_to_plot)]
        if step_df_sel.empty:
            continue

        df_nvtx   = filter_by_step_ranges(dataset.nvtx_df,     step_df_sel, global_start, use_cpu_step=True, rank=rank, offsets=offsets)
        df_nccl   = filter_by_step_ranges(dataset.nccl_df,     step_df_sel, global_start, rank=rank, offsets=offsets) if not dataset.nccl_df.empty   else pd.DataFrame()
        df_memcpy = filter_by_step_ranges(dataset.memcpy_df,   step_df_sel, global_start, rank=rank, offsets=offsets) if not dataset.memcpy_df.empty else pd.DataFrame()
        df_compute= filter_by_step_ranges(dataset.true_gpu_df, step_df_sel, global_start, rank=rank, offsets=offsets) if not dataset.true_gpu_df.empty else pd.DataFrame()

        print(df_compute)

        rank_offset = rank * lane_height

        # [A] gpu_compute row
        if not df_compute.empty and "gpu_compute" in y_map:

            y_base = y_map["gpu_compute"]

            for _, row in df_compute.iterrows():

                cpu_step = int(row.get("cpu_step", row["step"]))
                gpu_step = int(row.get("gpu_step", row["step"]))
                step_num = gpu_step if gpu_step != -1 else cpu_step

                bar_color = get_color(rank, step_num) 
                
                draw_bar_with_cpu_boundary(
                    ax=ax,
                    rel_start_ms=row["rel_start_ms"],
                    dur_ms=row["dur_ms"],
                    cpu_step=cpu_step,
                    gpu_step=gpu_step,
                    cpu_boundaries_ms=cpu_boundaries_ms,
                    y_pos=y_base + rank_offset,
                    lane_height=lane_height,
                    rank_color=bar_color,
                    alpha=0.9
                )
                if row["dur_ms"] > 10:
                    ax.text(
                        x=row["rel_start_ms"] + row["dur_ms"] / 2,
                        y=y_base + rank_offset + lane_height / 2,
                        s=f"S{step_num}",
                        ha='center', va='center',
                        fontsize=7, fontweight='bold',
                        color='white', clip_on=True
                    )
                if color_by == "step":
                    ax.text(
                        x=row["rel_start_ms"] + 1,
                        y=y_base + rank_offset + lane_height / 2,
                        s=f"R{rank}",
                        ha='left', va='center',
                        fontsize=6, fontweight='bold',
                        color='white', clip_on=True
                    )

        # [B] h2d row
        if "h2d" in y_map:
            y_base    = y_map["h2d"]
            current_y = y_base + rank_offset
            if not df_memcpy.empty:
                for _, row in df_memcpy.iterrows():

                    cpu_step = int(row.get("cpu_step", row["step"]))
                    gpu_step = int(row.get("gpu_step", row["step"]))
                    step_num = gpu_step if gpu_step != -1 else cpu_step

                    bar_color = get_color(rank, step_num)

                    draw_bar_with_cpu_boundary(
                        ax=ax,
                        rel_start_ms=row["rel_start_ms"],
                        dur_ms=row["dur_ms"],
                        cpu_step=cpu_step,
                        gpu_step=gpu_step,
                        cpu_boundaries_ms=cpu_boundaries_ms,
                        y_pos=current_y,
                        lane_height=lane_height,
                        rank_color=bar_color,
                        alpha=1.0
                    )
                    if row["dur_ms"] > 10:
                        ax.text(
                            x=row["rel_start_ms"] + row["dur_ms"] / 2,
                            y=current_y + lane_height / 2,
                            s=f"S{step_num}",
                            ha='center', va='center',
                            fontsize=7, fontweight='bold',
                            color='white', clip_on=True
                        )
                    if color_by == "step":
                        ax.text(
                            x=row["rel_start_ms"] + 1,
                            y=current_y + lane_height / 2,
                            s=f"R{rank}",
                            ha='left', va='center',
                            fontsize=6, fontweight='bold',
                            color='white', clip_on=True
                        )

        # [C] NCCL row
        if not df_nccl.empty and "NCCL" in y_map:
            y_base    = y_map["NCCL"]
            current_y = y_base + rank_offset
            for _, row in df_nccl.iterrows():

                cpu_step = int(row.get("cpu_step", row["step"]))
                gpu_step = int(row.get("gpu_step", row["step"]))
                step_num = gpu_step if gpu_step != -1 else cpu_step

                bar_color = get_color(rank, step_num)

                draw_bar_with_cpu_boundary(
                    ax=ax,
                    rel_start_ms=row["rel_start_ms"],
                    dur_ms=row["dur_ms"],
                    cpu_step=cpu_step,
                    gpu_step=gpu_step,
                    cpu_boundaries_ms=cpu_boundaries_ms,
                    y_pos=current_y,
                    lane_height=lane_height,
                    rank_color=bar_color,
                    alpha=0.9
                )
                if row["dur_ms"] > 10:
                    ax.text(
                        x=row["rel_start_ms"] + row["dur_ms"] / 2,
                        y=current_y + lane_height / 2,
                        s=f"S{step_num}",
                        ha='center', va='center',
                        fontsize=7, fontweight='bold',
                        color='white', clip_on=True
                    )
                if color_by == "step":
                    ax.text(
                        x=row["rel_start_ms"] + 1,
                        y=current_y + lane_height / 2,
                        s=f"R{rank}",
                        ha='left', va='center',
                        fontsize=6, fontweight='bold',
                        color='white', clip_on=True
                    )

        # [D] NVTX rows
        target_phases = ["data_wait","h2d", "zero_grad", "Forward", "Backward",
                         "Loss", "nccl_sync", "opt_step","NCCL_AllReduce"]

        if not df_nvtx.empty:
            for target_name in target_phases:

                mask     = df_nvtx["name"].str.contains(target_name, case=False, regex=False)
                cpu_rows = df_nvtx[mask]
                if cpu_rows.empty:
                    continue

                base_y = -1
                for key in y_map:
                    if target_name.lower() in key.lower():
                        base_y = y_map[key]
                        break
                if base_y == -1:
                    continue

                current_y = base_y + rank_offset

                for _, cpu_row in cpu_rows.iterrows():

                    cpu_start = cpu_row["rel_start_ms"]
                    cpu_dur   = cpu_row["dur_ms"]
                    step_num  = int(cpu_row["step"])

                    bar_color = get_color(rank, step_num)

                    ax.broken_barh(
                        [(cpu_start, cpu_dur)],
                        (current_y, lane_height),
                        facecolors=bar_color,
                        alpha=0.3,
                        linewidth=0
                    )
                    if color_by == "step":
                        ax.text(
                            x=cpu_start + 1,
                            y=current_y + lane_height / 2,
                            s=f"R{rank}",
                            ha='left', va='center',
                            fontsize=6, fontweight='bold',
                            color='black', clip_on=True
                        )

                    ax.vlines(
                        x=cpu_start + cpu_dur,
                        ymin=current_y,
                        ymax=current_y + lane_height,
                        colors='black',
                        linestyles='solid',
                        linewidth=1.5,
                        alpha=0.8
                    )

                    ax.text(
                        x=cpu_start + cpu_dur / 2,
                        y=current_y + lane_height / 2,
                        s=f"S{step_num}",
                        ha='center',
                        va='center',
                        fontsize=7,
                        fontweight='bold',
                        color='black',
                        clip_on=True
                    )

                if not df_compute.empty:
                    gpu_mask     = df_compute["name"].str.contains(target_name, case=False, regex=False)
                    relevant_gpu = df_compute[gpu_mask]

                    if not relevant_gpu.empty:
                        for _, gpu_row in relevant_gpu.iterrows():
                            cpu_step = int(gpu_row.get("cpu_step", gpu_row["step"]))
                            gpu_step = int(gpu_row.get("gpu_step", gpu_row["step"]))
                            step_num = gpu_step if gpu_step != -1 else cpu_step

                            bar_color = get_color(rank, step_num)

                            draw_bar_with_cpu_boundary(
                                ax=ax,
                                rel_start_ms=gpu_row["rel_start_ms"],
                                dur_ms=gpu_row["dur_ms"],
                                cpu_step=cpu_step,
                                gpu_step=gpu_step,
                                cpu_boundaries_ms=cpu_boundaries_ms,
                                y_pos=current_y,
                                lane_height=lane_height,
                                rank_color=bar_color,
                                alpha=1.0
                            )

                            ax.text(
                                x=gpu_row["rel_start_ms"] + gpu_row["dur_ms"] / 2,
                                y=current_y + lane_height / 2,
                                s=f"S{step_num}",
                                ha='center',
                                va='center',
                                fontsize=7,
                                fontweight='bold',
                                color='white',
                                clip_on=True
                            )
                            if color_by == "step":
                                ax.text(
                                    x=gpu_row["rel_start_ms"] + 1,
                                    y=current_y + lane_height / 2,
                                    s=f"R{rank}",
                                    ha='left', va='center',
                                    fontsize=6, fontweight='bold',
                                    color='white', clip_on=True
                                )

    # Step boundary lines
    for _, srow in target_stats.iterrows():
        x_cpu_start = (srow["earliest_start"] - global_start) / 1e6
        ax.axvline(x_cpu_start, color='k', ls='--', alpha=0.5)

        x_cpu_end = (srow["latest_end"] - global_start) / 1e6
        ax.axvline(x_cpu_end, color='k', ls='solid', alpha=0.5)

        x_gpu_end = (srow["gpu_latest_end"] - global_start) / 1e6
        ax.axvline(x_gpu_end, color='blue', ls='solid', alpha=0.3, linewidth=1.5)

        ax.text(
            (x_cpu_start + x_cpu_end) / 2,
            len(full_y_names),
            f"Step {int(srow['step'])}",
            ha='center', va='bottom', weight='bold'
        )

    ax.set_xlim(0, global_end_ms)
    ax.set_yticks([i + 0.5 for i in range(len(full_y_names))])
    ax.set_yticklabels(full_y_names, fontsize=11, fontweight='bold')

    for i in range(len(full_y_names) + 1):
        ax.axhline(y=i, color='black', linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Time (ms) relative to Global Step Start", fontsize=12)
    ax.set_ylim(0, len(full_y_names))

    all_gpu_names = []
    for dataset in gpu_data_map.values():
        if dataset.gpu_info:
            all_gpu_names.append(next(iter(dataset.gpu_info.values())))

    gpu_counts    = Counter(all_gpu_names)
    gpu_title_str = ", ".join([f"{name} ({count} GPUs)" for name, count in gpu_counts.items()]) if gpu_counts else "Unknown GPU"
    ax.set_title(f"[{gpu_title_str}] Multi-GPU Detailed Timeline | Steps: {steps_to_plot}", fontsize=14, fontweight='bold')

    ax.grid(True, axis='x', linestyle=':', alpha=0.5)
    ax.grid(False, axis='y')

    legend_elements = []
    for r in sorted_ranks:
        dataset  = gpu_data_map[r]
        gpu_name = next(iter(dataset.gpu_info.values())) if dataset.gpu_info else "Unknown GPU"
        legend_elements.append(Patch(facecolor=colors[r % len(colors)], label=f'Rank {r} ({gpu_name})'))

    legend_elements.append(Patch(facecolor='none', edgecolor='black', label='CPU step boundary (solid line)'))
    legend_elements.append(Patch(facecolor='none', edgecolor='red',   label='CPU/GPU step mismatch segment'))
    legend_elements.append(Patch(facecolor='none', edgecolor='blue',  label='GPU step end (blue line)'))

    ax.legend(handles=legend_elements, loc='upper right', title="GPU Ranks", bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved timeline to {out_png}")

    if show:
        plt.show()

# ==============================================================================
# 8. Full Analysis Pipeline
# ==============================================================================
def build_step_df_from_nvtx(nvtx_df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Robustly extract step ranges from NVTX data.
    Supported patterns: Batch_10, step 10, Iter-10, Iteration:10, Global Step 10
    Deduplication: for duplicate step numbers, take the outermost range.
    """
    if nvtx_df.empty:
        return pd.DataFrame()

    pattern = r"(?i)(?:Batch|step|Iter|Iteration|Global Step)[_\s:\-]*(\d+)"

    df = nvtx_df.copy()
    extracted_steps = df["name"].str.extract(pattern, expand=False)
    
    df["step"] = pd.to_numeric(extracted_steps, errors='coerce')
    
    valid_steps = df.dropna(subset=["step"])
    
    if valid_steps.empty:
        return pd.DataFrame(columns=["step", "start", "end"])

    step_df = valid_steps.groupby("step").agg(
        start=("start", "min"),
        end=("end", "max")
    ).reset_index()

    step_df["step"] = step_df["step"].astype(int)
    step_df = step_df.sort_values("start")

    return step_df[["step", "start", "end"]]


def filter_by_step_ranges(
    source_df: pd.DataFrame,
    step_df_sel: pd.DataFrame,
    global_start: int,
    use_cpu_step: bool = False,
    rank: int = 0,
    offsets: Optional[Dict] = None
    ) -> pd.DataFrame:
    """
    Filter events in source_df by step range.

    use_cpu_step=True  -> NVTX (CPU events): filter by cpu_step column
    use_cpu_step=False -> GPU kernels:        filter by gpu_step column
    Neither available  -> fallback to time range filtering

    rel_start_ms is always relative to global_start (clock-offset corrected per rank).
    """

    if offsets is not None and rank in offsets:
        adjusted_global_start = global_start - offsets[rank]
    else:
        adjusted_global_start = global_start

    rows = []

    for _, srow in step_df_sel.iterrows():
        s       = int(srow["step"])
        s_start = int(srow["start"])
        s_end   = int(srow["end"])

        if use_cpu_step and "cpu_step" in source_df.columns:
            d = source_df[source_df["cpu_step"] == s].copy()

        elif "gpu_step" in source_df.columns:
            d = source_df[source_df["gpu_step"] == s].copy()

        else:
            d = source_df[
                (source_df["start"] >= s_start) &
                (source_df["start"] <  s_end)
            ].copy()

        if d.empty:
            continue

        d["step"] = s
        rows.append(d)

    if not rows:
        return pd.DataFrame()

    filtered_df = pd.concat(rows, ignore_index=True)
    filtered_df = filtered_df.drop_duplicates(subset=["name", "start", "end"])

    filtered_df["rel_start_ms"] = (filtered_df["start"] - adjusted_global_start) / 1e6
    filtered_df["dur_ms"]       = (filtered_df["end"]   - filtered_df["start"]) / 1e6

    return filtered_df


def process_full_analysis(con: sqlite3.Connection,
                          steps: List[int]
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    nvtx_sch = find_nvtx_schema(con)
    k_sch    = find_kernel_schema(con)
    r_sch    = find_runtime_schema(con)
    m_sch    = find_memcpy_schema(con)
    s_sch    = find_stringids_schema(con)

    df_nvtx = load_nvtx_events(con, nvtx_sch) if nvtx_sch else pd.DataFrame()

    df_nccl = pd.DataFrame()
    if k_sch and s_sch:
        df_nccl = load_nccl_kernels(con, k_sch, s_sch)

    df_memcpy = pd.DataFrame()
    if m_sch:
        df_memcpy = load_memcpy_events(con, m_sch, want_kinds=[1], name="HtoD")

    df_true_gpu = pd.DataFrame()

    if nvtx_sch and r_sch and k_sch and not df_nvtx.empty:
        try:
            cpu_step_df    = build_step_df_from_nvtx(df_nvtx)
            df_runtime     = load_runtime_events_in_nvtx(con, df_nvtx, r_sch)
            df_kernels_all = load_all_kernels_for_span(con, k_sch, s_sch=s_sch)
            df_mapping     = map_nvtx_to_runtime(df_nvtx, df_runtime)

            df_true_gpu = compute_true_gpu_spans(df_mapping, df_kernels_all, step_df=cpu_step_df)

            if df_true_gpu.empty:
                raise ValueError("true_gpu_df is empty")

            gpu_batch_rows = df_true_gpu[
                df_true_gpu["name"].str.contains(r"\[GPU\] Batch", regex=True)
            ]

            if gpu_batch_rows.empty:
                raise ValueError("No [GPU] Batch_N rows found -> check NVTX markers for Batch_N")

            gpu_step_df = pd.DataFrame({
                "step":  gpu_batch_rows["cpu_step"].values,
                "start": gpu_batch_rows["start"].values,
                "end":   gpu_batch_rows["end"].values
            }).sort_values("step").reset_index(drop=True)

            def assign_cpu_step(event_start):
                for _, row in cpu_step_df.iterrows():
                    if row["start"] <= event_start < row["end"]:
                        return int(row["step"])
                return -1

            def assign_gpu_step(event_start):
                for _, row in gpu_step_df.iterrows():
                    if row["start"] <= event_start < row["end"]:
                        return int(row["step"])
                return -1

            if not df_nvtx.empty:
                df_nvtx["cpu_step"] = df_nvtx["start"].apply(assign_cpu_step)
                df_nvtx["gpu_step"] = df_nvtx["start"].apply(assign_gpu_step)

            if not df_nccl.empty:
                df_nccl["cpu_step"] = df_nccl["start"].apply(assign_cpu_step)
                df_nccl["gpu_step"] = df_nccl["start"].apply(assign_gpu_step)

            if not df_memcpy.empty:
                df_memcpy["cpu_step"] = df_memcpy["start"].apply(assign_cpu_step)
                df_memcpy["gpu_step"] = df_memcpy["start"].apply(assign_gpu_step)

        except Exception as e:
            print(f"  [Warning] Failed: {e}")
            import traceback
            traceback.print_exc()

    return df_nvtx, df_nccl, df_memcpy, df_true_gpu


# ==============================================================================
# 9. Main Execution
# ==============================================================================

if __name__ == "__main__":

    target_steps = []
    sqlite_dir   = "."

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--dir" and i + 1 < len(args):
            sqlite_dir = args[i + 1]
        elif arg.isdigit():
            target_steps.append(int(arg))

    all_files = sorted(glob.glob(os.path.join(sqlite_dir, "*.sqlite")))
    if not all_files:
        print(f"[Error] No .sqlite files found in '{sqlite_dir}'")
        sys.exit(1)

    from collections import defaultdict
    import re

    groups = defaultdict(list)
    for f in all_files:
        basename  = os.path.basename(f)
        group_key = re.sub(r'_?rank\d+', '', basename).replace('.sqlite', '')
        groups[group_key].append(f)

    group_keys = sorted(groups.keys())
    print("\nAvailable experiment groups:")
    for i, key in enumerate(group_keys):
        files = groups[key]
        print(f"  [{i}] {key}  ({len(files)} ranks)")
        for f in sorted(files):
            print(f"       - {os.path.basename(f)}")

    print()
    choice = input("Select group number: ").strip()
    try:
        idx          = int(choice)
        selected_key = group_keys[idx]
    except (ValueError, IndexError):
        print("[Error] Invalid input.")
        sys.exit(1)

    sqlite_files = sorted(groups[selected_key])

    if not target_steps:
        step_input   = input("Enter step numbers to plot (e.g. 2 3 4): ").strip()
        target_steps = [int(s) for s in step_input.split() if s.isdigit()]

    if not target_steps:
        target_steps = [1, 2, 3]

    print(f"\n--- Configuration ---")
    print(f"Selected experiment: {selected_key}")
    print(f"Target Steps:        {target_steps}")
    print(f"Files ({len(sqlite_files)}):")
    for f in sqlite_files:
        print(f"  - {f}")
    print("---------------------\n")

    gpu_data_map = {}

    for i, filepath in enumerate(sqlite_files):
        rank    = get_rank_from_filename(filepath, i)
        dataset = load_single_gpu(filepath, rank, target_steps)

        if dataset:
            gpu_data_map[rank] = dataset
        else:
            print(f"[Warning] Failed to load data for Rank {rank} ({filepath})")

    if not gpu_data_map:
        print("[Fatal Error] No valid GPU data loaded. Exiting.")
        sys.exit(1)
        
    try:
        offsets = calculate_clock_offsets(gpu_data_map, n_kernels=20)

        all_steps_map = get_all_step_intervals(gpu_data_map)
        
        final_df = get_global_earliest_steps(all_steps_map, gpu_data_map)
        
        if final_df.empty:
            print("[Error] Could not calculate global step intervals. Check if NVTX markers exist.")
            sys.exit(1)

        # --- Level 1: Wait Time Analysis ---
        print("\n--- Level 1: Wait Time Analysis ---")
        wait_df = load_all_gpu_compute_wait_time(gpu_data_map)

        print(wait_df["step"].value_counts().sort_index())
        print(f"\nstep=-1 ratio: {(wait_df['step']==-1).mean()*100:.1f}%")

        # Verify: at least one rank per bucket should have pure_wait ≈ 0
        step2 = wait_df[wait_df["step"] == 2].copy()
        print(step2[["rank", "bucket_idx", "pure_wait_ms"]]
            .sort_values(["bucket_idx", "rank"]))

        if wait_df.empty:
            print("[Warning] wait_df is empty. Check NVTX NCCL_AllReduce ranges.")
        else:
            print(f"  Total buckets analyzed: {len(wait_df)}")
            print(f"  Step range: {sorted(wait_df['step'].unique())}")

            output_filename = (
                f"level1_wait_time"
                f"_ranks{'_'.join(str(r) for r in sorted(gpu_data_map.keys()))}"
                f".png"
            )
            plot_wait_time_per_step(
                wait_df=wait_df,
                output_path=output_filename
            )
            print(f"Level 1 graph saved: {output_filename}")

        print("\n[Success] Analysis and visualization completed successfully.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Fatal Error] An unexpected error occurred: {e}")