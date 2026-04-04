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
    df_nvtx: pd.DataFrame      # CPU-side NVTX markers (ranges)
    df_nccl: pd.DataFrame      # NCCL communication kernels
    df_memcpy: pd.DataFrame    # H2D, D2H memory copies
    df_gpu_duration: pd.DataFrame  # Real GPU kernel span time mapped to NVTX
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

        df_nvtx, df_nccl, df_memcpy, df_gpu_duration = process_full_analysis(con, steps)
        
        return GpuDataset(
            rank=rank,
            filename=filepath,
            df_nvtx=df_nvtx,
            df_nccl=df_nccl,
            df_memcpy=df_memcpy,
            df_gpu_duration=df_gpu_duration,
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
        
        step_df = build_step_df_from_nvtx(dataset.df_nvtx)
        
        if not step_df.empty:
            all_steps_map[rank] = step_df
        else:
            print(f"[Warning] Rank {rank}: No step markers (Batch/Step/Iter) found in NVTX.")

    return all_steps_map
    """
    최종 결과 예시:

    입력 gpu_data_map:
      rank 0: nvtx_df에 cpu_batch_0_duration, cpu_batch_1_duration 등 포함
      rank 1: nvtx_df에 cpu_batch_0_duration, cpu_batch_1_duration 등 포함
      rank 2: nvtx_df에 cpu_batch_0_duration, cpu_batch_1_duration 등 포함
      rank 3: nvtx_df에 cpu_batch_0_duration, cpu_batch_1_duration 등 포함

    all_steps_map 결과:
    {
      0: step  start      end
         0     13800000   14200000   ← CPU 시계 (ns)
         1     14200000   14600000
         2     14600000   15000000

      1: step  start      end
         0     13800100   14200100   ← 같은 노드, 거의 동일
         1     14200100   14600100
         2     14600100   15000100

      2: step  start      end        ← 다른 노드, 100ms 차이
         0     13697000   14097000
         1     14097000   14497000
         2     14497000   14897000

      3: step  start      end        ← 다른 노드, 116ms 차이
         0     13684000   14084000
         1     14084000   14484000
         2     14484000   14884000
    }

    주의:
      rank 0, 1: 같은 노드 → start 차이 매우 작음 (수백 ns)
      rank 2, 3: 다른 노드 → CPU 시계 차이 ~100ms
                            (NTP 동기화로 어느정도 맞춰져 있음)

    이후 사용:
      compute_rank_order_per_step에서:
        모든 rank의 step 범위 비교
        → 각 step의 global earliest_start, latest_end 계산

      plot_timeline_custom_axis에서:
        특정 rank의 특정 step 범위 필터링
        step_df_sel = all_steps_map[rank][
            all_steps_map[rank]["step"].isin(steps_to_plot)
        ]
    """

def compute_rank_order_per_step(
    all_steps_map: Dict[int, pd.DataFrame],
    gpu_data_map: Dict[int, GpuDataset],
    offsets: Dict[int, int]
    ) -> pd.DataFrame:

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

        # offset 보정 적용
        # offset = rank 0 기준으로 맞추기 위한 보정값
        # start + offset → rank 0 시계 기준으로 변환
        offset = offsets.get(rank, 0)
        temp_df["start"] = temp_df["start"] + offset
        temp_df["end"]   = temp_df["end"]   + offset
        combined_list.append(temp_df)
    
        #     combined_list = [
        #     # df_rank0 (DataFrame)
        #     step  start      end        rank
        #     0     13800000   14200000   0
        #     1     14200000   14600000   0,

        #     # df_rank1 (DataFrame)
        #     step  start      end        rank
        #     0     13800100   14200100   1
        #     1     14200100   14600100   1,

        #     # df_rank2 (DataFrame)
        #     step  start      end        rank
        #     0     13697000   14097000   2
        #     1     14097000   14497000   2
        #     ]


    if not combined_list:
        return pd.DataFrame()
        
    big_df = pd.concat(combined_list, ignore_index=True)

    df_grouped = big_df.groupby("step")
    
    # step 값으로 그룹화
    #
    # df_grouped:
    #   step 0 그룹 (index 0, 2, 4, 6):
    #     index  step  start      end        rank
    #     0      0     13800000   14200000   0
    #     2      0     13800100   14200100   1
    #     4      0     13697000   14097000   2
    #     6      0     13684000   14084000   3
    #
    #   step 1 그룹 (index 1, 3, 5, 7):
    #     index  step  start      end        rank
    #     1      1     14200000   14600000   0
    #     3      1     14200100   14600100   1
    #     5      1     14097000   14497000   2
    #     7      1     14084000   14484000   3

    df_rank_order_per_step = df_grouped.agg(
        earliest_start=("start", "min"),
        latest_end=("end", "max")
    )
    
    # df_rank_order_per_step 결과:
    #   step  earliest_start  latest_end
    #   0     13799800        14200100
    #   1     14199800        14600100

    
    min_idx_series = df_grouped["start"].idxmin()
    max_idx_series = df_grouped["end"].idxmax()

    #   min_idx_series:
    #     step
    #     0    4   ← step 0에서 start 가장 작은 행의 index = 4 (rank 2)
    #     1    5   ← step 1에서 start 가장 작은 행의 index = 5 (rank 2)
    #
    #   max_idx_series:
    #     step
    #     0    6   ← step 0에서 end 가장 큰 행의 index = 6 (rank 3)
    #     1    7   ← step 1에서 end 가장 큰 행의 index = 7 (rank 3)

    
    fastest_ranks = big_df.loc[min_idx_series, ["step", "rank"]].rename(columns={"rank": "cpu_fastest_rank"})
    slowest_ranks = big_df.loc[max_idx_series, ["step", "rank"]].rename(columns={"rank": "cpu_slowest_rank"})
    
    df_rank_order_per_step = df_rank_order_per_step.merge(fastest_ranks, on="step", how="left")
    df_rank_order_per_step = df_rank_order_per_step.merge(slowest_ranks, on="step", how="left")
    
    # merge 전 df_rank_order_per_step:
    #   step  earliest_start  latest_end
    #   0     13799800        14200100
    #   1     14199800        14600100
    #
    # merge 후:
    #   step  earliest_start  latest_end  cpu_fastest_ranks cpu_sloweset_rank
    #   0     13799800        14200100    2
    #   1     14199800        14600100    2
    gpu_end_map   = {}
    gpu_start_map = {}
    gpu_end_rank_map   = {}   # ← 추가
    gpu_start_rank_map = {}   # ← 추가

    for rank, dataset in gpu_data_map.items():

        if dataset.df_gpu_duration.empty:
            continue

        gpu_batch_rows = dataset.df_gpu_duration[
            dataset.df_gpu_duration["name"].str.contains(r"gpu_batch_\d+_duration", regex=True)
        ]

        for _, row in gpu_batch_rows.iterrows():

            step  = int(row["data_batch_idx"])
            end   = int(row["gpu_end"])   + offsets.get(rank, 0)  # ← 추가
            start = int(row["gpu_start"]) + offsets.get(rank, 0)  # ← 추가


            # gpu_latest_end 업데이트
            if end > gpu_end_map.get(step, end - 1):
                gpu_end_map[step]      = end
                gpu_end_rank_map[step] = rank   # ← 가장 늦게 끝난 rank

            # gpu_earliest_start 업데이트
            if start < gpu_start_map.get(step, start + 1):
                gpu_start_map[step]      = start
                gpu_start_rank_map[step] = rank  # ← 가장 빨리 시작한 rank


    df_rank_order_per_step["gpu_latest_end"]     = df_rank_order_per_step["step"].map(gpu_end_map)
    df_rank_order_per_step["gpu_earliest_start"] = df_rank_order_per_step["step"].map(gpu_start_map)
    df_rank_order_per_step["gpu_slowest_rank"]   = df_rank_order_per_step["step"].map(gpu_end_rank_map)    # ← 추가
    df_rank_order_per_step["gpu_fastest_rank"]   = df_rank_order_per_step["step"].map(gpu_start_rank_map) # ← 추가


    return df_rank_order_per_step.sort_values("step")

    ## 📌 최종 df_rank_order_per_step
    """
    step  earliest_start  latest_end  cpu_fastest_rank  cpu_slowest_rank  gpu_latest_end  gpu_earliest_start  gpu_slowest_rank  gpu_fastest_rank
    0     13799800        14200100    2                 3                 116945          205                 3                 1
    1     14199800        14600100    2                 3                 132945          855                 3                 1
    """

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
            return KernelSchema(table=t, start_col=sc, end_col=ec, name_id_col=nid, corr_id_col=cid or "correlation_id")
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
    corr_id_col: str
    kind_col: Optional[str] = None
    bytes_col: Optional[str] = None

def find_memcpy_schema(con: sqlite3.Connection) -> Optional[MemcpySchema]:

    tables     = list_tables(con)
    candidates = [t for t in tables if "memcpy" in t.lower()]

    if not candidates: return None

    possible_start = ["start", "startNs", "timestamp_start"]
    possible_end   = ["end", "endNs", "timestamp_end"]
    possible_corr_id = ["correlationId", "correlation_id"] 
    possible_kind  = ["copyKind", "kind", "memcpyKind"]
    possible_bytes = ["bytes", "byteCount", "size"]

    for t in candidates:
        cols = set(table_columns(con, t))
        sc = next((c for c in possible_start if c in cols), None)
        ec = next((c for c in possible_end   if c in cols), None)
        cid = next((c for c in possible_corr_id if c in cols), None) 
        if not sc or not ec: continue

        kc = next((c for c in possible_kind  if c in cols), None)
        bc = next((c for c in possible_bytes if c in cols), None)
    
        return MemcpySchema(
            table=t,
            start_col=sc,
            end_col=ec,
            corr_id_col=cid,
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

    df["name"] = df["name"].apply(
        lambda x: rename_map.get(x, x)
    )

    # Batch_N → cpu_batch_N_duration
    df["name"] = df["name"].str.replace(
        r"^Batch_(\d+)$", r"cpu_batch_\1_duration", regex=True
    )

    return df

    """
    nvtx_df 결과 예시:

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
        k.{k_sch.start_col} AS kernel_start,
        k.{k_sch.end_col}   AS kernel_end,
        k.{k_sch.corr_id_col}  AS correlation_id
    FROM {k_sch.table} k
    JOIN {s_sch.table} s
      ON k.{k_sch.name_id_col} = s.{s_sch.id_col}
    WHERE k.{k_sch.end_col} > k.{k_sch.start_col}
      AND s.{s_sch.value_col} LIKE '{like_pattern}'
    """


    df = try_read_df(con, q)
    df["dur_ns"] = df["kernel_end"] - df["kernel_start"]
    return df
    """
     CUPTI_ACTIVITY_KIND_KERNEL 테이블:
      correlation_id   start    end      shortNameId
      3001            420      480      504
      3002            850      910      504
      3003            1280     1340     504

    StringIds 테이블:
      id    value
      504   ncclAllReduceRingLLKernel

    like_pattern = "%nccl%"
    → name LIKE '%nccl%' → ncclAllReduceRingLLKernel 매칭

    df_nccl 결과:
      name                        gpu_kernel_start  gpu_kernel_end  dur_ns correlation_id
      ncclAllReduceRingLLKernel   420               480             60                  ← step 0 bucket 0
      ncclAllReduceRingLLKernel   850               910             60                  ← step 0 bucket 1
      ncclAllReduceRingLLKernel   1280              1340            60                  ← step 1 bucket 0
    """

def load_memcpy_events(con: sqlite3.Connection,
                       sch: MemcpySchema,
                       want_kinds: Optional[List[int]] = None,
                       name: str ="gpu_h2d_duration") -> pd.DataFrame:
    q = f"""
    SELECT
        {sch.start_col} AS kernel_start,
        {sch.end_col}   AS kernel_end,
        {sch.corr_id_col} AS correlation_id,
        {sch.kind_col}    AS kind
    FROM {sch.table}
    WHERE {sch.end_col} > {sch.start_col}
    """

    df = try_read_df(con, q)

    if df.empty:
        return pd.DataFrame(columns=["name", "kernel_start", "kernel_end", "dur_ns","correlation_id"])

    if want_kinds is not None and sch.kind_col and "kind" in df.columns:
        df = df[df["kind"].isin(want_kinds)].copy()

    df["name"]   = name
    df["dur_ns"] = df["kernel_end"] - df["kernel_start"]

    return df[["name", "kernel_start", "kernel_end", "dur_ns","correlation_id","kind"]].copy()

    ## 📌 최종 df_memcpy
    """
    name               kernel_start  kernel_end  dur_ns  correlation_id
    gpu_h2d_duration   150           160         10      4001
    gpu_h2d_duration   160           168         8       4002
    gpu_h2d_duration   480           490         10      4004
    gpu_h2d_duration   490           498         8       4005
    gpu_h2d_duration   960           970         10      4006
    """

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
        {r_sch.corr_id_col} AS correlation_id,
        {r_sch.start_col} AS cpu_api_start, 
        {r_sch.end_col} AS cpu_api_end
    FROM {r_sch.table}
    WHERE {r_sch.start_col} >= {min_start} AND {r_sch.end_col} <= {max_end}
    """

    df = try_read_df(con, q)

    return df.sort_values("cpu_api_start") if not df.empty else df
    """
     correlation_id   cpu_api_start   cpu_api_end
      1001                        210             210.5       ← cpu_forward_launch(210~350) 안
      1002                        211             211.5       ← cpu_forward_launch 안
      1003                        212             212.5       ← cpu_forward_launch 안
      ...                         ...             ...
      2001                        361             361.5       ← cpu_backward_launch(360~480) 안
      2002                        362             362.5       ← cpu_backward_launch 안
      3001                        461             461.5       ← cpu_nccl_allreduce_launch(460~475) 안
    """


def load_gpu_kernel_timings(
    con: sqlite3.Connection, 
    k_sch: KernelSchema,
    s_sch: StringIdsSchema = None 
    ) -> pd.DataFrame:
    
    """Loads ALL kernels to calculate spans (needs correlation_id). GPU kernel time."""
    
    if s_sch:
        q = f"""
        SELECT 
            k.{k_sch.corr_id_col} AS correlation_id, 
            k.{k_sch.start_col}   AS kernel_start, 
            k.{k_sch.end_col}     AS kernel_end,
            s.{s_sch.value_col}   AS kernel_name
        FROM {k_sch.table} k
        LEFT JOIN {s_sch.table} s
            ON k.{k_sch.name_id_col} = s.{s_sch.id_col}
        WHERE k.{k_sch.end_col} > k.{k_sch.start_col}
        """
    
    #StringID 결과가 없을때 
    else:
        q = f"""
        SELECT 
            {k_sch.corr_id_col} AS correlation_id, 
            {k_sch.start_col}   AS kernel_start, 
            {k_sch.end_col}     AS kernel_end
        FROM {k_sch.table} 
        WHERE {k_sch.end_col} > {k_sch.start_col}
        """
    
    return try_read_df(con, q)
    """
    correlation_id   kernel_start   kernel_end       kernel_name
      1001            300           400             volta_sgemm_128x32_tn
      1002            400           500             volta_sgemm_128x32_tn
      1003            500           520             vectorized_elementwise
      2001            550           650             volta_sgemm_128x32_nt
      2002            650           750             volta_sgemm_128x32_nt
      3001            760           820             ncclAllReduceRingLL
    """

def map_nvtx_to_runtime(
    df_nvtx: pd.DataFrame,
    df_runtime_api_in_nvtx: pd.DataFrame
    ) -> pd.DataFrame:
  

    """Maps NVTX ranges to Runtime API correlation IDs."""
    if df_nvtx.empty or df_runtime_api_in_nvtx.empty:
        return pd.DataFrame()
    
    nvtx_sorted = df_nvtx.sort_values("start").reset_index(drop=True)
    rt_sorted   = df_runtime_api_in_nvtx.sort_values("cpu_api_start").reset_index(drop=True)

    rt_starts = rt_sorted["cpu_api_start"].values        
    rt_corrs  = rt_sorted["correlation_id"].values
    
    df_map_nvtx_api_launch  = []

    for _, row in nvtx_sorted.iterrows():
        cpu_launch_start = row["start"]
        cpu_launch_end   = row["end"]
        cpu_launch_in_nvtx = row["name"]

        # CPU가 launch한 GPU kernel의 correlation_id 수집
        mask = (rt_starts >= cpu_launch_start) & (rt_starts < cpu_launch_end)
        matched_corrs = rt_corrs[mask]
        
        for cid in matched_corrs:
            df_map_nvtx_api_launch.append({
                "cpu_launch_in_nvtx":     cpu_launch_in_nvtx,
                "cpu_launch_start":    cpu_launch_start,
                "cpu_launch_end":    cpu_launch_end,
                "correlation_id": cid
            })
            
    return pd.DataFrame(df_map_nvtx_api_launch)
    """
    cpu_launch_in_nvtx       cpu_launch_start  cpu_launch_end  correlation_id
      cpu_batch_0_duration          100         500       1001
      cpu_batch_0_duration          100         500       1002
      cpu_batch_0_duration          100         500       1003
      cpu_batch_0_duration          100         500       2001
      cpu_batch_0_duration          100         500       2002
      cpu_batch_0_duration          100         500       3001
      cpu_data_wait_launch          100         110       (없음, kernel launch 없음)
      cpu_h2d_launch                110         115       (없음, memcpy는 별도)
      cpu_train_compute_duration    200         490       1001
      cpu_train_compute_duration    200         490       1002
      cpu_train_compute_duration    200         490       1003
      cpu_train_compute_duration    200         490       2001
      cpu_train_compute_duration    200         490       2002
      cpu_train_compute_duration    200         490       3001
      cpu_forward_launch            210         350       1001
      cpu_forward_launch            210         350       1002
      cpu_forward_launch            210         350       1003
      cpu_backward_launch           360         480       2001
      cpu_backward_launch           360         480       2002
      cpu_nccl_allreduce_launch     460         475       3001
    """

def join_cpu_api_in_nvtx_with_gpu_kernel_timings(
    df_map_nvtx_to_cpu_api: pd.DataFrame, 
    df_kernels_all: pd.DataFrame,
    df_step_ranges_in_nvtx: pd.DataFrame = None
    ) -> pd.DataFrame:
    
    """Calculates [Min Kernel Start, Max Kernel End] for each NVTX."""

    if df_map_nvtx_to_cpu_api.empty or df_kernels_all.empty:
        return pd.DataFrame()

    print(f"Merging {len(df_map_nvtx_to_cpu_api)} CPU-Mappings with {len(df_kernels_all)} GPU-Kernels...")


    # =========================================================================
    # Step 1: correlation_id로 join
    # =========================================================================
    merged = pd.merge(df_map_nvtx_to_cpu_api, df_kernels_all, on="correlation_id", how="inner")


    # =========================================================================
    # Step 2: backward 범위 안의 NCCL kernel 제거
    # =========================================================================
    # backward NVTX 안에 NCCL kernel이 포함되어 있음
    # (DDP가 backward 중에 AllReduce를 실행하기 때문)
    # NCCL kernel은 nccl_df에서 별도로 로드하므로
    # true_gpu_df에는 포함하지 않음
    #
    # 예시:
    #   nvtx_name=cpu_backward_launch, name=ncclAllReduceRingLL → 제거 ❌
    #   nvtx_name=cpu_backward_launch, name=volta_sgemm_...     → 유지 ✅

    nccl_in_backward_mask = (
        merged["cpu_launch_in_nvtx"].str.contains("cpu_backward_launch", case=False) &
        merged["kernel_name"].str.contains("nccl", case=False)
    )
    merged = merged[~nccl_in_backward_mask]

    if merged.empty: 
        return pd.DataFrame()

    # =========================================================================
    # Step 3: NVTX 범위별로 min(k_start), max(k_end) 계산
    # =========================================================================

    def compute_nvtx_kernel_spans(group):
        # 각 NVTX 범위 안의 kernel들을 interval merge
        # → 겹치는 kernel 합치고 gap 제외
        df_kernels = pd.DataFrame({
            "start": group["kernel_start"].values,
            "end":   group["kernel_end"].values
        })
        intervals = get_merged_intervals(df_kernels)

        if not intervals:
            return pd.Series({
                "gpu_start": None,
                "gpu_end":   None,
                "dur_ns":    0
            })

        return pd.Series({
            "gpu_start": min(st for st, en in intervals),
            "gpu_end":   max(en for st, en in intervals),
            "dur_ns":    sum((en - st) for st, en in intervals)
        })



    result = merged.groupby(
        ["cpu_launch_in_nvtx", "cpu_launch_start", "cpu_launch_end"]
    ).apply(compute_nvtx_kernel_spans).reset_index()

    result = result.dropna(subset=["gpu_start", "gpu_end"])
    result["gpu_start"] = result["gpu_start"].astype(int)
    result["gpu_end"]   = result["gpu_end"].astype(int)
    result["dur_ns"]    = result["dur_ns"].astype(int)


    nvtx_to_gpu_name = {
        "cpu_forward_launch":         "gpu_forward_duration",
        "cpu_backward_launch":        "gpu_backward_duration",
        "cpu_loss_launch":            "gpu_loss_duration",
        "cpu_opt_step_launch":        "gpu_opt_step_duration",
        "cpu_zero_grad_launch":       "gpu_zero_grad_duration",
        "cpu_train_compute_duration": "gpu_train_compute_duration",
        "cpu_nccl_allreduce_launch":  "gpu_nccl_allreduce_duration",
    }


    def to_gpu_name(cpu_launch_in_nvtx: str) -> str:
        batch_match = re.match(r"cpu_batch_(\d+)_duration", cpu_launch_in_nvtx)
        if batch_match:
            return f"gpu_batch_{batch_match.group(1)}_duration"
        return nvtx_to_gpu_name.get(cpu_launch_in_nvtx, f"gpu_{cpu_launch_in_nvtx}")

    result["name"] = result["cpu_launch_in_nvtx"].apply(to_gpu_name)

    """
    result 전체 테이블:

    cpu_launch_in_nvtx           cpu_launch_start  cpu_launch_end  gpu_start  gpu_end  dur_ns  name
    cpu_batch_0_duration         100               500             300        950      600     gpu_batch_0_duration
    cpu_batch_1_duration         500               900             950        1600     600     gpu_batch_1_duration
    cpu_train_compute_duration   200               490             300        820      480     gpu_train_compute_duration
    cpu_zero_grad_launch         200               210             300        310      10      gpu_zero_grad_duration
    cpu_forward_launch           210               350             310        550      180     gpu_forward_duration
    cpu_loss_launch              350               360             550        560      10      gpu_loss_duration
    cpu_backward_launch          360               480             560        850      200     gpu_backward_duration
    cpu_opt_step_launch          480               490             850        950      80      gpu_opt_step_duration
    cpu_nccl_allreduce_launch    460               475             760        820      60      gpu_nccl_allreduce_duration
        주의:
      cpu_data_wait_launch:
        GPU kernel 없음 → intervals 비어있음
        → start=None, end=None, dur_ns=0
        → dropna 후 제거됨 ❌

      cpu_h2d_launch:
        cudaMemcpyAsync → MEMCPY 테이블
        → df_kernels_all(KERNEL 테이블)에 없음
        → join 시 매핑 안 됨
        → dropna 후 제거됨 ❌

      cpu_forward_launch:
        kernel 1: 310~400
        kernel 2: 450~500  ← gap 400~450
        kernel 3: 520~550  ← gap 500~520
        dur_ns = (400-310) + (500-450) + (550-520) = 180  ← gap 제외 ✅
    """

    if df_step_ranges_in_nvtx is not None and not df_step_ranges_in_nvtx.empty:
        # =====================================================================
        # Step 4: data_batch_idx 할당
        # =====================================================================
        # CPU 시계 기준 step 범위(df_step_ranges_in_nvtx)와
        # cpu_launch_start 시간을 비교하여 data_batch_idx 할당
        #
        # 예시:
        #   df_step_ranges_in_nvtx:
        #     step  start   end
        #     0     100     500   ← CPU 시계
        #     1     500     900
        #
        #   cpu_forward_launch: cpu_launch_start=210
        #   → 100 <= 210 < 500 → data_batch_idx=0
        #
        #   cpu_forward_launch: cpu_launch_start=600
        #   → 500 <= 600 < 900 → data_batch_idx=1

        def assign_data_batch_idx(cpu_launch_start):
            match = df_step_ranges_in_nvtx[
                (df_step_ranges_in_nvtx["start"] <= cpu_launch_start) &
                (df_step_ranges_in_nvtx["end"]   > cpu_launch_start)
            ]
            if not match.empty:
                return int(match.iloc[0]["step"])
            return -1

        result["data_batch_idx"] = result["cpu_launch_start"].apply(assign_data_batch_idx)
        result = result.sort_values(["data_batch_idx", "gpu_start"])

        return result[[
            "name",
            "gpu_start",
            "gpu_end",
            "dur_ns",
            "data_batch_idx"
        ]]
    return result[["name", "gpu_start", "gpu_end", "dur_ns"]]

    """
    최종 result 테이블 예시:

    data_batch_idx 있을 때:

    name                          gpu_start  gpu_end  dur_ns  data_batch_idx
    gpu_batch_0_duration          300        950      600     0
    gpu_train_compute_duration    300        820      480     0
    gpu_zero_grad_duration        300        310      10      0
    gpu_forward_duration          310        550      180     0    ← gap 제외
    gpu_loss_duration             550        560      10      0
    gpu_backward_duration         560        850      200     0
    gpu_opt_step_duration         850        950      80      0
    gpu_nccl_allreduce_duration   760        820      60      0
    gpu_batch_1_duration          950        1600     600     1
    gpu_train_compute_duration    950        1470     480     1
    gpu_zero_grad_duration        950        960      10      1
    gpu_forward_duration          960        1200     180     1
    gpu_loss_duration             1200       1210     10      1
    gpu_backward_duration         1210       1500     200     1
    gpu_opt_step_duration         1500       1600     80      1
    gpu_nccl_allreduce_duration   1410       1470     60      1

    주의:
      cpu_data_wait_launch → GPU kernel 없음 → dropna 후 제거 ❌
      cpu_h2d_launch       → MEMCPY 테이블  → join 시 제거 ❌

    data_batch_idx 없을 때 (df_step_ranges_in_nvtx=None):

    name                          gpu_start  gpu_end  dur_ns
    gpu_batch_0_duration          300        950      600
    gpu_train_compute_duration    300        820      480
    gpu_zero_grad_duration        300        310      10
    gpu_forward_duration          310        550      180
    gpu_loss_duration             550        560      10
    gpu_backward_duration         560        850      200
    gpu_opt_step_duration         850        950      80
    gpu_nccl_allreduce_duration   760        820      60
    gpu_batch_1_duration          950        1600     600
    """

# ==============================================================================
# 5. Clock Offset Calculation
# ==============================================================================

def get_allreduce_df(
    df_nccl: pd.DataFrame
    ) -> pd.DataFrame:

        if df_nccl.empty:

            return pd.DataFrame()

        return df_nccl[df_nccl["name"].str.contains("AllReduce", case=False)]
    
def calculate_clock_offsets(
    gpu_data_map: Dict[int, GpuDataset],
    n_kernels: int = 20
    ) -> Dict[int, int]:

    # 각 rank의 AllReduce end time을 순서대로 정렬
    rank_ends = {}
    for rank, dataset in gpu_data_map.items():
        allreduce_df = get_allreduce_df(dataset.df_nccl)
        if allreduce_df.empty:
            rank_ends[rank] = []
            continue
        # 시간순 정렬 후 end time 리스트
        ends = allreduce_df.sort_values("kernel_start")["kernel_end"].values
        rank_ends[rank] = ends

    # 같은 bucket index끼리 비교
    base_ends = rank_ends[0]  # rank 0 기준
    n = min(len(base_ends), n_kernels)

    offsets = {}
    offsets[0] = 0

    print("\n=== Starting Clock Offset Calculation ===")

    for rank in gpu_data_map.keys():
        if rank == 0:
            continue

        comp_ends = rank_ends[rank]
        m = min(n, len(comp_ends))

        # 같은 bucket index끼리 차이 계산
        per_bucket_diff = base_ends[:m] - comp_ends[:m]
        offset = int(np.median(per_bucket_diff))
        offsets[rank] = offset

        # print(f"\n  Rank {rank}:")
        # print(f"    per-bucket diff mean = {per_bucket_diff.mean()/1e6:.3f}ms")
        # print(f"    per-bucket diff std  = {per_bucket_diff.std()/1e6:.3f}ms")
        # print(f"    median offset        = {offset/1e6:+.3f}ms")

    print("\n=== Final Offsets ===")
    for rank, offset in offsets.items():
        print(f"  Rank {rank}: {offset/1e6:+.3f} ms")


    return offsets

def debug_clock_offsets(gpu_data_map: Dict[int, GpuDataset], n_kernels: int = 20):
    """
    각 rank의 같은 bucket index AllReduce start/end time을 직접 출력
    → offset이 clock 차이인지 작업 속도 차이인지 확인
    """
    print("\n=== Debug: Per-Bucket AllReduce Start/End Times ===")

    rank_allreduce = {}
    for rank, dataset in gpu_data_map.items():

        allreduce_df = get_allreduce_df(dataset.df_nccl)

        if allreduce_df.empty:
            print(f"Rank {rank}: NO NCCL DATA")
            continue

        sorted_df = allreduce_df.sort_values("kernel_start").head(n_kernels)
        rank_allreduce[rank] = sorted_df.reset_index(drop=True)

    ranks = sorted(rank_allreduce.keys())
    n = min(len(rank_allreduce[r]) for r in ranks)

    # print(f"\n{'bucket':>6} | " + " | ".join(
    #     [f"R{r}_start_ms   R{r}_end_ms  " for r in ranks]
    # ))
    # print("-" * (8 + 32 * len(ranks)))

    for i in range(n):
        row_parts = []
        for r in ranks:
            start_ms = rank_allreduce[r].iloc[i]["kernel_start"] / 1e6
            end_ms   = rank_allreduce[r].iloc[i]["kernel_end"]   / 1e6
            row_parts.append(f"{start_ms:12.3f}  {end_ms:12.3f}")
        # print(f"{i:>6} | " + " | ".join(row_parts))

    print("\n=== Per-Bucket End Time Diff (vs Rank 0) ===")
    base = rank_allreduce[0]
    for r in ranks:
        if r == 0:
            continue
        diffs = base["end"].values[:n] - rank_allreduce[r]["end"].values[:n]
        # print(f"\n  Rank 0 vs Rank {r}:")
        # print(f"    mean  = {diffs.mean()/1e6:+.3f} ms")
        # print(f"    std   = {diffs.std()/1e6:.3f} ms")
        # print(f"    min   = {diffs.min()/1e6:+.3f} ms")
        # print(f"    max   = {diffs.max()/1e6:+.3f} ms")

    print("\n=== Per-Bucket Start Time Diff (vs Rank 0) ===")
    for r in ranks:
        if r == 0:
            continue
        diffs = base["start"].values[:n] - rank_allreduce[r]["start"].values[:n]
        # print(f"\n  Rank 0 vs Rank {r}:")
        # print(f"    mean  = {diffs.mean()/1e6:+.3f} ms")
        # print(f"    std   = {diffs.std()/1e6:.3f} ms")
        # print(f"    min   = {diffs.min()/1e6:+.3f} ms")
        # print(f"    max   = {diffs.max()/1e6:+.3f} ms")
# ==============================================================================
# 6. Wait Time Analysis
# ==============================================================================

def get_allreduce_start_times_per_bucket(
    nvtx_df: pd.DataFrame,
    nccl_df: pd.DataFrame
    ) -> pd.DataFrame:

    nvtx_nccl_df = nvtx_df[
        nvtx_df["name"] == "cpu_nccl_allreduce_launch"
    ].sort_values("start").reset_index(drop=True)

    nccl_allreduce_df = nccl_df[
        nccl_df["name"].str.contains("AllReduce", case=False)
    ].sort_values("kernel_start").reset_index(drop=True)

    n_nvtx   = len(nvtx_nccl_df)
    n_kernel = len(nccl_allreduce_df)
    # print(f"NVTX ranges:       {n_nvtx}")
    # print(f"AllReduce kernels: {n_kernel}")

    if n_nvtx != n_kernel:
        print(f"[WARNING] Count mismatch: NVTX={n_nvtx}, kernels={n_kernel}")

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

def load_all_gpu_compute_wait_time(
    gpu_data_map: Dict[int, GpuDataset],
    offsets: Dict[int, int]
    ) -> pd.DataFrame:

    all_results = []

    for rank, gpu_data in gpu_data_map.items():

        # CPU 시계 기준 step 범위 추출
        # cpu_batch_N_duration에서 각 step의 CPU 시작/끝 시간 추출
        step_ranges = gpu_data.df_nvtx[
            gpu_data.df_nvtx["name"].str.contains(
                r"cpu_batch_\d+_duration", regex=True
            )
        ][["data_batch_idx", "start", "end"]].drop_duplicates("data_batch_idx")

        # CPU 시계 기준으로 step 할당
        nvtx_with_step = gpu_data.df_nvtx.copy()

        def assign_step(row_start):
            match = step_ranges[
                (step_ranges["start"] <= row_start) &
                (step_ranges["end"]   >= row_start)
            ]
            return int(match.iloc[0]["data_batch_idx"]) if not match.empty else -1

        nvtx_with_step["step"] = nvtx_with_step["start"].apply(assign_step)

        rank_df = get_allreduce_start_times_per_bucket(
            nvtx_df=nvtx_with_step,
            nccl_df=gpu_data.df_nccl,   # ← 수정
        )

        if rank_df.empty:
            print(f"[WARNING] Rank {rank}: No allreduce data found")
            continue

        rank_df["rank"] = rank

        # offset 보정
        rank_df["allreduce_start_corrected_ns"] = (
            rank_df["allreduce_start_ns"] + offsets.get(rank, 0)
        )

        all_results.append(rank_df)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    combined = combined.sort_values(["step", "rank", "bucket_idx"])
    combined["bucket_within_step"] = (
        combined.groupby(["step", "rank"]).cumcount()
    )

    max_allreduce_start = combined.groupby(
        ["step", "bucket_within_step"]
    )["allreduce_start_corrected_ns"].transform("max")

    combined["pure_wait_ms"] = (
        (max_allreduce_start - combined["allreduce_start_corrected_ns"])
        .clip(lower=0) / 1e6
    )

    return combined

# ==============================================================================
# 7. breakdown 
# ==============================================================================

def get_merged_intervals(df: pd.DataFrame):
    """DataFrame의 start, end 시간을 기반으로 겹치는 구간을 하나로 병합합니다."""
    if df is None or df.empty:
        return []

    start_col = next(
        (c for c in ["start", "kernel_start", "gpu_start"]
         if c in df.columns), None
    )
    end_col = next(
        (c for c in ["end", "kernel_end", "gpu_end"]
         if c in df.columns), None
    )
    if start_col is None or end_col is None:
        return []
    
    # 시간 순으로 정렬
    sorted_df = df.sort_values(start_col) 
    intervals = []
    
    for _, row in sorted_df.iterrows():
        st, en = row[start_col], row[end_col]
        if not intervals:
            intervals.append([st, en])
        else:
            last_st, last_en = intervals[-1]
            if st <= last_en:  # 이전 구간과 겹치면 병합
                intervals[-1][1] = max(last_en, en)
            else:
                intervals.append([st, en])
    return intervals

def get_total_duration_ms(intervals: list) -> float:
    """병합된 구간들의 총 길이를 ms 단위로 반환합니다."""
    return sum([en - st for st, en in intervals]) / 1e6

def get_intersection_ms(ints_a: list, ints_b: list) -> float:
    """두 구간 리스트(A와 B)가 동시에 진행된(Overlap) 시간(ms)을 계산합니다."""
    i, j = 0, 0
    overlap_ns = 0
    while i < len(ints_a) and j < len(ints_b):
        start_a, end_a = ints_a[i]
        start_b, end_b = ints_b[j]
        
        # 교집합 구간 찾기
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)
        
        if overlap_start < overlap_end:
            overlap_ns += (overlap_end - overlap_start)
            
        # 포인터 이동
        if end_a < end_b:
            i += 1
        else:
            j += 1
            
    return overlap_ns / 1e6


def aggregate_per_step_breakdown(
    gpu_data_map: Dict[int, GpuDataset],
    steps_to_plot: list,
    nccl_wait_df: pd.DataFrame,
    offsets: Dict[int, int],
    df_rank_order_per_step: pd.DataFrame,
    ) -> pd.DataFrame:

    target_stats = df_rank_order_per_step[
        df_rank_order_per_step["step"].isin(steps_to_plot)
    ]
    global_start = int(target_stats["earliest_start"].min())

    all_result = []

    for rank, gpu_data in gpu_data_map.items():

        df_nvtx         = gpu_data.df_nvtx
        df_nccl         = gpu_data.df_nccl
        df_memcpy       = gpu_data.df_memcpy
        df_gpu_duration = gpu_data.df_gpu_duration

        step_df     = build_step_df_from_nvtx(df_nvtx)
        step_df_sel = step_df[
            step_df["step"].isin(steps_to_plot)
        ].sort_values("start")

        if step_df_sel.empty:
            continue

        def safe_filter(df, use_data_batch_idx=False):
            if df is None or df.empty:
                return pd.DataFrame()
            res = filter_by_step_ranges(
                df, step_df_sel, global_start,
                use_data_batch_idx=use_data_batch_idx,
                rank=rank, offsets=offsets
            )
            return res if res is not None and not res.empty else pd.DataFrame()

        f_nvtx         = safe_filter(df_nvtx,         use_data_batch_idx=True)
        f_nccl         = safe_filter(df_nccl)
        f_memcpy       = safe_filter(df_memcpy,        use_data_batch_idx=True)
        f_gpu_duration = safe_filter(df_gpu_duration)

        for step in steps_to_plot:

            step_nvtx         = f_nvtx[f_nvtx["step"] == step]
            step_nccl         = f_nccl[f_nccl["step"] == step]
            step_memcpy       = f_memcpy[f_memcpy["step"] == step]
            step_gpu_duration = f_gpu_duration[f_gpu_duration["step"] == step]

            # ==============================================================
            # step 전체 시간
            # ==============================================================
            cpu_batch = step_nvtx[
                step_nvtx["name"].str.contains(
                    r"cpu_batch_\d+_duration", regex=True
                )
            ]
            gpu_batch = step_gpu_duration[
                step_gpu_duration["name"].str.contains(
                    r"gpu_batch_\d+_duration", regex=True
                )
            ]

            # CPU + GPU 둘 다 있는 경우 (정상)
            # step_start: CPU data_wait 시작 시간
            # step_end:   GPU 마지막 kernel 끝 시간
            if not cpu_batch.empty and not gpu_batch.empty:
                step_start    = cpu_batch["start"].min()
                step_end      = gpu_batch["gpu_end"].max()
                step_duration = (step_end - step_start) / 1e6

            # CPU만 있는 경우
            # GPU 데이터 없음 → CPU 시간만으로 계산
            elif not cpu_batch.empty:
                step_duration = (
                    cpu_batch["end"].max() - cpu_batch["start"].min()
                ) / 1e6

            # 둘 다 없는 경우
            else:
                step_duration = 0.0

            # ==============================================================
            # 항목별 필터링
            # ==============================================================
            df_data_wait = step_nvtx[
                step_nvtx["name"] == "cpu_data_wait_launch"
            ]
            df_h2d = step_memcpy[
                step_memcpy["name"] == "gpu_h2d_duration"
            ]
            df_compute = step_gpu_duration[
                step_gpu_duration["name"].str.contains(
                    r"gpu_forward_duration|gpu_backward_duration|"
                    r"gpu_loss_duration|gpu_opt_step_duration",
                    regex=True
                )
            ]
            df_nccl_step = step_nccl

            # ==============================================================
            # interval merge (겹치는 구간 병합)
            # ==============================================================
            # df_data_wait: start, end 컬럼 → 그대로 사용
            data_wait_ints = get_merged_intervals(
                df_data_wait
            ) if not df_data_wait.empty else []

            # df_h2d: kernel_start, kernel_end → start, end 변환
            h2d_ints = get_merged_intervals(
                df_h2d.rename(columns={
                    "kernel_start": "start",
                    "kernel_end":   "end"
                })
            ) if not df_h2d.empty else []

            # df_nccl: kernel_start, kernel_end → start, end 변환
            nccl_ints = get_merged_intervals(
                df_nccl_step.rename(columns={
                    "kernel_start": "start",
                    "kernel_end":   "end"
                })
            ) if not df_nccl_step.empty else []

            # df_compute: gpu_start, gpu_end → start, end 변환
            compute_ints = get_merged_intervals(
                df_compute.rename(columns={
                    "gpu_start": "start",
                    "gpu_end":   "end"
                })
            ) if not df_compute.empty else []

            # ==============================================================
            # 각 항목 전체 시간 (ms)
            # ==============================================================
            data_wait_total = get_total_duration_ms(data_wait_ints)
            h2d_total       = get_total_duration_ms(h2d_ints)
            compute_total   = get_total_duration_ms(compute_ints)
            nccl_total      = get_total_duration_ms(nccl_ints)

            # ==============================================================
            # pure_wait 계산
            # wait_df에서 해당 rank, step의 pure_wait_ms 합산
            # → NCCL 안에 포함된 시간 (overlap_compute_nccl 또는 nccl_excl 안에 있음)
            # → 별도 항목으로 표시만 (이중 계산 아님)
            # ==============================================================
            pure_wait = nccl_wait_df[
                (nccl_wait_df["rank"] == rank) &
                (nccl_wait_df["step"] == step)
            ]["pure_wait_ms"].sum()

            # ==============================================================
            # overlap 계산 (ms)
            # ==============================================================
            # 1. data_wait vs gpu_compute overlap
            # data_wait가 GPU 연산 중에 발생하는 경우
            # (GPU backward 중에 다음 step 데이터 준비)
            overlap_wait_compute = get_intersection_ms(data_wait_ints, compute_ints)

            # 2. h2d vs gpu_compute overlap
            # h2d가 GPU backward와 동시에 실행되는 경우
            # (non_blocking=True + 다른 stream)
            overlap_h2d_compute = get_intersection_ms(h2d_ints, compute_ints)

            # 3. gpu_compute vs nccl overlap
            # DDP overlap: backward 중 NCCL AllReduce 실행
            # pure_wait도 이 안에 포함될 수 있음
            overlap_compute_nccl = get_intersection_ms(compute_ints, nccl_ints)

            # ==============================================================
            # overlap 비율 계산
            # ==============================================================
            # data_wait 중 gpu_compute와 겹치는 비율
            overlap_wait_compute_ratio = (
                overlap_wait_compute / data_wait_total
                if data_wait_total > 0 else 0.0
            )

            # h2d 중 gpu_compute와 겹치는 비율
            overlap_h2d_compute_ratio = (
                overlap_h2d_compute / h2d_total
                if h2d_total > 0 else 0.0
            )

            # nccl 중 gpu_compute와 겹치는 비율
            # (pure_wait 포함 가능)
            overlap_compute_nccl_ratio = (
                overlap_compute_nccl / nccl_total
                if nccl_total > 0 else 0.0
            )

            # ==============================================================
            # 순수 시간 계산 (overlap 제외)
            # ==============================================================
            # data_wait: gpu_compute와 겹치는 부분 제외
            data_wait_excl = max(0.0, data_wait_total - overlap_wait_compute)

            # h2d: gpu_compute와 겹치는 부분 제외
            h2d_excl = max(0.0, h2d_total - overlap_h2d_compute)

            # nccl: gpu_compute와 겹치는 부분 제외
            # pure_wait는 overlap_compute_nccl 또는 nccl_excl 안에 있음
            # → 이중 계산 방지를 위해 추가로 빼지 않음
            nccl_excl = max(0.0, nccl_total - overlap_compute_nccl)

            # gpu_compute: 모든 overlap 제외
            compute_excl = max(
                0.0,
                compute_total
                - overlap_wait_compute
                - overlap_h2d_compute
                - overlap_compute_nccl
            )

            # ==============================================================
            # cpu_overhead
            # step 전체에서 설명된 시간 제외한 나머지
            # (zero_grad, loss CPU 시간, Python overhead 등)
            # ==============================================================
            explained = (
                data_wait_excl
                + h2d_excl
                + compute_excl
                + nccl_excl
                + overlap_wait_compute
                + overlap_h2d_compute
                + overlap_compute_nccl
            )
            cpu_overhead = max(0.0, step_duration - explained)

            # ==============================================================
            # 결과 저장
            # ==============================================================
            all_result.append({
                "rank":  rank,
                "step":  step,

                # 전체 시간
                "step_duration_ms": step_duration,

                # 각 항목 전체 시간
                "data_wait_total_ms": data_wait_total,
                "h2d_total_ms":       h2d_total,
                "compute_total_ms":   compute_total,
                "nccl_total_ms":      nccl_total,

                # overlap 시간
                "overlap_wait_compute_ms":  overlap_wait_compute,
                "overlap_h2d_compute_ms":   overlap_h2d_compute,
                "overlap_compute_nccl_ms":  overlap_compute_nccl,

                # overlap 비율
                "overlap_wait_compute_ratio":  overlap_wait_compute_ratio,
                "overlap_h2d_compute_ratio":   overlap_h2d_compute_ratio,
                "overlap_compute_nccl_ratio":  overlap_compute_nccl_ratio,

                # 순수 시간 (overlap 제외)
                "data_wait_excl_ms": data_wait_excl,
                "h2d_excl_ms":       h2d_excl,
                "nccl_excl_ms":      nccl_excl,
                "compute_excl_ms":   compute_excl,

                # pure_wait: nccl 안에 포함된 시간
                # overlap_compute_nccl 또는 nccl_excl 안에 있음
                # 이중 계산 아님 ✅
                "pure_wait_ms":    pure_wait,

                # cpu_overhead
                "cpu_overhead_ms": cpu_overhead,
            })

    return pd.DataFrame(all_result)

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

def draw_bar(
    ax,
    rel_start_ms: float,
    dur_ms: float,
    y_pos: float,
    lane_height: float,
    rank_color: str,
    alpha: float = 0.9
    ):
    ax.broken_barh(
        [(rel_start_ms, dur_ms)],
        (y_pos, lane_height),
        facecolors=rank_color,
        alpha=alpha,
        linewidth=0.5,
        edgecolor='black'
    )

def plot_timeline_custom_axis(
    gpu_data_map,
    df_rank_order_per_step,
    all_steps_map,
    steps_to_plot,
    out_png="timeline.png",
    show=True,
    color_by="rank",
    offsets: Optional[Dict] = None
    ):
    # ==============================================================
    # Set up start line and end line from df_rank_order_per_step
    # ==============================================================
    target_stats = df_rank_order_per_step[
        df_rank_order_per_step["step"].isin(steps_to_plot)
    ]

    if target_stats.empty:
        print(f"[Error] No data for steps {steps_to_plot}")
        return

    global_start         = int(target_stats["earliest_start"].min())
    global_end           = int(target_stats["gpu_latest_end"].max())
    timeline_duration_ms = (global_end - global_start) / 1e6

    print(f"  -> global_start (CPU): {global_start}")
    print(f"  -> global_end   (GPU): {global_end}")
    print(f"  -> Total timeline:     {timeline_duration_ms:.2f} ms")

    # ==============================================================
    # Set up y axis layout
    # ==============================================================
    base_names   = ["data_wait", "h2d", "gpu_compute", "NCCL"]
    fixed_nvtx   = ["zero_grad", "forward", "loss", "backward", "opt_step"]
    full_y_names = list(dict.fromkeys(base_names + fixed_nvtx))
    y_map        = {name: i for i, name in enumerate(full_y_names)}

    # ==============================================================
    # Set up color and fig size
    # ==============================================================
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

    # ==============================================================
    # Set up lane height
    # ==============================================================
    lane_height  = 0.8 / len(gpu_data_map)
    sorted_ranks = sorted(gpu_data_map.keys())

    # ==============================================================
    # Filter all data in the target range and plot
    # ==============================================================
    for rank in sorted_ranks:

        dataset    = gpu_data_map[rank]
        rank_color = colors[rank % len(colors)]

        if rank not in all_steps_map:
            continue

        step_df_sel = all_steps_map[rank][
            all_steps_map[rank]["step"].isin(steps_to_plot)
        ]

        if step_df_sel.empty:
            continue

        # filter by step ranges
        df_nvtx = filter_by_step_ranges(
            dataset.df_nvtx, step_df_sel, global_start,
            use_data_batch_idx=True, rank=rank, offsets=offsets
        )
        df_nccl = filter_by_step_ranges(
            dataset.df_nccl, step_df_sel, global_start,
            rank=rank, offsets=offsets
        ) if not dataset.df_nccl.empty else pd.DataFrame()

        df_memcpy = filter_by_step_ranges(
            dataset.df_memcpy, step_df_sel, global_start,
            use_data_batch_idx=True, rank=rank, offsets=offsets
        ) if not dataset.df_memcpy.empty else pd.DataFrame()

        df_gpu_duration = filter_by_step_ranges(
            dataset.df_gpu_duration, step_df_sel, global_start,
            rank=rank, offsets=offsets
        ) if not dataset.df_gpu_duration.empty else pd.DataFrame()

        # ==============================================================
        # PLOT
        # ==============================================================
        rank_offset = rank * lane_height

        # [A] gpu_compute row
        # df_gpu_duration의 모든 이벤트를 gpu_compute row에 그리기
        if not df_gpu_duration.empty and "gpu_compute" in y_map:
            y_base = y_map["gpu_compute"]

            for _, row in df_gpu_duration.iterrows():
                data_batch_idx = int(row.get("data_batch_idx", row["step"]))
                step_num       = data_batch_idx
                bar_color      = get_color(rank, step_num)

                draw_bar(
                    ax=ax,
                    rel_start_ms=row["rel_start_ms"],
                    dur_ms=row["dur_ms"],
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
        # df_memcpy의 이벤트를 h2d row에 그리기
        if "h2d" in y_map:
            y_base    = y_map["h2d"]
            current_y = y_base + rank_offset

            if not df_memcpy.empty:
                for _, row in df_memcpy.iterrows():
                    data_batch_idx = int(row.get("data_batch_idx", row["step"]))
                    step_num       = data_batch_idx
                    bar_color      = get_color(rank, step_num)

                    draw_bar(
                        ax=ax,
                        rel_start_ms=row["rel_start_ms"],
                        dur_ms=row["dur_ms"],
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
        # df_nccl의 이벤트를 NCCL row에 그리기
        if not df_nccl.empty and "NCCL" in y_map:
            y_base    = y_map["NCCL"]
            current_y = y_base + rank_offset

            for _, row in df_nccl.iterrows():
                data_batch_idx = int(row.get("data_batch_idx", row["step"]))
                step_num       = data_batch_idx
                bar_color      = get_color(rank, step_num)

                draw_bar(
                    ax=ax,
                    rel_start_ms=row["rel_start_ms"],
                    dur_ms=row["dur_ms"],
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
        # df_nvtx의 CPU 이벤트를 각 row에 그리기
        # + df_gpu_duration에서 매핑된 GPU 이벤트도 같은 row에 그리기
        target_phases = [
            "cpu_data_wait_launch",
            "cpu_h2d_launch",
            "cpu_zero_grad_launch",
            "cpu_forward_launch",
            "cpu_loss_launch",
            "cpu_backward_launch",
            "cpu_opt_step_launch",
            "cpu_nccl_allreduce_launch",
            "cpu_train_compute_duration",
        ]

        # target_phases → y_map 매핑
        phase_to_ymap = {
            "cpu_data_wait_launch":      "data_wait",
            "cpu_h2d_launch":            "h2d",
            "cpu_zero_grad_launch":      "zero_grad",
            "cpu_forward_launch":        "forward",
            "cpu_loss_launch":           "loss",
            "cpu_backward_launch":       "backward",
            "cpu_opt_step_launch":       "opt_step",
            "cpu_nccl_allreduce_launch": "NCCL",
            "cpu_train_compute_duration":"gpu_compute",
        }

        # gpu_duration → y_map 매핑
        gpu_to_ymap = {
            "gpu_forward_duration":         "forward",
            "gpu_backward_duration":        "backward",
            "gpu_loss_duration":            "loss",
            "gpu_opt_step_duration":        "opt_step",
            "gpu_zero_grad_duration":       "zero_grad",
            "gpu_train_compute_duration":   "gpu_compute",
            "gpu_nccl_allreduce_duration":  "NCCL",
        }

        if not df_nvtx.empty:
            for target_name in target_phases:

                mask     = df_nvtx["name"] == target_name
                cpu_rows = df_nvtx[mask]
                if cpu_rows.empty:
                    continue

                ymap_key = phase_to_ymap.get(target_name)
                if ymap_key is None or ymap_key not in y_map:
                    continue

                base_y    = y_map[ymap_key]
                current_y = base_y + rank_offset

                # CPU 이벤트 그리기 (투명하게)
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
                        ha='center', va='center',
                        fontsize=7, fontweight='bold',
                        color='black', clip_on=True
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

                # GPU 이벤트 그리기 (같은 row에 진하게)
                if not df_gpu_duration.empty:
                    gpu_name = next(
                        (k for k, v in gpu_to_ymap.items() if v == ymap_key),
                        None
                    )
                    if gpu_name:
                        gpu_rows = df_gpu_duration[
                            df_gpu_duration["name"] == gpu_name
                        ]
                        for _, gpu_row in gpu_rows.iterrows():
                            data_batch_idx = int(gpu_row.get("data_batch_idx", gpu_row["step"]))
                            step_num       = data_batch_idx
                            bar_color      = get_color(rank, step_num)

                            draw_bar(
                                ax=ax,
                                rel_start_ms=gpu_row["rel_start_ms"],
                                dur_ms=gpu_row["dur_ms"],
                                y_pos=current_y,
                                lane_height=lane_height,
                                rank_color=bar_color,
                                alpha=1.0
                            )
                            ax.text(
                                x=gpu_row["rel_start_ms"] + gpu_row["dur_ms"] / 2,
                                y=current_y + lane_height / 2,
                                s=f"S{step_num}",
                                ha='center', va='center',
                                fontsize=7, fontweight='bold',
                                color='white', clip_on=True
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

    # ==============================================================
    # Step boundary lines
    # ==============================================================
    for _, srow in target_stats.iterrows():
        # CPU step 시작 점선 (검정)
        x_cpu_start = (srow["earliest_start"] - global_start) / 1e6
        ax.axvline(x_cpu_start, color='k', ls='--', alpha=0.5)

        # CPU step 끝 실선 (검정)
        x_cpu_end = (srow["latest_end"] - global_start) / 1e6
        ax.axvline(x_cpu_end, color='k', ls='solid', alpha=0.5)

        # GPU step 끝 실선 (파랑)
        x_gpu_end = (srow["gpu_latest_end"] - global_start) / 1e6
        ax.axvline(x_gpu_end, color='blue', ls='solid', alpha=0.3, linewidth=1.5)

        ax.text(
            (x_cpu_start + x_cpu_end) / 2,
            len(full_y_names),
            f"Step {int(srow['step'])}",
            ha='center', va='bottom', weight='bold'
        )

    # ==============================================================
    # Axis settings
    # ==============================================================
    ax.set_xlim(0, timeline_duration_ms)
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
    gpu_title_str = ", ".join([
        f"{name} ({count} GPUs)"
        for name, count in gpu_counts.items()
    ]) if gpu_counts else "Unknown GPU"

    ax.set_title(
        f"[{gpu_title_str}] Multi-GPU Detailed Timeline | Steps: {steps_to_plot}",
        fontsize=14, fontweight='bold'
    )

    ax.grid(True, axis='x', linestyle=':', alpha=0.5)
    ax.grid(False, axis='y')

    # ==============================================================
    # Legend
    # ==============================================================
    legend_elements = []
    for r in sorted_ranks:
        dataset  = gpu_data_map[r]
        gpu_name = next(iter(dataset.gpu_info.values())) if dataset.gpu_info else "Unknown GPU"
        legend_elements.append(Patch(
            facecolor=colors[r % len(colors)],
            label=f'Rank {r} ({gpu_name})'
        ))

    legend_elements.append(Patch(
        facecolor='none', edgecolor='black',
        label='CPU step start (dashed) / end (solid)'
    ))
    legend_elements.append(Patch(
        facecolor='none', edgecolor='blue',
        label='GPU step end (blue line)'
    ))

    ax.legend(
        handles=legend_elements,
        loc='upper right',
        title="GPU Ranks",
        bbox_to_anchor=(1.05, 1)
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved timeline to {out_png}")

    if show:
        plt.show()

def plot_step_breakdown(
    df_breakdown: pd.DataFrame,
    output_prefix: str = "step_breakdown"
):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    # ==============================================================
    # 항목 정의
    # ==============================================================
    plot_cols = [
        "data_wait_excl_ms",
        "h2d_excl_ms",
        "compute_excl_ms",
        "overlap_wait_compute_ms",
        "overlap_h2d_compute_ms",
        "overlap_compute_nccl_ms",
        "nccl_excl_ms",
        "pure_wait_ms",
        "cpu_overhead_ms",
    ]

    col_labels = {
        "data_wait_excl_ms":       "Data Wait (excl)",
        "h2d_excl_ms":             "H2D (excl)",
        "compute_excl_ms":         "GPU Compute (excl)",
        "overlap_wait_compute_ms": "Overlap (Wait+Compute)",
        "overlap_h2d_compute_ms":  "Overlap (H2D+Compute)",
        "overlap_compute_nccl_ms": "Overlap (Compute+NCCL)",
        "nccl_excl_ms":            "NCCL (excl)",
        "pure_wait_ms":            "Pure Wait (straggler)",
        "cpu_overhead_ms":         "CPU Overhead",
    }

    colors = [
        '#1f77b4',  # data_wait_excl    파랑
        '#ff7f0e',  # h2d_excl          주황
        '#2ca02c',  # compute_excl      초록
        '#aec7e8',  # overlap_wait_compute 연파랑
        '#ffbb78',  # overlap_h2d_compute  연주황
        '#98df8a',  # overlap_compute_nccl 연초록
        '#d62728',  # nccl_excl         빨강
        '#e377c2',  # pure_wait         분홍
        '#7f7f7f',  # cpu_overhead      회색
    ]

    # ==============================================================
    # x축 레이블: R{rank}_S{step}
    # ==============================================================
    df_breakdown = df_breakdown.copy()
    df_breakdown["label"] = (
        "R" + df_breakdown["rank"].astype(str) +
        "_S" + df_breakdown["step"].astype(str)
    )
    df_breakdown = df_breakdown.sort_values(["step", "rank"])
    df_breakdown = df_breakdown.set_index("label")

    df_plot = df_breakdown[plot_cols].rename(columns=col_labels)

    step_totals = df_breakdown["step_duration_ms"]

    # ==============================================================
    # 1. Stacked Bar Chart
    # ==============================================================
    fig, ax = plt.subplots(figsize=(max(12, len(df_plot) * 0.8), 7))

    df_plot.plot(
        kind='bar',
        stacked=True,
        color=colors,
        edgecolor='black',
        linewidth=0.5,
        ax=ax
    )

    # 각 bar에 퍼센트 표시
    for n, label in enumerate(df_plot.index):
        total  = step_totals.loc[label]
        cum_val = 0
        for col_idx, col in enumerate(df_plot.columns):
            val = df_plot.loc[label, col]
            if val > 0:
                percent = (val / total) * 100
                y_pos   = cum_val + (val / 2)
                cum_val += val
                if percent >= 3.0:
                    ax.text(
                        n, y_pos,
                        f'{percent:.1f}%',
                        ha='center', va='center',
                        color='white', fontweight='bold', fontsize=8,
                        path_effects=[
                            path_effects.withStroke(
                                linewidth=2, foreground='black'
                            )
                        ]
                    )

    # step 경계선
    steps = df_breakdown["step"].unique()
    ranks = df_breakdown["rank"].unique()
    n_ranks = len(ranks)
    for i in range(1, len(steps)):
        ax.axvline(
            x=i * n_ranks - 0.5,
            color='black',
            linewidth=1.5,
            linestyle='--',
            alpha=0.5
        )
        ax.text(
            (i - 0.5) * n_ranks - 0.5,
            ax.get_ylim()[1] * 0.98,
            f"Step {steps[i-1]}",
            ha='center', va='top',
            fontsize=9, fontweight='bold'
        )

    ax.set_title(
        "Per-Step Time Breakdown\n"
        "(excl = overlap 제외 순수 시간, overlap = 동시 실행 구간)",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel("Rank_Step", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_xticklabels(df_plot.index, rotation=45, ha='right', fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1],
        title="Operations",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=9
    )

    plt.tight_layout()
    stacked_out = f"{output_prefix}_stacked.png"
    plt.savefig(stacked_out, dpi=200, bbox_inches='tight')
    print(f"Stacked bar saved: {stacked_out}")
    plt.show()
    plt.close()

    # ==============================================================
    # 2. Grouped Bar Chart (Log Scale)
    # rank별 비교
    # ==============================================================
    fig, ax = plt.subplots(figsize=(max(14, len(df_plot) * 0.8), 6))

    df_plot.plot(
        kind='bar',
        stacked=False,
        logy=True,
        color=colors,
        edgecolor='black',
        linewidth=0.5,
        ax=ax
    )

    ax.set_title(
        "Step Execution Time Components (Log Scale)",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel("Rank_Step", fontsize=11)
    ax.set_ylabel("Time (ms) [Log Scale]", fontsize=11)
    ax.set_xticklabels(df_plot.index, rotation=45, ha='right', fontsize=9)
    ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.legend(
        title="Operations",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=9
    )

    plt.tight_layout()
    grouped_out = f"{output_prefix}_grouped_log.png"
    plt.savefig(grouped_out, dpi=200, bbox_inches='tight')
    print(f"Grouped bar saved: {grouped_out}")
    plt.show()
    plt.close()

    # ==============================================================
    # 3. overlap 비율 Line Chart
    # ==============================================================
    fig, ax = plt.subplots(figsize=(max(12, len(df_plot) * 0.8), 5))

    ratio_cols = {
        "overlap_wait_compute_ratio": "Wait+Compute overlap ratio",
        "overlap_h2d_compute_ratio":  "H2D+Compute overlap ratio",
        "overlap_compute_nccl_ratio": "Compute+NCCL overlap ratio",
    }

    ratio_colors = ['#aec7e8', '#ffbb78', '#98df8a']

    for (col, label), color in zip(ratio_cols.items(), ratio_colors):
        if col in df_breakdown.columns:
            ax.plot(
                df_breakdown.index,
                df_breakdown[col] * 100,
                marker='o',
                label=label,
                color=color,
                linewidth=2
            )

    ax.set_title(
        "Overlap Ratios per Rank/Step (%)",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel("Rank_Step", fontsize=11)
    ax.set_ylabel("Overlap Ratio (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_xticklabels(df_breakdown.index, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100%')
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=9
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    ratio_out = f"{output_prefix}_overlap_ratio.png"
    plt.savefig(ratio_out, dpi=200, bbox_inches='tight')
    print(f"Overlap ratio chart saved: {ratio_out}")
    plt.show()
    plt.close()


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
    use_data_batch_idx: bool = False,
    rank: int = 0,
    offsets: Optional[Dict] = None
    ) -> pd.DataFrame:
    """
    Filter events in source_df by step range.

    use_data_batch_idx=True  -> filter by data_batch_idx column
                                NVTX, memcpy에 사용
    use_data_batch_idx=False -> data_batch_idx 있으면 사용
                                없으면 time range fallback
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

        if use_data_batch_idx and "data_batch_idx" in source_df.columns:
            d = source_df[source_df["data_batch_idx"] == s].copy()

        elif "data_batch_idx" in source_df.columns:
            d = source_df[source_df["data_batch_idx"] == s].copy()

        else:
            # fallback: 시간 범위로 필터링
            start_col_src = next(
                (c for c in ["start", "kernel_start", "gpu_start"]
                 if c in source_df.columns), None
            )
            if start_col_src is None:
                continue
            d = source_df[
                (source_df[start_col_src] >= s_start) &
                (source_df[start_col_src] <  s_end)
            ].copy()

        if d.empty:
            continue

        d["step"] = s
        rows.append(d)

    if not rows:
        return pd.DataFrame()

    filtered_df = pd.concat(rows, ignore_index=True)

    # 컬럼명 동적 탐색
    start_col = next(
        (c for c in ["start", "kernel_start", "gpu_start"]
         if c in filtered_df.columns), None
    )
    end_col = next(
        (c for c in ["end", "kernel_end", "gpu_end"]
         if c in filtered_df.columns), None
    )

    if start_col is None or end_col is None:
        return pd.DataFrame()

    filtered_df = filtered_df.drop_duplicates(
        subset=["name", start_col, end_col]
    )

    filtered_df["rel_start_ms"] = (
        filtered_df[start_col] - adjusted_global_start
    ) / 1e6

    filtered_df["dur_ms"] = (
        filtered_df[end_col] - filtered_df[start_col]
    ) / 1e6

    return filtered_df

    """
    공통 설정:
      rank = 2
      offset = +103000  (rank 2 시계가 rank 0보다 103ms 느림)
      global_start = 13800000  (rank 0 시계 기준)
      adjusted_global_start = 13800000 - 103000 = 13697000  (rank 2 시계 기준)

      step_df_sel:
        step  start      end
        0     13800000   14200000   ← rank 0 시계 기준
        1     14200000   14600000

    =====================================================================
    예시 1: df_nvtx (use_data_batch_idx=True)
    =====================================================================
    source_df (df_nvtx, rank 2):
      name                          start      end        data_batch_idx
      cpu_batch_0_duration          13697000   14097000   0
      cpu_forward_launch            13697210   13697350   0
      cpu_backward_launch           13697360   13697480   0
      cpu_nccl_allreduce_launch     13697460   13697475   0
      cpu_batch_1_duration          14097000   14497000   1
      cpu_forward_launch            14097210   14097350   1
      cpu_backward_launch           14097360   14097480   1

    use_data_batch_idx=True
    → data_batch_idx 기준 필터링

    s=0: d = rows where data_batch_idx==0
      name                          start      end        data_batch_idx  step
      cpu_batch_0_duration          13697000   14097000   0               0
      cpu_forward_launch            13697210   13697350   0               0
      cpu_backward_launch           13697360   13697480   0               0
      cpu_nccl_allreduce_launch     13697460   13697475   0               0

    s=1: d = rows where data_batch_idx==1
      name                          start      end        data_batch_idx  step
      cpu_batch_1_duration          14097000   14497000   1               1
      cpu_forward_launch            14097210   14097350   1               1
      cpu_backward_launch           14097360   14097480   1               1

    rel_start_ms 계산 (adjusted_global_start=13697000):
      cpu_batch_0_duration:  (13697000 - 13697000) / 1e6 = 0ms
      cpu_forward_launch:    (13697210 - 13697000) / 1e6 = 0.00021ms
      cpu_backward_launch:   (13697360 - 13697000) / 1e6 = 0.00036ms

    최종 filtered df_nvtx:
      name                      start      end        step  rel_start_ms  dur_ms
      cpu_batch_0_duration      13697000   14097000   0     0ms           0.4ms
      cpu_forward_launch        13697210   13697350   0     0.00021ms     0.00014ms
      cpu_backward_launch       13697360   13697480   0     0.00036ms     0.00012ms
      cpu_nccl_allreduce_launch 13697460   13697475   0     0.00046ms     0.000015ms
      cpu_batch_1_duration      14097000   14497000   1     0.4ms         0.4ms
      cpu_forward_launch        14097210   14097350   1     0.40021ms     0.00014ms
      cpu_backward_launch       14097360   14097480   1     0.40036ms     0.00012ms

    =====================================================================
    예시 2: df_nccl (use_data_batch_idx=False)
    =====================================================================
    source_df (df_nccl, rank 2):
      name                        kernel_start  kernel_end  dur_ns  correlation_id  data_batch_idx
      ncclAllReduceRingLLKernel   13697420      13697480    60      3001            0
      ncclAllReduceRingLLKernel   13697850      13697910    60      3002            0
      ncclAllReduceRingLLKernel   14097280      14097340    60      3003            1

    use_data_batch_idx=False
    → data_batch_idx 컬럼 있으므로 data_batch_idx 기준 필터링

    s=0: d = rows where data_batch_idx==0
      name                        kernel_start  kernel_end  data_batch_idx  step
      ncclAllReduceRingLLKernel   13697420      13697480    0               0
      ncclAllReduceRingLLKernel   13697850      13697910    0               0

    s=1: d = rows where data_batch_idx==1
      name                        kernel_start  kernel_end  data_batch_idx  step
      ncclAllReduceRingLLKernel   14097280      14097340    1               1

    start_col = "kernel_start"
    end_col   = "kernel_end"

    rel_start_ms 계산 (adjusted_global_start=13697000):
      ncclAllReduceRingLLKernel (420):  (13697420 - 13697000) / 1e6 = 0.00042ms
      ncclAllReduceRingLLKernel (850):  (13697850 - 13697000) / 1e6 = 0.00085ms

    최종 filtered df_nccl:
      name                        kernel_start  kernel_end  step  rel_start_ms  dur_ms
      ncclAllReduceRingLLKernel   13697420      13697480    0     0.00042ms     0.00006ms
      ncclAllReduceRingLLKernel   13697850      13697910    0     0.00085ms     0.00006ms
      ncclAllReduceRingLLKernel   14097280      14097340    1     0.40028ms     0.00006ms

    =====================================================================
    예시 3: df_memcpy (use_data_batch_idx=True)
    =====================================================================
    source_df (df_memcpy, rank 2):
      name               kernel_start  kernel_end  dur_ns  correlation_id  data_batch_idx
      gpu_h2d_duration   13697150      13697160    10      4001            0
      gpu_h2d_duration   13697160      13697168    8       4002            0
      gpu_h2d_duration   13697480      13697490    10      4004            1   ← GPU step 0에서 실행
      gpu_h2d_duration   13697490      13697498    8       4005            1     but data_batch_idx=1
      gpu_h2d_duration   14097960      14097970    10      4006            2

    use_data_batch_idx=True
    → data_batch_idx 기준 필터링
    → GPU step 0에서 실행됐어도 data_batch_idx=1이면 step 1로 분류 ✅

    s=0: d = rows where data_batch_idx==0
      name               kernel_start  kernel_end  data_batch_idx  step
      gpu_h2d_duration   13697150      13697160    0               0
      gpu_h2d_duration   13697160      13697168    0               0

    s=1: d = rows where data_batch_idx==1
      name               kernel_start  kernel_end  data_batch_idx  step
      gpu_h2d_duration   13697480      13697490    1               1   ← GPU step 0이지만
      gpu_h2d_duration   13697490      13697498    1               1     step=1로 분류 ✅

    start_col = "kernel_start"
    end_col   = "kernel_end"

    최종 filtered df_memcpy:
      name               kernel_start  kernel_end  step  rel_start_ms  dur_ms
      gpu_h2d_duration   13697150      13697160    0     0.00015ms     0.00001ms
      gpu_h2d_duration   13697160      13697168    0     0.00016ms     0.000008ms
      gpu_h2d_duration   13697480      13697490    1     0.00048ms     0.00001ms
      gpu_h2d_duration   13697490      13697498    1     0.00049ms     0.000008ms

    =====================================================================
    예시 4: df_gpu_duration (use_data_batch_idx=False)
    =====================================================================
    source_df (df_gpu_duration, rank 2):
      name                        gpu_start  gpu_end   dur_ns  data_batch_idx
      gpu_batch_0_duration        13697300   13697950  600     0
      gpu_forward_duration        13697310   13697550  180     0
      gpu_backward_duration       13697560   13697850  200     0
      gpu_opt_step_duration       13697850   13697950  80      0
      gpu_nccl_allreduce_duration 13697760   13697820  60      0
      gpu_batch_1_duration        13697950   14097600  600     1
      gpu_forward_duration        13697960   14097200  180     1
      gpu_backward_duration       14097210   14097500  200     1

    use_data_batch_idx=False
    → data_batch_idx 컬럼 있으므로 data_batch_idx 기준 필터링

    s=0: d = rows where data_batch_idx==0
      name                        gpu_start  gpu_end   data_batch_idx  step
      gpu_batch_0_duration        13697300   13697950  0               0
      gpu_forward_duration        13697310   13697550  0               0
      gpu_backward_duration       13697560   13697850  0               0
      gpu_opt_step_duration       13697850   13697950  0               0
      gpu_nccl_allreduce_duration 13697760   13697820  0               0

    start_col = "gpu_start"
    end_col   = "gpu_end"

    rel_start_ms 계산 (adjusted_global_start=13697000):
      gpu_batch_0_duration:  (13697300 - 13697000) / 1e6 = 0.0003ms
      gpu_forward_duration:  (13697310 - 13697000) / 1e6 = 0.00031ms
      gpu_backward_duration: (13697560 - 13697000) / 1e6 = 0.00056ms

    최종 filtered df_gpu_duration:
      name                        gpu_start  gpu_end   step  rel_start_ms  dur_ms
      gpu_batch_0_duration        13697300   13697950  0     0.0003ms      0.00065ms
      gpu_forward_duration        13697310   13697550  0     0.00031ms     0.00024ms
      gpu_backward_duration       13697560   13697850  0     0.00056ms     0.00029ms
      gpu_opt_step_duration       13697850   13697950  0     0.00085ms     0.0001ms
      gpu_nccl_allreduce_duration 13697760   13697820  0     0.00076ms     0.00006ms
      gpu_batch_1_duration        13697950   14097600  1     0.0009ms      0.4ms
      gpu_forward_duration        13697960   14097200  1     0.00096ms     0.4ms
      gpu_backward_duration       14097210   14097500  1     0.40021ms     0.00029ms
    """

def process_full_analysis(con: sqlite3.Connection,
                          steps: List[int]
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    nvtx_sch = find_nvtx_schema(con)
    k_sch    = find_kernel_schema(con)
    r_sch    = find_runtime_schema(con)
    m_sch    = find_memcpy_schema(con)
    s_sch    = find_stringids_schema(con)

    # ==========================================================================
    # 1. NVTX 이벤트 로드 (전체)
    # ==========================================================================
    df_nvtx = load_nvtx_events(con, nvtx_sch) if nvtx_sch else pd.DataFrame()

    # ==========================================================================
    # 2. NCCL 커널 로드(전체)
    # ==========================================================================
    df_nccl = pd.DataFrame()
    if k_sch and s_sch:
        df_nccl = load_nccl_kernels(con, k_sch, s_sch)

    # ==========================================================================
    # 3. Memcpy 이벤트 로드
    # ==========================================================================
    df_memcpy = pd.DataFrame()
    if m_sch:
        df_memcpy = load_memcpy_events(con, m_sch, want_kinds=[1], name="gpu_h2d_duration")

    # ==========================================================================
    # 4. True GPU Span 계산을 위한 준비
    # ==========================================================================

    df_gpu_duration = pd.DataFrame()

    if nvtx_sch and r_sch and k_sch and not df_nvtx.empty:
        try:

            # ======================================================================
            # Step 1: CPU step 범위 추출
            # ======================================================================
            # nvtx_df에서 "cpu_batch_N_duration" 패턴을 찾아
            # 각 step의 CPU 시작/끝 시간 추출
            #
            # 예시:
            #   nvtx_df:
            #     name                    start   end
            #     cpu_batch_0_duration    100     500
            #     cpu_batch_1_duration    500     900
            #     cpu_batch_2_duration    900     1300
            #
            #    df_step_ranges_in_nvtx 결과:
            #     step  start   end
            #     0     100     500    ← CPU 시계
            #     1     500     900
            #     2     900     1300
            df_step_ranges_in_nvtx  = build_step_df_from_nvtx(df_nvtx) 

            # ======================================================================
            # Step 2: NVTX 범위 안의 Runtime API 호출 로드
            # ======================================================================
            # CPU가 GPU kernel을 launch할 때 호출하는 Runtime API
            # (cudaLaunchKernel, cudaMemcpyAsync 등)
            # correlation_id로 GPU kernel과 연결됨
            #
            # 예시:
            #   df_runtime_api_in_nvtx:
            #     correlation_id     cpu_api_start      cpu_api_end
            #     1001              210                 211    ← cpu_forward_launch 범위 안
            #     1002              211                 212    ← cpu_forward_launch 범위 안
            #     1003              360                 361    ← cpu_backward_launch 범위 안
            #     ...
            #
            # 주의:
            #   start/end가 매우 짧음 (수 µs)
            #   CPU가 GPU stream에 명령만 내리는 시간
            df_runtime_api_in_nvtx     = load_runtime_events_in_nvtx(con, df_nvtx, r_sch)

            # ======================================================================
            # Step 3: 모든 GPU kernel 로드 + 커널이름 넣어주기
            # ======================================================================
            # GPU에서 실제로 실행된 모든 kernel + 커널 이름
            # 나중에 correlation_id로 Runtime API와 연결됨
            #
            # 예시:
            #   df_kernels_all:
            #     correlation_id   k_start   k_end    name
            #     1001            300       400      volta_sgemm_...  ← forward kernel
            #     1002            400       500      volta_sgemm_...  ← forward kernel
            #     1003            550       650      volta_sgemm_...  ← backward kernel
            #     ...
            #
            # 주의:
            #   k_start/k_end = GPU 시계 기준
            #   Runtime API start(CPU 시계)보다 나중에 실행됨

            df_kernels_all= load_gpu_kernel_timings(con, k_sch, s_sch=s_sch)

            # ======================================================================
            # Step 4: NVTX 범위 → correlation_id 매핑
            # ======================================================================
            # 각 NVTX 범위 안에서 CPU가 launch한 GPU kernel의
            # correlation_id를 매핑
            #
            # 예시:
            #   df_map_nvtx_to_cpu_api:
            #     nvtx_name            nvtx_start  nvtx_end  correlation_id
            #     cpu_forward_launch   210         350       1001
            #     cpu_forward_launch   210         350       1002
            #     cpu_forward_launch   210         350       1003
            #     cpu_backward_launch  360         480       2001
            #     cpu_backward_launch  360         480       2002
            #     cpu_nccl_allreduce_launch 460    475       3001
            #
            # 주의:
            #   cpu_batch_0_duration, cpu_train_compute_duration 같은
            #   wrapper NVTX는 하위 모든 kernel의 correlation_id 포함
            df_map_nvtx_to_cpu_api= map_nvtx_to_runtime(df_nvtx, df_runtime_api_in_nvtx)

            # ======================================================================
            # Step 5: NVTX 범위와 GPU kernel 시간 join
            # ======================================================================
            # df_map_nvtx_to_cpu_api (NVTX → cpu_launch_correlation_id) 와
            # df_kernels_all (correlation_id → GPU kernel 시간) 을 join하여
            # 각 NVTX 범위의 실제 GPU 실행 시간 계산
            #
            # 예시:
            #   df_map_nvtx_to_cpu_api:
            #     cpu_launch_in_nvtx   cpu_launch_correlation_id
            #     cpu_forward_launch   1001
            #     cpu_forward_launch   1002
            #     cpu_backward_launch  2001
            #
            #   df_kernels_all:
            #     correlation_id  k_start  k_end
            #     1001           310      400
            #     1002           450      550   ← gap 400~450
            #     2001           560      650
            #
            # 결과 df_gpu_duration:
            #   name                      gpu_start  gpu_end  dur_ns  data_batch_idx
            #   gpu_batch_0_duration      300        950      600     0
            #   gpu_forward_duration      310        550      180     0   ← gap 제외
            #   gpu_backward_duration     560        850      200     0
            #   gpu_opt_step_duration     850        950      80      0
            df_gpu_duration =  join_cpu_api_in_nvtx_with_gpu_kernel_timings(df_map_nvtx_to_cpu_api, df_kernels_all, df_step_ranges_in_nvtx= df_step_ranges_in_nvtx)

            if df_gpu_duration.empty:
                raise ValueError("true_gpu_df is empty")


            # gpu_batch_rows = df_true_gpu[
            #     df_true_gpu["name"].str.contains(r"\[GPU\] Batch", regex=True)
            # ]

            # if gpu_batch_rows.empty:
            #     raise ValueError("No [GPU] Batch_N rows found -> check NVTX markers for Batch_N")

            # gpu_step_df = pd.DataFrame({
            #     "step":  gpu_batch_rows["cpu_step"].values,
            #     "start": gpu_batch_rows["start"].values,
            #     "end":   gpu_batch_rows["end"].values
            # }).sort_values("step").reset_index(drop=True)
            # ======================================================================
            # Step 6: data_batch_idx 할당
            # ======================================================================
            # CPU 시계 기준 step 범위(df_step_ranges_in_nvtx)와
            # 각 이벤트의 start 시간을 비교하여 data_batch_idx 할당
            #
            # 예시:
            #   df_step_ranges_in_nvtx:
            #     step  start   end
            #     0     100     500   ← CPU 시계
            #     1     500     900
            #
            #   df_nvtx:
            #     name                  start   → data_batch_idx
            #     cpu_forward_launch    210     → 0  (100<=210<500)
            #     cpu_backward_launch   360     → 0  (100<=360<500)
            #     cpu_forward_launch    600     → 1  (500<=600<900)
            #
            #   df_nccl:
            #     name                      start   → data_batch_idx
            #     ncclAllReduceRingLL       420     → 0  (100<=420<500)  ← GPU 시계지만
            #                                                               CPU step 범위와 비교
            #
            #   df_memcpy:
            #     name        start   → data_batch_idx
            #     gpu_h2d     110     → 0  (100<=110<500)  ← step 0 데이터 전송
            #     gpu_h2d     480     → 0  (100<=480<500)  ← step 1 데이터지만
            #                                                 GPU에서 step 0 구간에 실행
            #                                                 → cpu_step 기준이므로 0
            #
            # 주의:
            #   df_nccl, df_memcpy는 GPU 시계 기준이지만
            #   data_batch_idx는 CPU step 범위와 비교
            #   → 노드간 clock offset이 있는 경우 부정확할 수 있음
            #   → 같은 노드 내에서는 문제없음


            def assign_data_batch_idx(event_start):
                for _, row in df_step_ranges_in_nvtx.iterrows():
                    if row["start"] <= event_start < row["end"]:
                        return int(row["step"])
                return -1  # 범위 밖 → 해당 step 없음


            if not df_nvtx.empty:
                df_nvtx["data_batch_idx"] = df_nvtx["start"].apply(assign_data_batch_idx)

            # df_nvtx (cpu_nccl_allreduce_launch만 필터링):
            # name                       start   end     data_batch_idx
            # cpu_nccl_allreduce_launch  460     475     0    ← step 0 bucket 0
            # cpu_nccl_allreduce_launch  820     835     0    ← step 0 bucket 1
            # cpu_nccl_allreduce_launch  1260    1275    1    ← step 1 bucket 0   

            if not df_nccl.empty and not df_nvtx.empty:

                 # cpu_nccl_allreduce_launch 범위 안의 correlation_id 찾기
                nvtx_nccl = df_nvtx[
                    df_nvtx["name"] == "cpu_nccl_allreduce_launch"
                ].sort_values("start")

                # df_map_nvtx_to_cpu_api:
                # cpu_launch_in_nvtx         cpu_launch_start  cpu_launch_correlation_id
                # cpu_forward_launch         210               1001
                # cpu_backward_launch        360               2001
                # cpu_nccl_allreduce_launch  460               3001    ← step 0 bucket 0
                # cpu_nccl_allreduce_launch  820               3002    ← step 0 bucket 1
                # cpu_nccl_allreduce_launch  1260              3003    ← step 1 bucket 0

                # Step 1: cpu_nccl_allreduce_launch만 필터링
                # nvtx_nccl:
                #     name                       start   end     data_batch_idx
                    # cpu_nccl_allreduce_launch  460     475     0
                    # cpu_nccl_allreduce_launch  820     835     0
                    # cpu_nccl_allreduce_launch  1260    1275    1

                nccl_corr_mapping = df_map_nvtx_to_cpu_api[
                    df_map_nvtx_to_cpu_api["cpu_launch_in_nvtx"] == "cpu_nccl_allreduce_launch"
                ][["cpu_launch_in_nvtx", "cpu_launch_start", "correlation_id"]]

                # nccl_corr_mapping (df_map_nvtx_to_cpu_api에서 필터링):
                # cpu_launch_in_nvtx         cpu_launch_start  cpu_launch_correlation_id
                # cpu_nccl_allreduce_launch  460               3001   ← nvtx_nccl start=460과 매칭
                # cpu_nccl_allreduce_launch  820               3002   ← nvtx_nccl start=820과 매칭
                # cpu_nccl_allreduce_launch  1260              3003   ← nvtx_nccl start=1260과 매칭

                nccl_corr_mapping = nccl_corr_mapping.merge(
                    nvtx_nccl[["start", "data_batch_idx"]],
                    left_on="cpu_launch_start",
                    right_on="start",
                    how="left"
                )

                # df_nccl에 data_batch_idx 할당
                df_nccl = df_nccl.merge(
                    nccl_corr_mapping[["correlation_id", "data_batch_idx"]],
                    left_on="correlation_id",
                    right_on="correlation_id",
                    how="left"
                )
            
                # df_nccl 최종:

                # name                        kernel_start  kernel_end  dur_ns  correlation_id  data_batch_idx
                # ncclAllReduceRingLLKernel   420               480             60      3001           0
                # ncclAllReduceRingLLKernel   850               910             60      3002           0
                # ncclAllReduceRingLLKernel   1280              1340            60      3003           1
                


            if not df_memcpy.empty:

                # =========================================================================
                # memcpy data_batch_idx 할당
                # =========================================================================
                # GPU 시계 기준인 kernel_start로는 CPU step 범위와 비교 불가
                # → cpu_h2d_launch NVTX의 correlation_id로 매핑하여
                #   CPU 시계 기준 data_batch_idx 할당
                #
                # 예시:
                #   nvtx_h2d:
                #     name              start   data_batch_idx
                #     cpu_h2d_launch    110     0
                #     cpu_h2d_launch    500     1
                #     cpu_h2d_launch    900     2
                #
                #   h2d_corr_mapping:
                #     cpu_launch_start  cpu_launch_correlation_id  data_batch_idx
                #     110               4001                       0
                #     500               4004                       1
                #     900               4006                       2
                #
                #   df_memcpy 최종:
                #     name               kernel_start  correlation_id  data_batch_idx
                #     gpu_h2d_duration   150           4001           0   ← GPU step 0에서 실행
                #     gpu_h2d_duration   480           4004           1   ← GPU step 0이지만
                #                                                          data_batch_idx=1 ✅
                
                # cpu_h2d_launch NVTX 필터링
                nvtx_h2d = df_nvtx[
                    df_nvtx["name"] == "cpu_h2d_launch"
                ].sort_values("start").reset_index(drop=True)

                # df_map_nvtx_to_cpu_api에서
                # cpu_h2d_launch에 해당하는 correlation_id 추출
                h2d_corr_mapping = df_map_nvtx_to_cpu_api[
                    df_map_nvtx_to_cpu_api["cpu_launch_in_nvtx"] == "cpu_h2d_launch"
                ][["cpu_launch_start", "correlation_id"]].merge(
                    nvtx_h2d[["start", "data_batch_idx"]],
                    left_on="cpu_launch_start",
                    right_on="start",
                    how="left"
                )
                # df_memcpy에 data_batch_idx 할당
                df_memcpy = df_memcpy.merge(
                    h2d_corr_mapping[["correlation_id", "data_batch_idx"]],
                    left_on="correlation_id",
                    right_on="correlation_id",
                    how="left"
                )


        except Exception as e:
            print(f"  [Warning] Failed: {e}")
            import traceback
            traceback.print_exc()

    return df_nvtx, df_nccl, df_memcpy, df_gpu_duration


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
    #==============================================================================
    # 9. Fetch data from sqlite
    # ==============================================================================

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

        for rank, dataset in gpu_data_map.items():
            print(f"\nRank {rank}:")
            print(f"  df_memcpy shape: {dataset.df_memcpy.shape}")
            print(f"  df_memcpy columns: {dataset.df_memcpy.columns.tolist()}")
            if not dataset.df_memcpy.empty:
                print(f"  data_batch_idx null: {dataset.df_memcpy['data_batch_idx'].isna().sum()}")
                print(f"  kind values: {dataset.df_memcpy['kind'].value_counts()}")


        for rank, dataset in gpu_data_map.items():
            print(f"\nRank {rank}:")
            print(f"  df_memcpy total:          {len(dataset.df_memcpy)}")
            print(f"  cpu_h2d_launch in nvtx:   {(dataset.df_nvtx['name'] == 'cpu_h2d_launch').sum()}")
            
            # h2d_corr_mapping 확인
            nvtx_h2d = dataset.df_nvtx[dataset.df_nvtx["name"] == "cpu_h2d_launch"]
            print(f"  nvtx_h2d rows:            {len(nvtx_h2d)}")
            print(f"  df_memcpy NaN:            {dataset.df_memcpy['data_batch_idx'].isna().sum()}")
            break
        
        rank0 = gpu_data_map[0]
        nan_memcpy = rank0.df_memcpy[rank0.df_memcpy["data_batch_idx"].isna()]
        valid_memcpy = rank0.df_memcpy[rank0.df_memcpy["data_batch_idx"].notna()]

        print(f"NaN memcpy dur_ns 평균:   {nan_memcpy['dur_ns'].mean():.0f}ns")
        print(f"Valid memcpy dur_ns 평균: {valid_memcpy['dur_ns'].mean():.0f}ns")
        print(f"\nNaN memcpy dur_ns 분포:\n{nan_memcpy['dur_ns'].describe()}")
        print(f"\nValid memcpy dur_ns 분포:\n{valid_memcpy['dur_ns'].describe()}")

        rank0 = gpu_data_map[0]
        df_memcpy = rank0.df_memcpy

        # large vs small 분리
        large_h2d = df_memcpy[df_memcpy["data_batch_idx"].notna()].copy()
        small_h2d = df_memcpy[df_memcpy["data_batch_idx"].isna()].copy()

        # step 범위 가져오기
        step_ranges = build_step_df_from_nvtx(rank0.df_nvtx)

        print("=== Large h2d (batch 데이터) ===")
        print(f"개수: {len(large_h2d)}")
        print(f"dur_ns 평균: {large_h2d['dur_ns'].mean()/1e6:.2f}ms")
        print(large_h2d[["kernel_start", "kernel_end", "dur_ns", "data_batch_idx"]].head(10))

        print("\n=== Small h2d (PyTorch 내부) ===")
        print(f"개수: {len(small_h2d)}")
        print(f"dur_ns 평균: {small_h2d['dur_ns'].mean()/1e6:.4f}ms")
        print(small_h2d[["kernel_start", "kernel_end", "dur_ns"]].head(10))

        print("\n=== Step 범위 (CPU 시계) ===")
        print(step_ranges.head(5))

        print("\n=== Large h2d가 어느 step GPU 구간에 있는지 ===")
        gpu_batch = rank0.df_gpu_duration[
            rank0.df_gpu_duration["name"].str.contains(r"gpu_batch_\d+_duration", regex=True)
        ][["data_batch_idx", "gpu_start", "gpu_end"]]
        print(gpu_batch.head(5))

        # large h2d의 kernel_start가 어느 GPU 구간에 있는지
        for _, row in large_h2d.head(6).iterrows():
            for _, grow in gpu_batch.iterrows():
                if grow["gpu_start"] <= row["kernel_start"] < grow["gpu_end"]:
                    print(f"large h2d kernel_start={row['kernel_start']} "
                        f"→ GPU step {int(grow['data_batch_idx'])} 구간 "
                        f"(data_batch_idx={int(row['data_batch_idx'])})")
                    break

        rank0 = gpu_data_map[0]
        small_h2d = rank0.df_memcpy[rank0.df_memcpy["data_batch_idx"].isna()].copy()

        print(f"Small h2d 전체 개수: {len(small_h2d)}")
        print(f"\nkernel_start 범위:")
        print(f"  min: {small_h2d['kernel_start'].min()}")
        print(f"  max: {small_h2d['kernel_start'].max()}")

        print(f"\nCPU step 범위:")
        step_ranges = build_step_df_from_nvtx(rank0.df_nvtx)
        print(step_ranges)

        print(f"\nSmall h2d vs step 범위 비교:")
        print(f"  Small h2d max:    {small_h2d['kernel_start'].max()}")
        print(f"  CPU step 0 start: {step_ranges['start'].min()}")
        print(f"  차이: {(step_ranges['start'].min() - small_h2d['kernel_start'].max())/1e6:.1f}ms")

        # step 1 구간의 small h2d 확인
        step1_start = 14008419466
        step1_end   = 14101811475

        small_in_step1 = small_h2d[
            (small_h2d["kernel_start"] >= step1_start) &
            (small_h2d["kernel_start"] <= step1_end)
        ]
        print(f"step 1 구간의 small h2d: {len(small_in_step1)}개")
        print(small_in_step1[["kernel_start", "kernel_end", "dur_ns"]])

        # step 0 이전 small h2d
        before_step0 = small_h2d[small_h2d["kernel_start"] < 12123528107]
        print(f"\nstep 0 이전 small h2d: {len(before_step0)}개")

        # step 0 구간의 small h2d
        step0_start = 12123528107
        step0_end   = 14007978371
        small_in_step0 = small_h2d[
            (small_h2d["kernel_start"] >= step0_start) &
            (small_h2d["kernel_start"] <= step0_end)
        ]
        print(f"step 0 구간의 small h2d: {len(small_in_step0)}개")


        rank0 = gpu_data_map[0]

        # data_wait NVTX 확인
        data_wait_nvtx = rank0.df_nvtx[
            rank0.df_nvtx["name"] == "cpu_data_wait_launch"
        ].copy()

        data_wait_nvtx["dur_ms"] = data_wait_nvtx["dur_ns"] / 1e6

        print(f"=== data_wait NVTX (rank 0) ===")
        print(f"개수: {len(data_wait_nvtx)}")
        print(data_wait_nvtx[["name", "start", "end", "dur_ms", "data_batch_idx"]].head(10).to_string())

        # large h2d와 비교
        large_h2d = rank0.df_memcpy[rank0.df_memcpy["data_batch_idx"].notna()].copy()
        large_h2d["dur_ms"] = large_h2d["dur_ns"] / 1e6

        print(f"\n=== large h2d (rank 0) ===")
        print(large_h2d[["kernel_start", "kernel_end", "dur_ms", "data_batch_idx"]].head(10).to_string())

        print(f"\n=== 비교: data_wait end vs h2d start ===")
        for step in [0, 1, 2, 3, 4]:
            dw = data_wait_nvtx[data_wait_nvtx["data_batch_idx"] == step]
            h2d = large_h2d[large_h2d["data_batch_idx"] == step]
            
            if not dw.empty and not h2d.empty:
                dw_end   = dw.iloc[0]["end"]
                h2d_start = h2d.iloc[0]["kernel_start"]
                gap_ms   = (h2d_start - dw_end) / 1e6
                print(f"  step {step}: data_wait_end={dw_end} h2d_start={h2d_start} gap={gap_ms:.3f}ms")

        offsets = calculate_clock_offsets(gpu_data_map, n_kernels=20)
        #debug_clock_offsets(gpu_data_map, n_kernels=20)

        all_steps_map = get_all_step_intervals(gpu_data_map)
        
        df_rank_order_per_step = compute_rank_order_per_step(all_steps_map, gpu_data_map,offsets)
        
        if df_rank_order_per_step.empty:
            print("[Error] Could not calculate global step intervals. Check if NVTX markers exist.")
            sys.exit(1)

        # --- Level 1: Wait Time Analysis ---
        print("\n--- Level 1: Wait Time Analysis ---")
        wait_df = load_all_gpu_compute_wait_time(gpu_data_map,offsets)


        if wait_df.empty:
            print("[Warning] wait_df is empty. Check NVTX NCCL_AllReduce ranges.")

        else:
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

            plot_timeline_custom_axis(
                gpu_data_map=gpu_data_map,
                df_rank_order_per_step=df_rank_order_per_step,
                all_steps_map=all_steps_map,
                steps_to_plot=target_steps,
                out_png="timeline",
                show=True,
                color_by="step", # set False on headless servers
                offsets=offsets 
            )
            #--- Step Breakdown Analysis ---
            print("\n--- Step Breakdown Analysis ---")

            df_breakdown = aggregate_per_step_breakdown(
                gpu_data_map           = gpu_data_map,
                steps_to_plot          = target_steps,
                nccl_wait_df           = wait_df,
                offsets                = offsets,
                df_rank_order_per_step = df_rank_order_per_step,
            )

            if df_breakdown.empty:
                print("[Warning] df_breakdown is empty.")
            else:
                print(df_breakdown.to_string())

                breakdown_filename = (
                    f"step_breakdown"
                    f"_ranks{'_'.join(str(r) for r in sorted(gpu_data_map.keys()))}"
                )

                plot_step_breakdown(
                    df_breakdown   = df_breakdown,
                    output_prefix  = breakdown_filename,
                )

                print(f"Step breakdown saved: {breakdown_filename}")

        print("\n[Success] Analysis and visualization completed successfully.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Fatal Error] An unexpected error occurred: {e}")