import sys
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
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
        # Call the detailed analysis function written by the user
        # (Note: we assume process_full_analysis is already defined)
        nvtx_df, nccl_df, memcpy_df, true_gpu_df = process_full_analysis(con, steps)
        
        # Create the object even if DataFrames are empty (they will just be empty DFs)
        return GpuDataset(
            rank=rank,
            filename=filepath,
            nvtx_df=nvtx_df,
            nccl_df=nccl_df,
            memcpy_df=memcpy_df,
            true_gpu_df=true_gpu_df
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
    
    # Sort filenames (rank0, rank1... order)
    file_list.sort()

    for idx, filepath in enumerate(file_list):
        # 1. Extract rank from filename
        rank = get_rank_from_filename(filepath, idx)
        
        # 2. Load data (including True GPU Span)
        dataset = load_single_gpu(filepath, rank, steps_to_analyze)
        
        if dataset:
            gpu_data_map[rank] = dataset
            
    print(f"\n[Done] Loaded data for {len(gpu_data_map)} GPUs.")

    return gpu_data_map

def get_all_step_intervals(gpu_data_map: Dict[int, 'GpuDataset']) -> Dict[int, pd.DataFrame]:
    """
    [Core Feature]
    Iterate over all loaded GPU datasets (GpuDataset),
    and for each GPU, extract the start/end timestamps of training steps
    recorded in NVTX markers (Step/Batch/Iteration).

    This information is later used to align multiple GPU timelines
    using the same step boundaries (Alignment).
    """
    all_steps_map = {}
    print("--- Extracting Step Intervals from NVTX ---")

    # gpu_data_map: dictionary of { rank_id: GpuDataset }
    for rank, dataset in gpu_data_map.items():
        
        # 1. Parse step information from NVTX DataFrame for each rank.
        #    Internally, build_step_df_from_nvtx parses strings like
        #    'Batch_100', 'step 101' into step numbers (100, 101) and their time ranges.
        step_df = build_step_df_from_nvtx(dataset.nvtx_df)
        
        # 2. Check if we got any valid step markers.
        if not step_df.empty:
            # Save the extracted step DataFrame (columns: step, start, end)
            # using rank as the key.
            all_steps_map[rank] = step_df
            
            # (Optional debugging) print how many steps were found for each rank.
            # print(f"Rank {rank}: Found {len(step_df)} steps.")
        else:
            # If NVTX does not contain keywords like 'Batch', 'Step', 'Iter',
            # we cannot define step boundaries, so warn the user.
            print(f"[Warning] Rank {rank}: No step markers (Batch/Step/Iter) found in NVTX.")

    # 3. Return a dictionary holding step intervals for every GPU rank.
    #    Example: { 0: step_df_rank0, 1: step_df_rank1, ... }
    return all_steps_map

def get_global_earliest_steps(all_steps_map: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    [Core Feature]
    Compare step timestamps across all ranks and compute, for each step:
    - Global Start (earliest start among all ranks)
    - Global End   (latest end among all ranks)
    - Fastest Rank (rank that started earliest)
    - Slowest Rank (rank that ended latest)
    """
    
    # 1. Combine all ranks into one big table by adding a 'rank' column.
    combined_list = []
    for rank, df in all_steps_map.items():
        temp_df = df.copy()
        temp_df["rank"] = rank
        combined_list.append(temp_df)
    
    if not combined_list:
        return pd.DataFrame()
        
    # big_df example columns: [step, start, end, rank]
    big_df = pd.concat(combined_list, ignore_index=True)
    
    # 2. Group by step and compute global statistics
    grouped = big_df.groupby("step")
    stats_df = grouped.agg(
        earliest_start=("start", "min"),
        latest_end=("end", "max")
    )
    
    # 3. Find the fastest and slowest ranks for each step
    # (1) Index of row with minimal 'start' per step, and maximal 'end' per step
    min_idx_series = grouped["start"].idxmin()
    max_idx_series = grouped["end"].idxmax()
    
    # (2) Fastest rank (earliest start)
    fastest_ranks = big_df.loc[min_idx_series, ["step", "rank"]].rename(columns={"rank": "fastest_rank"})
    
    # (3) Slowest rank (latest end)
    slowest_ranks = big_df.loc[max_idx_series, ["step", "rank"]].rename(columns={"rank": "slowest_rank"})
    
    # (4) Merge these back into stats_df
    final_df = stats_df.merge(fastest_ranks, on="step", how="left")
    final_df = final_df.merge(slowest_ranks, on="step", how="left")
    
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
    corr_id_col: str = "correlationId" # Default

def find_kernel_schema(con: sqlite3.Connection) -> Optional[KernelSchema]:
    tables = list_tables(con)
    kernel_tables = [t for t in tables if t.upper() == "CUPTI_ACTIVITY_KIND_KERNEL"]
    if not kernel_tables:
        kernel_tables = [t for t in tables if "kernel" in t.lower() and "cupti" in t.lower()]
    if not kernel_tables: return None

    possible_start = ["start", "startNs", "timestamp_start", "start_time"]
    possible_end   = ["end", "endNs", "timestamp_end", "end_time"]
    possible_name_id = ["shortName", "shortNameId", "nameId", "demangledName"]
    possible_corr = ["correlationId", "correlation_id"]

    for t in kernel_tables:
        cols = set(table_columns(con, t))
        sc = next((c for c in possible_start if c in cols), None)
        ec = next((c for c in possible_end if c in cols), None)
        nid = next((c for c in possible_name_id if c in cols), None)
        cid = next((c for c in possible_corr if c in cols), None)
        
        if sc and ec and nid:
            return KernelSchema(table=t, start_col=sc, end_col=ec, name_id_col=nid, corr_id_col=cid or "correlationId")
    return None

# --- NVTX Schema (Safe Version) ---
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
    possible_end = ["end", "endNs", "timestamp_end"]
    possible_name = ["text", "message", "name"]

    # 1. Check priority tables first
    for t_name in priority_candidates:
        actual_name = next((t for t in tables if t.upper() == t_name), None)
        if actual_name:
            cols = set(table_columns(con, actual_name))
            sc = next((c for c in possible_start if c in cols), None)
            ec = next((c for c in possible_end if c in cols), None)
            nc = next((c for c in possible_name if c in cols), None)
            if sc and ec and nc:
                return NvtxSchema(actual_name, nc, sc, ec)

    # 2. Fallback search
    fallback = [t for t in tables if "NVTX" in t.upper()]
    for t in fallback:
        cols = set(table_columns(con, t))
        sc = next((c for c in possible_start if c in cols), None)
        ec = next((c for c in possible_end if c in cols), None)
        nc = next((c for c in possible_name if c in cols), None)
        if sc and ec and nc:
            return NvtxSchema(t, nc, sc, ec)
            
    return None

# --- Memcpy Schema ---
@dataclass
class MemcpySchema:
    table: str
    start_col: str
    end_col: str
    kind_col: Optional[str] = None
    bytes_col: Optional[str] = None

def find_memcpy_schema(con: sqlite3.Connection) -> Optional[MemcpySchema]:
    tables = list_tables(con)
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
        return MemcpySchema(t, sc, ec, kc, bc)
    return None

# --- String IDs Schema ---
@dataclass
class StringIdsSchema:
    table: str
    id_col: str
    value_col: str

def find_stringids_schema(con: sqlite3.Connection) -> Optional[StringIdsSchema]:
    tables = list_tables(con)
    candidates = [t for t in tables if "string" in t.lower() and "id" in t.lower()]
    for t in candidates:
        cols = set(table_columns(con, t))
        if "id" in cols and "value" in cols: return StringIdsSchema(t, "id", "value")
    return None

# ==============================================================================
# 3. Data Loading Functions
# ==============================================================================
def load_nvtx_events(con: sqlite3.Connection, sch: NvtxSchema) -> pd.DataFrame:
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



def load_memcpy_events(
    con: sqlite3.Connection,
    sch: MemcpySchema,
    want_kinds: Optional[List[int]] = None,
    name: str = "gpu_memcpy_h2d"
) -> pd.DataFrame:

    select_cols = [
        f"{sch.start_col} AS start",
        f"{sch.end_col} AS end",
    ]
    if sch.kind_col:
        select_cols.append(f"{sch.kind_col} AS kind")
    if sch.bytes_col:
        select_cols.append(f"{sch.bytes_col} AS bytes")

    q = f"SELECT {', '.join(select_cols)} FROM {sch.table} WHERE {sch.end_col} > {sch.start_col}"
    df = try_read_df(con, q)
    if df.empty:
        return pd.DataFrame(columns=["name", "start", "end", "dur_ns"])

    # Kind filter
    if want_kinds is not None and sch.kind_col and "kind" in df.columns:
        df = df[df["kind"].isin(want_kinds)].copy()

    df["name"] = name
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
    
    """Loads Runtime API calls within NVTX ranges."""
    if nvtx_df.empty: 
        return pd.DataFrame()
    min_start = nvtx_df["start"].min()
    max_end = nvtx_df["end"].max()

    print(f"Loading Runtime events ({min_start} ~ {max_end})...")


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
    k_sch: KernelSchema
    ) -> pd.DataFrame:
    
    """Loads ALL kernels to calculate spans (needs correlationId)."""
    q = f"""
    SELECT 
        {k_sch.corr_id_col} AS correlationId, 
        {k_sch.start_col} AS k_start, 
        {k_sch.end_col} AS k_end 
    FROM {k_sch.table} 
    WHERE {k_sch.end_col} > {k_sch.start_col}"""
    
    return try_read_df(con, q)

def map_nvtx_to_runtime(
    nvtx_df: pd.DataFrame, 
    runtime_df: pd.DataFrame, 
    ):

    """Maps NVTX ranges to Runtime API correlation IDs."""
    if nvtx_df.empty or runtime_df.empty: 
        return pd.DataFrame()
    
    # Sort for masking
    nvtx_sorted = nvtx_df.sort_values("start").reset_index(drop=True)
    rt_sorted = runtime_df.sort_values("start").reset_index(drop=True)
    
    rt_starts = rt_sorted["start"].values
    rt_corrs = rt_sorted["correlationId"].values
    
    mapped_rows = []

    # Optimization: filter only relevant NVTX rows (e.g. forward, backward, loss)
    target_keywords = ["forward", "backward", "loss", "opt_step", "step", "Batch"]

    # Only process NVTX events that contain these keywords to save time
    
    for _, row in nvtx_sorted.iterrows():
        # Optional: skip tiny NVTX ranges or irrelevant names to speed up
        # if row["dur_ns"] < 1000: continue 
        
        nvtx_start = row["start"]
        nvtx_end = row["end"]
        nvtx_name = row["name"]
        

        # Runtime start must be inside NVTX
        mask = (rt_starts >= nvtx_start) & (rt_starts < nvtx_end)
        matched_corrs = rt_corrs[mask]
        
        for cid in matched_corrs:
            mapped_rows.append({
                "nvtx_name": nvtx_name,
                "nvtx_start": nvtx_start,
                "nvtx_end": nvtx_end,
                "correlationId": cid
            })
            
    return pd.DataFrame(mapped_rows)

def compute_true_gpu_spans(
    mapping_df: pd.DataFrame, 
    kernel_df: pd.DataFrame
    ) -> pd.DataFrame:
    
    """Calculates [Min Kernel Start, Max Kernel End] for each NVTX."""
    if mapping_df.empty or kernel_df.empty: 
        return pd.DataFrame()

    print(f"Merging {len(mapping_df)} CPU-Mappings with {len(kernel_df)} GPU-Kernels...")

    merged = pd.merge(mapping_df, kernel_df, on="correlationId", how="inner")
    
    if merged.empty: 
        return pd.DataFrame()

    # Group by NVTX identity (Name + Start Time)
    result = merged.groupby(["nvtx_name", "nvtx_start", "nvtx_end"], as_index=False).agg(
        true_start=("k_start", "min"),
        true_end=("k_end", "max")
    )
    
    result["dur_ns"] = result["true_end"] - result["true_start"]
    result = result.rename(columns={"true_start": "start", "true_end": "end"})
    
    # Rename for plotting: "[GPU] name"
    result["name"] = result["nvtx_name"].apply(lambda x: f"[GPU] {x}")
    return result[["name", "start", "end", "dur_ns"]]

# ==============================================================================
# 5. Plotting Logic
# ==============================================================================
def build_step_df_from_nvtx(nvtx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly extract step ranges from NVTX data.
    - Supported patterns: Batch_10, step 10, Iter-10, Iteration:10, Global Step 10 (case-insensitive)
    - Deduplication: if the same step number appears multiple times, take the "outermost" range
      (earliest start, latest end).
    """
    if nvtx_df.empty:
        return pd.DataFrame()

    # 1) Unified regex pattern to cover many naming styles
    # Explanation:
    #   (?i) : case-insensitive
    #   (?:Batch|step|Iter|Iteration|Global Step) : accepted keywords (non-capturing group)
    #   [_\s:\-]* : optional separators (underscore / space / colon / hyphen)
    #   (\d+) : capture the step number
    pattern = r"(?i)(?:Batch|step|Iter|Iteration|Global Step)[_\s:\-]*(\d+)"

    # 2) Extract the step number from the 'name' column
    df = nvtx_df.copy()
    extracted_steps = df["name"].str.extract(pattern, expand=False)
    
    # 3) Convert to numeric (non-parsable values become NaN)
    df["step"] = pd.to_numeric(extracted_steps, errors='coerce')
    
    # 4) Keep only rows that actually have a valid step number
    valid_steps = df.dropna(subset=["step"])
    
    if valid_steps.empty:
        # Return an empty DF instead of raising an error,
        # because some runs might not record step NVTX ranges.
        return pd.DataFrame(columns=["step", "start", "end"])

    # 5) Deduplicate: for each step, take the outermost range (min start, max end)
    step_df = valid_steps.groupby("step").agg(
        start=("start", "min"),
        end=("end", "max")
    ).reset_index()

    # 6) Convert step to int and sort by time
    step_df["step"] = step_df["step"].astype(int)
    step_df = step_df.sort_values("start")

    return step_df[["step", "start", "end"]]


def filter_by_step_ranges(source_df: pd.DataFrame, step_df_sel: pd.DataFrame, global_start: int) -> pd.DataFrame:
    """
    Filters events that fall within the selected step time ranges.
    Also calculates relative start time (ms) for plotting.
    """
    rows = []
    for _, srow in step_df_sel.iterrows():
        s = int(srow["step"])
        s_start = int(srow["start"])
        s_end = int(srow["end"])

        # Keep events that overlap with this step
        d = source_df[
            (source_df["start"] < s_end) & 
            (source_df["end"] > s_start)
        ].copy()
        
        if d.empty:
            continue
            
        d["step"] = s
        rows.append(d)
    
    if not rows:
        return pd.DataFrame()

    filtered = pd.concat(rows, ignore_index=True)
    
    # Drop duplicates in case an event spans across the boundary of two selected steps
    # We only want to draw it once per its unique identity (name, start, end)
    filtered = filtered.drop_duplicates(subset=["name", "start", "end"])

    # Convert to ms relative to the very first step's start
    filtered["rel_start_ms"] = (filtered["start"] - global_start) / 1e6
    filtered["dur_ms"] = (filtered["end"] - filtered["start"]) / 1e6
    return filtered

def plot_timeline_custom_axis(
    gpu_data_map: Dict[int, GpuDataset],
    final_df: pd.DataFrame,
    all_steps_map: Dict[int, pd.DataFrame],
    steps_to_plot: List[int],
    out_png="timeline_custom_axis.png", 
    show=True
):
    print(f"Plotting steps: {steps_to_plot} with Custom Y-Axis...")

    # ---------------------------------------------------------
    # 1. Preprocessing: scan all GPU datasets to build the final Y-axis label list
    # ---------------------------------------------------------
    
    # (1) Define base row names
    base_names = ["data_wait", "h2d", "gpu_compute", "NCCL"]
    
    # (2) Collect NCCL types (scan all ranks and only keep ones that exist)
    found_nccl_types = set()
    
    # def simplify_name(raw_name):
    #     lower = raw_name.lower()
    #     if "allreduce" in lower: return "NCCL AllReduce"
    #     if "broadcast" in lower: return "NCCL Broadcast"
    #     if "allgather" in lower: return "NCCL AllGather"
    #     if "reducescatter" in lower: return "NCCL ReduceScatter"
    #     if "send" in lower or "recv" in lower: return "NCCL P2P"
    #     return "NCCL Other"

    # for rank, dataset in gpu_data_map.items():
    #     if not dataset.nccl_df.empty:
    #         # Add simplified names into a set
    #         simplified = dataset.nccl_df["name"].apply(simplify_name).unique()
    #         found_nccl_types.update(simplified)
    
    # # Sort NCCL names alphabetically
    # nccl_names = sorted(list(found_nccl_types))

    # (3) Fixed NVTX markers
    fixed_nvtx = ['forward',  'loss', 'backward', 'opt_step', 'nccl_sync']
    
    # (4) Build the final Y-axis list (remove duplicates but preserve order)
    # Order: Base -> NCCL -> Fixed
    full_y_names = []
    
    # Base names (you can always include them, or filter later if desired)
    full_y_names.extend(base_names)
    full_y_names.extend(fixed_nvtx)
    # full_y_names.extend(nccl_names)
    
    
    # Remove duplicates (Python 3.7+ dict preserves insertion order)
    full_y_names = list(dict.fromkeys(full_y_names))
    
    # (5) Build mapping dict: name -> Y index
    # Example: {'data_wait': 0, 'h2d': 1, 'NCCL AllReduce': 2, ...}
    y_map = {name: i for i, name in enumerate(full_y_names)}
    
    print(f"--- Generated Y-Axis Labels ({len(full_y_names)}) ---")
    print(full_y_names)


    # ---------------------------------------------------------
    # 2. Prepare the figure canvas
    # ---------------------------------------------------------
    # Define the global reference time
    target_stats = final_df[final_df["step"].isin(steps_to_plot)]
    if target_stats.empty:
        print(f"[Error] No data for steps {steps_to_plot}")
        return
    global_start = target_stats["earliest_start"].min()

    # Figure setup
    # Height: (num items) * (height per item) + padding
    fig_height = len(full_y_names) * 1.5 + 2
    plt.figure(figsize=(24, fig_height))
    ax = plt.gca()

    # Rank colors (up to 8)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    
    # Bar height per lane (so we can stack ranks inside a single row)
    # Example: if row height is 0.8 and we have 4 ranks, each rank gets 0.2
    lane_height = 0.8 / len(gpu_data_map) 


    # ---------------------------------------------------------
    # 3. Iterate over GPUs and plot
    # ---------------------------------------------------------
    sorted_ranks = sorted(gpu_data_map.keys())

    for rank in sorted_ranks:
        dataset = gpu_data_map[rank]
        rank_color = colors[rank % len(colors)]
        
        # (1) Filter data (Time Alignment)
        if rank not in all_steps_map: 
            continue
        step_df_sel = all_steps_map[rank][all_steps_map[rank]["step"].isin(steps_to_plot)]
        if step_df_sel.empty: 
            continue
        
        # Load each dataset with filtering applied
        df_nvtx = filter_by_step_ranges(dataset.nvtx_df, step_df_sel, global_start)

        df_nccl = pd.DataFrame()
        if not dataset.nccl_df.empty:
            df_nccl = filter_by_step_ranges(dataset.nccl_df, step_df_sel, global_start)

        df_memcpy = pd.DataFrame()
        if not dataset.memcpy_df.empty:
            df_memcpy = filter_by_step_ranges(dataset.memcpy_df, step_df_sel, global_start)

        df_compute = pd.DataFrame()
        if not dataset.true_gpu_df.empty:
            df_compute = filter_by_step_ranges(dataset.true_gpu_df, step_df_sel, global_start)


        # -------------------------------------------------------
        # (2) Plotting: map each event type to the correct y_map row
        # -------------------------------------------------------
        
        # Small offset per rank (stack lanes: rank 0 bottom, rank 3 top)
        rank_offset = rank * lane_height

        # [A] GPU Compute -> map to 'gpu_compute' row
        if not df_compute.empty and "gpu_compute" in y_map:
            y_base = y_map["gpu_compute"]
            xranges = list(zip(df_compute["rel_start_ms"], df_compute["dur_ms"]))
            ax.broken_barh(xranges, (y_base + rank_offset, lane_height), 
                           facecolors=rank_color, alpha=0.9, linewidth=0.5, edgecolor='black')



        # [B] Memcpy -> map to 'h2d' row (assumption: memcpy is shown under h2d)
        # [B] Smart Overlay for Memcpy (CPU HtoD + GPU HtoD)
        # ---------------------------------------------------------
        # Requirement: y_map must contain the key "h2d"
        if "h2d" in y_map:
            y_base = y_map["h2d"]
            current_y = y_base + rank_offset
            
            # 1. Draw CPU (NVTX) background (light color)
            # -----------------------------------------------------
            if not df_nvtx.empty:
                # Find ranges in NVTX name that contain "h2d" (adjust keywords if needed)
                cpu_mask = df_nvtx["name"].str.contains("h2d", case=False, regex=True)
                cpu_rows = df_nvtx[cpu_mask]
                
                for _, cpu_row in cpu_rows.iterrows():
                    cpu_start = cpu_row["rel_start_ms"]
                    cpu_dur = cpu_row["dur_ms"]
                    
                    # Light background
                    ax.broken_barh([(cpu_start, cpu_dur)], 
                                   (current_y, lane_height), 
                                   facecolors=rank_color, 
                                   alpha=0.3,       # light (CPU command time)
                                   linewidth=0)

            # 2. Draw GPU (Actual Memcpy) segments (strong color)
            # -----------------------------------------------------
            if not df_memcpy.empty:
                # Assume df_memcpy is already filtered to kind=1 (HtoD)
                # If not, you can filter here as well.
                
                xranges = list(zip(df_memcpy["rel_start_ms"], df_memcpy["dur_ms"]))
                
                ax.broken_barh(xranges, 
                               (current_y, lane_height), 
                               facecolors=rank_color, 
                               alpha=1.0,           # strong (actual transfer time)
                               linewidth=0.5, 
                               edgecolor='black')   # outline for emphasis

        # [C] NCCL -> map to a single "NCCL" row
        if not df_nccl.empty and "NCCL" in y_map:
            y_base = y_map["NCCL"]
            current_y = y_base + rank_offset
            
            # Extract all NCCL time segments (without separating types)
            xranges = list(zip(df_nccl["rel_start_ms"], df_nccl["dur_ms"]))
            
            # Draw
            ax.broken_barh(xranges, 
                           (current_y, lane_height), 
                           facecolors=rank_color, 
                           alpha=0.9,           # strong (important)
                           linewidth=0.5, 
                           edgecolor='black')   # outline


        # [D] NVTX Markers -> map to matching rows by name
        # [E] Smart Overlay: CPU (light) + GPU (strong) + end marker line
        # Target phases to match against NVTX names
        target_phases = ["data_wait","Forward", "Backward", "Loss",  'nccl_sync', "opt_step"] 
        
        if not df_nvtx.empty:
            for target_name in target_phases:
                
                # 1. Draw CPU (NVTX) background
                mask = df_nvtx["name"].str.contains(target_name, case=False, regex=False)
                cpu_rows = df_nvtx[mask]
                
                if cpu_rows.empty: 
                    continue

                # Find Y-axis row
                base_y = -1
                for key in y_map:
                    if target_name.lower() in key.lower():
                        base_y = y_map[key]
                        break
                if base_y == -1: 
                    continue 

                current_y = base_y + rank_offset
                
                # CPU loop
                for _, cpu_row in cpu_rows.iterrows():
                    cpu_start = cpu_row["rel_start_ms"]
                    cpu_end   = cpu_start + cpu_row["dur_ms"] # start + duration
                    
                    # (1) Light CPU bar
                    ax.broken_barh([(cpu_start, cpu_row["dur_ms"])], 
                                   (current_y, lane_height), 
                                   facecolors=rank_color, 
                                   alpha=0.3, linewidth=0)
                    
                    # (2) Solid vertical line at CPU end
                    ax.vlines(x=cpu_end, 
                              ymin=current_y, ymax=current_y + lane_height,
                              colors='black', linestyles='solid', linewidth=1.5, alpha=0.8)

                # 2. Overlay GPU (True Spans)
                # Data already mapped by correlation ID in process_full_analysis
                if not df_compute.empty:
                    # df_compute["name"] format is "[GPU] Forward"
                    gpu_mask = df_compute["name"].str.contains(target_name, case=False, regex=False)
                    relevant_gpu = df_compute[gpu_mask]
                    
                    if not relevant_gpu.empty:
                        gpu_xranges = list(zip(relevant_gpu["rel_start_ms"], relevant_gpu["dur_ms"]))
                        
                        ax.broken_barh(gpu_xranges, 
                                       (current_y, lane_height), 
                                       facecolors=rank_color, 
                                       alpha=1.0,       # strong (real GPU work)
                                       linewidth=0.5, 
                                       edgecolor='white')
    
    # Step Lines
    for _, srow in step_df_sel.iterrows():
        x0 = (srow["start"] - global_start) / 1e6
        x1 = (srow["end"] - global_start) / 1e6
        ax.axvline(x0, color='k', ls='--', alpha=0.5)
        ax.axvline(x1, color='k', ls=':', alpha=0.5)
        ax.text((x0+x1)/2, len(full_y_names), f"Step {int(srow['step'])}", ha='center', va='bottom', weight='bold')
    # ---------------------------------------------------------
    # 4. Axis / label formatting
    # ---------------------------------------------------------
    
    # [Fix] Put Y tick labels at the center of each row lane
    ax.set_yticks([i + 0.5 for i in range(len(full_y_names))]) 
    ax.set_yticklabels(full_y_names, fontsize=11, fontweight='bold')
    
    # [Add] Row separators
    # Draw a horizontal line at each integer boundary
    for i in range(len(full_y_names) + 1):
        ax.axhline(y=i, color='black', linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Time (ms) relative to Global Step Start", fontsize=12)
    ax.set_title(f"Multi-GPU Detailed Timeline | Steps: {steps_to_plot}", fontsize=14)
    
    # Keep vertical grid lines for time as dotted
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)
    
    # Disable default y-grid since we draw separators ourselves
    ax.grid(False, axis='y') 

    # Legend (Rank colors)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[r % len(colors)], label=f'Rank {r}') for r in sorted_ranks]
    ax.legend(handles=legend_elements, loc='upper right', title="GPU Ranks", bbox_to_anchor=(1.05, 1))
    
    # Limit Y range to remove extra blank space
    ax.set_ylim(0, len(full_y_names))

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved timeline to {out_png}")
    if show: 
        plt.show()

# ==============================================================================
# 6. Main Execution
# ==============================================================================
def process_full_analysis(con: sqlite3.Connection, steps: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract NVTX, NCCL, Memcpy, and 'True GPU Spans' from a single GPU (SQLite) file.
    """
    
    # 1. Find schemas (auto-detect table names)
    nvtx_sch = find_nvtx_schema(con)
    k_sch = find_kernel_schema(con)
    r_sch = find_runtime_schema(con)  # for True GPU Span calculation
    m_sch = find_memcpy_schema(con)
    s_sch = find_stringids_schema(con) # for NCCL name mapping

    # -------------------------------------------------------
    # A. Load basic data (NVTX, NCCL, Memcpy)
    # -------------------------------------------------------
    df_nvtx = load_nvtx_events(con, nvtx_sch) if nvtx_sch else pd.DataFrame()
    
    df_nccl = pd.DataFrame()
    if k_sch and s_sch:
        df_nccl = load_nccl_kernels(con, k_sch, s_sch)
        
    df_memcpy = pd.DataFrame()
    if m_sch:
        df_memcpy = load_memcpy_events(con, m_sch, want_kinds=[1], name="HtoD")

    # -------------------------------------------------------
    # B. [Core] Compute True GPU Spans (Correlation ID matching)
    # -------------------------------------------------------
    df_true_gpu = pd.DataFrame()
    
    # Run only when all required schemas exist and NVTX is not empty
    if nvtx_sch and r_sch and k_sch and not df_nvtx.empty:
        try:
            print("  -> Calculating True GPU Spans (Correlation ID matching)...")
            
            # 1. Load Runtime API calls inside NVTX time span
            df_runtime = load_runtime_events_in_nvtx(con, df_nvtx, r_sch)
            
            # 2. Load all kernels (for mapping)
            df_kernels_all = load_all_kernels_for_span(con, k_sch)
            
            # 3. Build mapping (NVTX -> Runtime -> CorrelationId)
            df_mapping = map_nvtx_to_runtime(df_nvtx, df_runtime)
            
            # 4. Final computation (CorrelationId -> Kernel Time)
            df_true_gpu = compute_true_gpu_spans(df_mapping, df_kernels_all)
            
            print(f"  -> Found {len(df_true_gpu)} true GPU spans.")
            
        except Exception as e:
            print(f"  [Warning] Failed to compute True GPU Spans: {e}")
            # Even if it fails, we still return other data

    return df_nvtx, df_nccl, df_memcpy, df_true_gpu

if __name__ == "__main__":
    # Usage: python analyze_multi.py <step1> <step2> ... <rank0.sqlite> <rank1.sqlite> ...
    # Order does not matter; we separate by extension and digits.
    if len(sys.argv) < 3:
        print("Usage: python analyze_multi.py <step_N> <file1.sqlite> [file2.sqlite ...]")
        sys.exit(1)

    # 1. Parse args (split SQLite files and step numbers)
    sqlite_files = []
    target_steps = []

    for arg in sys.argv[1:]:
        if arg.endswith(".sqlite") or arg.endswith(".sqlite3"):
            sqlite_files.append(arg)
        elif arg.isdigit():
            target_steps.append(int(arg))
    
    if not sqlite_files:
        print("[Error] No SQLite files provided.")
        sys.exit(1)
        
    if not target_steps:
        print("[Warning] No steps specified. Defaulting to first detected step.")
        # You can set a default here if needed

    print(f"--- Configuration ---")
    print(f"Target Steps: {target_steps}")
    print(f"Files ({len(sqlite_files)}):")
    for f in sqlite_files:
        print(f"  - {f}")
    print("---------------------\n")

    # 2. Load multi-GPU data
    gpu_data_map = {} # { rank: GpuDataset }

    for i, filepath in enumerate(sqlite_files):
        # Extract rank from filename (example: 'nsys_rank0.sqlite' -> 0)
        # If no rank number exists in filename, use list index i as rank
        rank = get_rank_from_filename(filepath, i)
        
        # Load single GPU dataset (using load_single_gpu defined earlier)
        dataset = load_single_gpu(filepath, rank, target_steps)
        
        if dataset:
            gpu_data_map[rank] = dataset
            
        else:
            print(f"[Warning] Failed to load data for Rank {rank} ({filepath})")

    if not gpu_data_map:
        print("[Fatal Error] No valid GPU data loaded. Exiting.")
        sys.exit(1)

    try:
        # 3. Step synchronization (Time Alignment)
        # Extract step intervals for each GPU
        all_steps_map = get_all_step_intervals(gpu_data_map)
        
        # Build global step table (Earliest Start / Latest End)
        final_df = get_global_earliest_steps(all_steps_map)
        
        if final_df.empty:
            print("[Error] Could not calculate global step intervals. Check if NVTX markers exist.")
            sys.exit(1)
            
        print("\n--- Global Step Statistics ---")
        print(final_df.to_string(index=False))
        print("------------------------------\n")

        # 4. Plot the combined timeline (Plotting)
        output_filename = f"timeline_steps_{'_'.join(map(str, target_steps))}.png"
        
        plot_timeline_custom_axis(
            gpu_data_map=gpu_data_map,
            final_df=final_df,
            all_steps_map=all_steps_map,
            steps_to_plot=target_steps,
            out_png=output_filename,
            show=True  # set False on headless servers
        )
        
        print("\n[Success] Analysis and Visualization completed successfully.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Fatal Error] An unexpected error occurred: {e}")
