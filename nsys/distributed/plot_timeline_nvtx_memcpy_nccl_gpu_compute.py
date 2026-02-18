import sys
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Generic SQLite Helpers
#    (Basic utilities to inspect database tables and run queries)
# ==============================================================================
def list_tables(con: sqlite3.Connection) -> List[str]:

    """Returns a list of all tables in the SQLite database."""
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]

def table_columns(con: sqlite3.Connection, table: str) -> List[str]:

    """Returns a list of column names for a specific table."""
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [r[1] for r in rows]

def try_read_df(con: sqlite3.Connection, query: str) -> pd.DataFrame:
    
    """Safe wrapper for reading SQL into a DataFrame."""
    return pd.read_sql_query(query, con)


# ==============================================================================
# 2. Schemas & Finders (Runtime, NVTX, Kernel, etc.)
# ==============================================================================

# --- Runtime Schema (CPU API calls) ---
@dataclass
class RuntimeSchema:
    table: str
    cbid_col: str
    start_col: str
    end_col: str
    corr_id_col: str

def find_runtime_schema(con: sqlite3.Connection) -> Optional[RuntimeSchema]:

    #all tables from sqlite
    tables = list_tables(con)

    #all possible names for table in sqlite in case of using another version of nsys
    candidates = [t for t in tables if t == "CUPTI_ACTIVITY_KIND_RUNTIME"]

    if not candidates:
        candidates = [t for t in tables if "RUNTIME" in t.upper() and "CUPTI" in t.upper()]

    if not candidates: 
        return None

    #all possible names for columns in the table in case of using another version of nsys
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
    #all tables from sqlite
    tables = list_tables(con)

    #all possible names for table in sqlite in case of using another version of nsys
     candidates = [t for t in tables if t.upper() == "CUPTI_ACTIVITY_KIND_KERNEL"]

    #all possible names for table in sqlite in case of using another version of nsys
    if not  candidates:
         candidates = [t for t in tables if "kernel" in t.lower() and "cupti" in t.lower()]

    if not  candidates: 
        return None


    #all possible names for columns in the table in case of using another version of nsys
    possible_start = ["start", "startNs", "timestamp_start", "start_time"]
    possible_end   = ["end", "endNs", "timestamp_end", "end_time"]
    possible_name_id = ["shortName", "shortNameId", "nameId", "demangledName"]
    possible_corr = ["correlationId", "correlation_id"]

    for t in  candidates:

        cols = set(table_columns(con, t))
        sc = next((c for c in possible_start if c in cols), None)
        ec = next((c for c in possible_end if c in cols), None)
        nid = next((c for c in possible_name_id if c in cols), None)
        cid = next((c for c in possible_corr if c in cols), None)
        
        if sc and ec and nid:
            return KernelSchema(table=t, start_col=sc, end_col=ec, name_id_col=nid, corr_id_col=cid or "correlationId")

    return None


@dataclass
class NvtxSchema:
    table: str
    name_col: str
    start_col: str
    end_col: str

def find_nvtx_schema(con: sqlite3.Connection) -> Optional[NvtxSchema]:

    #all tables from sqlite
    tables = list_tables(con)

    #all possible names for table in sqlite in case of using another version of nsys
    priority_candidates = ["NVTX_EVENTS", "NVTX_PUSHPOP_RANGES"]
    
    #all possible names for columns in the table in case of using another version of nsys
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


@dataclass
class MemcpySchema:
    table: str
    start_col: str
    end_col: str
    kind_col: Optional[str] = None
    bytes_col: Optional[str] = None

def find_memcpy_schema(con: sqlite3.Connection) -> Optional[MemcpySchema]:

    #all tables from sqlite
    tables = list_tables(con)

    #all possible names for table in sqlite in case of using another version of nsys
    candidates = [t for t in tables if "memcpy" in t.lower()]

    if not candidates: 
        return None

    #all possible names for columns in the table in case of using another version of nsys
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


@dataclass
class StringIdsSchema:
    table: str
    id_col: str
    value_col: str

def find_stringids_schema(con: sqlite3.Connection) -> Optional[StringIdsSchema]:
  
    #all tables from sqlite
    tables = list_tables(con)

    #all possible names for table in sqlite in case of using another version of nsys
    candidates = [t for t in tables if t.lower() == "stringids"]

    if not candidates:
        candidates = [t for t in tables if "string" in t.lower() and "id" in t.lower()]

    #all possible names for columns in nvtx table in case of using another version of nsys
    possible_id_cols = ["id", "stringId", "string_id"]
    possible_value_cols = ["value", "text", "str", "string"]

    # Try candidates first
    for t in candidates:
        cols = set(table_columns(con, t))
        for ic in possible_id_cols:
            for vc in possible_value_cols:
                if ic in cols and vc in cols:
                    return StringIdsSchema(table=t, id_col=ic, value_col=vc)

    # 3) Fallback #2 (last resort):
    #    scan ALL tables and return the first table that has a matching (id_col, value_col)
    for t in tables:
        cols = set(table_columns(con, t))
        for ic in possible_id_cols:
            for vc in possible_value_cols:
                if ic in cols and vc in cols:
                    return StringIdsSchema(table=t, id_col=ic, value_col=vc)     

    return None

# ==============================================================================
# 3. Data Loading Functions
# ==============================================================================
def load_nvtx_events(
    con: sqlite3.Connection, 
    sch: NvtxSchema
    ) -> pd.DataFrame:

    q = f"""
    SELECT
        {sch.name_col} AS name,
        {sch.start_col} AS start,
        {sch.end_col} AS end
    FROM {sch.table}
    WHERE {sch.end_col} > {sch.start_col}
    """
    df = try_read_df(con, q)
    if df.empty:
        return pd.DataFrame(columns=["name", "start", "end", "dur_ns"])
    df["dur_ns"] = df["end"] - df["start"]
    return df

    #Ex) df_nvtx
    '''
    Example output df (after running load_nvtx_events):
           name     start      end    dur_ns
    0   step_0001  1000000  1200000  200000
    1   data_wait  1005000  1060000   55000
    2  gpu_compute  1060000  1195000  135000
    '''

def load_nccl_kernels(con: sqlite3.Connection,
                      kernel_sch: KernelSchema,
                      s_sch: StringIdsSchema,
                      like_pattern: str = "%nccl%") -> pd.DataFrame:
    """
    Loads kernel events that match a specific pattern (e.g., '%nccl%').
    Joins the Kernel table with the StringIds table to get human-readable names.
    """

    #query
    q = f"""
    SELECT
        s.{s_sch.value_col} AS name,
        k.{kernel_sch.start_col} AS start,
        k.{kernel_sch.end_col} AS end
    FROM {kernel_sch.table} k
    JOIN {s_sch.table} s
      ON k.{kernel_sch.name_id_col} = s.{s_sch.id_col}
    WHERE k.{kernel_sch.end_col} > k.{kernel_sch.start_col}
      AND s.{s_sch.value_col} LIKE '{like_pattern}'
    """

    df = try_read_df(con, q)

    df["dur_ns"] = df["end"] - df["start"]

    return df
    #Ex) df_nccl
    '''
    Example:
    - Kernel table has name IDs (not strings):
        k.start   k.end   k.shortNameId
        1000      1400    55
        2000      2600    77

    - StringIds table maps ID -> real string:
        s.id   s.value
        55     "ncclKernel_AllReduce_RING_LL"
        77     "ncclKernel_Broadcast_RING_LL"

    After JOIN + LIKE '%nccl%', the output df looks like:

                           name   start   end   dur_ns
    0  ncclKernel_AllReduce_RING_LL  1000  1400   400
    1  ncclKernel_Broadcast_RING_LL  2000  2600   600
    '''



def load_memcpy_events(
    con: sqlite3.Connection,
    sch: MemcpySchema,
    want_kinds: Optional[List[int]] = None,
    name: str = "gpu_memcpy_h2d"
) -> pd.DataFrame:

    #query
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


    # Kind filter
    if want_kinds is not None and sch.kind_col and "kind" in df.columns:
        df = df[df["kind"].isin(want_kinds)].copy()

    df["name"] = name
    df["dur_ns"] = df["end"] - df["start"]

    return df[["name", "start", "end", "dur_ns"]].copy()

    #Ex) df_memcpy 
    '''
    Example df from SQL (before filter):
        start   end   kind
     0  1000   1500   1
     1  2000   2600   2
    
    If want_kinds = [1], keep only kind==1:
        name            start   end   dur_ns
     0  gpu_memcpy_h2d  1000   1500  500
    '''


# ==============================================================================
# 4. [CORE LOGIC] True GPU Span Calculation
# ==============================================================================
def load_runtime_events_in_nvtx(
    con: sqlite3.Connection, 
    nvtx_df: pd.DataFrame, 
    run_sch: RuntimeSchema
) -> pd.DataFrame:
    
    """Loads Runtime API calls within NVTX ranges."""
    if nvtx_df.empty:
        return pd.DataFrame(columns=["correlationId", "start", "end", "cbid"])

    # We only load runtime events inside this window to reduce data size.
    min_start = nvtx_df["start"].min()
    max_end = nvtx_df["end"].max()

    print(f"Loading Runtime events ({min_start} ~ {max_end})...")


    select_cols = [
        f"{run_sch.corr_id_col} AS correlationId",
        f"{run_sch.start_col} AS start",
        f"{run_sch.end_col} AS end",
    ]
    if getattr(run_sch, "cbid_col", None):
        select_cols.append(f"{run_sch.cbid_col} AS cbid")

    q = f"""
    SELECT
        {", ".join(select_cols)}
    FROM {run_sch.table}
    WHERE {run_sch.start_col} >= {min_start}
      AND {run_sch.end_col} <= {max_end}
    """

    df = try_read_df(con, q)

    if df.empty:
        # Keep columns stable for callers
        cols = ["correlationId", "start", "end"]
        if "cbid" in df.columns or getattr(run_sch, "cbid_col", None):
            cols.append("cbid")
        return pd.DataFrame(columns=cols)

    return df.sort_values("start")

    # Ex) df_runtime
    '''
    Example output df (after running load_runtime_events_in_nvtx):

       correlationId     start      end   cbid
    0          1001   1002000  1002100     51
    1          1002   1006000  1006100     13
    2          1003   1060500  1060600     54
    '''


def load_all_kernels_for_span(
    con: sqlite3.Connection, 
    kernel_sch: KernelSchema
    ) -> pd.DataFrame:
    
    """
    Loads ALL GPU kernel events for "True GPU Span" calculation.
    We need correlationId to connect kernels to CPU Runtime calls.
    """

    q = f"""
    SELECT 
        {kernel_sch.corr_id_col} AS correlationId, 
        {kernel_sch.start_col} AS k_start, 
        {kernel_sch.end_col} AS k_end 
    FROM {kernel_sch.table} 
    WHERE {kernel_sch.end_col} > {kernel_sch.start_col}"""
    
    return try_read_df(con, q)
    # Ex) df_kernels_all
    '''
    Example output df (after running load_all_kernels_for_span):

       correlationId   k_start    k_end
    0          1001   1002200  1002600
    1          1001   1002700  1003100
    2          1002   1006200  1006900
    3          1003   1060700  1061500
    '''


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

    # Optimization: Filter only relevant NVTX rows (e.g. forward, backward, loss)
    target_keywords = ["step", "Batch","zero_grad","forward","loss", "backward",  "opt_step"]

    # Only process NVTX events that contain these keywords to save time
    
    for _, row in nvtx_sorted.iterrows():
        # Optional: Skip tiny NVTX ranges or irrelevant names to speed up
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

    #Ex) df_mapped_rows
    ''' Example output mapping_df (returned DataFrame):
    nvtx_name  nvtx_start  nvtx_end  correlationId
    0    step_1        1000      5000            101
    1    step_1        1000      5000            102
    2    step_1        1000      5000            103
    3    step_1        1000      5000            104
    4    step_1        1000      5000            105
    5    step_1        1000      5000            106
    6    step_1        1000      5000            107
    7   forward        1200      2500            102
    8   forward        1200      2500            103
    9      loss        2500      2700            104
    10 backward        2700      4200            105
    11 opt_step        4200      4800            106
    '''


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
    #Ex)true_gpu_df
    '''Example output result:
               name     start   end  dur_ns
    0  [GPU] forward    1100  1500     400
    1  [GPU] backward   2100  3400    1300
    '''

# -------------------------------------------------
# Step DF (only for drawing vertical lines / window)
# --------------------------------------------------
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
    #Ex) step_df
    '''
    Example input nvtx_df (simplified):
              name        start      end
    0     step_0001     1000000  1200000
    1     data_wait     1005000  1060000
    2    gpu_compute    1060000  1195000
    3     step 2        2000000  2300000
    4     Iter-2        2010000  2290000   # duplicate step=2 (inner range)
    
    Example output step_df (result of build_step_df_from_nvtx):
       step    start      end
    0    1   1000000  1200000
    1    2   2000000  2300000  # dedup => min start, max end among step=2 rows
    '''


def filter_by_step_ranges(source_df: pd.DataFrame, step_df_sel: pd.DataFrame, global_start: int) -> pd.DataFrame:
    """
    Filters events that fall within the selected step time ranges.
    Also calculates relative start time (ms) for plotting.
    """
    rows = []

    for _, srow in step_df_sel.iterrows():

        s = int(srow["step"])

        # s_start: start time of this step (ns)
        # s_end:   end time of this step (ns)
        s_start = int(srow["start"])
        s_end = int(srow["end"])

        
        # Keep all events that overlap with this step.
        # source_df can be NVTX / MEMCPY / NCCL events (they all have start/end timestamps).
        # Overlap condition:
        #   event.start < step.end   AND   event.end > step.start
        d = source_df[
            (source_df["start"] < s_end) & 
            (source_df["end"] > s_start)
        ].copy()
        
        if d.empty:
            continue
        
        # Tag these events with the current step number (so we know which step window caught them).
        d["step"] = s
        rows.append(d)
    
    if not rows:
        return pd.DataFrame()


    filtered_df = pd.concat(rows, ignore_index=True)
    
    # Drop duplicates in case an event spans across the boundary of two selected steps
    # We only want to draw it once per its unique identity (name, start, end)
    filtered_df = filtered_df.drop_duplicates(subset=["name", "start", "end"])

    # Convert to ms relative to the very first step's start
    filtered_df["rel_start_ms"] = (filtered_df["start"] - global_start) / 1e6
    filtered_df["dur_ms"] = (filtered_df["end"] - filtered_df["start"]) / 1e6

    return filtered_df
    # Ex) filtered_df
    '''

    filtered_df (final output):
            name     start      end    dur_ns  step  rel_start_ms  dur_ms
    0    data_wait  1005000  1060000   55000     1        0.5       55.0
    1   gpu_compute  1060000  1195000  135000     1        6.0      135.0
    2   gpu_compute  1995000  2050000   55000     2      995.0       55.0
    3   ncclKernelX  2295000  2310000   15000     2     1295.0       15.0
    4    data_wait  1199000  1201000    2000     1      199.0        2.0
    '''
    
    
def plot_timeline_combined(
    nvtx_df: pd.DataFrame,
    nccl_df: Optional[pd.DataFrame],
    memcpy_df: Optional[pd.DataFrame],
    true_gpu_df: Optional[pd.DataFrame],
    steps_to_plot: List[int],
    out_png="timeline_combined.png", 
    show=True
):

    print(f"Plotting steps: {steps_to_plot}...")

    # Steps are only used for the window range and the vertical boundary lines.
    try:
        step_df = build_step_df_from_nvtx(nvtx_df)
    except RuntimeError as e:
        print(f"[Error] {e}")
        return


    # Select only the steps we want to plot (steps_to_plot),
    # then sort them by start time so the timeline window is in correct order.
    step_df_sel = step_df[step_df["step"].isin(steps_to_plot)].sort_values("start")


    if step_df_sel.empty:
        print(f"No ranges found for steps={steps_to_plot}. Check your step numbers.")
        return

    # Define the overall time window for plotting:
    # global_start = the earliest start among the selected steps
    global_start = step_df_sel["start"].min()

    # Filter Data
    # 1) --- NVTX ---
    # Filter NVTX events so we only keep events that overlap the selected steps.
    plot_nvtx = filter_by_step_ranges(nvtx_df, step_df_sel, global_start)

    # 2) --- NCCL ---
    plot_nccl = pd.DataFrame()
    if nccl_df is not None and not nccl_df.empty:
        # filter NCCL kernel events the same way (keep only events overlapping selected steps).
        plot_nccl = filter_by_step_ranges(nccl_df, step_df_sel, global_start)

    # 3) --- MEMCPY ---
    plot_memcpy = pd.DataFrame()
    if memcpy_df is not None and not memcpy_df.empty:
        # filter memcpy events the same way (keep only events overlapping selected steps).
        plot_memcpy = filter_by_step_ranges(memcpy_df, step_df_sel, global_start)

    #4) ----GPU COMPUTE----------
    plot_true_gpu = pd.DataFrame()
    if true_gpu_df is not None and not true_gpu_df.empty:
        plot_true_gpu = filter_by_step_ranges(true_gpu_df, step_df_sel, global_start) 


    # ----------------------------------------
    # Determine Y-Axis Rows
    # ----------------------------------------
    # Priority: Base NVTX -> Layers (Compute) -> NCCL (Communication) -> MEMCPY (Transfer)
    base_names = ["data_wait", "h2d", "gpu_memcpy_h2d", "Optimizer", "opt_step", "loss", "backward", "forward","gpu_compute"]

    y_names = [n for n in base_names if n in plot_nvtx["name"].values or n in plot_memcpy["name"].values]
    
    # 2. True GPU Spans (Purple) - Insert them right after their CPU counterparts? 
    #    Or list them all together? Let's list them together for clarity or near the CPU one.
    #    Strategy: Add all [GPU] ... names
    gpu_names = sorted(plot_true_gpu["name"].unique())
    y_names += gpu_names
    
    # 3. NCCL
    if not plot_nccl.empty:
        # Simplify complex NCCL kernel names for cleaner plotting
        def simplify_name(raw_name):
            lower = raw_name.lower()
            if "allreduce" in lower: return "NCCL AllReduce"
            if "broadcast" in lower: return "NCCL Broadcast"
            if "allgather" in lower: return "NCCL AllGather"
            if "reducescatter" in lower: return "NCCL ReduceScatter"
            if "send" in lower or "recv" in lower: return "NCCL P2P"
            return "NCCL Other"

        plot_nccl["name"] = plot_nccl["name"].apply(simplify_name)
        
        # Add simplified NCCL names to Y-axis
        nccl_names = sorted(plot_nccl["name"].unique().tolist())
        y_names += nccl_names


    y_names = list(dict.fromkeys(y_names)) # Dedup
    y_map = {n: i for i, n in enumerate(y_names)}

    # Combine All
    combined = pd.concat([plot_nvtx, plot_nccl, plot_memcpy, plot_true_gpu], ignore_index=True)
    combined = combined[combined["name"].isin(y_names)]

    if combined.empty:
        print("Nothing to plot.")
        return

    # -----------------------------
    # Draw using Matplotlib
    # -----------------------------

    plt.figure(figsize=(18, max(5, 0.5 * len(y_names) + 2)))
    ax = plt.gca()

    for name in y_names:
        sub = combined[combined["name"] == name]
        segs = sub[["rel_start_ms", "dur_ms"]].to_numpy()
        if len(segs) == 0: continue
        
        y = y_map[name]
        
        # Color Logic
        if "[GPU]" in name:
            fc, alpha = 'tab:purple', 0.9  # True GPU Span
        elif "NCCL" in name:
            fc, alpha = 'tab:red', 0.8     # Comm
        elif "memcpy" in name:
            fc, alpha = 'tab:orange', 0.8  # Memcpy
        elif name in ["forward", "backward", "loss", "opt_step","gpu_compute"]:
            fc, alpha = 'tab:green', 0.5   # CPU High-level
        else:
            fc, alpha = 'tab:gray', 0.5    # Misc

        ax.broken_barh(segs, (y - 0.35, 0.7), facecolors=fc, alpha=alpha, edgecolor='black', linewidth=0.5)

    # Step Lines
    for _, srow in step_df_sel.iterrows():
        x0 = (srow["start"] - global_start) / 1e6
        x1 = (srow["end"] - global_start) / 1e6
        ax.axvline(x0, color='k', ls='--', alpha=0.5)
        ax.axvline(x1, color='k', ls=':', alpha=0.5)
        ax.text((x0+x1)/2, len(y_names), f"Step {int(srow['step'])}", ha='center', va='bottom', weight='bold')

    ax.set_yticks(range(len(y_names)))
    ax.set_yticklabels(y_names)
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"CPU vs True GPU Timeline | Steps: {steps_to_plot}")
    ax.grid(True, axis='x', linestyle=':', alpha=0.3)
    ax.set_ylim(-0.5, len(y_names) + 1.0)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved to {out_png}")
    if show: plt.show()

# ==============================================================================
# 6. Main Execution
# ==============================================================================
def process_full_analysis(con: sqlite3.Connection, steps: List[int]):
    print("--- 1. Analyzing Schemas ---")
    nvtx_sch = find_nvtx_schema(con)
    kernel_sch = find_kernel_schema(con)
    run_sch = find_runtime_schema(con)
    s_sch = find_stringids_schema(con)
    m_sch = find_memcpy_schema(con)
    
    if not (nvtx_sch and kernel_sch and run_sch):
        print("[Error] Critical tables (NVTX, Kernel, or Runtime) missing.")
        return

    print("--- 2. Loading Basic Events ---")
    nvtx_df = load_nvtx_events(con, nvtx_sch)
    
    nccl_df = pd.DataFrame()
    if s_sch:
        nccl_df = load_nccl_kernels(con, kernel_sch, s_sch)
        print(f"-> {len(nccl_df)} NCCL events.")

    memcpy_df = pd.DataFrame()
    if m_sch:
        memcpy_df = load_memcpy_events(con, m_sch, want_kinds=[1]) # H2D
        print(f"-> {len(memcpy_df)} Memcpy events.")

    print("--- 3. Computing True GPU Spans ---")
    # A. Runtime Events
    runtime_df = load_runtime_events_in_nvtx(con, nvtx_df, run_sch)
    # B. Link NVTX -> Runtime
    mapping_df = map_nvtx_to_runtime(nvtx_df, runtime_df)
    # C. Load All Kernels (for span calc)
    all_kernels = load_all_kernels_for_span(con, kernel_sch)
    # D. Compute Spans
    true_gpu_df = compute_true_gpu_spans(mapping_df, all_kernels)
    print(f"-> Generated {len(true_gpu_df)} true GPU spans.")

    # Plot
    plot_timeline_combined(
        nvtx_df=nvtx_df,
        nccl_df=nccl_df,
        memcpy_df=memcpy_df,
        true_gpu_df=true_gpu_df,
        steps_to_plot=steps,
        out_png="timeline_gpu_span.png"
    )

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_full.py <nsys.sqlite> <step_N> ...")
        sys.exit(1)

    sqlite_path = sys.argv[1]
    raw_steps = [int(x) for x in sys.argv[2:]]
    steps = [s for s in raw_steps if s > 0] if any(s > 0 for s in raw_steps) else raw_steps

    con = sqlite3.connect(sqlite_path)
    try:
        process_full_analysis(con, steps)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Fatal Error] {e}")
    finally:
        con.close()