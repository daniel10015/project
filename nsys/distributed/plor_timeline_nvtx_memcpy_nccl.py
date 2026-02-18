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

# -------------------------------------------
# MEMCPY extraction (Nsight Systems SQLite)
# -------------------------------------------
@dataclass
class MemcpySchema:
    table: str
    start_col: str
    end_col: str
    kind_col: Optional[str] = None   # e.g., copyKind
    bytes_col: Optional[str] = None  # e.g., bytes
    device_col: Optional[str] = None # optional


def find_memcpy_schema(con: sqlite3.Connection) -> Optional[MemcpySchema]:

    #all tables from sqlite
    tables = list_tables(con)

    candidates = []

    for t in tables:
        t_lower = t.lower()        
        if "memcpy" in t_lower:    
            candidates.append(t)   
    
    if not candidates:
        return None

    #all possible names for columns in the table in case of using another version of nsys
    possible_start = ["start", "startNs", "start_time", "timestamp_start"]
    possible_end   = ["end", "endNs", "end_time", "timestamp_end"]
    possible_kind  = ["copyKind", "kind", "memcpyKind", "srcKind"]
    possible_bytes = ["bytes", "byteCount", "size"]
    possible_dev   = ["deviceId", "device", "gpuId"]

    for t in candidates:

        #all columns from in the memcpy table
        cols = set(table_columns(con, t))

        start_col = next((c for c in possible_start if c in cols), None)
        end_col   = next((c for c in possible_end   if c in cols), None)
        if not start_col or not end_col:
            continue

        kind_col  = next((c for c in possible_kind  if c in cols), None)
        bytes_col = next((c for c in possible_bytes if c in cols), None)
        dev_col   = next((c for c in possible_dev   if c in cols), None)

        
        return MemcpySchema(
            table=t,
            start_col=start_col,
            end_col=end_col,
            kind_col=kind_col,
            bytes_col=bytes_col,
            device_col=dev_col,
        )
    return None


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

# ================================================================================
# StringIds & Kernel Logic (For NCCL Analysis)
# So for NCCL analysis, we need BOTH:
#   1) StringIds schema  (id -> string)
#   2) Kernel schema     (start/end + name_id column)
# Then we can JOIN them to get readable kernel names like "ncclKernel_AllReduce..."
# ================================================================================
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

@dataclass
class KernelSchema:
    table: str
    start_col: str
    end_col: str
    name_id_col: str

def find_kernel_schema(con: sqlite3.Connection) -> Optional[KernelSchema]:
    """
    Dynamically finds the CUDA Kernel table (CUPTI_ACTIVITY_KIND_KERNEL).
    """
     #all tables from sqlite
    tables = list_tables(con)


    #all possible names for table in sqlite in case of using another version of nsys
    candidates = [t for t in tables if t.upper() == "CUPTI_ACTIVITY_KIND_KERNEL"]
    if not candidates:
        candidates = [t for t in tables if "kernel" in t.lower() and "cupti" in t.lower()]
    if not candidates:
        return None

    #all possible names for columns in nvtx table in case of using another version of nsys
    possible_start = ["start", "startNs", "timestamp_start", "start_time"]
    possible_end   = ["end", "endNs", "timestamp_end", "end_time"]
    possible_name_id = [
        "shortName", "shortNameId", "nameId",
        "demangledName", "demangledNameId",
        "mangledName", "mangledNameId",
    ]


    for t in candidates:
        cols = set(table_columns(con, t))
        sc = next((c for c in possible_start if c in cols), None)
        ec = next((c for c in possible_end if c in cols), None)
        if not sc or not ec:
            continue
        nid = next((c for c in possible_name_id if c in cols), None)
        if nid:
            return KernelSchema(table=t, start_col=sc, end_col=ec, name_id_col=nid)

    return None

def load_nccl_kernels(con: sqlite3.Connection,
                      k_sch: KernelSchema,
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

# ==============================================================================
# NVTX Logic (For Timeline Plotting)
#    (Loads NVTX markers and identifies Step/Batch ranges)
# ==============================================================================
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
    
    #all possible names for columns in nvtx table in case of using another version of nsys
    possible_start = ["start", "startNs", "timestamp_start"]
    possible_end = ["end", "endNs", "timestamp_end"]
    possible_name = ["text", "message", "name"]


    for t_name in priority_candidates:
        actual_name = next((t for t in tables if t.upper() == t_name), None)

        if actual_name:
            cols = set(table_columns(con, actual_name))
            sc = next((c for c in possible_start if c in cols), None)
            ec = next((c for c in possible_end if c in cols), None)
            nc = next((c for c in possible_name if c in cols), None)

            if sc and ec and nc:
                return NvtxSchema(actual_name, nc, sc, ec)

    #fallback
    fallback = [t for t in tables if "NVTX" in t.upper()]
    
    for t in fallback:
        cols = set(table_columns(con, t))
        sc = next((c for c in possible_start if c in cols), None)
        ec = next((c for c in possible_end if c in cols), None)
        nc = next((c for c in possible_name if c in cols), None)

        if sc and ec and nc:
            return NvtxSchema(t, nc, sc, ec)


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

def _pick_layer_rows(nvtx_in_steps: pd.DataFrame, base_names: List[str], top_n: int) -> List[str]:
    """
    Selects the Top-N longest running layers (Compute) to plot.
    Filters out metadata names like 'Batch_', 'Iter', etc.
    """
    df = nvtx_in_steps.copy()
    
    df = df[~df["name"].isin(base_names)]
    df = df[~df["name"].str.match(r"step_\d+$", na=False)]
    df = df[~df["name"].str.startswith("Batch_", na=False)]
    df = df[~df["name"].str.startswith("Iter", na=False)]

    if df.empty:
        return []

    tot = df.groupby("name")["dur_ns"].sum().sort_values(ascending=False)
    return list(tot.head(top_n).index)


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


# ====================================
# Plotting (timeline / Gantt view)
# ====================================

def plot_timeline_combined(
    nvtx_df: pd.DataFrame,
    nccl_df: Optional[pd.DataFrame],
    memcpy_df: Optional[pd.DataFrame],
    steps_to_plot: List[int],
    out_png: str = "timeline_combined.png",
    show: bool = True,
    top_n_layers: int = 15,
    include_layers: bool = True
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
    step_df_sel = step_df[step_df["step"].isin(steps_to_plot)].sort_values("start").copy()

    if step_df_sel.empty:
        print(f"No ranges found for steps={steps_to_plot}. Check your step numbers.")
        return

    # Define the overall time window for plotting:
    # global_start = the earliest start among the selected steps
    global_start = step_df_sel["start"].min()

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

    # If all are empty, nothing to plot.
    if plot_nvtx.empty and plot_nccl.empty and plot_memcpy.empty:
        print("No events found in selected steps.")
        return

    # ----------------------------------------
    # Determine Y-Axis Rows
    # ----------------------------------------
    # Priority: Base NVTX -> Layers (Compute) -> NCCL (Communication) -> MEMCPY (Transfer)
    base_names = ["data_wait", "h2d", "gpu_compute", "Optimizer", "gpu_memcpy_h2d"]
    y_names = [n for n in base_names if (
        (not plot_nvtx.empty and n in plot_nvtx["name"].values) or
        (not plot_memcpy.empty and n in plot_memcpy["name"].values)
    )]

    # if include_layers and not plot_nvtx.empty:
    #     layer_names = _pick_layer_rows(plot_nvtx, base_names=base_names, top_n=top_n_layers)
    #     y_names += layer_names
    
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

    y_names += ['nccl_sync']
    y_names += ['forward']
    y_names += ['backward']
    y_names += ['loss']    
    y_names += ['opt_step']
   
    # Remove duplicates in y_names while preserving order
    y_names = list(dict.fromkeys(y_names))
    y_map = {name: i for i, name in enumerate(y_names)}
    print(y_names)

    # Prepare Final Data for Plotting
    combined_plot = pd.concat([plot_nvtx, plot_nccl, plot_memcpy], ignore_index=True)

    # Keep only the rows that has y_name as name
    combined_plot = combined_plot[combined_plot["name"].isin(y_names)].copy()
    
    if combined_plot.empty:
        print("Nothing to plot after filtering names.")
        return

    # -----------------------------
    # Draw using Matplotlib
    # -----------------------------
    # Dynamic height based on number of rows
    fig_h = max(4, 0.4 * len(y_names) + 2)
    plt.figure(figsize=(16, fig_h))
    ax = plt.gca()

    row_h = 0.7
    for name in y_names:
        # Extract segments for this specific row name
        sub = combined_plot[combined_plot["name"] == name]
        segs = sub[["rel_start_ms", "dur_ms"]].to_numpy()
        
        if len(segs) > 0:
            y = y_map[name]
            
            # [Color Logic]
            if "nccl" in name.lower():
                facecolor = 'tab:red'      # Communication
                alpha = 0.9
            elif "memcpy" in name.lower():
                facecolor = 'tab:orange'   # Transfer (Memcpy)
                alpha = 0.7
            elif name in base_names:
                facecolor = 'tab:green'    # System/Base
                alpha = 0.6
            else:
                facecolor = 'tab:blue'     # Computation
                alpha = 0.7
            
            # Draw the horizontal bars
            ax.broken_barh(segs, (y - row_h/2, row_h), 
                           facecolors=facecolor, alpha=alpha, 
                           edgecolor='black', linewidth=0.5)

    # Draw Step Boundaries & Labels
    for _, srow in step_df_sel.iterrows():
        s = int(srow["step"])
        x0 = (int(srow["start"]) - global_start) / 1e6
        x1 = (int(srow["end"]) - global_start) / 1e6
        
        # Dashed line for Start
        ax.axvline(x0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        # Dotted line for End
        ax.axvline(x1, color='black', linestyle=':', linewidth=1, alpha=0.5)
        
        # Step Label at the top
        ax.text((x0 + x1) / 2, len(y_names) + 0.2, f"Step {s}", 
            ha="center", va="bottom", fontsize=11, weight='bold', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
    # Final Formatting
    ax.set_yticks(range(len(y_names)))
    ax.set_yticklabels(y_names, fontsize=9)
    ax.set_xlabel("Time (ms) [relative to first selected step start]")
    ax.set_title(f"Combined Timeline (NVTX + NCCL + MEMCPY) | Steps: {steps_to_plot}")
    ax.grid(True, axis='x', linestyle=':', alpha=0.3)
    ax.set_ylim(-0.5, len(y_names) + 1.0) # Add some breathing room top/bottom

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"[Plot] Saved to {out_png}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close()




# ==============================================================================
# 5. Main Execution
# ==============================================================================
def process_full_analysis(con: sqlite3.Connection, steps: List[int]):
    """
    Orchestrates the entire analysis:
    1. Load NCCL data.
    2. Load NVTX data.
    3. Laod memcpy data.
    3. Merge and Plot.
    """
    print("--- 1. Loading NCCL Kernels ---")
    nccl_df = pd.DataFrame()
    
    # Try finding Schema
    s_sch = find_stringids_schema(con)
    k_sch = find_kernel_schema(con)
    
    if s_sch and k_sch:
        print(f"[Schema] Found StringIds: {s_sch.table}")
        print(f"[Schema] Found Kernels: {k_sch.table}")
        nccl_df = load_nccl_kernels(con, k_sch, s_sch, like_pattern="%nccl%")
        print(f"-> Found {len(nccl_df)} NCCL events.")
    else:
        print("[Warning] Could not find Kernel/String tables. Skipping NCCL.")

    print(f"\n--- 2. Loading NVTX Events & Plotting Steps: {steps} ---")
    nvtx_sch = find_nvtx_schema(con)
    if not nvtx_sch:
        print("[Error] NVTX table not found. Cannot proceed.")
        return

    print(f"[Schema] Found NVTX: {nvtx_sch.table}")
    nvtx_df = load_nvtx_events(con, nvtx_sch)

    print(f"\n--- 3. Loading MEMCPY Events ---")
    memcpy_df = pd.DataFrame()

    memcpy_sch = find_memcpy_schema(con)
    if not memcpy_sch:
        print("[Warning] Could not find a memcpy table. Skipping MEMCPY.")
    else:
        print(f"[Schema] Found MEMCPY: {memcpy_sch.table} kind_col={memcpy_sch.kind_col}")

        # Choose only the HtoD kinds based on your printed distribution.
        # Example: HtoD kind might be [1] (depends on nsys version / schema)
        HtoD_KIND = [1]

        memcpy_df = load_memcpy_events(con, memcpy_sch, want_kinds=HtoD_KIND, name="gpu_memcpy_h2d")
        print(f"-> Found {len(memcpy_df)} MEMCPY events.")

    # Call plotting with NVTX + NCCL + MEMCPY
    plot_timeline_combined(
        nvtx_df=nvtx_df,
        nccl_df=nccl_df,
        memcpy_df=memcpy_df,
        steps_to_plot=steps,
        out_png="timeline_nvtx_memcpy_nccl.png",
        show=True,
        top_n_layers=15,
        include_layers=True
    )



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python combined_analysis.py <nsys_export.sqlite> <step_N> [step_M ...]")
        print("Example: python combined_analysis.py report.sqlite 10 11 12")
        sys.exit(1)

    sqlite_path = sys.argv[1]
    raw_steps = [int(x) for x in sys.argv[2:]]
    
    # Filter out step 0 if needed, or keep it.
    steps = [s for s in raw_steps if s > 0]
    if not steps:
         print("[Info] No steps > 0 provided. Using provided steps including 0.")
         steps = raw_steps

    con = sqlite3.connect(sqlite_path)
    
    try:
        process_full_analysis(con, steps)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Fatal Error] {e}")
    finally:
        con.close()

