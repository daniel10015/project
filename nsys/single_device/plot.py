import sys
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Generic SQLite Helpers
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
# 2. StringIds & Kernel Logic (For NCCL Analysis)
#    (Detects schema versions and loads GPU kernel data)
# ==============================================================================
@dataclass
class StringIdsSchema:
    table: str
    id_col: str
    value_col: str

def find_stringids_schema(con: sqlite3.Connection) -> Optional[StringIdsSchema]:
    """
    Dynamically finds the table responsible for String ID mapping.
    Nsys schema changes often, so we probe for likely table/column names.
    """
    tables = list_tables(con)
    candidates = [t for t in tables if t.lower() == "stringids"]
    if not candidates:
        candidates = [t for t in tables if "string" in t.lower() and "id" in t.lower()]

    possible_id_cols = ["id", "stringId", "string_id"]
    possible_value_cols = ["value", "text", "str", "string"]

    for t in candidates:
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
    tables = list_tables(con)
    kernel_tables = [t for t in tables if t.upper() == "CUPTI_ACTIVITY_KIND_KERNEL"]
    if not kernel_tables:
        kernel_tables = [t for t in tables if "kernel" in t.lower() and "cupti" in t.lower()]
    if not kernel_tables:
        return None

    possible_start = ["start", "startNs", "timestamp_start", "start_time"]
    possible_end   = ["end", "endNs", "timestamp_end", "end_time"]
    possible_name_id = [
        "shortName", "shortNameId", "nameId",
        "demangledName", "demangledNameId",
        "mangledName", "mangledNameId",
    ]

    for t in kernel_tables:
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






# ==============================================================================
# 3. NVTX Logic (For Timeline Plotting)
#    (Loads NVTX markers and identifies Step/Batch ranges)
# ==============================================================================
@dataclass
class NvtxSchema:
    table: str
    name_col: str
    start_col: str
    end_col: str

def find_nvtx_schema(con: sqlite3.Connection) -> Optional[NvtxSchema]:
    """Dynamically finds the NVTX events table."""
    tables = list_tables(con)
    possible_name_cols = ["text", "name", "message", "label"]
    possible_start_cols = ["start", "startNs", "start_time", "timestamp_start"]
    possible_end_cols = ["end", "endNs", "end_time", "timestamp_end"]

    nvtx_tables = [t for t in tables if "nvtx" in t.lower()]
    fallback_tables = tables

    def probe(table_list: List[str]) -> Optional[NvtxSchema]:
        for t in table_list:
            cols = set(table_columns(con, t))
            for nc in possible_name_cols:
                for sc in possible_start_cols:
                    for ec in possible_end_cols:
                        if nc in cols and sc in cols and ec in cols:
                            return NvtxSchema(table=t, name_col=nc, start_col=sc, end_col=ec)
        return None

    sch = probe(nvtx_tables)
    if sch: return sch
    return probe(fallback_tables)

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

def build_step_df_from_nvtx(nvtx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies step boundaries by looking for 'Batch_X' or 'step_X' patterns.
    Currently hardcoded to look for 'Batch_'.
    """
    # Looks for 'Batch_0', 'Batch_1', etc.
    step_ranges = nvtx_df[nvtx_df["name"].str.match(r"Batch_\d+$", na=False)].copy()
    
    if step_ranges.empty:
        # Fallback: Try looking for 'step_' if 'Batch_' is missing
        step_ranges = nvtx_df[nvtx_df["name"].str.match(r"step_\d+$", na=False)].copy()
        if step_ranges.empty:
            raise RuntimeError("Couldn't find NVTX step ranges (checked 'Batch_X' and 'step_X').")
        
        # Extract step number from 'step_X'
        step_ranges["step"] = step_ranges["name"].str.extract(r"step_(\d+)").astype(int)
    else:
        # Extract step number from 'Batch_X'
        step_ranges["step"] = step_ranges["name"].str.extract(r"Batch_(\d+)").astype(int)

    step_ranges = step_ranges.sort_values("start")
    return step_ranges[["step", "start", "end"]].copy()


# ==============================================================================
# 4. Plotting Logic
# ==============================================================================
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

def plot_timeline_combined(
    nvtx_df: pd.DataFrame,
    nccl_df: Optional[pd.DataFrame],
    steps_to_plot: List[int],
    out_png: str = "timeline_combined.png",
    show: bool = True,
    top_n_layers: int = 15,
    include_layers: bool = True
):
    """
    Main plotting function. Merges NVTX and NCCL data and draws a Gantt chart.
    """
    print(f"Plotting steps: {steps_to_plot}...")

    # 1. Identify Step Ranges
    try:
        step_df = build_step_df_from_nvtx(nvtx_df)
    except RuntimeError as e:
        print(f"[Error] {e}")
        return

    # Filter for user-requested steps
    step_df_sel = step_df[step_df["step"].isin(steps_to_plot)].sort_values("start").copy()

    if step_df_sel.empty:
        print(f"No ranges found for steps={steps_to_plot}. Check your step numbers.")
        return

    global_start = step_df_sel["start"].min()

    # 2. Filter Data (NVTX & NCCL)
    plot_nvtx = filter_by_step_ranges(nvtx_df, step_df_sel, global_start)

    plot_nccl = pd.DataFrame()
    if nccl_df is not None and not nccl_df.empty:
        plot_nccl = filter_by_step_ranges(nccl_df, step_df_sel, global_start)

    if plot_nvtx.empty and plot_nccl.empty:
        print("No events found in selected steps.")
        return

    # 3. Determine Y-Axis Rows
    #    Priority: Base NVTX -> Layers (Compute) -> NCCL (Communication)
    base_names = ["data_wait", "h2d", "gpu_compute", "Optimizer"]
    y_names = [n for n in base_names if n in plot_nvtx["name"].values] # Only include if present

    if include_layers and not plot_nvtx.empty:
        # [FIX] Changed 'in_steps' to 'plot_nvtx'
        layer_names = _pick_layer_rows(plot_nvtx, base_names=base_names, top_n=top_n_layers)
        y_names += layer_names
        
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
    
    # Remove duplicates in y_names while preserving order
    y_names = list(dict.fromkeys(y_names))
    y_map = {name: i for i, name in enumerate(y_names)}

    # 4. Prepare Final Data for Plotting
    combined_plot = pd.concat([plot_nvtx, plot_nccl], ignore_index=True)
    combined_plot = combined_plot[combined_plot["name"].isin(y_names)].copy()
    
    if combined_plot.empty:
        print("Nothing to plot after filtering names.")
        return

    # -----------------------------
    # 5. Draw using Matplotlib
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

    # 6. Draw Step Boundaries & Labels
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
    ax.set_title(f"Combined Timeline (NVTX + NCCL) | Steps: {steps_to_plot}")
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
    
    # [FIX] Call the unified plotting function with both DataFrames
    plot_timeline_combined(
        nvtx_df=nvtx_df,
        nccl_df=nccl_df,      # Pass the loaded NCCL data here
        steps_to_plot=steps,
        out_png="timeline_combined.png",
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