import sys
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: sqlite inspection
# -----------------------------

# take all tables and table names in the DB
def list_tables(con: sqlite3.Connection) -> List[str]:
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]

# take all columns and column names in a table
def table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    # row: (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows]


def find_table_with_columns(
    con: sqlite3.Connection,
    must_have: List[str],
    name_hint_keywords: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Find a table that contains all columns in must_have.
    Optionally prefer tables whose name contains any of name_hint_keywords.
    """
    tables = list_tables(con)
    candidates = []
    for t in tables:
        cols = set(table_columns(con, t))
        if all(c in cols for c in must_have):
            score = 0
            if name_hint_keywords:
                low = t.lower()
                for kw in name_hint_keywords:
                    if kw.lower() in low:
                        score += 1
            candidates.append((score, t))
    if not candidates:
        return None
    # Prefer highest score, then stable name
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def try_read_df(con: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, con)


# -----------------------------
# NVTX extraction
# -----------------------------
@dataclass
class NvtxSchema:
    table: str
    name_col: str
    start_col: str
    end_col: str

def find_nvtx_schema(con: sqlite3.Connection) -> Optional[NvtxSchema]:
    """
    Find NVTX table
    Try common NVTX table/column patterns.
    We want: 1) name/text/message,  2) start, 3) end.
    """
    tables = list_tables(con)

    # Common NVTX event tables often have these columns:
    # start, end, text OR name OR message
    possible_name_cols = ["text", "name", "message", "label"]
    possible_start_cols = ["start", "startNs", "start_time", "timestamp_start"]
    possible_end_cols = ["end", "endNs", "end_time", "timestamp_end"]

    # Prefer tables with "nvtx" in the name
    nvtx_tables = []
    for table_name in tables:
        lower_name = table_name.lower()
        if "nvtx" in lower_name:
            nvtx_tables.append(table_name)

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
    if sch:
        return sch
    return probe(fallback_tables)

def load_nvtx_events(con: sqlite3.Connection, sch: NvtxSchema) -> pd.DataFrame:
    """
    Load data from nvtx table and return df
    1) message, 2) start time, 3) end time, 4) duration
    """
    q = f"""
    SELECT
        {sch.name_col} AS name,
        {sch.start_col} AS start,
        {sch.end_col} AS end
    FROM {sch.table}
    WHERE {sch.end_col} > {sch.start_col}
    """
    df = try_read_df(con, q)
    df["dur_ns"] = df["end"] - df["start"]
    return df


# -----------------------------
# MEMCPY extraction (HtoD)
# -----------------------------
@dataclass
class MemcpySchema:
    table: str
    start_col: str
    end_col: str
    kind_col: str

def find_memcpy_schema(con: sqlite3.Connection) -> Optional[MemcpySchema]:
    """
    Find CUPTI_ACTIVITY_KIND_MEMCPY table
    We want a table with:  1) start, 2) end, 3) copyKind (or kind)
    Common Nsight Systems export tables:
      - CUPTI_ACTIVITY_KIND_MEMCPY: start, end, bytes, copyKind
    """
    # Try the most common first
    common = [
        ("CUPTI_ACTIVITY_KIND_MEMCPY", ["start", "end", "copyKind"]),
        ("CUPTI_ACTIVITY_KIND_MEMCPY2", ["start", "end", "copyKind"]),
    ]

    # Get the schema immediately if the table name matches "CUPTI_ACTIVITY_KIND_MEMCPY"
    tables = list_tables(con)

    for tname, req_cols in common:
        if tname in tables:
            cols = set(table_columns(con, tname))
            if all(c in cols for c in req_cols):
                return MemcpySchema(table=tname, start_col="start", end_col="end", kind_col="copyKind")

    # Try to find other tables
    # Heuristic search: any table that looks like memcpy activity
    # Must have start/end + a "copyKind" or "kind" column
    for t in tables:
        low = t.lower()
        if "memcpy" not in low and "copy" not in low:
            continue
        cols = set(table_columns(con, t))
        if "start" in cols and "end" in cols:
            if "copyKind" in cols:
                return MemcpySchema(table=t, start_col="start", end_col="end", kind_col="copyKind")
            if "kind" in cols:
                return MemcpySchema(table=t, start_col="start", end_col="end", kind_col="kind")

    # Final fallback: try any table with start/end/copyKind
    t = find_table_with_columns(con, ["start", "end", "copyKind"], name_hint_keywords=["memcpy", "cupti"])
    if t:
        return MemcpySchema(table=t, start_col="start", end_col="end", kind_col="copyKind")

    return None

def load_h2d_memcpy_events(con: sqlite3.Connection, sch: MemcpySchema) -> pd.DataFrame:
    # CUPTI copyKind codes: 1 = HtoD, 2 = DtoH, 8 = DtoD, etc.
    q = f"""
    SELECT
        {sch.start_col} AS start,
        {sch.end_col} AS end,
        {sch.kind_col} AS kind
    FROM {sch.table}
    WHERE {sch.end_col} > {sch.start_col}
    """
    df = try_read_df(con, q)
    df["dur_ns"] = df["end"] - df["start"]

    # HtoD only
    df = df[df["kind"] == 1].copy()
    return df


# -----------------------------
# Step matching & aggregation
# -----------------------------
def match_events_to_steps(step_df: pd.DataFrame, ev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each event to a step by time containment:
      step.start <= ev.start and ev.end <= step.end
      e.g.) 1) data_wait, 2) GPU compute, 3) HtoD, 4) CPU overhead  ========> step_000X
    Uses merge_asof on start time for speed.
    """

    # If ev_df or step_df is empty, no events will be assigned to steps
    if ev_df.empty or step_df.empty:
        out = ev_df.copy()
        out["step"] = np.nan
        return out


    # Ordering by start time
    step_sorted = step_df.sort_values("start")[["step", "start", "end"]].copy()
    ev_sorted = ev_df.sort_values("start").copy()

    # New merged dataframe
    merged = pd.merge_asof(
        ev_sorted,
        step_sorted,
        on="start",
        direction="backward",
        suffixes=("", "_step"),
    )

    # ev.end must be within step.end
    merged = merged[merged["end"] <= merged["end_step"]].copy()
    merged.rename(columns={"end_step": "step_end"}, inplace=True)
    return merged


def build_step_df_from_nvtx(nvtx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step ranges are named like "step_0003"
    """

    step_ranges = nvtx_df[nvtx_df["name"].str.match(r"step_\d{4}$", na=False)].copy()
    if step_ranges.empty:
        raise RuntimeError("Couldn't find NVTX step ranges named like 'step_0000'. Please check your NVTX step range names.")

    step_ranges["step"] = step_ranges["name"].str.extract(r"step_(\d{4})").astype(int)
    step_ranges = step_ranges.sort_values("start")
    # If multiple step ranges with the same step index exist, sum by step later
    return step_ranges[["step", "start", "end"]].copy()

def aggregate_graph_a(nvtx_df: pd.DataFrame, h2d_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine NVTX + HtoD
    """
    step_df = build_step_df_from_nvtx(nvtx_df)

    # 1) NVTX categories (data_wait, gpu_compute)
    cats = nvtx_df[nvtx_df["name"].isin(["data_wait", "gpu_compute"])].copy()
    cats_matched = match_events_to_steps(step_df, cats)
    nvtx_sum = cats_matched.groupby(["step", "name"])["dur_ns"].sum().unstack(fill_value=0)

    # 2) HtoD from memcpy (true transfer time)
    h2d_matched = match_events_to_steps(step_df, h2d_df)
    h2d_sum = h2d_matched.groupby("step")["dur_ns"].sum()

    # 3) Step total from NVTX step ranges
    step_total = (step_df["end"] - step_df["start"]).groupby(step_df["step"]).sum()

    out = pd.DataFrame(index=sorted(step_total.index))
    out["step_total_ns"] = step_total

    out["data_wait_ns"] = nvtx_sum["data_wait"] if "data_wait" in nvtx_sum.columns else 0
    out["gpu_compute_ns"] = nvtx_sum["gpu_compute"] if "gpu_compute" in nvtx_sum.columns else 0
    out["h2d_ns"] = h2d_sum.reindex(out.index).fillna(0).astype(np.int64)

    # CPU overhead = remainder
    out["cpu_overhead_ns"] = out["step_total_ns"] - (out["data_wait_ns"] + out["gpu_compute_ns"] + out["h2d_ns"])
    out["cpu_overhead_ns"] = out["cpu_overhead_ns"].clip(lower=0)

    # ns -> ms
    for c in out.columns:
        out[c.replace("_ns", "_ms")] = out[c] / 1e6

    return out.reset_index().rename(columns={"index": "step"})


# -----------------------------
# Plot
# -----------------------------
def plot_graph_a(df: pd.DataFrame, max_steps: int = 20, out_png: str = "graphA_step_breakdown.png", show: bool = True):
    dfp = df.sort_values("step").head(max_steps)

    steps = dfp["step"].to_numpy()
    data_wait = dfp["data_wait_ms"].to_numpy()
    h2d = dfp["h2d_ms"].to_numpy()
    gpu = dfp["gpu_compute_ms"].to_numpy()
    cpu = dfp["cpu_overhead_ms"].to_numpy()

    plt.figure(figsize=(12, 5))
    plt.bar(steps, data_wait, label="Data wait (NVTX)")
    plt.bar(steps, h2d, bottom=data_wait, label="HtoD memcpy (CUDA trace)")
    plt.bar(steps, gpu, bottom=data_wait + h2d, label="GPU compute (NVTX)")
    plt.bar(steps, cpu, bottom=data_wait + h2d + gpu, label="CPU overhead (remainder)")


    plt.yscale("log")
    plt.ylabel("Time (ms) [log scale]")

    plt.xlabel("Step")
    #plt.ylabel("Time (ms)")
    plt.title("Graph A: Step Breakdown (Nsight Systems sqlite)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"[Saved] {out_png}")


def plot_graph_a_grouped(df, max_steps=20, out_png="graphA_grouped.png", show=True):
    dfp = df.sort_values("step").head(max_steps)

    steps = dfp["step"].to_numpy()
    data_wait = dfp["data_wait_ms"].to_numpy()
    h2d = dfp["h2d_ms"].to_numpy()
    gpu = dfp["gpu_compute_ms"].to_numpy()
    cpu = dfp["cpu_overhead_ms"].to_numpy()

    width = 0.2  # bar width
    x = np.arange(len(steps))  # 0,1,2,... (step index)

    plt.figure(figsize=(12, 5))
    plt.bar(x - 1.5*width, data_wait, width, label="Data wait")
    plt.bar(x - 0.5*width, h2d,      width, label="HtoD")
    plt.bar(x + 0.5*width, gpu,      width, label="GPU compute")
    plt.bar(x + 1.5*width, cpu,      width, label="CPU overhead")

    plt.xticks(x, steps)  # set x-axis labels to step numbers
    plt.xlabel("Step")
    plt.yscale("log")
    plt.ylabel("Time (ms) [log scale]")
    #plt.ylabel("Time (ms)")
    plt.title("Graph A (Grouped): Step Breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"[Saved] {out_png}")


def plot_graph_a_broken_axis(
    df,
    max_steps: int = 20,
    out_png: str = "graphA_step_breakdown_broken_axis.png",
    show: bool = True,
    top_pad_ratio: float = 0.05,   # extra y-axis padding for the top plot
    bot_pad_ratio: float = 0.10,   # extra y-axis padding for the bottom plot
):
    dfp = df.sort_values("step").head(max_steps)

    steps = dfp["step"].to_numpy()
    data_wait = dfp["data_wait_ms"].to_numpy()
    h2d = dfp["h2d_ms"].to_numpy()
    gpu = dfp["gpu_compute_ms"].to_numpy()
    cpu = dfp["cpu_overhead_ms"].to_numpy()

    total = data_wait + h2d + gpu + cpu

    # --- Bottom (zoom-in) y-limit: pick based on typical steps excluding the largest outlier (e.g., step0)
    sorted_total = np.sort(total)
    if len(sorted_total) >= 2:
        typical_max = sorted_total[-2]   # second largest value
    else:
        typical_max = sorted_total[-1]

    bot_ymax = max(1.0, typical_max * (1.0 + bot_pad_ratio))  # ensure at least 1ms
    top_ymin = bot_ymax * 1.02  # start the top plot slightly above the bottom plot's max
    top_ymax = total.max() * (1.0 + top_pad_ratio)

    # --- figure with 2 axes (broken axis style)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 2]}
    )

    def draw_stacked(ax):
        ax.bar(steps, data_wait, label="Data wait (NVTX)")
        ax.bar(steps, h2d, bottom=data_wait, label="HtoD memcpy (CUDA trace)")
        ax.bar(steps, gpu, bottom=data_wait + h2d, label="GPU compute (NVTX)")
        ax.bar(steps, cpu, bottom=data_wait + h2d + gpu, label="CPU overhead (remainder)")

    # Draw the same stacked bars on both axes, but crop them using different y-limits
    draw_stacked(ax_top)
    draw_stacked(ax_bot)

    ax_bot.set_ylim(0, bot_ymax)
    ax_top.set_ylim(top_ymin, top_ymax)

    # --- diagonal break marks
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)  # hide x labels on the top plot
    ax_bot.xaxis.tick_bottom()

    d = 0.008  # diagonal size
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # left
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # right

    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)        # left
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # right

    # labels
    ax_bot.set_xlabel("Step")
    ax_bot.set_ylabel("Time (ms)")
    ax_top.set_title("Graph A: Step Breakdown (Broken Axis)")

    # show legend only once (on the top axis)
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"[Saved] {out_png}")

# -----------------------------
# Main
# -----------------------------
def main(sqlite_path: str, max_steps: int = 20):
    con = sqlite3.connect(sqlite_path)

    # Find NVTX
    nvtx_sch = find_nvtx_schema(con)
    if not nvtx_sch:
        raise RuntimeError("NVTX table not found. Did you run nsys with NVTX tracing enabled?")

    print(f"[NVTX] Using table={nvtx_sch.table}, name_col={nvtx_sch.name_col}, start_col={nvtx_sch.start_col}, end_col={nvtx_sch.end_col}")
    nvtx_df = load_nvtx_events(con, nvtx_sch)

    # Find MEMCPY (HtoD)
    memcpy_sch = find_memcpy_schema(con)
    if not memcpy_sch:
        print("[WARN] Could not find a MEMCPY table. HtoD will be set to 0.")
        h2d_df = pd.DataFrame(columns=["start", "end", "dur_ns", "kind"])
    else:
        print(f"[MEMCPY] Using table={memcpy_sch.table}, start={memcpy_sch.start_col}, end={memcpy_sch.end_col}, kind={memcpy_sch.kind_col}")
        h2d_df = load_h2d_memcpy_events(con, memcpy_sch)

    con.close()

    # Aggregate
    out = aggregate_graph_a(nvtx_df, h2d_df)

    # Print preview
    cols_show = ["step", "data_wait_ms", "h2d_ms", "gpu_compute_ms", "cpu_overhead_ms", "step_total_ms"]
    print(out[cols_show].head(10).to_string(index=False))

    # Plot
    plot_graph_a(out, max_steps=max_steps, out_png="graphA_step_breakdown.png", show=True)
    plot_graph_a_grouped(out, max_steps=max_steps, out_png="graphA_grouped.png", show=True)
    plot_graph_a_broken_axis(out, max_steps=max_steps, out_png="graphA_broken.png", show=True)
    plot_timeline_nvtx(nvtx_df, steps_to_plot=steps, out_png="timeline_nvtx.png", show=True, top_n_layers=12, include_layers=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_graph_a_from_nsys_sqlite.py <nsys_export.sqlite> [max_steps]")
        sys.exit(1)

    sqlite_path = sys.argv[1]
    max_steps = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
    main(sqlite_path, max_steps=max_steps)
