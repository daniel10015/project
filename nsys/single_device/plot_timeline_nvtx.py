import sys
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: sqlite inspection
# -----------------------------
def list_tables(con: sqlite3.Connection) -> List[str]:
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]

def table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [r[1] for r in rows]

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
    if sch:
        return sch
    return probe(fallback_tables)

def load_nvtx_events(con: sqlite3.Connection, sch: NvtxSchema) -> pd.DataFrame:
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
# Step DF
# -----------------------------
def build_step_df_from_nvtx(nvtx_df: pd.DataFrame) -> pd.DataFrame:
    step_ranges = nvtx_df[nvtx_df["name"].str.match(r"step_\d{4}$", na=False)].copy()
    if step_ranges.empty:
        raise RuntimeError("Couldn't find NVTX step ranges like 'step_0000'. Check your NVTX naming.")

    step_ranges["step"] = step_ranges["name"].str.extract(r"step_(\d{4})").astype(int)
    step_ranges = step_ranges.sort_values("start")
    return step_ranges[["step", "start", "end"]].copy()


# -----------------------------
# Timeline plotting (NVTX)
# -----------------------------
def _pick_layer_rows(nvtx_in_steps: pd.DataFrame, base_names: List[str], top_n: int) -> List[str]:
    """
    Pick top-N layer-like NVTX names by total duration.
    Exclude base_names and exclude step_XXXX ranges.
    """
    df = nvtx_in_steps.copy()
    df = df[~df["name"].isin(base_names)]
    df = df[~df["name"].str.match(r"step_\d{4}$", na=False)]

    if df.empty:
        return []

    tot = df.groupby("name")["dur_ns"].sum().sort_values(ascending=False)
    return list(tot.head(top_n).index)

def plot_timeline_nvtx(
    nvtx_df: pd.DataFrame,
    steps_to_plot: List[int],
    out_png: str = "timeline_nvtx.png",
    show: bool = True,
    top_n_layers: int = 12,
    include_layers: bool = True,
):
    """
    Plot a Gantt-like NVTX timeline for selected steps.
    Rows:
      - data_wait, h2d, gpu_compute (always)
      - Top-N layer NVTX names (optional)

    X-axis is time in ms relative to the first selected step start.
    """
    step_df = build_step_df_from_nvtx(nvtx_df)

    # Filter step ranges for selected steps
    step_df_sel = step_df[step_df["step"].isin(steps_to_plot)].sort_values("start").copy()
    if step_df_sel.empty:
        raise RuntimeError(f"No step ranges found for steps={steps_to_plot}")

    global_start = step_df_sel["start"].min()
    global_end = step_df_sel["end"].max()

    # NVTX events that overlap the global window (limit data)
    dfw = nvtx_df[(nvtx_df["end"] > global_start) & (nvtx_df["start"] < global_end)].copy()

    # Only keep events fully inside each step window (cleaner)
    # We'll attach each event to a step by containment.
    # Simple containment check by iterating steps (steps_to_plot is usually small)
    rows = []
    for _, srow in step_df_sel.iterrows():
        s = int(srow["step"])
        s_start = int(srow["start"])
        s_end = int(srow["end"])

        d = dfw[(dfw["start"] >= s_start) & (dfw["end"] <= s_end)].copy()
        if d.empty:
            continue
        d["step"] = s
        rows.append(d)

    if not rows:
        raise RuntimeError("No NVTX events found inside the selected steps.")

    in_steps = pd.concat(rows, ignore_index=True)
    in_steps["rel_start_ms"] = (in_steps["start"] - global_start) / 1e6
    in_steps["dur_ms"] = (in_steps["end"] - in_steps["start"]) / 1e6

    base_names = ["data_wait", "h2d", "gpu_compute"]

    # Decide rows (y categories)
    y_names = base_names.copy()
    if include_layers:
        layer_names = _pick_layer_rows(in_steps, base_names=base_names, top_n=top_n_layers)
        y_names += layer_names

    # Keep only chosen names + step boundaries (we don’t plot step_XXXX itself as row)
    plot_df = in_steps[in_steps["name"].isin(y_names)].copy()

    # Build y mapping
    y_map = {name: i for i, name in enumerate(y_names)}
    plot_df["y"] = plot_df["name"].map(y_map)

    # -----------------------------
    # Draw
    # -----------------------------
    fig_h = max(4, 0.35 * len(y_names) + 2)
    plt.figure(figsize=(14, fig_h))
    ax = plt.gca()

    row_h = 0.8
    for name in y_names:
        segs = plot_df[plot_df["name"] == name][["rel_start_ms", "dur_ms"]].to_numpy()
        y = y_map[name]
        # broken_barh expects (xmin, width)
        ax.broken_barh(segs, (y - row_h/2, row_h))

    # Step boundary vertical lines + labels
    for _, srow in step_df_sel.iterrows():
        s = int(srow["step"])
        x0 = (int(srow["start"]) - global_start) / 1e6
        x1 = (int(srow["end"]) - global_start) / 1e6
        ax.axvline(x0, linewidth=1)
        ax.axvline(x1, linewidth=1)
        ax.text((x0 + x1) / 2, len(y_names) + 0.2, f"step {s}", ha="center", va="bottom")

    ax.set_yticks(range(len(y_names)))
    ax.set_yticklabels(y_names)
    ax.set_xlabel("Time (ms)  [relative to first selected step start]")
    ax.set_ylabel("NVTX ranges")
    ax.set_title("NVTX Timeline (Gantt view)")

    ax.set_ylim(-1, len(y_names) + 1)
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
def main(sqlite_path: str, steps: List[int]):
    con = sqlite3.connect(sqlite_path)

    nvtx_sch = find_nvtx_schema(con)
    if not nvtx_sch:
        raise RuntimeError("NVTX table not found. Did you run nsys with NVTX tracing enabled?")

    print(f"[NVTX] table={nvtx_sch.table} name={nvtx_sch.name_col} start={nvtx_sch.start_col} end={nvtx_sch.end_col}")
    nvtx_df = load_nvtx_events(con, nvtx_sch)
    con.close()

    # Timeline plot
    plot_timeline_nvtx(
        nvtx_df,
        steps_to_plot=steps,
        out_png="timeline_nvtx_data_wait_syn_H2D_asyn.png",
        show=True,
        top_n_layers=12,
        include_layers=True,
    )

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_timeline_nvtx.py <nsys_export.sqlite> <step0> [step1 step2 ...]")
        sys.exit(1)

    sqlite_path = sys.argv[1]
    steps = [int(x) for x in sys.argv[2:]]

    # step 0 skip
    steps = [s for s in steps if s != 0]

    if not steps:
        raise RuntimeError("After excluding step 0, no steps remain to plot. Please pass step >= 1.")


    main(sqlite_path, steps)
