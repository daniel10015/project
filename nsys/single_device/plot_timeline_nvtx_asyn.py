import sys
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

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
# MEMCPY extraction (Nsight Systems SQLite)
# -----------------------------
@dataclass
class MemcpySchema:
    table: str
    start_col: str
    end_col: str
    kind_col: Optional[str] = None   # e.g., copyKind
    bytes_col: Optional[str] = None  # e.g., bytes
    device_col: Optional[str] = None # optional


def find_memcpy_schema(con: sqlite3.Connection) -> Optional[MemcpySchema]:
    tables = list_tables(con)

    candidates = [t for t in tables if "memcpy" in t.lower()]
    if not candidates:
        return None

    possible_start = ["start", "startNs", "start_time", "timestamp_start"]
    possible_end   = ["end", "endNs", "end_time", "timestamp_end"]
    possible_kind  = ["copyKind", "kind", "memcpyKind", "srcKind"]
    possible_bytes = ["bytes", "byteCount", "size"]
    possible_dev   = ["deviceId", "device", "gpuId"]

    for t in candidates:
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

    # kind 필터
    if want_kinds is not None and sch.kind_col and "kind" in df.columns:
        df = df[df["kind"].isin(want_kinds)].copy()

    df["name"] = name
    df["dur_ns"] = df["end"] - df["start"]
    return df[["name", "start", "end", "dur_ns"]].copy()


def debug_print_memcpy_kinds(con: sqlite3.Connection, sch: MemcpySchema):
    if not sch.kind_col:
        print("[MEMCPY] kind_col 없음 (HtoD 필터 숫자 확인 불가).")
        return
    q = f"""
    SELECT {sch.kind_col} AS kind, COUNT(*) AS n
    FROM {sch.table}
    GROUP BY {sch.kind_col}
    ORDER BY n DESC
    """
    df = try_read_df(con, q)
    print("[MEMCPY] kind distribution:")
    print(df.to_string(index=False))


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
    if df.empty:
        return pd.DataFrame(columns=["name", "start", "end", "dur_ns"])
    df["dur_ns"] = df["end"] - df["start"]
    return df


# -----------------------------
# Step DF (only for drawing vertical lines / window)
# -----------------------------
def build_step_df_from_nvtx(nvtx_df: pd.DataFrame) -> pd.DataFrame:
    step_ranges = nvtx_df[nvtx_df["name"].str.match(r"step_\d{4}$", na=False)].copy()
    if step_ranges.empty:
        raise RuntimeError("Couldn't find NVTX step ranges like 'step_0000'. Check your NVTX naming.")
    step_ranges["step"] = step_ranges["name"].str.extract(r"step_(\d{4})").astype(int)
    step_ranges = step_ranges.sort_values("start")
    return step_ranges[["step", "start", "end"]].copy()


# -----------------------------
# Timeline plotting (NO step clipping)
# -----------------------------
def _pick_layer_rows(df_window: pd.DataFrame, base_names: List[str], top_n: int) -> List[str]:
    df = df_window.copy()
    df = df[~df["name"].isin(base_names)]
    df = df[~df["name"].str.match(r"step_\d{4}$", na=False)]
    if df.empty:
        return []
    tot = df.groupby("name")["dur_ns"].sum().sort_values(ascending=False)
    return list(tot.head(top_n).index)


def plot_timeline_full_no_clip(
    all_df: pd.DataFrame,         # nvtx + memcpy 합친 df
    nvtx_df: pd.DataFrame,        # step 라인 찾기용 (NVTX만)
    steps_to_plot: List[int],
    out_png: str = "timeline_full_no_clip.png",
    show: bool = True,
    top_n_layers: int = 12,
    include_layers: bool = True,
):
    # step은 “창(window)”와 “세로선” 용도만
    step_df = build_step_df_from_nvtx(nvtx_df)
    step_df_sel = step_df[step_df["step"].isin(steps_to_plot)].sort_values("start").copy()
    if step_df_sel.empty:
        raise RuntimeError(f"No step ranges found for steps={steps_to_plot}")

    global_start = int(step_df_sel["start"].min())
    global_end   = int(step_df_sel["end"].max())

    # ✅ 중요: 이벤트를 step으로 자르지 않는다.
    # 그냥 window와 겹치는 것만 필터링하고, start/end는 원본 유지
    dfw = all_df[(all_df["end"] > global_start) & (all_df["start"] < global_end)].copy()
    if dfw.empty:
        raise RuntimeError("No events found in the selected window.")

    dfw["rel_start_ms"] = (dfw["start"] - global_start) / 1e6
    dfw["dur_ms"] = (dfw["end"] - dfw["start"]) / 1e6

    # base rows (네가 보고 싶은 row만)
    base_names = [
        "data_wait",
        "pre_data_loading",
        "h2d",
        "h2d_async_prefetch",
        "gpu_memcpy_h2d",   # ✅ memcpy name을 이걸로 넣음
        "gpu_compute",
    ]

    y_names = base_names.copy()
    if include_layers:
        layer_names = _pick_layer_rows(dfw, base_names=base_names, top_n=top_n_layers)
        y_names += layer_names

    plot_df = dfw[dfw["name"].isin(y_names)].copy()

    y_map = {name: i for i, name in enumerate(y_names)}
    plot_df["y"] = plot_df["name"].map(y_map)

    fig_h = max(4, 0.35 * len(y_names) + 2)
    plt.figure(figsize=(14, fig_h))
    ax = plt.gca()

    row_h = 0.8
    for name in y_names:
        segs = plot_df[plot_df["name"] == name][["rel_start_ms", "dur_ms"]].to_numpy()
        if len(segs) == 0:
            continue
        y = y_map[name]
        ax.broken_barh(segs, (y - row_h/2, row_h))

    # step boundary lines (표시만)
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
    ax.set_ylabel("Events")
    ax.set_title("Timeline (NO step clipping)")

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

    # --- NVTX ---
    nvtx_sch = find_nvtx_schema(con)
    if not nvtx_sch:
        raise RuntimeError("NVTX table not found. Did you run nsys with NVTX tracing enabled?")

    print(f"[NVTX] table={nvtx_sch.table} name={nvtx_sch.name_col} start={nvtx_sch.start_col} end={nvtx_sch.end_col}")
    nvtx_df = load_nvtx_events(con, nvtx_sch)

    # --- MEMCPY ---
    memcpy_sch = find_memcpy_schema(con)
    if not memcpy_sch:
        print("[MEMCPY] memcpy 테이블을 못 찾음. memcpy row는 비어있음.")
        memcpy_htod = pd.DataFrame(columns=["name", "start", "end", "dur_ns"])
    else:
        print(f"[MEMCPY] table={memcpy_sch.table} start={memcpy_sch.start_col} end={memcpy_sch.end_col} kind={memcpy_sch.kind_col}")
        debug_print_memcpy_kinds(con, memcpy_sch)

        # ✅ 네 출력 보고 HtoD kind만 지정
        # 예: HtoD가 copyKind=1이면:
        HtoD_KIND = [1]  # 필요하면 [1, 8] 처럼 가능

        # ✅ 절대 clip하지 않고 원본 이벤트 그대로
        memcpy_htod = load_memcpy_events(con, memcpy_sch, want_kinds=HtoD_KIND, name="gpu_memcpy_h2d")

    con.close()

    # --- 합치기 (NVTX + MEMCPY 그대로) ---
    all_df = pd.concat([nvtx_df, memcpy_htod], ignore_index=True)

    plot_timeline_full_no_clip(
        all_df=all_df,
        nvtx_df=nvtx_df,
        steps_to_plot=steps,
        out_png="timeline_nvtx_plus_memcpy_no_clip.png",
        show=True,
        top_n_layers=12,
        include_layers=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_timeline_no_clip.py <nsys_export.sqlite> <step0> [step1 step2 ...]")
        sys.exit(1)

    sqlite_path = sys.argv[1]
    steps = [int(x) for x in sys.argv[2:]]

    # step 0 skip (원하면 지워도 됨)
    steps = [s for s in steps if s != 0]
    if not steps:
        raise RuntimeError("After excluding step 0, no steps remain to plot. Please pass step >= 1.")

    main(sqlite_path, steps)
