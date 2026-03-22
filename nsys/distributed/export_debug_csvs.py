
import sys
import sqlite3
from pathlib import Path

import pandas as pd

# ── re-use all helpers from the main analysis script ──────────────────────────
# Both files must live in the same directory.
from plot_timeline_nvtx_mecmpy_nccl_gpu_compute_combined import (
    list_tables, table_columns, try_read_df,
    find_nvtx_schema, find_kernel_schema, find_runtime_schema,
    find_stringids_schema, find_memcpy_schema,
    load_nvtx_events, load_runtime_events_in_nvtx,
    load_all_kernels_for_span, load_nccl_kernels, load_memcpy_events,
    map_nvtx_to_runtime, compute_true_gpu_spans,
    build_step_df_from_nvtx,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, path: Path, label: str):
    """Save DataFrame as CSV and print a one-line summary."""
    if df is None or df.empty:
        print(f"  [SKIP]  {label} — empty or None")
        pd.DataFrame().to_csv(path, index=False)   # write empty file for visibility
        return
    df.to_csv(path, index=False)
    print(f"  [OK]    {label} — {len(df):,} rows  →  {path.name}")


def raw_table_csv(con: sqlite3.Connection, table: str, out_dir: Path):
    """Dump a raw table to CSV (capped at 200 k rows)."""
    try:
        df = try_read_df(con, f"SELECT * FROM \"{table}\" LIMIT 200000")
        save(df, out_dir / f"RAW_{table}.csv", f"RAW {table}")
    except Exception as exc:
        print(f"  [WARN]  Could not dump RAW {table}: {exc}")


# ── main export ───────────────────────────────────────────────────────────────

def export_all(sqlite_path: str, out_dir: str = "debug_tables"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(sqlite_path)
    print(f"\n{'─'*60}")
    print(f"  Source : {sqlite_path}")
    print(f"  Output : {out.resolve()}")
    print(f"{'─'*60}\n")

    # ── 00 · All tables ───────────────────────────────────────────────────────
    tables = list_tables(con)
    df_tables = pd.DataFrame({
        "table_name": tables,
        "row_count":  [
            con.execute(f"SELECT COUNT(*) FROM \"{t}\"").fetchone()[0]
            for t in tables
        ],
        "columns": [", ".join(table_columns(con, t)) for t in tables],
    })
    save(df_tables, out / "00_all_tables.csv", "All tables")

    # ── 01 · Schema detection ─────────────────────────────────────────────────
    nvtx_sch   = find_nvtx_schema(con)
    kernel_sch = find_kernel_schema(con)
    run_sch    = find_runtime_schema(con)
    s_sch      = find_stringids_schema(con)
    m_sch      = find_memcpy_schema(con)

    schema_rows = []
    def _row(name, obj):
        schema_rows.append({
            "schema":  name,
            "found":   obj is not None,
            "details": str(obj.__dict__) if obj else "NOT FOUND",
        })
    _row("NvtxSchema",      nvtx_sch)
    _row("KernelSchema",    kernel_sch)
    _row("RuntimeSchema",   run_sch)
    _row("StringIdsSchema", s_sch)
    _row("MemcpySchema",    m_sch)
    save(pd.DataFrame(schema_rows), out / "01_schema_detection.csv", "Schema detection")

    # ── 02 · NVTX events ──────────────────────────────────────────────────────
    nvtx_df = pd.DataFrame()
    if nvtx_sch:
        nvtx_df = load_nvtx_events(con, nvtx_sch)
        save(nvtx_df, out / "02_nvtx_events.csv", "NVTX events")
    else:
        print("  [SKIP]  NVTX events — schema not found")

    # ── 03 · Runtime (CPU API) events ────────────────────────────────────────
    runtime_df = pd.DataFrame()
    if run_sch and not nvtx_df.empty:
        runtime_df = load_runtime_events_in_nvtx(con, nvtx_df, run_sch)
        save(runtime_df, out / "03_runtime_events.csv", "Runtime events")
    else:
        print("  [SKIP]  Runtime events — missing schema or empty NVTX")

    # ── 04 · All kernel events ────────────────────────────────────────────────
    kernel_df = pd.DataFrame()
    if kernel_sch:
        kernel_df = load_all_kernels_for_span(con, kernel_sch)
        save(kernel_df, out / "04_kernel_events.csv", "Kernel events (all)")
    else:
        print("  [SKIP]  Kernel events — schema not found")

    # ── 05 · NCCL kernels ────────────────────────────────────────────────────
    nccl_df = pd.DataFrame()
    if kernel_sch and s_sch:
        nccl_df = load_nccl_kernels(con, kernel_sch, s_sch)
        save(nccl_df, out / "05_nccl_kernels.csv", "NCCL kernels")
    else:
        print("  [SKIP]  NCCL kernels — missing kernel or StringIds schema")

    # ── 06 · Memcpy events (all kinds) ───────────────────────────────────────
    memcpy_df = pd.DataFrame()
    if m_sch:
        memcpy_df = load_memcpy_events(con, m_sch, want_kinds=None, name="gpu_memcpy")
        # Also attach the raw kind column for debugging
        if not memcpy_df.empty and m_sch.kind_col:
            raw_mc = try_read_df(
                con,
                f"SELECT {m_sch.start_col} AS start, {m_sch.kind_col} AS kind "
                f"FROM {m_sch.table} WHERE {m_sch.end_col} > {m_sch.start_col}"
            )
            memcpy_df = memcpy_df.merge(raw_mc, on="start", how="left")
        save(memcpy_df, out / "06_memcpy_events.csv", "Memcpy events")
    else:
        print("  [SKIP]  Memcpy events — schema not found")

    # ── 07 · StringIds ────────────────────────────────────────────────────────
    if s_sch:
        sid_df = try_read_df(con, f"SELECT * FROM {s_sch.table}")
        save(sid_df, out / "07_stringids.csv", "StringIds")
    else:
        print("  [SKIP]  StringIds — schema not found")

    # ── 08 · Step ranges ─────────────────────────────────────────────────────
    if not nvtx_df.empty:
        step_df = build_step_df_from_nvtx(nvtx_df)
        if not step_df.empty:
            step_df["dur_ms"] = (step_df["end"] - step_df["start"]) / 1e6
        save(step_df, out / "08_step_ranges.csv", "Step ranges")
    else:
        print("  [SKIP]  Step ranges — empty NVTX")

    # ── 09 · NVTX → Runtime mapping ──────────────────────────────────────────
    mapping_df = pd.DataFrame()
    if not nvtx_df.empty and not runtime_df.empty:
        mapping_df = map_nvtx_to_runtime(nvtx_df, runtime_df)
        save(mapping_df, out / "09_mapping_nvtx_to_runtime.csv", "NVTX→Runtime mapping")
    else:
        print("  [SKIP]  NVTX→Runtime mapping — empty NVTX or Runtime df")

    # ── 10 · True GPU spans ───────────────────────────────────────────────────
    if not mapping_df.empty and not kernel_df.empty:
        true_gpu_df = compute_true_gpu_spans(mapping_df, kernel_df)
        if not true_gpu_df.empty:
            true_gpu_df["dur_ms"] = true_gpu_df["dur_ns"] / 1e6
        save(true_gpu_df, out / "10_true_gpu_spans.csv", "True GPU spans")
    else:
        print("  [SKIP]  True GPU spans — missing mapping or kernel df")

    # ── RAW table dumps ───────────────────────────────────────────────────────
    # print("\n  Dumping raw tables...")
    # for t in tables:
    #     raw_table_csv(con, t, out)

    con.close()
    print(f"\n{'─'*60}")
    print(f"  ✓  Export complete → {out.resolve()}")
    print(f"{'─'*60}\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_debug_csvs.py <nsys.sqlite> [output_dir]")
        sys.exit(1)

    sqlite_path = sys.argv[1]
    out_dir     = sys.argv[2] if len(sys.argv) >= 3 else "debug_tables"
    export_all(sqlite_path, out_dir)