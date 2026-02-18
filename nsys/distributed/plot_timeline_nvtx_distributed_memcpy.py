import sys
import os
import re
import sqlite3
import argparse
import glob
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# ==============================================================================
# 1. Generic SQLite Helpers
# ==============================================================================

def list_tables(con: sqlite3.Connection) -> List[str]:
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]

def table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [r[1] for r in rows]

def try_read_df(con: sqlite3.Connection, query: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, con)
    except Exception:
        return pd.DataFrame()


@dataclass
class NvtxSchema:
    table: str
    name_col: str
    start_col: str
    end_col: str

def find_nvtx_schema(con: sqlite3.Connection) -> Optional[NvtxSchema]:
    tables = list_tables(con)
    # NVTX 관련 테이블 후보군
    candidates = ["NVTX_EVENTS", "NVTX_PUSHPOP_RANGES"]
    for t_name in candidates + [t for t in tables if "NVTX" in t.upper()]:
        real_name = next((t for t in tables if t.upper() == t_name.upper()), None)
        if real_name:
            cols = set(table_columns(con, real_name))
            if "start" in cols and "end" in cols and "text" in cols:
                return NvtxSchema(real_name, "text", "start", "end")
            if "timestamp_start" in cols and "timestamp_end" in cols and "text" in cols: # Old schema
                 return NvtxSchema(real_name, "text", "timestamp_start", "timestamp_end")
    return None

@dataclass
class KernelSchema:
    table: str
    start_col: str
    end_col: str
    name_id_col: str

def find_kernel_schema(con: sqlite3.Connection) -> Optional[KernelSchema]:
    tables = list_tables(con)
    kt = next((t for t in tables if "KERNEL" in t.upper() and "CUPTI" in t.upper()), None)
    if not kt: return None
    cols = set(table_columns(con, kt))
    return KernelSchema(kt, "start", "end", "demangledName") if "demangledName" in cols else KernelSchema(kt, "start", "end", "shortName")

@dataclass
class MemcpySchema:
    table: str
    start_col: str
    end_col: str
    kind_col: str

def find_memcpy_schema(con: sqlite3.Connection) -> Optional[MemcpySchema]:
    tables = list_tables(con)
    mt = next((t for t in tables if "MEMCPY" in t.upper() and "CUPTI" in t.upper()), None)
    if not mt: return None
    return MemcpySchema(mt, "start", "end", "copyKind")

# ==============================================================================
# 2. Data Loaders
# ==============================================================================
def load_nvtx(con: sqlite3.Connection, sch: NvtxSchema) -> pd.DataFrame:
    q = f"SELECT {sch.name_col} as name, {sch.start_col} as start, {sch.end_col} as end FROM {sch.table} WHERE end > start"
    return try_read_df(con, q)

def load_nccl(con: sqlite3.Connection, k_sch: KernelSchema) -> pd.DataFrame:
    # Kernel 테이블에서 NCCL 이름만 필터링 (StringID 조인 없이 demangledName 사용 가정)
    # 만약 StringId 테이블이 별도라면 복잡해지므로, 여기서는 demangledName이 있는 최신 스키마 가정
    # 또는 간단히 NVTX에서 NCCL을 찾을 수도 있음. 여기서는 Kernel 기반 시도.
    cols = table_columns(con, k_sch.table)
    name_col = k_sch.name_id_col
    
    # String table join이 필요한 경우 (구버전 nsys)
    if "demangledName" not in cols and "shortName" not in cols:
        # 간단하게 NVTX에서 NCCL을 찾는 것으로 대체 (안전장치)
        return pd.DataFrame() 

    q = f"SELECT {name_col} as name, {k_sch.start_col} as start, {k_sch.end_col} as end FROM {k_sch.table}"
    df = try_read_df(con, q)
    if df.empty: return df
    
    # String lookup (if needed) - 생략하고 NVTX 우선 사용 권장하지만, 사용자가 Kernel을 원함.
    # 여기서는 DataFrame 상에서 필터링
    df = df[df["name"].astype(str).str.contains("nccl", case=False, na=False)].copy()
    return df

def load_memcpy(con: sqlite3.Connection, sch: MemcpySchema) -> pd.DataFrame:
    q = f"SELECT {sch.start_col} as start, {sch.end_col} as end, {sch.kind_col} as kind FROM {sch.table}"
    df = try_read_df(con, q)
    if df.empty: return df
    # H2D = 1 (Kind) usually
    df = df[df["kind"] == 1].copy()
    df["name"] = "gpu_memcpy_h2d"
    return df[["name", "start", "end"]]

# ==============================================================================
# 3. Multi-GPU Processing
# ==============================================================================
def get_rank_from_filename(filename: str, idx: int) -> int:
    match = re.search(r"rank_?(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else idx

def process_file(filepath: str, rank: int, steps: List[int]) -> pd.DataFrame:
    con = sqlite3.connect(filepath)
    dfs = []
    
    try:
        # 1. NVTX
        nvtx_sch = find_nvtx_schema(con)
        if nvtx_sch:
            df = load_nvtx(con, nvtx_sch)
            if not df.empty: dfs.append(df)
            
        # 2. NCCL (from Kernels or NVTX)
        k_sch = find_kernel_schema(con)
        if k_sch:
            df = load_nccl(con, k_sch)
            if not df.empty: dfs.append(df)

        # 3. Memcpy
        m_sch = find_memcpy_schema(con)
        if m_sch:
            df = load_memcpy(con, m_sch)
            if not df.empty: dfs.append(df)
            
    finally:
        con.close()
        
    if not dfs: return pd.DataFrame()
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["rank"] = rank
    full_df["dur_ns"] = full_df["end"] - full_df["start"]
    
    # --- Step Filtering (Local) ---
    # 각 랭크별로 'step' NVTX를 찾아 기준점을 잡습니다.
    # 사용자의 코드 로직을 따름
    step_pattern = r"(?i)(?:Batch|step|Iter|Iteration|Global Step)[_\s:\-]*(\d+)"
    full_df["step_id"] = pd.to_numeric(full_df["name"].str.extract(step_pattern, expand=False), errors='coerce')
    
    valid_steps = full_df.dropna(subset=["step_id"])
    if valid_steps.empty: return pd.DataFrame()

    # 사용자가 요청한 Step의 범위 찾기
    req_steps_df = valid_steps[valid_steps["step_id"].isin(steps)]
    if req_steps_df.empty: return pd.DataFrame()
    
    min_start = req_steps_df["start"].min()
    max_end = req_steps_df["end"].max()
    
    # 해당 시간 범위 내의 모든 이벤트 필터링
    # 여유분(margin) 조금 둠
    margin = 5_000_000 # 5ms
    mask = (full_df["end"] >= min_start - margin) & (full_df["start"] <= max_end + margin)
    filtered = full_df[mask].copy()
    
    # 시간 정규화 (이 랭크의 첫 번째 선택된 Step 시작점을 0으로)
    # *주의*: 멀티 GPU 정렬을 위해선 Global Reference가 필요하지만, 
    # 상세 뷰에서는 보통 "각 랭크의 Step 시작점"을 0으로 맞춰서 Jitter를 보거나(Align), 
    # 절대 시간으로 비교합니다. 여기서는 "절대 시간" 보존을 위해 Raw Start를 남겨두고 나중에 정렬합니다.
    return filtered

# ==============================================================================
# 4. Plotting (Advanced Stacked View)
# ==============================================================================
def simplify_nccl_name(name: str) -> str:
    lower = name.lower()
    if "allreduce" in lower: return "NCCL AllReduce"
    if "broadcast" in lower: return "NCCL Broadcast"
    if "allgather" in lower: return "NCCL AllGather"
    if "reducescatter" in lower: return "NCCL ReduceScatter"
    if "send" in lower or "recv" in lower: return "NCCL P2P"
    if "nccl" in lower: return "NCCL Other"
    return name

def plot_detailed_multi_gpu(combined_df: pd.DataFrame, steps: List[int], output_file: str):
    if combined_df.empty:
        print("No data available for plotting.")
        return

    # 1. Global Time Alignment
    # 모든 랭크 중 가장 먼저 시작된 Step의 시작시간을 0으로 잡습니다.
    # 이를 위해 각 랭크별로 요청된 첫 step의 시작 시간을 구합니다.
    start_times = combined_df[combined_df["step_id"] == steps[0]]["start"]
    if start_times.empty:
        global_t0 = combined_df["start"].min()
    else:
        global_t0 = start_times.min()

    combined_df["rel_start_ms"] = (combined_df["start"] - global_t0) / 1e6
    combined_df["dur_ms"] = combined_df["dur_ns"] / 1e6

    # 2. Y-Axis Categories (사용자 지정 리스트 + 자동 발견)
    # 사용자가 원했던 리스트 순서 적용
    priority_order = [
        "data_wait", "h2d", "gpu_memcpy_h2d", 
        "opt_step", "loss", "backward", "forward", 
        "nccl_sync"
    ]
    
    # 데이터에 있는 실제 이름들 정리
    combined_df["plot_name"] = combined_df["name"].apply(simplify_nccl_name)
    
    # 실제 데이터에 존재하는 것만 남김
    present_names = set(combined_df["plot_name"].unique())
    
    y_categories = []
    # 1) 우선순위 목록에 있는 것 먼저 추가
    for p in priority_order:
        if p in present_names:
            y_categories.append(p)
            present_names.remove(p)
            
    # 2) NCCL 관련 추가
    nccl_names = sorted([n for n in present_names if "NCCL" in n])
    y_categories.extend(nccl_names)
    for n in nccl_names: present_names.discard(n)
    
    # 3) 나머지 (기타 커널 등) - 너무 많으면 지저분하므로 중요 키워드만
    others = sorted(list(present_names))
    # 필터링: 너무 자잘한건 제외하거나 필요시 추가
    wanted_others = ["gpu_compute", "Optimizer"]
    for w in wanted_others:
        if w in others:
            y_categories.append(w)

    if not y_categories:
        print("No matching categories found in data.")
        return

    print(f"Y-Axis Categories: {y_categories}")

    # 3. Plot Configuration
    ranks = sorted(combined_df["rank"].unique())
    n_ranks = len(ranks)
    n_cats = len(y_categories)
    
    # 랭크가 많으면 그래프가 너무 뚱뚱해지지 않게 서브 바 높이 조절
    row_height = 0.8
    sub_bar_height = row_height / n_ranks
    
    fig_h = max(6, n_cats * 1.2) # 동적 높이
    fig, ax = plt.subplots(figsize=(18, fig_h))
    
    # 색상 팔레트
    colors = {
        "NCCL": "tab:red",
        "Memcpy": "tab:green",
        "Compute": "tab:blue",
        "Base": "gray"
    }

    # 4. Drawing
    for row_idx, cat_name in enumerate(y_categories):
        # Base Y position for this category
        y_base = row_idx - 0.5 + (1 - row_height)/2 # Center logic
        
        # Color decision
        if "NCCL" in cat_name: c = colors["NCCL"]
        elif "memcpy" in cat_name.lower() or "h2d" in cat_name.lower(): c = colors["Memcpy"]
        elif "forward" in cat_name or "backward" in cat_name or "compute" in cat_name: c = colors["Compute"]
        else: c = colors["Base"]

        # Loop Ranks (Sub-rows)
        for r_i, rank in enumerate(ranks):
            # Rank Order: Rank 0 at TOP of the row, Rank N at BOTTOM
            # y_pos = row_idx (integer) is center? No, let's use 0..N integers as centers
            # Let's map row_idx to y-center.
            
            # Logic: y_cat_center = row_idx
            # range: [row_idx - 0.4, row_idx + 0.4]
            # Rank 0: Top -> row_idx + 0.4 - sub_h
            
            y_offset = (n_ranks - 1 - r_i) * sub_bar_height # Stack from top
            final_y = (row_idx - 0.4) + y_offset # start from bottom of the row
            
            # Data subset
            sub_df = combined_df[(combined_df["plot_name"] == cat_name) & (combined_df["rank"] == rank)]
            if sub_df.empty: continue
            
            xranges = list(zip(sub_df["rel_start_ms"], sub_df["dur_ms"]))
            
            # Draw
            ax.broken_barh(xranges, (final_y, sub_bar_height), 
                           facecolors=c, edgecolor='black', linewidth=0.3, alpha=0.85)

    # 5. Decoration
    # Step boundaries (using Rank 0 as reference)
    ref_df = combined_df[(combined_df["rank"] == ranks[0]) & (combined_df["step_id"].notna())]
    if not ref_df.empty:
        step_starts = ref_df.groupby("step_id")["rel_start_ms"].min()
        for step_num, start_t in step_starts.items():
            if step_num not in steps: continue
            ax.axvline(x=start_t, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.text(start_t, n_cats - 0.5, f"Step {int(step_num)}", ha='left', va='bottom', fontsize=10, fontweight='bold', color='black')

    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(y_categories, fontsize=11, fontweight='bold')
    ax.set_ylim(-0.5, n_cats)
    ax.set_xlabel(f"Time (ms) - Relative to Rank {ranks[0]} Start")
    ax.set_title(f"Detailed Multi-GPU Timeline (Steps {steps}) | {n_ranks} Ranks Stacked per Row", fontsize=14)
    ax.grid(True, axis='x', linestyle=':', alpha=0.3)
    
    # Legend construction manually
    patches = [
        mpatches.Patch(color=colors["Compute"], label='Compute (Fwd/Bwd)'),
        mpatches.Patch(color=colors["NCCL"], label='Communication (NCCL)'),
        mpatches.Patch(color=colors["Memcpy"], label='Memory (H2D/Memcpy)'),
        mpatches.Patch(color='none', label=' '), # Spacer
        mpatches.Patch(edgecolor='black', facecolor='white', label='Top Strip: Rank 0'),
        mpatches.Patch(edgecolor='black', facecolor='white', label=f'Btm Strip: Rank {ranks[-1]}'),
    ]
    ax.legend(handles=patches, loc='upper right', ncol=3)

    plt.tight_layout()
    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("Done.")

# ==============================================================================
# 5. Main
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Sqlite files")
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="timeline_detailed_multi.png")
    args = parser.parse_args()

    # Expand globs
    file_list = []
    inputs = args.files if isinstance(args.files, list) else [args.files]
    for f in inputs:
        file_list.extend(glob.glob(f))
    file_list = sorted(list(set(file_list)))
    
    if not file_list:
        print("No files found.")
        sys.exit(1)
        
    print(f"Loading {len(file_list)} files...")
    
    all_dfs = []
    for idx, f in enumerate(file_list):
        rank = get_rank_from_filename(f, idx)
        print(f"Processing Rank {rank}: {f}")
        df = process_file(f, rank, args.steps)
        if not df.empty:
            all_dfs.append(df)
            
    if not all_dfs:
        print("No valid data found in given steps.")
        sys.exit(1)
        
    combined = pd.concat(all_dfs, ignore_index=True)
    
    plot_detailed_multi_gpu(combined, args.steps, args.output)