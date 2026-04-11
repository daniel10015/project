# schemas.py
# ─────────────────────────────────────────────────────────────────────────────
# Dataclass definitions for all profiling schemas (NVTX, Kernel, Memcpy, etc.)
# and the find_* functions that auto-detect column names in SQLite databases.
#
# DEPENDENCIES:
#   db_helpers.py  ← list_tables(), table_columns()
#
# WHY THIS FILE EXISTS:
#   Nsight Systems changes its SQLite column names between versions.
#   For example, the kernel start time might be called "start", "startNs",
#   or "timestamp_start" depending on the version.
#   The find_* functions detect which columns actually exist so the rest of
#   the code doesn't break when the schema changes.
#
# Simple analogy:
#   Imagine you receive a box labelled "carrots" but sometimes the label says
#   "carrot", "Carrot", or "carrots_fresh". The find_* functions look inside
#   the box and tell you the exact label name, so you can always grab the
#   right item regardless of label variation.
# ─────────────────────────────────────────────────────────────────────────────

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .db_helpers import list_tables, table_columns


# ==============================================================================
# 1. GpuDataset  (top-level container for one rank's profiling data)
# ==============================================================================

@dataclass
class GpuDataset:
    """
    Holds all profiling data loaded from one GPU rank's SQLite file.

    Simple example:
        You run training with 4 GPUs (rank 0, 1, 2, 3).
        Each GPU produces one .sqlite file.
        Each file is loaded into one GpuDataset object.

        gpu_data_map = {
            0: GpuDataset(rank=0, df_nvtx=..., df_nccl=..., ...),
            1: GpuDataset(rank=1, ...),
            2: GpuDataset(rank=2, ...),
            3: GpuDataset(rank=3, ...),
        }

    Fields:
        rank                         → which GPU (0, 1, 2, 3, …)
        filename                     → path to the .sqlite file
        df_nvtx                      → CPU-side timing markers (NVTX ranges)
        df_nccl                      → NCCL communication kernel events
        df_memcpy                    → Host-to-Device memory copies
        df_gpu_duration              → True GPU kernel spans mapped to NVTX
        gpu_info                     → {gpu_id: gpu_name} dict
        df_kernel_occupancy_per_step → per-step kernel occupancy stats
        df_all_kernels_with_sm       → all kernels with SM% computed
    """
    rank:                         int
    filename:                     str
    df_nvtx:                      pd.DataFrame
    df_nccl:                      pd.DataFrame
    df_memcpy:                    pd.DataFrame
    df_gpu_duration:              pd.DataFrame
    gpu_info:                     Dict[int, str]
    df_kernel_occupancy_per_step: pd.DataFrame
    df_all_kernels_with_sm:       pd.DataFrame


# ==============================================================================
# 2. RuntimeSchema  (CPU Runtime API calls — cudaLaunchKernel, etc.)
# ==============================================================================

@dataclass
class RuntimeSchema:
    """
    Describes the CUPTI_ACTIVITY_KIND_RUNTIME table columns.

    This table records every time the CPU calls a CUDA runtime function such
    as cudaLaunchKernel or cudaMemcpyAsync.  Each call has a correlation_id
    that links it to the actual GPU kernel that ran later.

    Simple example:
        CPU calls cudaLaunchKernel at t=210ns → correlation_id = 1001
        GPU runs the kernel   at t=300ns with correlation_id = 1001
        → We can link CPU time 210ns to GPU time 300ns via correlation_id 1001

    
    Fields:
        table
            Actual table name found in the SQLite file.
            Usually 'CUPTI_ACTIVITY_KIND_RUNTIME'.
 
        cbid_col
            Callback ID column. An integer that identifies which CUDA API
            function was called.
            Example values:
                cbid=200  →  cudaLaunchKernel
                cbid=201  →  cudaMemcpyAsync
 
        start_col
            Column name for the CPU timestamp when this API call started.
            Unit: nanoseconds, CPU wall-clock.
 
        end_col
            Column name for the CPU timestamp when this API call ended.
            end - start is typically only a few microseconds, because the CPU
            only sends a command to the GPU stream — it does NOT wait for
            the GPU to finish.
 
        corr_id_col
            Column name for the integer that links this CPU call to its
            GPU kernel in CUPTI_ACTIVITY_KIND_KERNEL.
            Example: corr_id=1001 here matches corr_id=1001 in the kernel table.
    """
    table:       str
    cbid_col:    str
    start_col:   str
    end_col:     str
    corr_id_col: str


def find_runtime_schema(con: sqlite3.Connection) -> Optional[RuntimeSchema]:

    """
    Auto-detect the Runtime API table and its column names.

    Returns None if the table does not exist in this database.
    """

    tables     = list_tables(con)
    candidates = [t for t in tables if t == "CUPTI_ACTIVITY_KIND_RUNTIME"]

    if not candidates:
        candidates = [t for t in tables
                      if "RUNTIME" in t.upper() and "CUPTI" in t.upper()]
    if not candidates:
        return None

    for t in candidates:

        cols = set(table_columns(con, t))
        cid  = next((c for c in ["correlationId", "correlation_id"] if c in cols), None)
        sc   = next((c for c in ["start", "startNs", "timestamp_start"] if c in cols), None)
        ec   = next((c for c in ["end",   "endNs",   "timestamp_end"]   if c in cols), None)

        if cid and sc and ec:

            return RuntimeSchema(
                table=t, 
                cbid_col="cbid",
                start_col=sc, 
                end_col=ec, 
                corr_id_col=cid,
            )

    return None


# ==============================================================================
# 3. KernelSchema  (GPU kernel execution — CUPTI_ACTIVITY_KIND_KERNEL)
# ==============================================================================

@dataclass
class KernelSchema:
    """
    Describes the CUPTI_ACTIVITY_KIND_KERNEL table columns.

    This table records every GPU kernel that ran on the device, including
    its start/end timestamps, occupancy (block/grid sizes, registers, etc.),
    and a name ID that links to the StringIds table.

    Simple example:
        One row =  one GPU kernel execution:
            start=300ns, end=400ns (ran for 100ns)
            blockX=128, blockY=1, blockZ=1 (128 threads per block)
            gridX=256,  gridY=1, gridZ=1  (256 blocks total = 32768 threads)
            registersPerThread=32
            correlationId=1001 (links back to the cudaLaunchKernel call)
    Fields:
        table
            Actual table name, usually 'CUPTI_ACTIVITY_KIND_KERNEL'.
 
        start_col
            GPU timestamp when this kernel started executing.
            Unit: nanoseconds, GPU device clock (NOT CPU clock).
 
        end_col
            GPU timestamp when this kernel finished executing.
            end - start = true GPU execution time.
 
        name_id_col
            Integer ID that points to a row in the StringIds table, which
            holds the human-readable kernel name.
            Example: nameId=504  →  StringIds[504] = 'ncclAllReduceRingLL'
 
        corr_id_col
            Correlation ID linking this kernel to the CPU runtime call
            that launched it. Matches RuntimeSchema.corr_id_col.
 
        stream_id_col
            CUDA stream ID this kernel ran on.
            Kernels on different streams can run concurrently.
            Example: compute kernels on stream 7, NCCL kernels on stream 13.
 
        block_x_col / block_y_col / block_z_col
            Thread count in each dimension of one thread block.
            threads_per_block = blockX x blockY x blockZ.
 
        grid_x_col / grid_y_col / grid_z_col
            Block count in each dimension of the kernel grid.
            total_blocks = gridX x gridY x gridZ.
 
        registers_col
            Registers used per thread.
 
        static_shared_col
            Static shared memory allocated per thread block (bytes).
 
    """
    table:              str
    start_col:          str
    end_col:            str
    name_id_col:        str
    corr_id_col:        str = "correlationId"
    stream_id_col:      str = "streamId"
    block_x_col:        str = "blockX"
    block_y_col:        str = "blockY"
    block_z_col:        str = "blockZ"
    grid_x_col:         str = "gridX"
    grid_y_col:         str = "gridY"
    grid_z_col:         str = "gridZ"
    registers_col:      str = "registersPerThread"
    static_shared_col:  str = "staticSharedMemory"
    dynamic_shared_col: str = "dynamicSharedMemory"


def find_kernel_schema(con: sqlite3.Connection) -> Optional[KernelSchema]:
    """
    Auto-detect the GPU kernel table and its column names.

    Checks for 'CUPTI_ACTIVITY_KIND_KERNEL' first, then falls back to any
    table whose name contains 'kernel' and 'cupti'.

    Returns None if no matching table is found.
    """

    tables        = list_tables(con)

    kernel_tables = [t for t in tables if t.upper() == "CUPTI_ACTIVITY_KIND_KERNEL"]

    if not kernel_tables:

        kernel_tables = [t for t in tables
                         if "kernel" in t.lower() and "cupti" in t.lower()]

    if not kernel_tables:
        return None

    possible_start          = ["start", "startNs", "timestamp_start", "start_time"]
    possible_end            = ["end",   "endNs",   "timestamp_end",   "end_time"]
    possible_name_id        = ["shortName", "shortNameId", "nameId", "demangledName"]
    possible_corr           = ["correlationId", "correlation_id"]
    possible_stream         = ["streamId", "stream_id", "streamid"]
    possible_block_x        = ["blockX",              "block_x"]
    possible_block_y        = ["blockY",              "block_y"]
    possible_block_z        = ["blockZ",              "block_z"]
    possible_grid_x         = ["gridX",               "grid_x"]
    possible_grid_y         = ["gridY",               "grid_y"]
    possible_grid_z         = ["gridZ",               "grid_z"]
    possible_registers      = ["registersPerThread",  "registers_per_thread"]
    possible_static_shared  = ["staticSharedMemory",  "static_shared_memory"]
    possible_dynamic_shared = ["dynamicSharedMemory", "dynamic_shared_memory"]

    for t in kernel_tables:
        
        cols = set(table_columns(con, t))

        sc  = next((c for c in possible_start   if c in cols), None)
        ec  = next((c for c in possible_end     if c in cols), None)
        nid = next((c for c in possible_name_id if c in cols), None)
        cid = next((c for c in possible_corr    if c in cols), None)
        sid = next((c for c in possible_stream  if c in cols), None)

        if not (sc and ec and nid):
            continue

        bx  = next((c for c in possible_block_x        if c in cols), None)
        by  = next((c for c in possible_block_y        if c in cols), None)
        bz  = next((c for c in possible_block_z        if c in cols), None)
        gx  = next((c for c in possible_grid_x         if c in cols), None)
        gy  = next((c for c in possible_grid_y         if c in cols), None)
        gz  = next((c for c in possible_grid_z         if c in cols), None)
        reg = next((c for c in possible_registers      if c in cols), None)
        ss  = next((c for c in possible_static_shared  if c in cols), None)
        ds  = next((c for c in possible_dynamic_shared if c in cols), None)

        return KernelSchema(
            table              = t,
            start_col          = sc,
            end_col            = ec,
            name_id_col        = nid,
            corr_id_col        = cid or "correlationId",
            stream_id_col      = sid or "streamId",
            block_x_col        = bx  or "blockX",
            block_y_col        = by  or "blockY",
            block_z_col        = bz  or "blockZ",
            grid_x_col         = gx  or "gridX",
            grid_y_col         = gy  or "gridY",
            grid_z_col         = gz  or "gridZ",
            registers_col      = reg or "registersPerThread",
            static_shared_col  = ss  or "staticSharedMemory",
            dynamic_shared_col = ds  or "dynamicSharedMemory",
        )

    return None


# ==============================================================================
# 4. NvtxSchema  (CPU-side NVTX range markers)
# ==============================================================================

@dataclass
class NvtxSchema:
    """
    Describes the NVTX_EVENTS table columns.

    NVTX markers are the human-readable labels your training code pushes, like
    'forward', 'backward', 'Batch_0'.  They record CPU wall-clock time.

    Simple example:
        Row: name='forward', start=210ns, end=350ns
        → CPU spent 140ns executing forward-pass launch calls

    Fields:
        table
            Actual table name found in the SQLite file.
            Usually 'NVTX_EVENTS' or 'NVTX_PUSHPOP_RANGES'.
 
        name_col
            Column holding the marker label string.
            Example values: 'forward', 'backward', 'Batch_0', 'NCCL_AllReduce'
            Possible column names across Nsight versions: 'text', 'message', 'name'
 
        start_col
            Column for the CPU timestamp when the marker was pushed (range opened).
            Unit: nanoseconds, CPU wall-clock.
 
        end_col
            Column for the CPU timestamp when the marker was popped (range closed).
            end - start = CPU time spent in this phase (launch overhead only).
    """
    table:     str
    name_col:  str
    start_col: str
    end_col:   str


def find_nvtx_schema(con: sqlite3.Connection) -> Optional[NvtxSchema]:
    """
    Auto-detect the NVTX events table and its column names.

    Tries 'NVTX_EVENTS' and 'NVTX_PUSHPOP_RANGES' first, then falls back
    to any table whose name contains 'NVTX'.
    """

    tables             = list_tables(con)
    priority_candidates = ["NVTX_EVENTS", "NVTX_PUSHPOP_RANGES"]
    possible_start     = ["start", "startNs", "timestamp_start"]
    possible_end       = ["end",   "endNs",   "timestamp_end"]
    possible_name      = ["text",  "message", "name"]

    for t_name in priority_candidates:
        actual = next((t for t in tables if t.upper() == t_name), None)
        if actual:
            cols = set(table_columns(con, actual))
            sc   = next((c for c in possible_start if c in cols), None)
            ec   = next((c for c in possible_end   if c in cols), None)
            nc   = next((c for c in possible_name  if c in cols), None)

            if sc and ec and nc:

                return NvtxSchema(actual, nc, sc, ec)

    return None


# ==============================================================================
# 5. MemcpySchema  (Host-to-Device / Device-to-Host memory copies)
# ==============================================================================

@dataclass
class MemcpySchema:
    """
    Describes the CUPTI memcpy table columns.

    Each row records one DMA transfer (e.g. data from CPU RAM → GPU VRAM).
    The 'kind' column distinguishes H2D (kind=1), D2H (kind=2), etc.

    Simple example:
        H2D transfer: CPU sends one batch of images to the GPU
            start=150ns, end=160ns, bytes=25165824 (24MB), kind=1
 
    Fields:
        table
            → Actual table name, usually contains 'memcpy'.
            → Example: 'CUPTI_ACTIVITY_KIND_MEMCPY'.
 
        start_col
            → GPU timestamp when the DMA transfer started.
            → Unit: nanoseconds.
 
        end_col
            → GPU timestamp when the DMA transfer finished.
            → end - start = transfer duration.
 
        corr_id_col
            → Correlation ID linking this transfer to the cudaMemcpyAsync
            → CPU call that triggered it.
 
        kind_col
            → Integer identifying the transfer direction.
                1 = H2D  (Host → Device, CPU → GPU)   ← training input data
                2 = D2H  (Device → Host, GPU → CPU)   ← evaluation metrics
                8 = D2D  (Device → Device, within GPU)
            → None if this column does not exist in this Nsight version.
 
        bytes_col
            → Number of bytes transferred in this DMA operation.
            → None if this column does not exist in this Nsight version.
 
        stream_id_col
            → CUDA stream ID the DMA transfer used.
            → H2D transfers usually run on a dedicated copy stream separate
            → from the compute stream.
            → None if this column does not exist in this Nsight version.
    """
    table:         str
    start_col:     str
    end_col:       str
    corr_id_col:   str
    kind_col:      Optional[str] = None
    bytes_col:     Optional[str] = None
    stream_id_col: Optional[str] = None

def find_memcpy_schema(con: sqlite3.Connection) -> Optional[MemcpySchema]:
    """
    Auto-detect the memcpy table and its column names.

    Returns None if no memcpy table is found.
    """

    tables     = list_tables(con)
    candidates = [t for t in tables if "memcpy" in t.lower()]

    if not candidates:
        return None

    possible_start   = ["start", "startNs", "timestamp_start"]
    possible_end     = ["end",   "endNs",   "timestamp_end"]
    possible_corr_id = ["correlationId", "correlation_id"]
    possible_kind    = ["copyKind", "kind", "memcpyKind"]
    possible_bytes   = ["bytes", "byteCount", "size"]
    possible_stream  = ["streamId", "stream_id", "streamid"]

    for t in candidates:
        cols = set(table_columns(con, t))
        sc   = next((c for c in possible_start   if c in cols), None)
        ec   = next((c for c in possible_end     if c in cols), None)
        cid  = next((c for c in possible_corr_id if c in cols), None)
        if not sc or not ec:
            continue

        kc  = next((c for c in possible_kind   if c in cols), None)
        bc  = next((c for c in possible_bytes  if c in cols), None)
        sid = next((c for c in possible_stream if c in cols), None)

        return MemcpySchema(
            table         = t,
            start_col     = sc,
            end_col       = ec,
            corr_id_col   = cid,
            kind_col      = kc,
            bytes_col     = bc,
            stream_id_col = sid,
        )

    return None


# ==============================================================================
# 6. StringIdsSchema  (maps integer name IDs → human-readable kernel names)
# ==============================================================================

@dataclass
class StringIdsSchema:
    """
    Describes the StringIds lookup table.

    CUPTI stores kernel names as integer IDs to save space.
    This table maps those IDs back to readable strings.

    Simple example:
        StringIds table:
            id=504,  value='ncclAllReduceRingLLKernel'
            id=1023, value='volta_sgemm_128x32_tn'

        Kernel table row: nameId=504 → look up → 'ncclAllReduceRingLLKernel'
    
    Fields:
        table
             → Actual table name, usually 'StringIds' or 'STRINGIDS'.
 
        id_col
            →  Integer primary key column (the ID stored in the kernel table).
            →  In practice always named 'id'.
 
        value_col
            → String column holding the human-readable name.
            → In practice always named 'value'.
    """
    table:     str
    id_col:    str
    value_col: str


def find_stringids_schema(con: sqlite3.Connection) -> Optional[StringIdsSchema]:
    """
    Auto-detect the StringIds lookup table.

    Returns None if no table with 'string' and 'id' in its name is found.
    """
    tables     = list_tables(con)
    candidates = [t for t in tables
                  if "string" in t.lower() and "id" in t.lower()]
    for t in candidates:
        cols = set(table_columns(con, t))
        if "id" in cols and "value" in cols:
            return StringIdsSchema(t, "id", "value")
    return None

