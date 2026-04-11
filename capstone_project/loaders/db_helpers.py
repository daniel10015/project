# db_helpers.py
# ─────────────────────────────────────────────────────────────────────────────
# Generic SQLite helper functions and stream-type classifiers.
#
# DEPENDENCIES:  stdlib + pandas only.
#                Does NOT import from any other file in this project.
#
# Every other file (schemas.py, data_loader.py, …) imports from here.
# Think of this as the "foundation" layer.
# ─────────────────────────────────────────────────────────────────────────────

import sqlite3
from typing import Dict, List

import pandas as pd


# ==============================================================================
# 1. Generic SQLite Helpers
# ==============================================================================

def list_tables(con: sqlite3.Connection) -> List[str]:
    """
    Return a list of every table name in this SQLite database.

    Simple example:
        Database has tables: NVTX_EVENTS, CUPTI_ACTIVITY_KIND_KERNEL, StringIds
        → returns ['NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds']
    """
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return [r[0] for r in rows]


def table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    """
    Return the column names for the given table.

    Simple example:
        Table 'NVTX_EVENTS' has columns: id, text, start, end
        → returns ['id', 'text', 'start', 'end']
    """
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [r[1] for r in rows]


def try_read_df(con: sqlite3.Connection, query: str) -> pd.DataFrame:
    """
    Run a SQL query and return the result as a pandas DataFrame.

    Simple example:
        query = "SELECT text, start, end FROM NVTX_EVENTS"
        → returns a DataFrame with those 3 columns
    """
    return pd.read_sql_query(query, con)


# ==============================================================================
# 2. Stream-Type Classifiers
# ==============================================================================
# These two functions look at a kernel name string and decide what kind of work
# the GPU was doing at that moment.
#
# Simple analogy:
#   A factory has workers doing different jobs:
#     - "nccl" workers  →  network communication (sending gradients)
#     - "adam" workers  →  optimizer step (updating weights)
#     - "sgemm" workers →  matrix math (forward/backward pass)
#   We read the worker's name tag and classify their job.
 
def classify_stream_type(kernel_name: str) -> str:
    """
    Classify a GPU kernel name into one of these categories:
        'nccl'      → NCCL communication kernel (AllReduce, etc.)
        'optimizer' → Optimizer kernels (Adam, SGD, etc.)
        'h2d'       → Host-to-Device memory copy
        'compute'   → Matrix math, convolution, batch-norm, etc.
        'unknown'   → Does not match any of the above
 
    Simple example:
        classify_stream_type("ncclAllReduceRingLL")   → 'nccl'
        classify_stream_type("volta_sgemm_128x32_tn") → 'compute'
        classify_stream_type("fused_adam_cuda")       → 'optimizer'
        classify_stream_type("memcpy_htod")           → 'h2d'
    """
    if not isinstance(kernel_name, str):
        return "unknown"
 
    n = kernel_name.lower()
 
    if "nccl" in n:
        return "nccl"
 
    if any(p in n for p in ["adam", "sgd", "lamb", "optimizer",
                              "multi_tensor", "fused_adam"]):
        return "optimizer"
 
    if any(p in n for p in ["memcpy", "memset", "htod", "h2d"]):
        return "h2d"
 
    if any(p in n for p in [
        "sgemm", "hgemm", "gemm",
        "cudnn", "conv", "bn_", "batchnorm",
        "elementwise", "reduce", "softmax",
        "at::native", "aten::", "volta_", "ampere_",
        "cutlass", "fmha", "flash_attn",
    ]):
        return "compute"
 
    return "unknown"
 
 
def _stream_type_from_nvtx_name(name: str) -> str:
    """
    Classify a GPU-duration name (derived from NVTX marker) into a stream type.
 
    These names come from the NVTX markers your training code emits, not raw
    kernel names. They are already human-readable (e.g. 'gpu_forward_duration').
 
    Simple example:
        _stream_type_from_nvtx_name("gpu_forward_duration")  → 'forward'
        _stream_type_from_nvtx_name("gpu_backward_duration") → 'backward'
        _stream_type_from_nvtx_name("gpu_opt_step_duration") → 'opt_step'
        _stream_type_from_nvtx_name("NCCL_AllReduce_1")      → 'nccl_active'
    """
    n = name.lower()
 
    if "nccl"      in n: return "nccl_active"
    if "opt_step"  in n: return "opt_step"
    if "zero_grad" in n: return "opt_step"
    if "loss"      in n: return "forward"
    if "forward"   in n: return "forward"
    if "backward"  in n: return "backward"
 
    return "compute"
