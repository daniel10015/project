# main.py


import sys
import os
import glob
import re
import json
import argparse
from collections import defaultdict

# Force a non-GUI matplotlib backend early to avoid GTK/Wayland segfaults.
# Must happen before any module imports matplotlib.pyplot.
if os.environ.get("MPLBACKEND") is None:
    os.environ["MPLBACKEND"] = "Agg"

from loaders.data_loader import load_single_gpu,get_all_step_intervals,compute_rank_order_per_step
from analysis.clock_offset import calculate_clock_offsets
from analysis.wait_analysis import load_all_gpu_compute_wait_time
from analysis.sm_analysis import build_sm_timeline_df

from viz.plot_sm_timeline import plot_sm_timeline
from viz.plot_wait_time   import plot_wait_time_summary
from viz.plot_timeline    import plot_timeline_custom_axis

from viz.plot_timeline import get_merged_intervals

from viz.plot_memory import (
    load_mem_csvs,
    plot_memory_per_steps,
    plot_phase_memory_by_rank,
    plot_memory_breakdown_per_phase,
)

from viz.plot_step_breakdown import (
    aggregate_per_step_breakdown,
    plot_step_breakdown,
)

 

# ─────────────────────────────────────────────────────────────────────────────
# GPU Hardware Specs
# Edit this if you are using a different GPU (e.g. H100, V100).
# ─────────────────────────────────────────────────────────────────────────────
GPU_SPECS_A100 = {
    "sm_count":          108,   # Number of SMs (Streaming Multiprocessors)
    "sm_max_threads":    2048,  # Max threads per SM
    "sm_max_blocks":     32,    # Max thread blocks per SM
    "sm_max_registers":  65536, # Max registers per SM
    "sm_max_shared_mem": 49152, # Max shared memory per SM (bytes)
}
GPU_SPECS_A30 = {
    "sm_count":          56,    # Number of SMs
    "sm_max_threads":    2048,  # Max threads per SM
    "sm_max_blocks":     32,    # Max thread blocks per SM
    "sm_max_registers":  65536, # Max registers per SM
    "sm_max_shared_mem": 49152, # Max shared memory per SM (bytes)
}


# ─────────────────────────────────────────────────────────────────────────────
# get_rank_from_filename  
#
# Simple example:
#   'experiment_bs128_rank0.sqlite' → 0
#   'experiment_bs128_rank3.sqlite' → 3
#   'some_file_without_rank.sqlite' → index (e.g. 2 if it is the 3rd file)
# ─────────────────────────────────────────────────────────────────────────────
def get_rank_from_filename(filename: str, index: int) -> int:
    """
    Extract the GPU rank number from a filename by searching for patterns
    like 'rank0', 'rank_1', 'RANK2' (case-insensitive).
 
    Returns:
        int rank number
    """
    match = re.search(r"rank_?(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return index

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: File Selection
# Finds all .sqlite files in a directory and groups them by experiment name.
#
# Simple example of what "grouping" means:
#   Files in the folder:
#     experiment_A_rank0.sqlite
#     experiment_A_rank1.sqlite
#     experiment_B_rank0.sqlite
#     experiment_B_rank1.sqlite
#
#   Groups:
#     [0] experiment_A  (2 ranks)
#     [1] experiment_B  (2 ranks)
#
#   You type "0" → loads experiment_A_rank0 and experiment_A_rank1.
# ─────────────────────────────────────────────────────────────────────────────
def select_sqlite_files(sqlite_dir: str) -> list[str]:

    # Find all .sqlite files in the given directory
    all_files = sorted(glob.glob(os.path.join(sqlite_dir, "*.sqlite")))

    if not all_files:
        print(f"\n[Error] No .sqlite files found in: '{sqlite_dir}'")
        print(f"  → Make sure you passed the correct directory with --dir")
        sys.exit(1)

    # Group files by experiment name
    # (strips "_rankN" from the filename to get the group key)
    groups = defaultdict(list)
    for f in all_files:
        basename  = os.path.basename(f)
        group_key = re.sub(r"_?rank\d+", "", basename).replace(".sqlite", "")
        groups[group_key].append(f)

    group_keys = sorted(groups.keys())

    # Print the menu
    print("\n" + "=" * 60)
    print("  Available experiment groups:")
    print("=" * 60)
    for i, key in enumerate(group_keys):
        files = sorted(groups[key])
        print(f"\n  [{i}]  {key}  ({len(files)} rank(s))")
        for f in files:
            print(f"        - {os.path.basename(f)}")
    print("=" * 60)

    # Ask user to pick one group
    choice = input("\nSelect group number: ").strip()
    try:
        selected_key = group_keys[int(choice)]
    except (ValueError, IndexError):
        print(f"[Error] '{choice}' is not a valid group number. Exiting.")
        sys.exit(1)

    return sorted(groups[selected_key])

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Step Selection
# Asks which training steps you want to analyze.
#
# Simple example:
#   Your training ran 50 steps (0 ~ 49).
#   You only want to look at steps 2, 3, 4.
#   → Type: 2 3 4
# ─────────────────────────────────────────────────────────────────────────────
def select_steps(steps_from_args: list[int]) -> list[int]:

    if steps_from_args:
        # Already provided via command line, no need to ask
        return steps_from_args

    step_input = input(
        "Enter step numbers to analyze (space-separated, e.g.  2 3 4): "
    ).strip()

    steps = [int(s) for s in step_input.split() if s.isdigit()]

    if not steps:
        print("[Warning] No steps entered. Using default: [1, 2, 3]")
        steps = [1, 2, 3]

    return steps

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Data Loading
# Loads the SQLite files you selected and builds GpuDataset objects.
#
# Simple example:
#   You selected 4 files (rank0, rank1, rank2, rank3).
#   This function loads each file and returns:
#     gpu_data_map = {
#       0: GpuDataset(rank=0, df_nvtx=..., df_nccl=..., ...),
#       1: GpuDataset(rank=1, ...),
#       2: GpuDataset(rank=2, ...),
#       3: GpuDataset(rank=3, ...),
#     }
# ─────────────────────────────────────────────────────────────────────────────
def load_data(sqlite_files: list[str], 
              target_steps: list[int],
              gpu_specs: dict
              ) -> dict:

    print("\n" + "=" * 60)
    print("  Loading GPU data ...")
    print("=" * 60)

    gpu_data_map = {}

    for i, filepath in enumerate(sqlite_files):
        rank    = get_rank_from_filename(filepath, i)
        dataset = load_single_gpu(filepath, rank, target_steps, gpu_specs)

        if dataset:
            gpu_data_map[rank] = dataset
            print(f"  ✓ Rank {rank} loaded successfully: {os.path.basename(filepath)}")
        else:
            print(f"  ✗ [Warning] Failed to load Rank {rank}: {filepath}")

    if not gpu_data_map:
        print("\n[Fatal Error] No valid GPU data was loaded. Exiting.")
        sys.exit(1)

    print(f"\n  → Loaded {len(gpu_data_map)} rank(s): {sorted(gpu_data_map.keys())}")
    return gpu_data_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Quick Sanity Check
# Prints a short summary of what was loaded so you can verify correctness
# before running heavy analysis.
# ─────────────────────────────────────────────────────────────────────────────



def print_data_summary(gpu_data_map: dict):

    print("\n" + "=" * 60)
    print("  Data Summary (Sanity Check)")
    print("=" * 60)
    

    for rank, ds in gpu_data_map.items():

        if "data_batch_idx" in ds.df_gpu_duration.columns:
            print(f"    gpu steps found  : {sorted(ds.df_gpu_duration['data_batch_idx'].dropna().unique().astype(int).tolist())}")
        else:
            print(f"    gpu steps found  : data_batch_idx column missing")
        
        print(f"\n  Rank {rank}  ({os.path.basename(ds.filename)})")
        print(f"    df_nvtx          : {ds.df_nvtx.shape[0]:>6} rows")
        print(f"    df_nccl          : {ds.df_nccl.shape[0]:>6} rows")
        print(f"    df_memcpy        : {ds.df_memcpy.shape[0]:>6} rows")
        print(f"    df_gpu_duration  : {ds.df_gpu_duration.shape[0]:>6} rows")
        print(f"    df_all_kernels   : {ds.df_all_kernels_with_sm.shape[0]:>6} rows")

        # Show which steps are present
        if not ds.df_gpu_duration.empty and "data_batch_idx" in ds.df_gpu_duration.columns:
            steps_found = sorted(ds.df_gpu_duration["data_batch_idx"].dropna().unique().astype(int).tolist())
            print(f"    steps found      : {steps_found}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Export to CSV
#
# Saves every DataFrame from every rank into a structured folder.
#
# Output folder structure:
#   csv_export/
#     rank0/
#       nvtx.csv
#       nccl.csv
#       memcpy.csv
#       gpu_duration.csv
#       kernel_occupancy.csv
#       all_kernels_with_sm.csv
#     rank1/
#       ...
#
# Simple example:
#   gpu_data_map has rank 0 and rank 1.
#   → creates csv_export/rank0/ and csv_export/rank1/
#   → each folder has one CSV per DataFrame
#   → empty DataFrames are skipped with a warning message
#
# Usage:
#   export_to_csv(gpu_data_map)                       → saves to ./csv_export/
#   export_to_csv(gpu_data_map, out_dir="my_output")  → saves to ./my_output/
# ─────────────────────────────────────────────────────────────────────────────
def export_to_csv(gpu_data_map: dict, out_dir: str = "csv_export"):
    """
    Write every DataFrame in every GpuDataset to a CSV file.
 
    Parameters:
        gpu_data_map  → the dict returned by load_data()
        out_dir       → root folder for output (created if it does not exist)
    """
 
    # Each DataFrame gets a short filename and a description for the log message
    dataframes_to_export = [
        ("nvtx.csv",              "df_nvtx",                      "CPU NVTX markers"),
        ("nccl.csv",              "df_nccl",                      "NCCL kernels"),
        ("memcpy.csv",            "df_memcpy",                    "H2D memory copies"),
        ("gpu_duration.csv",      "df_gpu_duration",              "GPU kernel durations"),
        ("kernel_occupancy.csv",  "df_kernel_occupancy_per_step", "kernel occupancy stats"),
        ("all_kernels_sm.csv",    "df_all_kernels_with_sm",       "all kernels + SM%"),
    ]
 
    print("\n" + "=" * 60)
    print(f"  Exporting CSVs → {os.path.abspath(out_dir)}/")
    print("=" * 60)
 
    for rank, ds in gpu_data_map.items():
 
        # Create one subfolder per rank
        rank_dir = os.path.join(out_dir, f"rank{rank}")
        os.makedirs(rank_dir, exist_ok=True)
 
        print(f"\n  Rank {rank}:")
 
        for filename, attr_name, description in dataframes_to_export:
 
            df = getattr(ds, attr_name)
 
            out_path = os.path.join(rank_dir, filename)
 
            if df.empty:
                print(f"    ⚠  {filename:<28} skipped  ({description} is empty)")
                continue
 
            df.to_csv(out_path, index=False)
            print(f"    ✓  {filename:<28} {len(df):>7} rows  →  {out_path}")
 
    print(f"\n  Done. All CSVs saved to: {os.path.abspath(out_dir)}/")



# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Create plot output directory
#
# Creates a structured folder to save all plots.
#
# Output folder structure:
#   plots/
#     sm_timeline/      ← SM utilization plots (one per step)
#     wait_time/        ← NCCL wait time summary
#     timeline/         ← detailed Gantt-chart timeline
#
# Simple example:
#   make_plot_dir()            → creates ./plots/ and subfolders
#   make_plot_dir("my_plots")  → creates ./my_plots/ and subfolders
#
# Returns the root path so you can pass it to each plot function.
# ─────────────────────────────────────────────────────────────────────────────
def make_plot_dir(root: str = "plots") -> dict:
    """
    Create the output directory structure for all plots.
 
    Parameters:
        root  → root folder name (created relative to current working directory)
 
    Returns:
        dict with keys:
            root         → root folder path
            sm_timeline  → path for SM utilization plots
            wait_time    → path for wait time summary plots
            timeline     → path for Gantt-chart timeline plots
 
    Simple example:
        dirs = make_plot_dir("plots")
        →  plots/
               sm_timeline/
               wait_time/
               timeline/
 
        Use the returned paths:
            plot_sm_timeline(..., output_path=dirs["sm_timeline"] + "/sm_step2.png")
            plot_wait_time_summary(..., output_path=dirs["wait_time"] + "/summary.png")
    """
    subdirs = {
        "root":        root,
        "sm_timeline": os.path.join(root, "sm_timeline"),
        "wait_time":   os.path.join(root, "wait_time"),
        "timeline":    os.path.join(root, "timeline"),
        "breakdown":    os.path.join(root, "breakdown"),
    }
 
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)
 
    print("\n" + "=" * 60)
    print(f"  Plot output directory: {os.path.abspath(root)}/")
    print("=" * 60)
    for key, path in subdirs.items():
        if key == "root":
            continue
        print(f"    {key:<14} → {path}/")
 
    return subdirs

# ─────────────────────────────────────────────────────────────────────────────
# parse_experiment_info
#
# Extracts model name, batch size, image size, and max batches from the
# SQLite filename so they can be included in every plot filename.
#
# Simple example:
#   filename: 'profile_result_resnet_50_fin_bs256_img224_mb50_rank0.sqlite'
#   → returns:
#       model = 'resnet_50'
#       bs    = '256'
#       img   = '224'
#       mb    = '50'
#       tag   = 'resnet_50_bs256_img224_mb50'
#
# The tag is appended to every PNG name so you know exactly which
# experiment each plot came from without opening the file.
# ─────────────────────────────────────────────────────────────────────────────
def parse_experiment_info(sqlite_files: list[str]) -> dict:
    """
    Parse experiment metadata from a SQLite filename.
 
    Looks for these patterns (case-insensitive):
        bs<N>    → batch size per GPU
        img<N>   → input image size
        mb<N>    → max batches profiled per epoch
        resnet_50, vgg16, etc. → model name (heuristic)
 
    Parameters:
        sqlite_files → list of .sqlite file paths (uses the first file)
 
    Returns dict with keys:
        model  → model name string, e.g. 'resnet_50'
        bs     → batch size string, e.g. '256'
        img    → image size string, e.g. '224'
        mb     → max batches string, e.g. '50'
        tag    → combined tag for filenames, e.g. 'resnet_50_bs256_img224_mb50'
 
    If a field cannot be parsed, it is set to 'unknown'.
 
    Simple example:
        files = ['profile_result_resnet_50_fin_bs256_img224_mb50_rank0.sqlite']
        info  = parse_experiment_info(files)
        info["tag"] → 'resnet_50_bs256_img224_mb50'
 
        filename: f"sm_timeline_step2_{info['tag']}.png"
        →         'sm_timeline_step2_resnet_50_bs256_img224_mb50.png'
    """
    if not sqlite_files:
        return {"model": "unknown", "bs": "unknown",
                "img": "unknown", "mb": "unknown",
                "tag": "unknown"}
 
    # Use the first file — all ranks share the same experiment config
    basename = os.path.basename(sqlite_files[0]).lower()
 
    # Extract bs, img, mb with regex
    bs_match  = re.search(r"bs(\d+)",  basename)
    img_match = re.search(r"img(\d+)", basename)
    mb_match  = re.search(r"mb(\d+)",  basename)
 
    bs  = bs_match.group(1)  if bs_match  else "unknown"
    img = img_match.group(1) if img_match else "unknown"
    mb  = mb_match.group(1)  if mb_match  else "unknown"
 
    # Extract model name — look for known model keywords
    # Remove rank suffix and common prefixes first
    clean = re.sub(r"_?rank_?\d+", "", basename)
    clean = re.sub(r"\.sqlite$",   "", clean)
    clean = re.sub(r"profile_result_", "", clean)
    clean = re.sub(r"_bs\d+.*",    "", clean)   # strip everything from bs onwards
    clean = clean.strip("_")
 
    # Map common variations to clean names
    model_map = {
        "resnet_50":  "resnet_50",
        "resnet50":   "resnet_50",
        "resnet_101": "resnet_101",
        "vgg16":      "vgg16",
        "vgg_16":     "vgg16",
        "bert":       "bert",
        "gpt":        "gpt",
    }
    model = "unknown"
    for key, val in model_map.items():
        if key in clean:
            model = val
            break
    if model == "unknown" and clean:
        # Fallback: use whatever is left after stripping
        model = clean[:30]   # cap length to keep filenames reasonable
 
    tag = f"{model}_bs{bs}_img{img}_mb{mb}"
 
    print(f"\n  Experiment info parsed from filename:")
    print(f"    model : {model}")
    print(f"    bs    : {bs}")
    print(f"    img   : {img}")
    print(f"    mb    : {mb}")
    print(f"    tag   : {tag}")
 
    return {"model": model, "bs": bs, "img": img, "mb": mb, "tag": tag}

# ─────────────────────────────────────────────────────────────────────────────
# DEBUG SECTIONS
# Each section is wrapped in:
#   if DEBUG_<NAME>:
#       ...
#
# Set the flag to True to enable that section.
# This way you can turn each section on/off without deleting code.
#
# Simple example:
#   You only want to test clock offset calculation today.
#   → Set DEBUG_CLOCK_OFFSET = True, everything else = False.
# ─────────────────────────────────────────────────────────────────────────────
DEBUG_CLOCK_OFFSET   = True   # Calculate + print clock offsets between ranks
DEBUG_WAIT_TIME      = True  # Analyze NCCL wait time per step
DEBUG_SM_TIMELINE    = True  # Build SM utilization timeline (slow)
DEBUG_PLOT           = True  # Generate and save plots


def get_all_steps_from_loaded_data(gpu_data_map: dict) -> list[int]:
    """
    Best-effort step discovery for --all.
    Prefers df_gpu_duration.data_batch_idx; falls back to NVTX data_batch_idx.
    """
    steps: set[int] = set()
    for _, ds in gpu_data_map.items():
        if getattr(ds, "df_gpu_duration", None) is not None and not ds.df_gpu_duration.empty:
            if "data_batch_idx" in ds.df_gpu_duration.columns:
                for v in ds.df_gpu_duration["data_batch_idx"].dropna().unique().tolist():
                    try:
                        steps.add(int(v))
                    except Exception:
                        pass
        if getattr(ds, "df_nvtx", None) is not None and not ds.df_nvtx.empty:
            if "data_batch_idx" in ds.df_nvtx.columns:
                for v in ds.df_nvtx["data_batch_idx"].dropna().unique().tolist():
                    try:
                        steps.add(int(v))
                    except Exception:
                        pass
    return sorted(steps)


def _rgba_floats(color) -> list[float]:
    return [float(color[0]), float(color[1]), float(color[2]), float(color[3])]


def _total_duration_ms(intervals: list) -> float:
    return float(sum(en - st for st, en in intervals) / 1e6) if intervals else 0.0


def export_phase_rank_step_json(
    gpu_data_map: dict,
    steps_to_export: list[int],
    df_rank_order_per_step,
    all_steps_map: dict,
    offsets: dict,
    out_json: str,
    title: str,
) -> None:
    """
    Timeline-as-stacked-bar JSON in the viewer's `type="bar"` format:
      - bar = one "lane" from the matplotlib timeline (row_name + rank)
      - segments are laid out in TIME ORDER from global_start → global_end
      - gaps between events are encoded as transparent "NOP" segments, so the
        spacing matches the matplotlib timeline (forward blocks are separated, etc.)
    """
    from viz.plot_timeline import (
        filter_by_step_ranges,
        aggregate_gpu_kernels_by_nvtx,
        RANK_COLORS,
    )
    import matplotlib.colors as mcolors

    steps_sorted = sorted(steps_to_export)
    if not steps_sorted:
        raise ValueError("No steps to export.")

    # Keep ordering identical to plot_timeline_custom_axis()
    rows_in_order: list[str] = [
        "data_wait",
        "h2d",
        "gpu_compute",
        "NCCL",
        "zero_grad",
        "forward",
        "loss",
        "backward",
        "opt_step",
    ]

    target_stats = df_rank_order_per_step[df_rank_order_per_step["step"].isin(steps_sorted)]
    if target_stats.empty:
        raise ValueError(f"No df_rank_order_per_step rows for steps {steps_sorted}")
    global_start = int(target_stats["earliest_start"].min())
    global_end = int(target_stats["bwd_latest_end"].max())
    timeline_duration_ms = float((global_end - global_start) / 1e6)

    # Legend: match matplotlib y-axis order (top → bottom).
    # In `plot_timeline_custom_axis()`, y=0 is the bottom row and the last item is the top row.
    legend_names_top_to_bottom = list(reversed(rows_in_order))

    # IMPORTANT: Do NOT change the JSON schema. The renderer only supports:
    #   legend[*] = { name, color } and bar.segments[*] = { value, legendIndex }.
    #
    # To represent matplotlib's "faint CPU launch" bars, we emit separate legend
    # entries for CPU lanes with the same RGB but alpha=0.3, and output separate
    # bars labeled "(CPU)".
    cpu_overlay_rows = {
        "data_wait",
        "h2d",
        "gpu_compute",
        "NCCL",
        "zero_grad",
        "forward",
        "loss",
        "backward",
        "opt_step",
    }

    legend: list[dict] = []
    legend_index_for_lane: dict[tuple[str, str], int] = {}  # (row_name, kind) -> legendIndex
    for i, name in enumerate(legend_names_top_to_bottom):
        # Use the exact same base palette as the matplotlib timeline.
        # (timeline uses tab: colors from `RANK_COLORS`, either per-rank or per-step)
        base_color = RANK_COLORS[i % len(RANK_COLORS)] if RANK_COLORS else "tab:blue"
        base_rgba = list(mcolors.to_rgba(base_color, alpha=1.0))
        # GPU (solid)
        legend_index_for_lane[(name, "gpu")] = len(legend)
        legend.append({"name": name, "color": [float(x) for x in base_rgba]})

        # CPU (faint) where applicable
        if name in cpu_overlay_rows:
            legend_index_for_lane[(name, "cpu")] = len(legend)
            cpu_rgba = list(mcolors.to_rgba(base_color, alpha=0.3))
            legend.append({"name": f"{name} (CPU)", "color": [float(x) for x in cpu_rgba]})

    nop_legend_index = len(legend)
    legend.append({"name": "NOP", "color": [0.0, 0.0, 0.0, 0.0]})

    # Row naming consistent with matplotlib timeline
    phase_to_row = {
        "cpu_data_wait_launch":       "data_wait",
        "cpu_h2d_launch":             "h2d",
        "cpu_zero_grad_launch":       "zero_grad",
        "cpu_forward_launch":         "forward",
        "cpu_loss_launch":            "loss",
        "cpu_backward_launch":        "backward",
        "cpu_opt_step_launch":        "opt_step",
        "cpu_nccl_allreduce_launch":  "NCCL",
        "cpu_train_compute_wrapper":  "gpu_compute",
    }
    gpu_to_row = {
        "gpu_forward_duration":         "forward",
        "gpu_backward_duration":        "backward",
        "gpu_loss_duration":            "loss",
        "gpu_opt_step_duration":        "opt_step",
        "gpu_zero_grad_duration":       "zero_grad",
        "gpu_train_compute_duration":   "gpu_compute",
        "gpu_nccl_allreduce_duration":  "NCCL",
    }

    sorted_ranks = sorted(gpu_data_map.keys())

    def _build_segments_from_events(
        lane_events: list[tuple[float, float, int]],
        total_ms: float,
        nop_idx: int,
    ) -> list[dict]:
        """
        Convert sorted lane events into stacked segments with transparent gaps.
        lane_events: [(start_ms, dur_ms, legendIndex), ...]
        """
        segs: list[dict] = []
        cursor = 0.0
        for st, dur, li in lane_events:
            st = float(st)
            dur = float(dur)
            if dur <= 0:
                continue
            if st > cursor:
                segs.append({"value": float(st - cursor), "legendIndex": nop_idx})
            segs.append({"value": dur, "legendIndex": int(li)})
            cursor = max(cursor, st + dur)
        if total_ms > cursor:
            segs.append({"value": float(total_ms - cursor), "legendIndex": nop_idx})
        return segs

    # Uniform bar geometry: identical y/w/h; only x differs
    bar_w = 90
    rank_stride = 115  # within-group spacing
    group_gap = 140    # between row groups

    bars: list[dict] = []
    x_cursor = 0

    # JSON-only: mirror ONLY the "type" lanes (the overview rows),
    # but keep the detailed phase lanes in the exact same ordering.
    base_names = ["data_wait", "h2d", "gpu_compute", "NCCL"]
    fixed_nvtx = ["zero_grad", "forward", "loss", "backward", "opt_step"]
    json_rows_for_bars = list(reversed(base_names)) + fixed_nvtx

    for row_name in json_rows_for_bars:
        # NVTX name for the faint CPU-launch lane (if this row has one)
        overlay_nvtx_name = next((k for k, v in phase_to_row.items() if v == row_name), None)
        # gpu_duration name for the solid GPU execution lane (if this row has one)
        gpu_name_for_row = next((k for k, v in gpu_to_row.items() if v == row_name), None)

        for ri, rank in enumerate(sorted_ranks):
            ds = gpu_data_map.get(rank)
            if ds is None or rank not in all_steps_map:
                continue
            step_df_sel = all_steps_map[rank][all_steps_map[rank]["step"].isin(steps_sorted)]
            if step_df_sel.empty:
                continue

            df_nvtx = (
                filter_by_step_ranges(
                    ds.df_nvtx, step_df_sel, global_start,
                    use_data_batch_idx=True, rank=rank, offsets=offsets,
                )
                if getattr(ds, "df_nvtx", None) is not None and not ds.df_nvtx.empty
                else pd.DataFrame()
            )
            df_memcpy = (
                filter_by_step_ranges(
                    ds.df_memcpy, step_df_sel, global_start,
                    use_data_batch_idx=True, rank=rank, offsets=offsets,
                )
                if getattr(ds, "df_memcpy", None) is not None and not ds.df_memcpy.empty
                else pd.DataFrame()
            )
            df_nccl = (
                filter_by_step_ranges(
                    ds.df_nccl, step_df_sel, global_start,
                    rank=rank, offsets=offsets,
                )
                if getattr(ds, "df_nccl", None) is not None and not ds.df_nccl.empty
                else pd.DataFrame()
            )

            df_gpu_agg = (
                aggregate_gpu_kernels_by_nvtx(ds.df_gpu_duration)
                if getattr(ds, "df_gpu_duration", None) is not None and not ds.df_gpu_duration.empty
                else pd.DataFrame()
            )
            df_gpu = (
                filter_by_step_ranges(
                    df_gpu_agg, step_df_sel, global_start,
                    rank=rank, offsets=offsets,
                )
                if df_gpu_agg is not None and not df_gpu_agg.empty
                else pd.DataFrame()
            )

            gpu_events: list[tuple[float, float, int]] = []
            cpu_events: list[tuple[float, float, int]] = []

            li_gpu = legend_index_for_lane.get((row_name, "gpu"), nop_legend_index)
            li_cpu = legend_index_for_lane.get((row_name, "cpu"), nop_legend_index)

            if row_name == "data_wait":
                if not df_nvtx.empty:
                    cpu_rows = df_nvtx[df_nvtx["name"] == "cpu_data_wait_launch"]
                    for _, r in cpu_rows.iterrows():
                        cpu_events.append((float(r["rel_start_ms"]), float(r["dur_ms"]), li_cpu))

            elif row_name == "h2d":
                if not df_memcpy.empty:
                    for _, r in df_memcpy.iterrows():
                        gpu_events.append((float(r["rel_start_ms"]), float(r["dur_ms"]), li_gpu))

            elif row_name == "NCCL":
                if not df_nccl.empty:
                    for _, r in df_nccl.iterrows():
                        gpu_events.append((float(r["rel_start_ms"]), float(r["dur_ms"]), li_gpu))

            elif row_name == "gpu_compute":
                if not df_gpu.empty:
                    for _, r in df_gpu.iterrows():
                        gpu_events.append((float(r["rel_start_ms"]), float(r["dur_ms"]), li_gpu))

            else:
                # Per-phase GPU spans (preferred)
                if gpu_name_for_row and not df_gpu.empty:
                    gpu_rows = df_gpu[df_gpu["name"] == gpu_name_for_row]
                    for _, r in gpu_rows.iterrows():
                        gpu_events.append((float(r["rel_start_ms"]), float(r["dur_ms"]), li_gpu))

                # Fallback to CPU launch spans if GPU spans missing
                if not gpu_events and not df_nvtx.empty and overlay_nvtx_name:
                    cpu_rows = df_nvtx[df_nvtx["name"] == overlay_nvtx_name]
                    for _, r in cpu_rows.iterrows():
                        cpu_events.append((float(r["rel_start_ms"]), float(r["dur_ms"]), li_cpu))

            # Always include CPU launch spans where available (light bars in matplotlib).
            if overlay_nvtx_name and not df_nvtx.empty:
                cpu_rows = df_nvtx[df_nvtx["name"] == overlay_nvtx_name]
                for _, r in cpu_rows.iterrows():
                    cpu_events.append((float(r["rel_start_ms"]), float(r["dur_ms"]), li_cpu))

            gpu_events.sort(key=lambda t: (t[0], t[1]))
            cpu_events.sort(key=lambda t: (t[0], t[1]))

            if gpu_events:
                bars.append({
                    "x": x_cursor + ri * rank_stride,
                    "y": 0,
                    "w": bar_w,
                    "h": timeline_duration_ms,
                    "label": f"{row_name} R{rank}",
                    "segments": _build_segments_from_events(gpu_events, timeline_duration_ms, nop_legend_index),
                })
            if cpu_events:
                bars.append({
                    "x": x_cursor + ri * rank_stride,
                    "y": 0,
                    "w": bar_w,
                    "h": timeline_duration_ms,
                    "label": f"{row_name} (CPU) R{rank}",
                    "segments": _build_segments_from_events(cpu_events, timeline_duration_ms, nop_legend_index),
                })

        x_cursor += len(sorted_ranks) * rank_stride + group_gap

    payload = {
        "type": "bar",
        "title": title,
        "unit": "ms",
        "legend": legend,
        "bars": bars,
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[json] Saved: {out_json}")


def run_analysis(
    gpu_data_map: dict,
    target_steps: list[int],
    exp_info: dict = None,
    json_only: bool = False,
):

    tag = exp_info["tag"] if exp_info else "experiment"

    # ── Clock Offset ──────────────────────────────────────────────────────────
    # Different GPUs on different nodes may have slightly different clocks.
    # This step calculates how much each rank's clock differs from rank 0.
    #
    # Simple example:
    #   Rank 0 clock: 1000 ms
    #   Rank 2 clock: 1100 ms  ← 100ms ahead
    #   offset[2] = -100ms     ← subtract 100ms to align with rank 0
    if DEBUG_CLOCK_OFFSET:
        print("\n" + "─" * 60)
        print("  [DEBUG] Clock Offset Calculation")
        print("─" * 60)
        offsets = calculate_clock_offsets(gpu_data_map, n_kernels=20)

    else:
        # Default: all offsets = 0 (no clock correction)
        offsets = {rank: 0 for rank in gpu_data_map}

    from analysis.clock_offset import debug_clock_offsets
    debug_clock_offsets(gpu_data_map, n_kernels=10)

    # ── Step Intervals ────────────────────────────────────────────────────────
    all_steps_map          = get_all_step_intervals(gpu_data_map)
    
    df_rank_order_per_step = compute_rank_order_per_step(all_steps_map, gpu_data_map, offsets)

    if df_rank_order_per_step.empty:
        print("[Error] Could not compute step intervals. Check NVTX markers.")
        return

    print("\n  Rank order per step:")
    print(df_rank_order_per_step.to_string())


    # ── Wait Time Analysis ────────────────────────────────────────────────────
    # Measures how long each rank waits for NCCL AllReduce to start.
    # A rank that finishes its backward pass early must wait for slower ranks.
    #
    # Simple example:
    #   Rank 0 finishes backward at t=100ms → AllReduce starts at t=120ms
    #   Wait time for Rank 0 = 120 - 100 = 20ms  ← this is wasted time
    if (not json_only) and DEBUG_WAIT_TIME:
        print("\n" + "─" * 60)
        print("  [DEBUG] NCCL Wait Time Analysis")
        print("─" * 60)
        wait_df = load_all_gpu_compute_wait_time(gpu_data_map, offsets)

        if wait_df.empty:
            print("  [Warning] wait_df is empty. Check NVTX NCCL_AllReduce ranges.")
        else:
            print(f"  wait_df shape: {wait_df.shape}")
            print(wait_df.head(20).to_string())
    else:
        wait_df = None

    # ── SM Timeline ───────────────────────────────────────────────────────────
    # Builds a per-bucket SM utilization timeline for each rank and step.
    # This is the core data for the SM utilization plot.
    if (not json_only) and DEBUG_SM_TIMELINE:
        print("\n" + "─" * 60)
        print("  [DEBUG] Building SM Timeline")
        print("─" * 60)
        df_sm_timeline = build_sm_timeline_df(
            gpu_data_map = gpu_data_map,
            offsets      = offsets,
            steps        = target_steps,
            n_buckets    = 350,
        )
        print(f"  df_sm_timeline shape: {df_sm_timeline.shape}")
        print(df_sm_timeline.head(5).to_string())
    else:
        df_sm_timeline = None

    # # ── Plots ─────────────────────────────────────────────────────────────────
    if (not json_only) and DEBUG_PLOT:
        print("\n" + "─" * 60)
        print("  [DEBUG] Generating Plots")
        print("─" * 60)


        # Create output directory structure
        dirs = make_plot_dir("plots")

        # ── SM Timeline plots (one PNG per step) ──────────────────────────────
        if df_sm_timeline is not None:

            for step in target_steps:
                out = os.path.join(dirs["sm_timeline"], f"sm_timeline_step{step}_{tag}.png")
                plot_sm_timeline(
                    df_sm_timeline = df_sm_timeline,
                    step           = step,
                    n_buckets      = 350,
                    figsize        = (22, 12),
                    output_path    = out,
                    dpi            = 150,
                    show           = False,
                )
            print("  aved: sm_timeline.png")

        else:
            print("  [Skip] SM timeline not built. Set DEBUG_SM_TIMELINE = True first.")


        # ── Wait time summary plot ────────────────────────────────────────────
        if wait_df is not None and not wait_df.empty:

            out = os.path.join(dirs["wait_time"], f"wait_time_summary_{tag}.png")

            plot_wait_time_summary(
                wait_df     = wait_df,
                output_path = out,
            )
            print("  Saved: wait_time_summary.png")
        else:
            print("  [Skip] wait_df is empty.")


        # ── Gantt-chart timeline plot ─────────────────────────────────────────
        ranks_str = "_".join(str(r) for r in sorted(gpu_data_map.keys()))
        steps_str = "_".join(str(s) for s in sorted(target_steps))
        out = os.path.join(
            dirs["timeline"],
            f"timeline_ranks{ranks_str}_steps{steps_str}_{tag}.png"
        )
        plot_timeline_custom_axis(
            gpu_data_map           = gpu_data_map,
            df_rank_order_per_step = df_rank_order_per_step,
            all_steps_map          = all_steps_map,
            steps_to_plot          = target_steps,
            out_png                = out,
            color_by               = "step",
            offsets                = offsets,
            show                   = False,
        )
 

        # ── breakdown plot─────────────────────────────────────────
        df_breakdown = aggregate_per_step_breakdown(
            gpu_data_map           = gpu_data_map,
            steps_to_plot          = target_steps,
            nccl_wait_df           = wait_df,
            offsets                = offsets,
            df_rank_order_per_step = df_rank_order_per_step,
        )
        plot_step_breakdown(
            df_breakdown  = df_breakdown,
            output_prefix = os.path.join(dirs["breakdown"], f"step_breakdown_{tag}"),
            dpi           = 200,
        )

        print(f"\n  All plots saved to: {os.path.abspath('plots')}/")

    if json_only:
        dirs = make_plot_dir("plots")
        out_prefix = os.path.join(dirs["breakdown"], f"step_breakdown_{tag}")
        export_phase_rank_step_json(
            gpu_data_map=gpu_data_map,
            steps_to_export=target_steps,
            df_rank_order_per_step=df_rank_order_per_step,
            all_steps_map=all_steps_map,
            offsets=offsets,
            out_json=out_prefix + ".json",
            title="Per-phase durations per rank (vertical layout: x = phase/rank, y = time)",
        )


    print("\n[Done] Analysis complete.")

# ─────────────────────────────────────────────────────────────────────────────
# Memory analysis entry point
#
# Loads all rank CSV files from the mem/ directory, runs the three memory
# plots, and saves them to plots/<tag>/memory/.
#
# Usage:
#   run_memory_analysis(mem_dir="mem", exp_info=exp_info)
#
# Simple example:
#   mem/ contains:
#       mem_log_bs256_img224_mb50_rank0.csv
#       mem_log_bs256_img224_mb50_rank1.csv
#       mem_log_bs256_img224_mb50_rank2.csv
#       mem_log_bs256_img224_mb50_rank3.csv
#
#   run_memory_analysis("mem", exp_info)
#   → plots/resnet_50_bs256_img224_mb50/memory/
#         memory_trend_resnet_50_bs256_img224_mb50.png
#         memory_phase_resnet_50_bs256_img224_mb50.png
#         memory_step2_breakdown_resnet_50_bs256_img224_mb50.png  (if steps given)
# ─────────────────────────────────────────────────────────────────────────────
def run_memory_analysis(
    mem_dir:      str        = "mem",
    exp_info:     dict       = None,
    target_steps: list[int]  = None,
    ) -> None:

    """
    Load memory CSV files and produce all three memory plots.
 
    Parameters:
        mem_dir      → directory containing mem_log_*_rank*.csv files
        exp_info     → dict from parse_experiment_info() — used for filenames
        target_steps → optional list of steps for the breakdown plot
    """
 
    tag = exp_info["tag"] if exp_info else "experiment"
    
    if not os.path.isdir(mem_dir):
        print(f"\n  [Skip] Memory plots: '{mem_dir}/' directory not found.")
        return

    # ── Find CSV files ────────────────────────────────────────────────────────
    # pattern   = os.path.join(mem_dir, "*.csv")
    # csv_files = sorted(glob.glob(pattern))
 
    # if not csv_files:
    #     print(f"[Warning] run_memory_analysis: no CSV files found in '{mem_dir}'")
    #     print(f"  → Make sure mem/ contains mem_log_*_rank*.csv files")
    #     return
 
    print("\n" + "=" * 60)
    print(f"  Memory Analysis — loading from {mem_dir}/")
    print("=" * 60)
 
    # load_mem_csvs() handles glob, per-file loading, and concatenation
    try:
        rank_dfs, combined_df = load_mem_csvs(mem_dir)

    except Exception as e:
        print(f"  [Warning] Failed to load memory CSVs: {e}")
        return
 
    if not rank_dfs:
        print("  [Warning] No memory data loaded. Skipping memory plots.")
        return
 
    # ── Create output directory ───────────────────────────────────────────────
    out_dir = os.path.join("plots", tag, "memory")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Saving memory plots to: {os.path.abspath(out_dir)}/")
 

 
    # ── Plot 1: Memory per phase for one reference step ───────────────────────
    plot_phase_memory_by_rank(
        combined_df = combined_df,
        out_path    = os.path.join(out_dir, f"memory_phase_{tag}.png"),
    )
 
    # ── Plot 2: Breakdown per step (only if steps specified) ──────────────────
    # if target_steps:
    #     plot_memory_breakdown_per_phase(
    #         rank_dfs     = rank_dfs,
    #         target_steps = target_steps,
    #         out_dir      = out_dir,
    #         tag          = tag,
    #     )
    # else:
    #     print("  [Info] No target steps given — skipping breakdown plot.")
    #     print("         Pass step numbers as args: python main.py --dir sqlite/ 2 10 49")
 
    print(f"\n  Memory analysis complete.")

# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Nsight Systems DDP traces.")
    parser.add_argument("--dir", default=".", help="Directory containing per-rank .sqlite files")
    parser.add_argument("--json", action="store_true", help="Write JSON only (no matplotlib outputs)")
    parser.add_argument("--all", action="store_true", help="Analyze all steps found (no interactive prompt)")
    parser.add_argument("steps", nargs="*", type=int, help="Step numbers to analyze")
    cli = parser.parse_args()

    sqlite_dir = cli.dir
    steps_from_args = list(cli.steps) if cli.steps else []

    # ── File selection ────────────────────────────────────────────────────────
    sqlite_files = select_sqlite_files(sqlite_dir)

    # ── Print config summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Configuration")
    print("=" * 60)
    print(f"  Directory    : {os.path.abspath(sqlite_dir)}")
    if cli.all:
        print("  Target steps : --all (discover after load)")
    else:
        print(f"  Target steps : {steps_from_args if steps_from_args else '(prompt)'}")
    print(f"  Files ({len(sqlite_files)}):")
    for f in sqlite_files:
        print(f"    - {os.path.basename(f)}")

    # # ── Load data ─────────────────────────────────────────────────────────────
    ACTIVE_GPU_SPECS = GPU_SPECS_A100 
    gpu_data_map = load_data(sqlite_files, steps_from_args, ACTIVE_GPU_SPECS)

    if cli.all:
        target_steps = get_all_steps_from_loaded_data(gpu_data_map)
        if not target_steps:
            print("[Error] --all requested but no steps were discovered.")
            sys.exit(1)
    else:
        target_steps = select_steps(steps_from_args)

    

    # # ── Sanity check ─────────────────────────────────────────────────────────
    print_data_summary(gpu_data_map)

    # # ── Run analysis ──────────────────────────────────────────────────────────
    exp_info = parse_experiment_info(sqlite_files)
    run_analysis(gpu_data_map, target_steps, exp_info, json_only=cli.json)

    if not cli.json:
        # # ── export csv ─────────────────────────────────────────────────────
        export_to_csv(gpu_data_map, out_dir="csv_export")

        # ── Memory analysis ─────────────────────────────────────────────────
        # Reads CSV files from mem/ directory and produces memory plots.
        run_memory_analysis(
            mem_dir      = "mem",
            exp_info     = exp_info,
            target_steps = steps_from_args if steps_from_args else None,
        )