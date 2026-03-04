import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to mem_log.csv")
    ap.add_argument("--skip-step0", action="store_true", help="Exclude step==0 (warmup)")
    ap.add_argument("--steps", type=int, nargs="*", default=None, help="Only plot these steps (e.g. --steps 1 2 3)")
    ap.add_argument("--out", default="memory_graph.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)

    # ---- pick time col
    t_col = pick_col(df, ["t_ms", "t_us", "t_ns"])
    if t_col is None:
        raise RuntimeError("CSV is missing one of: t_ms / t_us / t_ns")

    # normalize time to ms
    if t_col == "t_ns":
        df["t_ms_norm"] = df["t_ns"] / 1e6
    elif t_col == "t_us":
        df["t_ms_norm"] = df["t_us"] / 1e3
    else:
        df["t_ms_norm"] = df["t_ms"]

    # ---- pick allocated/reserved columns (prefer MB)
    alloc_col = pick_col(df, ["allocated_MB", "allocated_B"])
    reserv_col = pick_col(df, ["reserved_MB", "reserved_B"])
    max_alloc_col = pick_col(df, ["max_allocated_MB", "max_allocated_B"])
    max_reserv_col = pick_col(df, ["max_reserved_MB", "max_reserved_B"])

    if alloc_col is None or reserv_col is None:
        raise RuntimeError(
            "CSV is missing allocated/reserved columns. (Check: allocated_MB/reserved_MB or allocated_B/reserved_B)"
        )

    # convert to MB if needed
    def to_mb(series, colname):
        if colname.endswith("_B"):
            return series / (1024**2)
        return series

    df["allocated_plot"] = to_mb(df[alloc_col], alloc_col)
    df["reserved_plot"] = to_mb(df[reserv_col], reserv_col)

    if max_alloc_col is not None:
        df["max_alloc_plot"] = to_mb(df[max_alloc_col], max_alloc_col)
    else:
        df["max_alloc_plot"] = np.nan

    if max_reserv_col is not None:
        df["max_reserv_plot"] = to_mb(df[max_reserv_col], max_reserv_col)
    else:
        df["max_reserv_plot"] = np.nan

    # fragmentation / cache slack
    df["slack_plot"] = (df["reserved_plot"] - df["allocated_plot"]).clip(lower=0)

    # ---- optional filters
    if "step" in df.columns:
        if args.skip_step0:
            df = df[df["step"] != 0].copy()
        if args.steps is not None and len(args.steps) > 0:
            df = df[df["step"].isin(args.steps)].copy()

    df = df.sort_values("t_ms_norm")

    # ---- Plot 1: time-series lines
    plt.figure(figsize=(14, 6))
    plt.plot(df["t_ms_norm"], df["allocated_plot"], label="allocated (MB)")
    plt.plot(df["t_ms_norm"], df["reserved_plot"], label="reserved (MB)")
    plt.plot(df["t_ms_norm"], df["slack_plot"], label="reserved - allocated (MB)")

    # tag markers (optional)
    if "tag" in df.columns:
        # show only a few tags to avoid clutter
        important_tags = [
            "step_start", "after_h2d", "after_forward", "after_backward", "after_opt_step", "step_end"
        ]
        for tag in important_tags:
            sub = df[df["tag"] == tag]
            if not sub.empty:
                plt.scatter(sub["t_ms_norm"], sub["allocated_plot"], s=18, label=f"tag:{tag}")

    plt.xlabel("time (ms) [relative to first mark]")
    plt.ylabel("memory (MB)")
    title = "PyTorch CUDA Memory over Time"
    if "step" in df.columns:
        title += " (with steps)"
    plt.title(title)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    if args.show:
        plt.show()
    else:
        plt.close()
    print(f"[Saved] {args.out}")

    # ---- Plot 2 (optional): per-step peak summary
    if "step" in df.columns:
        g = df.groupby("step", as_index=False).agg(
            peak_alloc_MB=("allocated_plot", "max"),
            peak_reserv_MB=("reserved_plot", "max"),
            peak_slack_MB=("slack_plot", "max"),
        ).sort_values("step")

        out2 = args.out.replace(".png", "_per_step.png")
        plt.figure(figsize=(14, 5))
        plt.plot(g["step"], g["peak_alloc_MB"], label="peak allocated (MB)")
        plt.plot(g["step"], g["peak_reserv_MB"], label="peak reserved (MB)")
        plt.plot(g["step"], g["peak_slack_MB"], label="peak reserved - allocated (MB)")
        plt.xlabel("step")
        plt.ylabel("peak memory (MB)")
        plt.title("Per-step Peak Memory")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out2, dpi=200)
        if args.show:
            plt.show()
        else:
            plt.close()
        print(f"[Saved] {out2}")

if __name__ == "__main__":
    main()
