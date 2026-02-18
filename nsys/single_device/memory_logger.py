# memory_logger.py
import csv
from dataclasses import dataclass
from time import time_ns

import torch


@dataclass
class MemRecord:
    t_ns: int
    step: int
    tag: str
    allocated: int
    reserved: int
    max_allocated: int
    max_reserved: int


class MemoryLogger:
    def __init__(self, device="cuda", out_csv="mem_log.csv"):
        self.device = torch.device(device)
        self.out_csv = out_csv
        self.rows = []
        self.t0 = None

    def _now(self) -> int:
        return time_ns()

    def reset_step_peak(self):
        """Call at step start to measure per-step peak."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def mark(self, step: int, tag: str):
        """Record current allocated/reserved + peak stats."""
        if self.device.type != "cuda":
            return

        t = self._now()
        if self.t0 is None:
            self.t0 = t

        alloc = torch.cuda.memory_allocated(self.device)
        reserv = torch.cuda.memory_reserved(self.device)
        max_alloc = torch.cuda.max_memory_allocated(self.device)
        max_reserv = torch.cuda.max_memory_reserved(self.device)

        self.rows.append(
            MemRecord(
                t_ns=t,
                step=step,
                tag=tag,
                allocated=alloc,
                reserved=reserv,
                max_allocated=max_alloc,
                max_reserved=max_reserv,
            )
        )

    def dump(self):
        """Write all records to CSV."""
        if not self.rows:
            print("[MemoryLogger] No rows to dump.")
            return

        with open(self.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t_ns", "t_ms", "step", "tag",
                "allocated_B", "reserved_B", "max_allocated_B", "max_reserved_B",
                "allocated_MB", "reserved_MB", "max_allocated_MB", "max_reserved_MB"
            ])

            for r in self.rows:
                t_ms = (r.t_ns - self.t0) / 1e6
                w.writerow([
                    r.t_ns, t_ms, r.step, r.tag,
                    r.allocated, r.reserved, r.max_allocated, r.max_reserved,
                    r.allocated / 1e6, r.reserved / 1e6, r.max_allocated / 1e6, r.max_reserved / 1e6
                ])

        print(f"[MemoryLogger] Saved: {self.out_csv} ({len(self.rows)} records)")
