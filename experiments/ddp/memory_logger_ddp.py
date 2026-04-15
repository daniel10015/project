# memory_logger.py
import csv
import os
import pickle
from dataclasses import dataclass, field
from time import time_ns
from typing import Optional
import torch
import torch.distributed as dist   


@dataclass
class MemRecord:
    timestamp_ns: int
    step: int
    tag: str
    rank: int

    allocated: int
    reserved: int
    max_allocated: int
    max_reserved: int

    #component
    param_MB:        float  # sum of model.parameters()
    grad_MB:         float  # sum of p.grad
    opt_state_MB:    float  # sum of optimizer.state
    activation_MB:   float  # total - above three (estimated)

    #nccl
    nccl_buffer_B:    int   # NCCL buffer size
    bytes_sent_B:     int   # bytes sent in this step
    bytes_recv_B:     int   # bytes received in this step
    comm_duration_ns: int   # communication time (nanoseconds)
    comm_mem_delta_B: int   # memory increase during communication



class MemoryLogger:
    """
    DDP training memory logger.

    Features:
      1. mark() records a per-phase memory snapshot (async, minimal overhead)
      2. _measure_components() breaks down memory by component (params/grads/opt_state/activation)
      3. profile_steps: enables precise _record_memory_history profiling for specified steps only
      4. make_comm_hook() automatically records AllReduce communication info

    Usage:
      memlog = MemoryLogger(
          device=local_rank,
          out_csv="mem_log.csv",
          profile_steps=[0, 5, 10],   # steps to run precise profiling on
          snapshot_dir="snapshots",
      )
      ddp_model.register_comm_hook(state=None, hook=memlog.make_comm_hook())

      for step in range(max_steps):
          memlog.reset_step_peak(step)
          memlog.mark(step, "before_forward", model=ddp_model, optimizer=optimizer)
          pred = model(data)
          memlog.mark(step, "after_forward",  model=ddp_model, optimizer=optimizer)
          ... 
      
      memlog.dump()
    """

    def __init__(self, 
                device="cuda", 
                out_csv: str = "mem_log.csv",
                batch_size:  Optional[int] = None,
                image_size:  Optional[int] = None,
                max_batches: Optional[int] = None,
                profile_steps: Optional[list] = None,
                snapshot_dir:  str = "memory_snapshots",
                ):

        self.device = torch.device(device)
        self.snapshot_dir = snapshot_dir
        self.out_csv = out_csv
        self.rows = []
        self.t0 = None

        self.profile_steps = set(profile_steps or [])
        self._is_recording = False


        if batch_size is not None and image_size is not None and max_batches is not None:
            base = out_csv.replace(".csv", "")
            self.out_csv = f"{base}_bs{batch_size}_img{image_size}_mb{max_batches}.csv"
        else:
            self.out_csv = out_csv


        # ── Auto-detect rank info ──
        # If dist is initialized, get rank automatically.
        # If not initialized, treat as single GPU (rank=0).
        if dist.is_available() and dist.is_initialized():
            self.rank       = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank       = 0
            self.world_size = 1
        
        self._bytes_sent_B     = 0   # cumulative bytes sent in this step
        self._bytes_recv_B     = 0   # cumulative bytes received in this step
        self._comm_duration_ns = 0   # cumulative communication time
        self._comm_mem_delta_B = 0   # cumulative memory change during communication

    # ──────────────────────────────────────────────────────────
    # Internal utilities
    # ──────────────────────────────────────────────────────────

    def _now(self) -> int:

        return time_ns()

    # ──────────────────────────────────────────────────────────
    # Step start
    # ──────────────────────────────────────────────────────────
    def reset_step_peak(self, step: int):
        """
        Must be called at the start of each step.
        Resets peak memory stats + starts recording if this is a profile step.
        """

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # Reset communication counters at the start of each step
        self._bytes_sent_B     = 0
        self._bytes_recv_B     = 0
        self._comm_duration_ns = 0
        self._comm_mem_delta_B = 0

        # If this step is in profile_steps, start precise memory recording
        if step in self.profile_steps:
            torch.cuda.memory._record_memory_history(max_entries=100000)
            self._is_recording = True
            if self.rank == 0:
                print(f"[MemoryLogger] Step {step}: memory recording started")
    
    # ──────────────────────────────────────────────────────────
    # Per-component direct measurement
    # ──────────────────────────────────────────────────────────
    def _measure_components(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ) -> dict:
        """
        Semantic memory breakdown.

        params     → direct sum of model.parameters()       (exact)
        grads      → direct sum of p.grad                   (exact)
        opt_state  → recursive traversal of optimizer.state (exact)
        activation → total - above three                    (estimated)

        Notes:
          - optimizer state is not created until after the first step()
          - grad is None after zero_grad(set_to_none=True) or before forward → measured as 0
          - activation estimate may include DataLoader input and internal temp buffers
        """

        if self.device.type != "cuda":
            
            return {"param_MB": 0.0, 
                    "grad_MB": 0.0,
                    "opt_state_MB": 0.0,
                    "activation_MB": 0.0}

        # 1. params
        param_bytes = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )

        # 2. gradients
        #    p.grad is None after zero_grad(set_to_none=True) or before forward
        grad_bytes = sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters()
            if p.grad is not None
        )

        # 3. optimizer state
        #    recursive traversal → accurate for any optimizer type
        def _count_bytes(obj) -> int:
            if isinstance(obj, torch.Tensor):
                return obj.numel() * obj.element_size()
            if isinstance(obj, dict):
                return sum(_count_bytes(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return sum(_count_bytes(v) for v in obj)
            return 0


        opt_bytes = _count_bytes(optimizer.state)


        # 4. total allocated
        total_bytes = torch.cuda.memory_allocated(self.device)

        # 5. activation estimate
        #    = total - params - grads - opt_state
        #    clamp to 0 to prevent negatives from measurement error
        activation_bytes = max(0,
            total_bytes - param_bytes - grad_bytes - opt_bytes
        )

        return {
            "param_MB":      round(param_bytes      / 1e6, 2),
            "grad_MB":       round(grad_bytes        / 1e6, 2),
            "opt_state_MB":  round(opt_bytes         / 1e6, 2),
            "activation_MB": round(activation_bytes  / 1e6, 2),
        }


    def _get_nccl_buffer(self) -> int:
        """
        Estimate NCCL buffer size.

        The NCCL buffer lives inside VRAM but PyTorch does not expose
        its exact size directly. We approximate it using memory_stats()
        internal fields.

        Approximation method:
            active_bytes - allocated ≈ NCCL buffer + internal padding.
        PyTorch has no public API that returns the exact NCCL buffer size.
        """

        try:
            stats     = torch.cuda.memory_stats(self.device)
            # active_bytes = sum of all currently active memory blocks
            # slightly different from allocated (includes alignment padding)
            active    = stats.get("active_bytes.all.current", 0)
            allocated = torch.cuda.memory_allocated(self.device)
            # the difference is the estimate of NCCL buffer + internal padding
            return max(0, active - allocated)
        except Exception:
            return 0


    # ──────────────────────────────────────────────────────────
    # AllReduce comm hook
    # ──────────────────────────────────────────────────────────

    def make_comm_hook(self):
        """
        Returns a DDP comm hook.
        Automatically records communication info at the start and end of AllReduce.

        Usage:
          ddp_model.register_comm_hook(state=None, hook=memlog.make_comm_hook())

        Note:
          Using register_comm_hook replaces DDP's default AllReduce.
          Therefore this hook must call all_reduce directly.
        """
        logger = self

        def hook(state, bucket):
            import torch.distributed as dist
            import torch.cuda.nvtx as nvtx

            # Record AllReduce start time
            comm_start_t   = logger._now()
            mem_before     = torch.cuda.memory_allocated(logger.device)
            param_bytes    = (bucket.buffer().numel()
                              * bucket.buffer().element_size())

            nvtx.range_push("NCCL_AllReduce")

            fut = dist.all_reduce(
                bucket.buffer(), async_op=True
            ).get_future()

            def on_complete(fut):
                # Record AllReduce completion time
                duration              = logger._now() - comm_start_t
                mem_after             = torch.cuda.memory_allocated(logger.device)

                logger._comm_duration_ns += duration
                logger._comm_mem_delta_B += max(0, mem_after - mem_before)
                logger._bytes_sent_B     += param_bytes
                logger._bytes_recv_B     += param_bytes

                nvtx.range_pop()
                return fut.value()[0]

            return fut.then(on_complete)

        return hook

    # ──────────────────────────────────────────────────────────
    # Snapshot save (_record_memory_history result)
    # ──────────────────────────────────────────────────────────

    def _save_snapshot(self, step: int):
        """
        Called during profile_steps steps.
        Saves the result of torch.cuda.memory._snapshot().

        The saved file can later be analyzed with:
          import torch.cuda._memory_viz as viz
          with open("step5_rank0.pickle", "rb") as f:
              snapshot = pickle.load(f)
          with open("timeline.html", "w") as f:
              f.write(viz.trace_plot(snapshot))
        """
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(
            self.snapshot_dir,
            f"step{step}_rank{self.rank}.pickle"
        )

        snapshot = torch.cuda.memory._snapshot()

        with open(path, "wb") as f:
            pickle.dump(snapshot, f)

        print(f"[MemoryLogger] Rank {self.rank}: "
              f"snapshot saved → {path}")


    def mark(self, 
            step: int, 
            tag: str,
            model:     Optional[torch.nn.Module]        = None,
            optimizer: Optional[torch.optim.Optimizer]  = None,
            ):

        """
        Record a memory snapshot at the current point in training.

        No synchronize() call → GPU pipeline is preserved, overhead is minimal.
        (Async measurement — a few MB of error is acceptable.)

        If model and optimizer are provided, per-component breakdown is also recorded.
        If not provided, param_MB and similar fields are recorded as 0.

        If tag is 'step_end' and this is a profile step, snapshot is saved automatically.
        """

        if self.device.type != "cuda":
            return

        t = self._now()

        if self.t0 is None:
            self.t0 = t

        alloc      = torch.cuda.memory_allocated(self.device)
        reserv     = torch.cuda.memory_reserved(self.device)
        max_alloc  = torch.cuda.max_memory_allocated(self.device)
        max_reserv = torch.cuda.max_memory_reserved(self.device)
        nccl_buf   = self._get_nccl_buffer()

        # Component breakdown (only when model and optimizer are provided)
        components = {"param_MB": 0.0, "grad_MB": 0.0,
                      "opt_state_MB": 0.0, "activation_MB": 0.0}

        # Only measure components on profile steps to minimize overhead
        if (model is not None and optimizer is not None and step in self.profile_steps):
            components = self._measure_components(model, optimizer)

        self.rows.append(
            MemRecord(
                timestamp_ns=t,
                step=step,
                tag=tag,
                rank=self.rank,
                allocated=alloc,
                reserved=reserv,
                max_allocated=max_alloc,
                max_reserved=max_reserv,
                param_MB=components["param_MB"],
                grad_MB=components["grad_MB"],
                opt_state_MB=components["opt_state_MB"],
                activation_MB=components["activation_MB"],
                nccl_buffer_B=nccl_buf,
                bytes_sent_B=self._bytes_sent_B,
                bytes_recv_B=self._bytes_recv_B,
                comm_duration_ns=self._comm_duration_ns,
                comm_mem_delta_B=self._comm_mem_delta_B,
            )
        )
        # If tag is step_end and recording is active, save snapshot and stop recording
        if tag == "step_end" and self._is_recording:
            self._save_snapshot(step)
            torch.cuda.memory._record_memory_history(enabled=None)
            self._is_recording = False


    # ──────────────────────────────────────────────────────────
    # CSV save
    # ──────────────────────────────────────────────────────────


    def dump(self):
        """Write all records to CSV."""
        if not self.rows:
            print(f"[MemoryLogger] Rank {self.rank}: No rows to dump.")
            return
        
        out_path = self.out_csv.replace(".csv", f"_rank{self.rank}.csv")

        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            
            # 1. Header order matches the write order below
            w.writerow([
                "timestamp_ns", "elapsed_ms", "step", "tag", "rank",
                "allocated_B",     "reserved_B",
                "max_allocated_B", "max_reserved_B",
                "allocated_MB",    "reserved_MB",       # <-- moved here
                "max_allocated_MB","max_reserved_MB",   # <-- moved here
                "param_MB", "grad_MB", "opt_state_MB", "activation_MB", # <-- moved to end
                "nccl_buffer_B",   "nccl_buffer_MB",
                "bytes_sent_B",    "bytes_sent_MB",
                "bytes_recv_B",    "bytes_recv_MB",
                "comm_duration_ms",
                "comm_mem_delta_B","comm_mem_delta_MB",
            ])

            for r in self.rows:
                elapsed_ms = (r.timestamp_ns - self.t0) / 1e6
                # 2. Write values in the same order as the header
                w.writerow([
                    r.timestamp_ns, round(elapsed_ms, 3),
                    r.step, r.tag, r.rank,
                    r.allocated,     r.reserved,       
                    r.max_allocated, r.max_reserved,   
                    round(r.allocated     / 1e6, 2),
                    round(r.reserved      / 1e6, 2),
                    round(r.max_allocated / 1e6, 2),
                    round(r.max_reserved  / 1e6, 2),
                    r.param_MB, r.grad_MB, r.opt_state_MB, r.activation_MB,
                    r.nccl_buffer_B,
                    round(r.nccl_buffer_B   / 1e6, 2),
                    r.bytes_sent_B,
                    round(r.bytes_sent_B    / 1e6, 2),
                    r.bytes_recv_B,
                    round(r.bytes_recv_B    / 1e6, 2),
                    round(r.comm_duration_ns / 1e6, 3),
                    r.comm_mem_delta_B,
                    round(r.comm_mem_delta_B / 1e6, 2),
                ])

        print(f"[MemoryLogger] Rank {self.rank}: Saved {out_path} ({len(self.rows)} records)")