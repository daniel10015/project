# memory_logger.py
import csv
from dataclasses import dataclass
from time import time_ns

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

    #nccl
    nccl_buffer_B:    int   # NCCL이 잡아먹는 버퍼 크기
    bytes_sent_B:     int   # 이 스텝에서 보낸 bytes
    bytes_recv_B:     int   # 이 스텝에서 받은 bytes
    comm_duration_ns: int   # 통신에 걸린 시간 (나노초)
    comm_mem_delta_B: int   # 통신 중 메모리 증가량



class MemoryLogger:
    def __init__(self, device="cuda", out_csv="mem_log.csv",
                 batch_size=None, image_size=None, max_batches=None):
        self.device = torch.device(device)
        self.out_csv = out_csv
        self.rows = []
        self.t0 = None


        if batch_size is not None and image_size is not None and max_batches is not None:
            base = out_csv.replace(".csv", "")
            self.out_csv = f"{base}_bs{batch_size}_img{image_size}_mb{max_batches}.csv"
        else:
            self.out_csv = out_csv


        # ── Rank 정보 자동 감지 ──
        # dist가 초기화됐으면 rank 자동으로 가져옴
        # 초기화 안 됐으면 단일 GPU로 간주 (rank=0)
        if dist.is_available() and dist.is_initialized():
            self.rank       = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank       = 0
            self.world_size = 1
        
        self._bytes_sent_B     = 0   # 이 스텝에서 보낸 bytes 누적
        self._bytes_recv_B     = 0   # 이 스텝에서 받은 bytes 누적
        self._comm_duration_ns = 0   # 통신 시간 누적
        self._comm_mem_delta_B = 0   # 통신 중 메모리 변화 누적
        self._comm_start_t     = 0   # 통신 시작 시각 (임시)
        self._comm_mem_before  = 0   # 통신 직전 메모리 (임시)

    def _now(self) -> int:

        return time_ns()

    def reset_step_peak(self):
        
        """Call at step start to measure per-step peak."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # 통신 카운터도 스텝마다 초기화
        self._bytes_sent_B     = 0
        self._bytes_recv_B     = 0
        self._comm_duration_ns = 0
        self._comm_mem_delta_B = 0
    

    def _get_nccl_buffer(self) -> int:
        """
        NCCL 버퍼 크기 추정

        NCCL 버퍼는 VRAM 안에 잡히는데
        PyTorch가 직접 크기를 알려주지 않음
        → memory_stats()의 세부 항목으로 추정
        """

        try:
            stats     = torch.cuda.memory_stats(self.device)
            # active_bytes = 실제로 활성화된 메모리 블록들의 합
            # allocated와 미묘하게 다름 (정렬 패딩 등 포함)
            active    = stats.get("active_bytes.all.current", 0)
            allocated = torch.cuda.memory_allocated(self.device)
            # 그 차이가 NCCL 버퍼 + 내부 패딩의 추정치
            return max(0, active - allocated)
        except Exception:
            return 0

    def mark_comm_start(self):
        """
        AllReduce 등 통신 시작 직전에 호출
        → 통신 시작 시각, 메모리 기록
        """
        if self.device.type != "cuda":
            return
        self._comm_start_t    = self._now()
        self._comm_mem_before = torch.cuda.memory_allocated(self.device)
    
    def mark_comm_end(self, param_bytes: int = 0):
        """
        AllReduce 등 통신 완료 직후에 호출
        param_bytes: 통신한 파라미터의 총 bytes 크기

        DDP AllReduce는 그래디언트를 보내고(sent) 받음(recv)
        → 같은 크기가 sent, recv 양쪽에 기록됨
        """
        if self.device.type != "cuda":
            return

        duration              = self._now() - self._comm_start_t
        mem_after             = torch.cuda.memory_allocated(self.device)

        self._comm_duration_ns += duration
        self._comm_mem_delta_B += max(0, mem_after - self._comm_mem_before)
        self._bytes_sent_B     += param_bytes
        self._bytes_recv_B     += param_bytes  # AllReduce는 보내고 받음

    def register_ddp_hooks(self, model: torch.nn.Module):
        """
        DDP 모델의 그래디언트 훅 등록
        → AllReduce 자동 감지

        사용법:
          model = DDP(model)
          logger.register_ddp_hooks(model)
        """
        logger_ref = self   # 클로저용 참조

        def make_hook(param: torch.Tensor):

            def hook(grad: torch.Tensor):

                # 그래디언트가 AllReduce되는 순간 호출됨
                byte_size = grad.numel() * grad.element_size()
                logger_ref._bytes_sent_B += byte_size
                logger_ref._bytes_recv_B += byte_size
                return grad

            return hook

        hooked = 0
        for name, param in model.named_parameters():

            if param.requires_grad:
                param.register_hook(make_hook(param))
                hooked += 1

        print(f"[MemoryLogger] Rank {self.rank}: "
              f"{hooked}개 파라미터에 통신 훅 등록 완료")

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
        nccl_buf   = self._get_nccl_buffer()

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
                nccl_buffer_B=nccl_buf,
                bytes_sent_B=self._bytes_sent_B,
                bytes_recv_B=self._bytes_recv_B,
                comm_duration_ns=self._comm_duration_ns,
                comm_mem_delta_B=self._comm_mem_delta_B,
            )
        )

    def dump(self):
        """Write all records to CSV."""

        if not self.rows:

            print(f"[MemoryLogger] Rank {self.rank}: No rows to dump.")
            return
        
        out_path = self.out_csv.replace(".csv", f"_rank{self.rank}.csv")


        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                 # 기존 컬럼들
                "timestamp_ns",
                "elapsed_ms",
                "step", "tag", "rank",
                #memor_B
                "allocated_B",     "reserved_B",
                "max_allocated_B", "max_reserved_B",
                #memory_MB
                "allocated_MB",    "reserved_MB",
                "max_allocated_MB","max_reserved_MB",
                # 통신 관련 새 컬럼들
                "nccl_buffer_B",   "nccl_buffer_MB",
                "bytes_sent_B",    "bytes_sent_MB",
                "bytes_recv_B",    "bytes_recv_MB",
                "comm_duration_ms",
                "comm_mem_delta_B","comm_mem_delta_MB",
            ])

            for r in self.rows:
                elapsed_ms = (r.timestamp_n - self.t0) / 1e6
                w.writerow([
                    r.timestamp_ns, elapsed_ms, r.step, r.tag, r.rank,
                    r.allocated,     r.reserved,
                    r.max_allocated, r.max_reserved,
                    r.allocated     / 1e6, r.reserved      / 1e6,
                    r.max_allocated / 1e6, r.max_reserved  / 1e6,
                    r.nccl_buffer_B,       r.nccl_buffer_B / 1e6,
                    r.bytes_sent_B,        r.bytes_sent_B  / 1e6,
                    r.bytes_recv_B,        r.bytes_recv_B  / 1e6,
                    r.comm_duration_ns / 1e6,
                    r.comm_mem_delta_B,    r.comm_mem_delta_B / 1e6,
                ])

        print(f"[MemoryLogger] Rank {self.rank}: "
              f"Saved {out_path} ({len(self.rows)} records)")