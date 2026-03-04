import torch
import time

# PyTorch NVTX (설치 필요 없음)
class nvtx_range:
    def __init__(self, msg): self.msg = msg
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(self.msg)
    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

print("cuda available:", torch.cuda.is_available())

# GPU 없으면 종료 (CUDA HW는 GPU 있어야 나옴)
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU on this node.")

device = "cuda"
A = torch.randn(4096, 4096, device=device)
B = torch.randn(4096, 4096, device=device)

with nvtx_range("Program"):
    for i in range(3):
        with nvtx_range(f"Step {i}"):
            with nvtx_range("Matmul"):
                C = A @ B
            # GPU 작업이 끝나도록 기다리기 (타임라인 정렬이 쉬워짐)
            with nvtx_range("Sync"):
                torch.cuda.synchronize()
            time.sleep(0.05)

print("done")
