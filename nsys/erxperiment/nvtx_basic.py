import time
import torch

# 간단한 context manager 만들기
class nvtx_range:
    def __init__(self, msg):
        self.msg = msg
    def __enter__(self):
        torch.cuda.nvtx.range_push(self.msg)
    def __exit__(self, exc_type, exc, tb):
        torch.cuda.nvtx.range_pop()

def fake_work(name, sec):
    with nvtx_range(name):
        time.sleep(sec)

with nvtx_range("Program"):
    fake_work("Load Data", 0.2)
    fake_work("Forward", 0.3)
    fake_work("Backward", 0.1)
    fake_work("Optimizer Step", 0.15)

print("done")

