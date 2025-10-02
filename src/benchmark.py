
from time import perf_counter, perf_counter_ns, sleep
from enum import Enum, unique
from cupti import cupti
"""
A wrapper class for train/test loops defined as functions.
"""
@unique
class Metric(Enum):
    CPU_TIME_TOTAL = auto()
    CPU_MEM_COPY = auto()
    GPU_MEM_COPY = auto()
    

class LoopFnWrap():
    METRIC_CUPTI_ENUM_MAP = {
            "GPU_MEM_COPY":cupti.ActivityKind.MEMCPY
    }
    def cupti_func_buffer_requested(self):
        buffer_size = 8 * 1024 * 1024  # 8MB buffer
        max_num_records = 0            # no bound on # of activity records
        return buffer_size, max_num_records

    def cupti_func_buffer_completed(self, activities: list):
        self.cupti_activities = activities

    def __init__(self, fn, 
                 metrics=("GPU_MEM_COPY")):
        assert callable(fn), f"{fn.__name__} is not callable"
        self.fn = fn
        self.metrics = metrics  # TODO: Add metrics/activity kind
        self.cupti_activities = None  # activity objects returned from cupti callback
        # TODO: Context?

        # enable cupti activities based on metrics
        

    def __call__(self):
        
        # start profiling
        self.fn()
        # end profiling and print/return results

def benchmark_s(func, *args) -> float:
  """
  Assumes function is blocking, so when the function returns the execution is finished
  """
  start_time = perf_counter()
  func(*args)
  return perf_counter() - start_time

def benchmark_ns(func, *args) -> int:
  """
  Assumes function is blocking, so when the function returns the execution is finished
  """
  start_time = perf_counter_ns()
  func(*args)
  return perf_counter_ns() - start_time

# Sample functions to test capabilities
def sleep2second():
  sleep(2)

def long_loop(start=0, n=1024):
  k = 0
  for i in range(start,n):
    for ii in range(start,n):
      k += i + ii
  return k  

if __name__ == '__main__':
  print(f'sleep takes {benchmark_s(sleep2second)} seconds')
  print(f'long_loop takes {benchmark_s(long_loop, 1<<2, 1<<10)} seconds')
  print(f'sleep takes {benchmark_ns(sleep2second)} nanoseconds')
  print(f'long_loop takes {benchmark_ns(long_loop, 1<<2, 1<<10)} nanoseconds')
