
from time import perf_counter, perf_counter_ns, sleep
    

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
