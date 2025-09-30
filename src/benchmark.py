from time import perf_counter, sleep

def benchmark(func, *args) -> float:
  """
  Assumes function is blocking, so when the function returns the execution is finished
  """
  start_time = perf_counter()
  func(*args)
  return perf_counter() - start_time

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
  print(f'sleep takes {benchmark(sleep2second)} seconds')
  print(f'long_loop takes {benchmark(long_loop, 1<<2, 1<<10)} seconds')