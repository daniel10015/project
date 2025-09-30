from time import perf_counter, sleep

# TODO figure out how to arbitrarily pass in params
def benchmark(func, weak = False) -> float:
  """
  Assumes function is blocking, so when the function returns the execution is finished
  """
  start_time = perf_counter()
  func()
  return perf_counter() - start_time

def sleep2second():
  sleep(2)

def long_loop(n=1024):
  k = 0
  for i in range(1024):
    for ii in range(1024):
      k += i + ii
  return k  

if __name__ == '__main__':
  print(f'sleep takes {benchmark(sleep2second)} seconds')
  print(f'long_loop takes {benchmark(long_loop)} seconds')