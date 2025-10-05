from . import metric_info
from . import util
from cupti import cupti
import matplotlib.pyplot as plt
import numpy as np

class metric_callback:

  def memcpy(self, activity) -> str:
    return (f'{metric_info.MEMCPY_KIND_STR[activity.copy_kind]} of {activity.bytes} bytes '
    f'on stream {activity.stream_id}, at {util.ns_to_s(activity.start) - self.start_time_s}s '
    f'from duration = {activity.end - activity.start}ns')
  
  def memory(self, activity) -> str:
    isAlloc = activity.memory_operation_type == cupti.ActivityMemoryOperationType.ALLOCATION
    opTime = util.ns_to_s(activity.timestamp) - self.start_time_s
    opType = 'malloc' if isAlloc else 'free'
    size = activity.bytes
    addr = activity.address
    if self.memory_info.get(addr):
      assert (not isAlloc) and self.memory_info[addr]['size'] == size, "activity should be a free but it's not"
      self.memory_info[addr]['end'] = opTime
    else:
      self.memory_info[addr] = {'size':size, 'start':opTime}
    return (f'memory operation ({opType}) address {addr} at {opTime} of size {size}')

  def render_memory(self):
    events = []

    # Create (time, delta_size) pairs
    for info in self.memory_info.values():
        start, size = info['start'], info['size']
        events.append((start, size))  # allocation event
        if 'end' in info:
            events.append((info['end'], -size))  # free event 

    if not events:
        print("No events to plot.")
        return
    
    # Sort events by time
    events.sort(key=lambda x: x[0])

    # Compute cumulative utilization over time
    times, sizes = zip(*events)
    times = np.array(times)
    deltas = np.array(sizes)
    utilization = np.cumsum(deltas)

    # Convert to MB
    utilization_MB = utilization / (1024 ** 2)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.step(times, utilization_MB, where="post", lw=2)
    plt.fill_between(times, utilization_MB, step="post", alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MB)")
    plt.title('Memory Utilization Over Time')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

  def __init__(self, start_time_s: float):
    self.start_time_s = start_time_s
    self.router = {
      'MEMCPY': self.memcpy,
      'MEMORY': self.memory,
    }
    self.renders = {
      'MEMORY': self.render_memory
    }

    self.memory_info = {}

  def render_type(self, metric_type: str):
     self.renders[metric_type]()

  def route(self, activity):
    return self.router.get(metric_info.CUPTI_TO_METRIC[activity.kind])(activity)