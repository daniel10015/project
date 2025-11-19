from . import metric_info
from . import util
from cupti import cupti
import matplotlib.pyplot as plt
import numpy as np

debug = True

class metric_collection:

  def memcpy(self, activity) -> str:
    if debug:
      print(f'activity at ({activity.start}) copies {activity.bytes} bytes via {metric_info.MEMCPY_KIND_STR[activity.copy_kind]} for {activity.end-activity.start}ns')
    self.memcpy_info.append((activity.start, activity.bytes))
    self.memcpy_info.append((activity.end, -activity.bytes))
  
  def memory(self, activity) -> str:
    isAlloc = activity.memory_operation_type == cupti.ActivityMemoryOperationType.ALLOCATION
    opTime = activity.timestamp
    opType = 'malloc' if isAlloc else 'free'
    size = activity.bytes
    addr = activity.address
    if self.memory_info.get(addr):
      if isAlloc:
        assert self.memory_info[addr][-1].get('end', 1e20) < opTime, f'previous end time is {self.memory_info[addr][-1].get("end", 1e20)} but next alloc time is {opTime}'
        self.memory_info[addr].append({'size':size, 'start':opTime})
      else:
        assert self.memory_info.get(addr)[-1].get('end') == None, f"writing end before start: {self.memory_info.get(addr)}"
        assert size == self.memory_info.get(addr)[-1].get('size'), f'size different, free size is {size} but alloc size is {self.memory_info.get(addr)[-1].get("size")}'
        self.memory_info[addr][-1]['end'] = opTime
    else:
      self.memory_info[addr] = [{'size':size, 'start':opTime}]
    return (f'memory operation ({opType}) address {addr} at {opTime} of size {size}')

  def render_memory(self):
    events = []

    # Create (time, delta_size) pairs
    for info_list in self.memory_info.values():
        for info in info_list:
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
    times = times - np.min(times) # offset to 0
    times, units = util.scale_time_units(times_ns=times)
    deltas = np.array(sizes)
    utilization = np.cumsum(deltas)

    # Convert to MB
    utilization_MB = utilization / (1024 ** 2)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.step(times, utilization_MB, where="post", lw=2)
    plt.fill_between(times, utilization_MB, step="post", alpha=0.3)
    plt.xlabel(f"Time ({units})")
    plt.ylabel("Memory (MB)")
    plt.title('Memory Utilization Over Time')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

  def render_memcpy(self):
    if not self.memcpy_info:
        print("No events to plot.")
        return
    
    # Sort events by time
    self.memcpy_info.sort(key=lambda x: x[0])

    # Compute cumulative utilization over time
    times, sizes = zip(*self.memcpy_info)
    times = np.array(times)
    times = times - np.min(times) # offset to 0
    times, units = util.scale_time_units(times_ns=times)
    deltas = np.array(sizes)
    utilization = np.cumsum(deltas)

    # Convert to MB
    utilization_MB = utilization / (1024 ** 2)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.step(times, utilization_MB, where="post", lw=2)
    plt.fill_between(times, utilization_MB, step="post", alpha=0.3)
    plt.xlabel(f"Time ({units})")
    plt.ylabel("Memory (MB)")
    plt.title('Memory Copies Over Time')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

  def __init__(self, start_time_s: float):
    self.start_time_s = start_time_s
    self.router = {
      metric_info.Metric.MEMCPY: self.memcpy,
      metric_info.Metric.MEMORY: self.memory,
    }
    self.renders = {
      metric_info.Metric.MEMORY: self.render_memory,
      metric_info.Metric.MEMCPY: self.render_memcpy,
    }

    self.cupti_enabled = set()

    self.memory_info = {}
    self.memcpy_info = []

  def render_type(self, metric_type: metric_info.Metric):
     self.renders[metric_type]()

  def route(self, activity):
    print(f'kind {activity.kind} res {self.router.get(metric_info.CUPTI_TO_METRIC[activity.kind])(activity)}')
    return self.router.get(metric_info.CUPTI_TO_METRIC[activity.kind])(activity)
  
  def enable(self, metric):
     if metric_info.is_metric_cupti(metric) and not metric in self.cupti_enabled:
      self.cupti_enabled.add(metric)
      cupti.activity_enable(metric_info.METRIC_TO_CUPTI[metric])
  def disable(self, metric):
     if metric_info.is_metric_cupti(metric) and metric in self.cupti_enabled:
      self.cupti_enabled.remove(metric)
      cupti.activity_disable(metric_info.METRIC_TO_CUPTI[metric])