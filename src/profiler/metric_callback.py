from . import metric_info
from . import util
from cupti import cupti
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # (Imported for Gantt chart legend)
import numpy as np
from typing import Optional

class metric_callback:

  def __init__(self, start_time_s: Optional[float] = None):
    # A variable to store the first timestamp in ns (integer).
    self.start_time_ns: Optional[int] = int(start_time_s * 1e9) if start_time_s is not None else None
    
    self.router = {
      'MEMCPY': self.memcpy,
      'MEMORY': self.memory,
      'KERNEL': self.kernel,
    }
    self.renders = {
      'MEMORY': self.render_memory,
      'MEMCPY': self.render_memcpy,
      'KERNEL': self.render_timeline # ⭐️ (Modified) Uncommented and linked new function
    }
    self.memory_info = {}
    self.memcpy_events = []
    
    # ⭐️ (Added) List to store events for timeline visualization
    self.timeline_events = [] 

  #
  def memcpy(self, activity) -> str:
    
    # If start_time_ns is not set, set it to the current time
    if self.start_time_ns is None:
        self.start_time_ns = activity.start
    
    # Calculate relative time (seconds) based on start_time_ns
    at_s = (activity.start - self.start_time_ns) / 1e9

    delta = 0
    if activity.copy_kind == 1: # Host -> Device
        delta = activity.bytes
        copy_type = 'MemCpy (HtoD)'
    elif activity.copy_kind == 2: # Device -> Host
        delta = -activity.bytes
        copy_type = 'MemCpy (DtoH)'
    else:
        copy_type = f'MemCpy (Kind {activity.copy_kind})' # Other copy

    # Store data for the original render_memcpy
    self.memcpy_events.append((activity.start, delta))

    # ⭐️ (Added) Store data for render_timeline
    self.timeline_events.append({
        'type': copy_type,
        'stream': str(activity.stream_id),
        'start_ns': activity.start,
        'end_ns': activity.end
    })

    return (f'{metric_info.MEMCPY_KIND_STR[activity.copy_kind]} of {activity.bytes} bytes '
    # Use at_s variable
    f'on stream {activity.stream_id}, at {at_s:.9f}s '
    f'from duration = {activity.end - activity.start}ns')

  #
  def memory(self, activity) -> str:
    # (No modifications to this method)
    
    if self.start_time_ns is None:
        self.start_time_ns = activity.timestamp

    opTime_ns = activity.timestamp
    at_s = (opTime_ns - self.start_time_ns) / 1e9
    
    opTime_absolute_s = util.ns_to_s(opTime_ns)

    isAlloc = activity.memory_operation_type == cupti.ActivityMemoryOperationType.ALLOCATION
    opType = 'malloc' if isAlloc else 'free'
    size = activity.bytes
    addr = activity.address
    ctx = getattr(activity, 'context_id', None)
    stream = getattr(activity, 'stream_id', None)

    key = (ctx, stream, addr)

    if self.memory_info.get(key):
      assert (not isAlloc) and self.memory_info[key]['size'] == size, \
        f"activity ({activity.memory_operation_type}) should be a free but it's not"
      self.memory_info[key]['end'] = opTime_absolute_s
    else:
      self.memory_info[key] = {'size': size, 'start': opTime_absolute_s}

    return (f'memory operation ({opType}) context={ctx}, stream={stream}, '
            f'address={addr} at {at_s:.9f}s of size {size}')

  def kernel(self, activity) -> str:
      
      def _get(a, key, default=None):
          if isinstance(a, dict):
              return a.get(key, default)
          return getattr(a, key, default)

      name     = _get(activity, "name", "<unknown>")
      stream   = _get(activity, "stream_id", "-")
      start    = _get(activity, "start", 0)
      end      = _get(activity, "end", 0)
      duration = end - start

      if self.start_time_ns is None:
          self.start_time_ns = start
          
      at_s = (start - self.start_time_ns) / 1e9

      # ⭐️ (Added) Store data for render_timeline
      self.timeline_events.append({
          'type': 'KERNEL',
          'stream': str(stream),
          'start_ns': start,
          'end_ns': end,
          'name': name
      })

      return (
          f'Kernel    : "{name}"\n'
          f'Stream ID : {stream}\n'
          f'Start(ns) : {start}\n'
          f'End(ns)   : {end}\n'
          f'Start at  : {at_s:.6f}s \n' # Start from 0
          f'Duration  : {duration}ns \n'
      )
  
  
  def render_memory(self):
    # (No modifications to this method)
    events = []
    for info in self.memory_info.values():
        start, size = info['start'], info['size']
        events.append((start, size))
        if 'end' in info:
            events.append((info['end'], -size))

    if not events:
        print("No events to plot.")
        return
    
    events.sort(key=lambda x: x[0])

    times, sizes = zip(*events)
    times = np.array(times)
    times = times - np.min(times) 
    times, units = util.scale_time_units(times_ns=times)
    deltas = np.array(sizes)
    utilization = np.cumsum(deltas)
    utilization_MB = utilization / (1024 ** 2)

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
    # (No modifications to this method)
    events = self.memcpy_events 
    if not events:
        print("No memcpy events to plot.")
        return
    
    events.sort(key=lambda x: x[0])
    times_ns, sizes = zip(*events)
    
    times = np.array(times_ns)
    times = times - np.min(times) 
    times, units = util.scale_time_units(times_ns=times)
    
    deltas = np.array(sizes) 
    deltas_KB = deltas / 1024  

    plt.figure(figsize=(8, 4))
    plt.bar(times, deltas_KB, width=0.1) 
    plt.xlabel(f"Time ({units})")
    plt.ylabel("Individual Copy Size (KB)") 
    plt.title('Individual MemCpy Operations Over Time') 
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("memory_utilization.png")
    print("Graph saved to 'memory_utilization.png'.")
    plt.show()

  # ⭐️ ================================================================
  # ⭐️ (Added) KERNEL & MEMCPY Timeline Visualization Method
  # ⭐️ ================================================================
  


  def render_timeline(self):
      """
      Visualizes all KERNEL and MEMCPY events stored in self.timeline_events 
      as a Gantt chart.
      """
      events = self.timeline_events
      if not events:
          print("No KERNEL or MEMCPY events captured to render a timeline.")
          return

      # 🌟 [Mod 1]
      # Moved min_start_ns calculation up *before* the debug print
      # to calculate 'relative start time'.
      try:
          # Use .get() to handle cases without 'start_ns' (set to infinity to exclude from min)
          min_start_ns = min(e.get('start_ns', float('inf')) for e in events)
          if min_start_ns == float('inf'):
              print("No valid events with 'start_ns' found.")
              return
      except ValueError:
          print("No valid events found.")
          return


      # --- [Debug Code Start] ---
      print(f"\n--- Captured CUDA Events (Total {len(events)}) ---")
      
      try:
          events_sorted = sorted(events, key=lambda e: e.get('start_ns', 0))
      except TypeError:
          print("Error sorting events. Printing in original order.")
          events_sorted = events

      max_print = 100
      
      for i, event in enumerate(events_sorted[:max_print]):
          name = event.get('name', 'N/A')
          e_type = event.get('type', 'N/A')
          stream = event.get('stream', 'N/A')
          start_ns = event.get('start_ns', 0)
          end_ns = event.get('end_ns', 0)
          duration_ns = end_ns - start_ns
          
          # 🌟 [Mod 2]
          # Calculate 'relative start time (s)' same as graph X-axis
          # Display with 9 decimal places (nanosecond precision)
          relative_start_s = (start_ns - min_start_ns) / 1e9
          
          # 🌟 [Mod 3]
          # Added Start (s) (relative time) to f-string print
          # Removed original Start (ns) (absolute time) as it was too long.
          print(f"[{i+1:03d}] "
                f"Stream: {stream:<6} | "
                f"Type: {e_type:<15} | "
                f"Start (s): {relative_start_s:<12.9f} | "
                f"Duration (ns): {duration_ns:<12,d} | "
                f"Name: {name}")

      if len(events) > max_print:
          print(f"... {len(events) - max_print} more events omitted ...")
          
      print("------------------------------------------------------\n")
      # --- [Debug Code End] ---


      # 🌟 [Mod 4]
      # This block is removed as min_start_ns calculation was moved up.
      # try:
      #     min_start_ns = min(e['start_ns'] for e in events)
      # ...


      # Create a sorted list of stream IDs to use as Y-axis labels
      stream_ids = sorted(
          list(set(e['stream'] for e in events)), 
          key=lambda x: int(x) if x.isdigit() else x
      )
      stream_to_y = {stream_id: i for i, stream_id in enumerate(stream_ids)}
      
      # Color mapping by event type
      color_map = {
          'KERNEL': '#1f77b4',       # Blue
          'MemCpy (HtoD)': '#2ca02c', # Green
          'MemCpy (DtoH)': '#d62728'  # Red
      }

      fig, ax = plt.subplots(figsize=(20, max(8, len(stream_ids) * 0.5))) 

      # Draw each event as a horizontal bar (barh)
      for event in events:
          # 🌟 [Mod 5]
          # Exclude events without a 'stream' key from the graph (error prevention)
          stream = event.get('stream')
          if stream is None: 
              continue
              
          y_pos = stream_to_y[stream]
          start_sec = (event.get('start_ns', 0) - min_start_ns) / 1e9
          duration_sec = (event.get('end_ns', 0) - event.get('start_ns', 0)) / 1e9
          
          if duration_sec <= 0: continue
              
          color = color_map.get(event.get('type'), 'gray') 
          
          ax.barh(
              y_pos,
              duration_sec,
              left=start_sec,
              height=0.6,
              color=color,
              alpha=0.8,
              edgecolor='black'
          )

      # --- Chart Styling ---
      ax.set_yticks(range(len(stream_ids)))
      ax.set_yticklabels([f"Stream {s}" for s in stream_ids])
      ax.set_xlabel('Time (seconds relative to first event)', fontsize=12)
      ax.set_ylabel('CUDA Stream', fontsize=12)
      ax.set_title('CUDA Activity Timeline (KERNEL & MEMCPY)', fontsize=16, fontweight='bold')
      
      max_end_sec = (max(e['end_ns'] for e in events) - min_start_ns) / 1e9
      # Fix X-axis range to 0 ~ 0.2s for zoom-in (previous request)
      ax.set_xlim(0, max_end_sec*1.05)
      #ax.set_xlim(0, 0.2) 

      # Create Legend
      legend_patches = [
          mpatches.Patch(color='#1f77b4', label='KERNEL (Compute)'),
          mpatches.Patch(color='#2ca02c', label='MemCpy (CPU -> GPU)'),
          mpatches.Patch(color='#d62728', label='MemCpy (GPU -> CPU)')
      ]
      ax.legend(handles=legend_patches, loc='upper left', fontsize=10)
      
      ax.grid(True, axis='x', linestyle='--', alpha=0.6)
      plt.tight_layout()
      
      plt.savefig('cuda_activity_timeline.png', dpi=150)
      print("Timeline graph saved to 'cuda_activity_timeline.png'.")
      plt.show()

  # -----------------------------------------------------------------
  
  def render_type(self, metric_type: str):
     # render_type follows the self.renders dictionary, no modification needed
     self.renders[metric_type]()

  def route(self, activity):
    return self.router.get(metric_info.CUPTI_TO_METRIC[activity.kind])(activity)
