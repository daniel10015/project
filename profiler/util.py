import numpy as np


def ns_to_s(val: float) -> float:
  return val*1e-9

def b_to_mb(val: int) -> float:
   return val * 1e-6

def scale_time_units(times_ns):
    """
    Scales time units s.t. it's in the highest possible unit
    that is > 0
    """
    units = ["ns", "µs", "ms", "s"]
    times = np.array(times_ns, dtype=float)
    idx = 0

    while np.max(times) > 1e4 and idx < len(units) - 1:
        times *= 1e-3
        idx += 1

    return times, units[idx]