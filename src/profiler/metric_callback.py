from . import metric_info
from . import util

class metric_callback:

  def memcpy(self, activity) -> str:
    print('collect metric')
    return (f'{metric_info.MEMCPY_KIND_STR[activity.copy_kind]} of {activity.bytes} bytes'
    f'on stream {activity.stream_id}, at {util.ns_to_s(activity.start) - self.start_time_s}s '
    f'from duration = {activity.end - activity.start}ns')
  
  def __init__(self, start_time_s: float):
    self.start_time_s = start_time_s
    self.router = {
      'MEMCPY': self.memcpy
    }

  def route(self, activity):
    print('route')
    return self.router.get(metric_info.CUPTI_TO_METRIC[activity.kind])(activity)