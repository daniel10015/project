
from time import perf_counter, perf_counter_ns, time
from enum import Enum, unique, auto
from cupti import cupti
from .metric_callback import metric_callback
from . import metric_info

"""
#A wrapper class for train/test loops defined as functions.
@unique
class Metric(Enum):
    CPU_TIME_TOTAL = auto()
    CPU_MEM_COPY = auto()
    GPU_MEM_COPY = auto()
"""

class profiler():
    
    def cupti_func_buffer_requested(self):
        buffer_size = 8 * 1024 * 1024  # 8MB buffer
        max_num_records = 0            # no bound on # of activity records
        return buffer_size, max_num_records

    def cupti_func_buffer_completed(self, activities: list):
        for activity in activities:
            metric_name = metric_info.CUPTI_TO_METRIC[activity.kind]
            self.profile_out[metric_name].append(self.metric_callback.route(activity))

    def __init__(self, fn, metrics: tuple[str, ...]):
        assert callable(fn), f"{fn.__name__} is not callable"
        self.fn = fn
        self.metrics = metrics
        self.profile_out = {metric: [] for metric in metrics}
        self.metric_callback = metric_callback(start_time_s=0)
        
        # TODO: Context?

        # enable cupti activities based on metrics
        cupti.activity_register_callbacks(self.cupti_func_buffer_requested, self.cupti_func_buffer_completed)
        
    def __call__(self, *args):
        """
        start profiling here
        """
        for metric in self.metrics:
            cupti.activity_enable(metric_info.METRIC_TO_CUPTI[metric])
        ret = self.fn(*args)
        cupti.activity_flush_all(1)
        for metric in self.metrics:
            cupti.activity_disable(metric_info.METRIC_TO_CUPTI[metric])
        return ret

    def visualize(self, metric_types: tuple[str,...] = None):
        """
        Pass in the metrics you want to visualize.

        Or, if nothing gets passed in, visualize on all the metrics you processed
        """
        if metric_types == None:
            metric_types = self.metrics
        for metric_type in metric_types:
            self.metric_callback.render_type(metric_type)

    def spill(self):
        return self.profile_out