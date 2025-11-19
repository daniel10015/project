
from enum import Enum
from cupti import cupti
from .metric_callback import metric_collection
from .metric_info import *

# should just always be true but maybe one day it won't be true? I guess if switching out of nvda hw
def is_cupti_required(metrics: tuple[Metric, ...]):
    return True

class profiler():
    
    def cupti_func_buffer_requested(self):
        buffer_size = 8 * 1024 * 1024  # 8MB buffer
        max_num_records = 0            # no bound on # of activity records
        return buffer_size, max_num_records

    def cupti_func_buffer_completed(self, activities: list):
        for activity in activities:
            metric_name = CUPTI_TO_METRIC[activity.kind]
            self.profile_out[metric_name].append(self.metric_collection.route(activity))

    def __init__(self, fn, metrics: tuple[Metric, ...]):
        assert callable(fn), f"{fn.__name__} is not callable"
        self.fn = fn
        self.metrics = metrics
        self.profile_out = {metric: [] for metric in metrics} # str representation
        self.metric_collection = metric_collection(start_time_s=0)
        
        # TODO: Context?

        # enable cupti activities based on metrics
        if is_cupti_required(metrics):
            self.cupti_enabled = True
            cupti.activity_register_callbacks(self.cupti_func_buffer_requested, self.cupti_func_buffer_completed)
        else:
            self.cupti_enabled = False
        
    def __call__(self, *args):
        """
        start profiling here
        """
        for metric in self.metrics:
            print(f'enabling {metric}')
            self.metric_collection.enable(metric)
        ret = self.fn(*args)
        
        if self.cupti_enabled:
            cupti.activity_flush_all(1)
        for metric in self.metrics:
            self.metric_collection.disable(metric)
        return ret

    def visualize(self, metric_types: tuple[Metric,...]):
        """
        Pass in the metrics you want to visualize.

        Or, if nothing gets passed in, visualize on all the metrics you processed
        """
        if len(metric_types) == 0:
            metric_types = self.metrics
        print(f'metric type: {metric_types}')
        for metric_type in metric_types:
            self.metric_collection.render_type(metric_type)

    def spill(self):
        return self.profile_out