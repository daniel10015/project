from cupti import cupti
from enum import Enum, auto

class Metric(Enum):
    MEMORY = auto()
    MEMCPY = auto()
    DURATION = auto()

# maps metric type to string meaning
MEMCPY_KIND_STR = {
    0: "Unknown",
    1: "Host -> Device",
    2: "Device -> Host",
    3: "Host -> Array",
    4: "Array -> Host",
    5: "Array -> Array",
    6: "Array -> Device",
    7: "Device -> Array",
    8: "Device -> Device",
    9: "Host -> Host",
    10: "Peer -> Peer",
    2147483647: "FORCE_INT"
}

# thin wrapper around dictionary call, maybe more readable
def is_metric_cupti(metric: Metric) -> bool:
    return True if METRIC_TO_CUPTI.get(metric) != None else False

# TODO: replace strings with enum so they're caught by linters better?
# or maybe just always use cupti.activitykind?
METRIC_TO_CUPTI = {
  Metric.MEMCPY:cupti.ActivityKind.MEMCPY,
  Metric.MEMORY:cupti.ActivityKind.MEMORY2,
}

CUPTI_TO_METRIC = {
  cupti.ActivityKind.MEMCPY:Metric.MEMCPY,
  cupti.ActivityKind.MEMORY2:Metric.MEMORY,
}