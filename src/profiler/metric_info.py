from cupti import cupti

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

# TODO: replace strings with enum so they're caught by linters better?
# or maybe just always use cupti.activitykind?
METRIC_TO_CUPTI = {
  'MEMCPY':cupti.ActivityKind.MEMCPY
}

CUPTI_TO_METRIC = {
  cupti.ActivityKind.MEMCPY:'MEMCPY'
}