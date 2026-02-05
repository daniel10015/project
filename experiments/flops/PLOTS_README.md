# Modular Plotting System for CUPTI Profiling

This module provides 5 modular plotting functions for analyzing CUPTI profiling data.

## Overview

The plotting system is located in `plots.py` and provides:

1. **Bottleneck Analysis** - Identifies which kernels take the most time
2. **Stacktrace Visualization** - Shows Python call stack with kernel runtimes
3. **Layer Runtime Statistics** - Box plots showing mean/median/min/max/quartiles for forward/backward passes
4. **Throughput Analysis** - Throughput metrics for forward/backward passes
5. **Timeline Visualization** - Memory copies (2D) and forward/backward passes (1D)

## Integration with Existing Code

### Using with `experiments/thru_com/main.py`

```python
from experiments.flops.plots import (
    KernelDataCollector,
    setup_cupti_kernels,
    finalize_cupti_kernels,
    plot_bottleneck_analysis,
    plot_layer_runtime_stats,
    plot_throughput_analysis,
    plot_timeline_memcpy_and_passes
)
from experiments.thru_com.main import MEMCPY_KIND_STR

# Initialize kernel collector
kernel_collector = KernelDataCollector()

# Setup CUPTI for kernels (in addition to existing MEMCPY setup)
setup_cupti_kernels(kernel_collector)

# Your existing training code...
# ... model training ...

# Finalize CUPTI kernels
finalize_cupti_kernels()

# Generate plots
plot_bottleneck_analysis(kernel_collector)
plot_layer_runtime_stats(profile)  # profile is your ExtractModel instance
plot_throughput_analysis(profile)
plot_timeline_memcpy_and_passes(memcpy_info, profile, MEMCPY_KIND_STR)
```

### Using with `experiments/thru_com/main_ddp.py`

The same approach works, but note that `memcpy_info` is a list in `main_ddp.py`:

```python
# In main_ddp.py, memcpy_info is a list, not a MemoryCopy object
plot_timeline_memcpy_and_passes(memcpy_info, profile, MEMCPY_KIND_STR)
```

## Data Collectors

### KernelDataCollector

Collects kernel launch data from `cupti.ActivityKind.CONCURRENT_KERNEL`:

```python
kernel_collector = KernelDataCollector()
setup_cupti_kernels(kernel_collector)
# ... run your code ...
finalize_cupti_kernels()
```

### StacktraceCollector

Collects Python call stack information (optional):

```python
stacktrace_collector = StacktraceCollector()
import sys
sys.settrace(stacktrace_collector.trace_calls)
# ... run your code ...
sys.settrace(None)
```

## Plot Functions

### 1. plot_bottleneck_analysis(kernel_collector, top_n=20)

Shows which kernels consume the most time. Displays:
- Total duration by kernel (bar chart)
- Mean duration vs launch frequency (scatter plot)

### 2. plot_stacktrace_with_kernels(stacktrace_collector, kernel_collector)

Visualizes Python call stack depth over time with kernel runtimes overlaid.

### 3. plot_layer_runtime_stats(profile, forward_markers=None)

Creates box plots showing runtime statistics (mean, median, min, max, quartiles) for:
- Forward pass layers
- Backward pass layers

### 4. plot_throughput_analysis(profile, forward_markers=None)

Shows throughput (TFLOPs/s) for forward and backward passes using box plots.

### 5. plot_timeline_memcpy_and_passes(memcpy_info, profile, MEMCPY_KIND_STR)

Two-panel plot:
- **Top panel**: Memory copy timeline (2D) showing cumulative memory utilization
- **Bottom panel**: Forward/backward pass timeline (1D) showing layer execution

## Notes

- The plotting functions handle both `MemoryCopy` objects and lists for `memcpy_info`
- Forward/backward pass identification uses a simple heuristic (first half = forward, second half = backward)
- You can provide `forward_markers` to improve forward/backward identification
- All time units are automatically scaled (ns → µs → ms → s)
- Kernel durations are collected from `cupti.ActivityKind.CONCURRENT_KERNEL` which gives kernel launches and durations

## Example

See `example_usage.py` for a complete example.
