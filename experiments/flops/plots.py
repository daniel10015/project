"""
Modular plotting system for CUPTI profiling data.

This module provides 5 main plotting functions:
1. Bottleneck analysis on cupti data
2. Stacktrace visualization of python calls and kernel runtimes
3. Layer runtime statistics (mean/median/min/max/quartiles) for forward/backward
4. Throughput analysis for forward/backward passes
5. Timeline of memory copies (2D) and forward/backward passes (1D)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import sys
from cupti import cupti


# ==============================
# Data Collectors
# ==============================

class KernelDataCollector:
    """Collects kernel launch data from CUPTI CONCURRENT_KERNEL activities."""
    
    def __init__(self):
        self.kernels = []  # List of (start_ns, end_ns, name, duration_ns)
        self.kernel_by_name = defaultdict(list)  # Group kernels by name
        
    def collect(self, activity):
        """Collect a kernel activity."""
        if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
            duration_ns = activity.end - activity.start
            kernel_info = {
                'start': activity.start,
                'end': activity.end,
                'name': activity.name,
                'duration_ns': duration_ns
            }
            self.kernels.append(kernel_info)
            self.kernel_by_name[activity.name].append(kernel_info)
    
    def get_total_duration(self):
        """Get total duration of all kernels."""
        if not self.kernels:
            return 0
        return max(k['end'] for k in self.kernels) - min(k['start'] for k in self.kernels)
    
    def get_kernel_statistics(self):
        """Get statistics grouped by kernel name."""
        stats = {}
        for name, kernels in self.kernel_by_name.items():
            durations = [k['duration_ns'] for k in kernels]
            stats[name] = {
                'count': len(durations),
                'total_duration': sum(durations),
                'mean_duration': np.mean(durations),
                'median_duration': np.median(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'std_duration': np.std(durations)
            }
        return stats


class StacktraceCollector:
    """Collects Python call stack information."""
    
    def __init__(self):
        self.call_stack = []
        self.call_history = []  # List of (timestamp, event, function_name, depth)
        self.depth = 0
        
    def trace_calls(self, frame, event, arg):
        """Tracer function for sys.settrace."""
        if event == "call":
            func_name = frame.f_code.co_name
            class_name = None
            
            # Extract class name if available
            locals_ = frame.f_locals
            if 'self' in locals_:
                class_name = locals_['self'].__class__.__name__
            elif 'cls' in locals_ and isinstance(locals_['cls'], type):
                class_name = locals_['cls'].__name__
            
            full_name = f"{class_name}.{func_name}" if class_name else func_name
            call_info = {
                "name": full_name,
                "frame": frame,
                "id": id(frame),
                "filename": frame.f_code.co_filename,
                "lineno": frame.f_lineno,
                "depth": self.depth
            }
            self.call_stack.append(call_info)
            self.call_history.append(("call", full_name, self.depth))
            self.depth += 1
            
        elif event == "return":
            if self.call_stack:
                popped_call = self.call_stack.pop()
                self.call_history.append(("return", popped_call["name"], self.depth))
                self.depth -= 1
                
        return self.trace_calls


# ==============================
# Utility Functions
# ==============================

def scale_time_units(times_ns):
    """
    Scales time values so that they are expressed in the largest possible unit
    (ns, µs, ms, s) such that the values remain > 0.
    """
    units = ["ns", "µs", "ms", "s"]
    times = np.array(times_ns, dtype=float)
    idx = 0
    
    while np.max(times) > 1e4 and idx < len(units) - 1:
        times *= 1e-3
        idx += 1
    
    return times, units[idx]


def identify_forward_backward(layer_timeline: List[Tuple], forward_markers: Optional[List[str]] = None):
    """
    Identify forward and backward passes from layer timeline.
    Returns separate lists for forward and backward passes.
    """
    if forward_markers is None:
        # Heuristic: assume first half is forward, second half is backward
        # This is a simple heuristic and may need refinement
        mid_point = len(layer_timeline) // 2
        forward = layer_timeline[:mid_point]
        backward = layer_timeline[mid_point:]
    else:
        # More sophisticated: use layer names to identify
        forward = []
        backward = []
        for event in layer_timeline:
            layer_name = event[-1] if len(event) > 3 else ""
            if any(marker in layer_name for marker in forward_markers):
                forward.append(event)
            else:
                backward.append(event)
    
    return forward, backward


# ==============================
# Plot 1: Bottleneck Analysis
# ==============================

def plot_bottleneck_analysis(kernel_collector: KernelDataCollector, 
                             top_n: int = 20,
                             figsize: Tuple[int, int] = (12, 8)):
    """
    Plot bottleneck analysis showing which kernels take the most time.
    
    Args:
        kernel_collector: KernelDataCollector instance with collected kernel data
        top_n: Number of top kernels to display
        figsize: Figure size tuple
    """
    if not kernel_collector.kernels:
        print("No kernel data available for bottleneck analysis")
        return
    
    stats = kernel_collector.get_kernel_statistics()
    
    # Sort by total duration
    sorted_kernels = sorted(stats.items(), 
                            key=lambda x: x[1]['total_duration'], 
                            reverse=True)[:top_n]
    
    kernel_names = [name for name, _ in sorted_kernels]
    total_durations = [stat['total_duration'] for _, stat in sorted_kernels]
    mean_durations = [stat['mean_duration'] for _, stat in sorted_kernels]
    counts = [stat['count'] for _, stat in sorted_kernels]
    
    # Convert to microseconds for readability
    total_durations_us = [d / 1e3 for d in total_durations]
    mean_durations_us = [d / 1e3 for d in mean_durations]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Total duration by kernel
    y_pos = np.arange(len(kernel_names))
    ax1.barh(y_pos, total_durations_us, align='center')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([name[:50] + '...' if len(name) > 50 else name 
                         for name in kernel_names], fontsize=8)
    ax1.set_xlabel('Total Duration (µs)')
    ax1.set_title('Top Kernels by Total Duration')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean duration vs count
    scatter = ax2.scatter(counts, mean_durations_us, s=100, alpha=0.6)
    ax2.set_xlabel('Kernel Launch Count')
    ax2.set_ylabel('Mean Duration (µs)')
    ax2.set_title('Kernel Duration vs Launch Frequency')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Annotate top kernels
    for i, (name, stat) in enumerate(sorted_kernels[:5]):
        ax2.annotate(name[:30], 
                    (stat['count'], stat['mean_duration'] / 1e3),
                    fontsize=7, alpha=0.7)
    
    plt.tight_layout()
    return fig


# ==============================
# Plot 2: Stacktrace Visualization
# ==============================

def plot_stacktrace_with_kernels(stacktrace_collector: StacktraceCollector,
                                 kernel_collector: KernelDataCollector,
                                 figsize: Tuple[int, int] = (16, 10)):
    """
    Visualize Python call stack with kernel runtimes overlaid.
    
    Args:
        stacktrace_collector: StacktraceCollector with call history
        kernel_collector: KernelDataCollector with kernel data
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize timestamps
    if kernel_collector.kernels:
        kernel_times = [k['start'] for k in kernel_collector.kernels]
        t0 = min(kernel_times)
    else:
        t0 = 0
    
    # Plot call stack depth over time (simplified)
    if stacktrace_collector.call_history:
        # Create a timeline of stack depth
        # This is a simplified version - in practice you'd need timestamps
        depths = [depth for _, _, depth in stacktrace_collector.call_history]
        call_names = [name for _, name, _ in stacktrace_collector.call_history]
        
        # Plot stack depth
        x_positions = np.arange(len(depths))
        ax.plot(x_positions, depths, 'b-', alpha=0.3, label='Call Stack Depth', linewidth=1)
    
    # Plot kernels as horizontal bars
    if kernel_collector.kernels:
        y_kernel = 0
        kernel_y_offset = max(depths) + 2 if stacktrace_collector.call_history else 0
        
        for i, kernel in enumerate(kernel_collector.kernels):
            start_rel = (kernel['start'] - t0) / 1e6  # Convert to ms
            duration_ms = kernel['duration_ns'] / 1e6
            
            # Map kernel time to x-axis (simplified - would need proper time mapping)
            # For now, use index-based positioning
            ax.barh(y_kernel + kernel_y_offset, duration_ms, 
                   left=start_rel, height=0.5, alpha=0.6,
                   color='red', edgecolor='black')
            
            if i < 10:  # Label first 10 kernels
                ax.text(start_rel + duration_ms/2, y_kernel + kernel_y_offset,
                       kernel['name'][:20], fontsize=6, ha='center', va='center')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Stack Depth / Kernel Index')
    ax.set_title('Python Call Stack with Kernel Runtimes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ==============================
# Plot 3: Layer Runtime Statistics
# ==============================

def plot_layer_runtime_stats(profile, 
                            forward_markers: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (14, 8)):
    """
    Plot layer runtime statistics (mean/median/min/max/quartiles) for forward and backward.
    
    Args:
        profile: ModelProfile instance with layer timing data
        forward_markers: List of strings to identify forward pass layers
        figsize: Figure size tuple
    """
    if not hasattr(profile, 'layers') or not profile.layers:
        print("No layer data available")
        return
    
    # Separate forward and backward
    forward_times = {}
    backward_times = {}
    
    # Identify forward vs backward (simplified heuristic)
    layer_names = list(profile.layers.keys())
    mid_point = len(layer_names) // 2
    
    for i, (layer_name, layer_data) in enumerate(profile.layers.items()):
        times = layer_data.get('time_ns', [])
        if not times:
            continue
            
        times_us = [t / 1e3 for t in times]  # Convert to microseconds
        
        if i < mid_point:
            forward_times[layer_name] = times_us
        else:
            backward_times[layer_name] = times_us
    
    # Calculate statistics
    def calc_stats(times_list):
        if not times_list:
            return {}
        arr = np.array(times_list)
        return {
            'mean': np.mean(arr),
            'median': np.median(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'q25': np.percentile(arr, 25),
            'q75': np.percentile(arr, 75)
        }
    
    forward_stats = {name: calc_stats(times) for name, times in forward_times.items()}
    backward_stats = {name: calc_stats(times) for name, times in backward_times.items()}
    
    # Create box plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Forward pass
    if forward_stats:
        forward_data = [forward_times[name] for name in forward_stats.keys()]
        forward_labels = [name[:30] for name in forward_stats.keys()]
        
        bp1 = ax1.boxplot(forward_data, labels=forward_labels, vert=True, patch_artist=True)
        ax1.set_ylabel('Runtime (µs)')
        ax1.set_title('Forward Pass Layer Runtimes')
        ax1.set_xticklabels(forward_labels, rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Color boxes
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
    
    # Backward pass
    if backward_stats:
        backward_data = [backward_times[name] for name in backward_stats.keys()]
        backward_labels = [name[:30] for name in backward_stats.keys()]
        
        bp2 = ax2.boxplot(backward_data, labels=backward_labels, vert=True, patch_artist=True)
        ax2.set_ylabel('Runtime (µs)')
        ax2.set_title('Backward Pass Layer Runtimes')
        ax2.set_xticklabels(backward_labels, rotation=45, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Color boxes
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
    
    plt.tight_layout()
    return fig


# ==============================
# Plot 4: Throughput Analysis
# ==============================

def plot_throughput_analysis(profile,
                            forward_markers: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (14, 8)):
    """
    Plot throughput analysis for forward and backward passes.
    
    Args:
        profile: ModelProfile instance with layer timing and FLOPs data
        forward_markers: List of strings to identify forward pass layers
        figsize: Figure size tuple
    """
    if not hasattr(profile, 'layers') or not profile.layers:
        print("No layer data available")
        return
    
    forward_throughput = []
    backward_throughput = []
    forward_labels = []
    backward_labels = []
    
    layer_names = list(profile.layers.keys())
    mid_point = len(layer_names) // 2
    
    for i, (layer_name, layer_data) in enumerate(profile.layers.items()):
        times = layer_data.get('time_ns', [])
        flops = layer_data.get('flops', None)
        
        if not times or flops is None or flops <= 0:
            continue
        
        # Calculate throughput (FLOPs per second)
        throughputs = []
        for time_ns in times:
            time_s = time_ns / 1e9
            if time_s > 0:
                throughput = flops / time_s  # FLOPs per second
                throughputs.append(throughput / 1e12)  # Convert to TFLOPs
        
        if not throughputs:
            continue
        
        if i < mid_point:
            forward_throughput.append(throughputs)
            forward_labels.append(layer_name)
        else:
            backward_throughput.append(throughputs)
            backward_labels.append(layer_name)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Forward pass throughput
    if forward_throughput:
        bp1 = ax1.boxplot(forward_throughput, labels=forward_labels, vert=True, patch_artist=True)
        ax1.set_ylabel('Throughput (TFLOPs/s)')
        ax1.set_title('Forward Pass Throughput')
        ax1.set_xticklabels(forward_labels, rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        for patch in bp1['boxes']:
            patch.set_facecolor('lightgreen')
    
    # Backward pass throughput
    if backward_throughput:
        bp2 = ax2.boxplot(backward_throughput, labels=backward_labels, vert=True, patch_artist=True)
        ax2.set_ylabel('Throughput (TFLOPs/s)')
        ax2.set_title('Backward Pass Throughput')
        ax2.set_xticklabels(backward_labels, rotation=45, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        for patch in bp2['boxes']:
            patch.set_facecolor('lightyellow')
    
    plt.tight_layout()
    return fig


# ==============================
# Plot 5: Timeline of Memory Copies and Forward/Backward Passes
# ==============================

def plot_timeline_memcpy_and_passes(memcpy_info: List[Tuple],
                                   profile,
                                   MEMCPY_KIND_STR: Dict,
                                   forward_markers: Optional[List[str]] = None,
                                   figsize: Tuple[int, int] = (16, 10)):
    """
    Plot timeline showing memory copies (2D) and forward/backward passes (1D).
    
    Args:
        memcpy_info: List of (time_ns, bytes, copy_kind) tuples or MemoryCopy object
        profile: ModelProfile/ExtractModel instance with timeline data
        MEMCPY_KIND_STR: Dictionary mapping copy_kind to string
        forward_markers: List of strings to identify forward pass layers
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Handle memcpy_info as list or object with memcpy_info attribute
    if hasattr(memcpy_info, 'memcpy_info'):
        memcpy_list = memcpy_info.memcpy_info
    else:
        memcpy_list = memcpy_info
    
    # ========== Memory Copy Timeline (2D) ==========
    if memcpy_list:
        memcpy_info_sorted = sorted(memcpy_list, key=lambda x: x[0])
        
        # Handle different tuple lengths (with or without device_id)
        if len(memcpy_info_sorted[0]) >= 3:
            times, sizes, kinds = zip(*[(t[0], t[1], t[2]) for t in memcpy_info_sorted])
        else:
            times, sizes = zip(*memcpy_info_sorted)
            kinds = [0] * len(times)  # Default kind
        
        times = np.array(times)
        t0_memcpy = np.min(times)
        times = times - t0_memcpy
        times_scaled, time_units = scale_time_units(times)
        
        utilization = np.cumsum(np.array(sizes))
        
        memcpy_colors = {
            "Host -> Device": "tab:green",
            "Device -> Host": "tab:blue",
            "Other": "tab:gray"
        }
        
        for kind_str in ["Host -> Device", "Device -> Host", "Other"]:
            mask = np.array([
                (MEMCPY_KIND_STR.get(k, "Other") == kind_str)
                if k in MEMCPY_KIND_STR else False
                for k in kinds
            ])
            if not np.any(mask):
                continue
            
            ax1.step(times_scaled[mask], utilization[mask], where="post",
                    lw=2, label=f"Memcpy: {kind_str}", color=memcpy_colors[kind_str])
        
        ax1.set_ylabel("Memory Utilization (bytes)")
        ax1.set_yscale("log")
        ax1.set_title("Memory Copy Timeline (2D)")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.4)
    
    # ========== Forward/Backward Pass Timeline (1D) ==========
    if hasattr(profile, 'timeline') and profile.timeline:
        timeline = sorted(profile.timeline, key=lambda x: x[0])
        
        if timeline:
            # Handle timeline format: (start, end, flops, layer) or (device_id, start, end, flops, layer)
            if len(timeline[0]) == 5:
                # DDP format with device_id
                timeline_clean = [(start, end, flops, layer) for _, start, end, flops, layer in timeline]
            else:
                timeline_clean = timeline
            
            t0_timeline = timeline_clean[0][0]
            
            # Align with memcpy timeline if available
            if memcpy_list:
                t0_memcpy = min(t[0] for t in memcpy_list)
                t0_aligned = min(t0_memcpy, t0_timeline)
            else:
                t0_aligned = t0_timeline
            
            # Separate forward and backward
            # Use profile.base_time if available for alignment
            if hasattr(profile, 'base_time') and profile.base_time > 0:
                base_time = profile.base_time
            else:
                base_time = t0_aligned
            
            # Heuristic: layers appearing in first half of timeline are forward
            layer_names = list(profile.layers.keys()) if hasattr(profile, 'layers') else []
            mid_point = len(layer_names) // 2 if layer_names else len(timeline_clean) // 2
            layer_to_index = {name: i for i, name in enumerate(layer_names)} if layer_names else {}
            
            forward_events = []
            backward_events = []
            
            for i, (start, end, flops, layer) in enumerate(timeline_clean):
                layer_idx = layer_to_index.get(layer, i)
                start_rel = (start - base_time) / 1e6  # Convert to ms
                end_rel = (end - base_time) / 1e6
                duration = end_rel - start_rel
                
                if layer_idx < mid_point or i < len(timeline_clean) // 2:
                    forward_events.append((start_rel, duration, layer))
                else:
                    backward_events.append((start_rel, duration, layer))
            
            # Plot forward pass
            if forward_events:
                y_forward = 1.0
                for start, duration, layer in forward_events:
                    ax2.barh(y_forward, duration, left=start, height=0.3,
                            color='blue', alpha=0.7, edgecolor='black')
                    if duration > 0.1:  # Only label if bar is large enough
                        ax2.text(start + duration/2, y_forward, layer[:15],
                                ha='center', va='center', fontsize=7)
            
            # Plot backward pass
            if backward_events:
                y_backward = 0.5
                for start, duration, layer in backward_events:
                    ax2.barh(y_backward, duration, left=start, height=0.3,
                            color='red', alpha=0.7, edgecolor='black')
                    if duration > 0.1:  # Only label if bar is large enough
                        ax2.text(start + duration/2, y_backward, layer[:15],
                                ha='center', va='center', fontsize=7)
            
            ax2.set_ylim(0, 1.5)
            ax2.set_yticks([0.5, 1.0])
            ax2.set_yticklabels(['Backward', 'Forward'])
            ax2.set_ylabel("Pass Type")
            ax2.set_xlabel(f"Time ({time_units})")
            ax2.set_title("Forward/Backward Pass Timeline (1D)")
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ==============================
# Helper function to setup CUPTI for kernels
# ==============================

def setup_cupti_kernels(kernel_collector: KernelDataCollector):
    """Setup CUPTI to collect CONCURRENT_KERNEL activities."""
    def func_buffer_requested():
        buffer_size = 8 * 1024 * 1024  # 8MB buffer
        max_num_records = 0
        return buffer_size, max_num_records
    
    def func_buffer_completed(activities: list):
        for activity in activities:
            kernel_collector.collect(activity)
    
    cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    
    return func_buffer_requested, func_buffer_completed


def finalize_cupti_kernels():
    """Finalize CUPTI kernel collection."""
    cupti.activity_flush_all(1)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
