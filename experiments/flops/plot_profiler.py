"""
Profiler API for the plotting system.

Similar to src/profiler/profiler.py, but collects all metrics automatically
and provides visualization capabilities.
"""

import sys
import time
from time import perf_counter_ns, time_ns
from typing import Callable, Any, List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from cupti import cupti

from plots import (
    KernelDataCollector,
    scale_time_units
)

# MEMCPY_KIND_STR mapping (same as in main.py)
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


class TimedStacktraceCollector:
    """Collects Python call stack information with timestamps and depth limit."""
    
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.call_stack = []
        self.call_events = []  # List of (timestamp_ns, event_type, function_name, depth, frame_id)
        self.depth = 0
        self.start_time = None
        
    def trace_calls(self, frame, event, arg):
        """Tracer function for sys.settrace."""
        if self.start_time is None:
            self.start_time = time_ns()
        
        current_time = time_ns()
        frame_id = id(frame)
        
        if event == "call":
            # Only track up to max_depth
            if self.depth < self.max_depth:
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
                    "id": frame_id,
                    "filename": frame.f_code.co_filename,
                    "lineno": frame.f_lineno,
                    "depth": self.depth,
                    "start_time": current_time
                }
                self.call_stack.append(call_info)
                self.call_events.append(("call", full_name, self.depth, current_time, frame_id))
                self.depth += 1
            else:
                # Still increment depth but don't track
                self.depth += 1
                
        elif event == "return":
            # The frame in a return event is the frame that's returning
            # The top of call_stack should be the matching call (LIFO order)
            if self.call_stack:
                # Pop from stack - should match the returning frame
                popped_call = self.call_stack.pop()
                # Use the frame_id from the return event's frame to ensure correct matching
                self.call_events.append(("return", popped_call["name"], popped_call["depth"], current_time, frame_id))
                self.depth -= 1
            elif self.depth > 0:
                # Handle case where we didn't track the call (depth >= max_depth)
                self.depth -= 1
                
        return self.trace_calls
    
    def get_call_ranges(self):
        """Get call ranges as (start_ns, end_ns, name, depth) tuples.
        
        Matches call and return events by frame_id to correctly calculate function durations.
        Each function call has a unique frame_id, and the return event matches that same frame_id.
        """
        ranges = []
        active_calls = {}  # frame_id -> (start_time, name, depth)
        
        for event_type, name, depth, timestamp, frame_id in self.call_events:
            if event_type == "call":
                # Store the call with its frame_id
                active_calls[frame_id] = (timestamp, name, depth)
            elif event_type == "return":
                # Match return with the corresponding call by frame_id
                if frame_id in active_calls:
                    start_time, call_name, call_depth = active_calls.pop(frame_id)
                    ranges.append((start_time, timestamp, call_name, call_depth))
        
        # Handle any remaining active calls (functions that didn't return)
        end_time = time_ns()
        for frame_id, (start_time, name, depth) in active_calls.items():
            ranges.append((start_time, end_time, name, depth))
        
        return ranges


class MemoryCopyCollector:
    """Collects memory copy operations from CUPTI."""
    
    def __init__(self):
        self.memcpy_info = []
        
    def collect(self, activity):
        """Collect a memcpy activity."""
        if activity.kind == cupti.ActivityKind.MEMCPY:
            self.memcpy_info.append((activity.start, activity.bytes, activity.copy_kind))
            self.memcpy_info.append((activity.end, -activity.bytes, activity.copy_kind))


class PlotProfiler:
    """
    Profiler that collects all metrics (kernels, memcpy, stacktrace) automatically
    and provides visualization capabilities.
    """
    
    MAX_STACKTRACE_DEPTH = 10  # Maximum depth for stacktrace visualization
    
    def __init__(self, fn: Callable):
        """
        Initialize the profiler with a function to profile.
        
        Args:
            fn: Callable function to profile
        """
        assert callable(fn), f"{fn} is not callable"
        self.fn = fn
        
        # Initialize collectors
        self.kernel_collector = KernelDataCollector()
        self.memcpy_collector = MemoryCopyCollector()
        self.stacktrace_collector = TimedStacktraceCollector(max_depth=10)
        
        # Setup CUPTI callbacks
        cupti.activity_register_callbacks(
            self._cupti_buffer_requested,
            self._cupti_buffer_completed
        )
        
        self._tracing_enabled = False
        
    def _cupti_buffer_requested(self):
        """CUPTI buffer request callback."""
        buffer_size = 8 * 1024 * 1024  # 8MB buffer
        max_num_records = 0
        return buffer_size, max_num_records
    
    def _cupti_buffer_completed(self, activities: list):
        """CUPTI buffer completion callback."""
        for activity in activities:
            self.kernel_collector.collect(activity)
            self.memcpy_collector.collect(activity)
    
    def __call__(self, *args, **kwargs):
        """
        Run the profiled function and collect all metrics.
        
        Args:
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Return value of the profiled function
        """
        # Enable CUPTI activities
        cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
        cupti.activity_enable(cupti.ActivityKind.MEMCPY)
        
        # Enable stacktrace collection
        sys.settrace(self.stacktrace_collector.trace_calls)
        self._tracing_enabled = True
        
        try:
            # Run the function
            ret = self.fn(*args, **kwargs)
        finally:
            # Disable stacktrace
            sys.settrace(None)
            self._tracing_enabled = False
            
            # Flush and disable CUPTI
            cupti.activity_flush_all(1)
            cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
            cupti.activity_disable(cupti.ActivityKind.MEMCPY)
        
        return ret
    
    def visualize_stacktrace_and_kernels(self, figsize: Tuple[int, int] = (16, 10)):
        """
        Visualize stacktrace (dynamic depth, 1-10) and kernels (top row) as horizontal bar chart.
        
        Layout:
        - Top row: GPU Kernels
        - Bottom rows: Python stack depths (dynamically determined from data, max 10)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get all events with timestamps
        call_ranges = self.stacktrace_collector.get_call_ranges()
        kernels = self.kernel_collector.kernels
        
        if not call_ranges and not kernels:
            print("No data to visualize")
            return fig
        
        # Dynamically determine the maximum depth from call_ranges (capped at MAX_STACKTRACE_DEPTH)
        depth = 0
        if call_ranges:
            depth = max(d for _, _, _, d in call_ranges)
            depth = min(depth, self.MAX_STACKTRACE_DEPTH)
        
        max_depth = depth  # Use for the rest of the function
        
        # Find time range
        all_times = []
        if call_ranges:
            all_times.extend([start for start, _, _, _ in call_ranges])
            all_times.extend([end for _, end, _, _ in call_ranges])
        if kernels:
            all_times.extend([k['start'] for k in kernels])
            all_times.extend([k['end'] for k in kernels])
        
        if not all_times:
            print("No timestamp data available")
            return fig
        
        t0 = min(all_times)
        t_max = max(all_times)
        total_duration = t_max - t0
        
        # Convert to appropriate time units for display
        times_scaled, time_units = scale_time_units([total_duration])
        # scale_factor converts nanoseconds to the display unit
        scale_factor = times_scaled[0] / total_duration if total_duration > 0 else 1
        
        # Row positions: dynamically calculated based on max_depth
        row_height = 0.8
        row_spacing = 0.2
        
        # Generate dynamic color map for depths (1-10)
        # Use a colormap that provides distinct colors
        if max_depth > 0:
            depth_colormap = plt.cm.Set3 if max_depth <= 12 else plt.cm.tab20
            depth_colors = {depth: depth_colormap(depth / max(max_depth, 1)) 
                          for depth in range(max_depth + 1)}
        else:
            depth_colors = {}
        
        print(f"Plotting {len(call_ranges)} call ranges across depths 0-{max_depth}")
        for idx, (start_ns, end_ns, name, depth) in enumerate(call_ranges):
            # Only plot depths within our range (0 to max_depth, capped at 10)
            if depth > max_depth or depth < 0:
                continue
            
            duration_ns = end_ns - start_ns
            start_scaled = (start_ns - t0) * scale_factor
            duration_scaled = duration_ns * scale_factor
            
            y_pos = depth * (row_height + row_spacing)
            
            print(f"  Range {idx}: {name} at depth {depth}, duration: {duration_ns/1e6:.3f} ms, "
                  f"start_scaled: {start_scaled:.6f}, duration_scaled: {duration_scaled:.6f}")
            
            # Draw bar - ensure minimum width for visibility
            if duration_scaled < 1e-10:  # If duration is essentially zero, make it visible
                duration_scaled = max(duration_scaled, (t_max - t0) * scale_factor * 0.001)  # 0.1% of total
            
            rect = Rectangle(
                (start_scaled, y_pos),
                duration_scaled,
                row_height,
                facecolor=depth_colors.get(depth, 'lightgray'),
                edgecolor='black',
                linewidth=0.5,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add label if bar is wide enough
            if duration_scaled > 0.01 * (t_max - t0) * scale_factor:  # Only label if > 1% of total
                ax.text(
                    start_scaled + duration_scaled / 2,
                    y_pos + row_height / 2,
                    name[:30] + '...' if len(name) > 30 else name,
                    ha='center',
                    va='center',
                    fontsize=7,
                    rotation=0
                )
        
        # Plot GPU kernels (top row, above all Python depths)
        kernel_y = (max_depth + 1) * (row_height + row_spacing)
        kernel_colors = plt.cm.tab20(np.linspace(0, 1, min(len(kernels), 20)))
        
        for i, kernel in enumerate(kernels):
            start_scaled = (kernel['start'] - t0) * scale_factor
            duration_scaled = kernel['duration_ns'] * scale_factor
            
            color = kernel_colors[i % len(kernel_colors)]
            
            # Draw bar
            rect = Rectangle(
                (start_scaled, kernel_y),
                duration_scaled,
                row_height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label if bar is wide enough
            if duration_scaled > 0.01 * (t_max - t0) * scale_factor:
                kernel_name_short = kernel['name'][:25] + '...' if len(kernel['name']) > 25 else kernel['name']
                ax.text(
                    start_scaled + duration_scaled / 2,
                    kernel_y + row_height / 2,
                    kernel_name_short,
                    ha='center',
                    va='center',
                    fontsize=7,
                    rotation=0
                )
        
        # Set up axes - dynamically calculate max_y based on max_depth
        max_y = (max_depth + 1) * (row_height + row_spacing) + row_height
        ax.set_xlim(0, (t_max - t0) * scale_factor)
        ax.set_ylim(-0.1, max_y + 0.1)
        ax.set_xlabel(f"Time ({time_units})")
        ax.set_ylabel("Stack Depth / Kernels")
        
        # Set y-ticks dynamically based on max_depth
        y_ticks = []
        y_labels = []
        for depth in range(max_depth + 1):
            y_pos = depth * (row_height + row_spacing) + row_height / 2
            y_ticks.append(y_pos)
            y_labels.append(f"Python Depth {depth}")
        # Add kernel row
        y_ticks.append(kernel_y + row_height / 2)
        y_labels.append("GPU Kernels")
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_title(f"Stacktrace (Python, depths 0-{max_depth}) and Kernel Timeline")
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def visualize_all(self, show: bool = False):
        """
        Generate all 5 plots.
        
        Args:
            show: If True, call plt.show() for each plot
        """
        from plots import (
            plot_bottleneck_analysis,
            plot_layer_runtime_stats,
            plot_throughput_analysis,
            plot_timeline_memcpy_and_passes
        )
        
        # Note: plot_layer_runtime_stats and plot_throughput_analysis need a profile object
        # which we don't have. We'll skip those for now or create a minimal adapter.
        
        print("Generating Plot 1: Bottleneck Analysis...")
        fig1 = plot_bottleneck_analysis(self.kernel_collector)
        if show:
            plt.show()
        else:
            plt.close(fig1)
        
        print("Generating Plot 2: Stacktrace with Kernels...")
        fig2 = self.visualize_stacktrace_and_kernels()
        if show:
            plt.show()
        else:
            plt.close(fig2)
        
        print("Generating Plot 3: Layer Runtime Statistics...")
        # Create a minimal profile object - PlotProfiler doesn't collect layer timing data
        # So this plot will be empty, but we'll still generate it
        class MinimalProfile:
            def __init__(self):
                self.layers = {}  # Empty - no layer data collected
                self.timeline = []
                self.base_time = 0
        
        profile = MinimalProfile()
        fig3 = plot_layer_runtime_stats(profile)
        if fig3 is not None:
            if show:
                plt.show()
            else:
                plt.close(fig3)
        
        print("Generating Plot 4: Throughput Analysis...")
        fig4 = plot_throughput_analysis(profile)
        if fig4 is not None:
            if show:
                plt.show()
            else:
                plt.close(fig4)
        
        print("Generating Plot 5: Timeline of Memory Copies...")
        # Create a minimal profile-like object for the timeline plot
        t0 = 0
        if self.kernel_collector.kernels:
            t0 = min([k['start'] for k in self.kernel_collector.kernels])
        elif self.memcpy_collector.memcpy_info:
            t0 = min([t[0] for t in self.memcpy_collector.memcpy_info])
        
        class MinimalProfile:
            def __init__(self, base_time):
                self.timeline = []
                self.layers = {}
                self.base_time = base_time
        
        profile = MinimalProfile(t0)
        fig5 = plot_timeline_memcpy_and_passes(
            self.memcpy_collector.memcpy_info,
            profile,
            MEMCPY_KIND_STR
        )
        if show:
            plt.show()
        else:
            plt.close(fig5)
        
        print("All plots generated!")
    
    def get_data(self):
        """Return collected data for manual inspection."""
        return {
            'kernels': self.kernel_collector.kernels,
            'kernel_stats': self.kernel_collector.get_kernel_statistics(),
            'memcpy': self.memcpy_collector.memcpy_info,
            'stacktrace': self.stacktrace_collector.get_call_ranges(),
            'call_events': self.stacktrace_collector.call_events
        }
