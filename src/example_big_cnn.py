import torch
from torch import nn
from profiler.profiler import profiler
from profiler import benchmark
import time

# -----------------------------------------------------------------
# 1. Make the model much heavier (BigCNN)
# -----------------------------------------------------------------
class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Greatly increase computation, similar to VGG.
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 32x32 -> 16x16
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 16x16 -> 8x8
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1) # 8x8 -> 1x1
        )
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -----------------------------------------------------------------
# 2. 'Single Stream' (Baseline) function for comparison
# -----------------------------------------------------------------
def do_something_single_stream_baseline():
    """
    Process a total of 4096 batches in a single stream (default stream).
    This is our comparison baseline.
    """
    # Include model creation inside the function for fair benchmarking
    model = BigCNN().to("cuda")
    
    # 4096 = 2048 (S1) + 2048 (S2)
    # Same total workload as multi-stream
    total_batch_size = 4096 
    
    x_cpu = torch.randn(total_batch_size, 3, 32, 32)
    
    # Sequential execution on the default (synchronous) stream
    # (HtoD Copy) -> (Compute) -> (DtoH Copy)
    x_gpu = x_cpu.to("cuda")
    output_gpu = model(x_gpu)
    output_cpu = output_gpu.to("cpu")
    
    # CPU waits for all operations to finish (required)
    torch.cuda.synchronize()
    # print(f"Baseline Result: {output_cpu.device}, Shape: {output_cpu.shape}")
    
    torch.cuda.empty_cache()


# -----------------------------------------------------------------
# 3. 'Multi Stream' function (asynchronous execution)
# -----------------------------------------------------------------
def do_something_multi_stream():
    """
    Process a total of 4096 batches asynchronously in 2 streams of 2048 each.
    """
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    model = BigCNN().to("cuda") # Use BigCNN

    batch_size_per_stream = 2048

    # Use Pinned Memory for asynchronous copy (required)
    x_cpu1 = torch.randn(batch_size_per_stream, 3, 32, 32).pin_memory()
    x_cpu2 = torch.randn(batch_size_per_stream, 3, 32, 32).pin_memory()

    outputs_cpu = [] # List to hold results

    # --- Schedule Stream 1 tasks ---
    with torch.cuda.stream(s1):
        x_gpu1 = x_cpu1.to("cuda", non_blocking=True) # (S1) HtoD
        output_gpu1 = model(x_gpu1) # (S1) Compute
        output_cpu1 = output_gpu1.to("cpu", non_blocking=True) # (S1) DtoH
        outputs_cpu.append(output_cpu1)

    # --- Schedule Stream 2 tasks ---
    # (While S1 is executing, CPU schedules S2 tasks)
    with torch.cuda.stream(s2):
        x_gpu2 = x_cpu2.to("cuda", non_blocking=True) # (S2) HtoD
        output_gpu2 = model(x_gpu2) # (S2) Compute
        output_cpu2 = output_gpu2.to("cpu", non_blocking=True) # (S2) DtoH
        outputs_cpu.append(output_cpu2)

    # 7. CPU waits for all GPU tasks (s1, s2) to complete
    torch.cuda.synchronize()

    # 8. Check results
    # print(f"Result 1 device: {outputs_cpu[0].device}, Shape: {outputs_cpu[0].shape}")
    # print(f"Result 2 device: {outputs_cpu[1].device}, Shape: {outputs_cpu[1].shape}")
    
    torch.cuda.empty_cache()

# -----------------------------------------------------------------
# 4. Main execution and benchmark
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    # --- Warmup ---
    # Run a sufficiently large task beforehand to "warm up" the GPU.
    print("Warmup... (running single stream to setup CUDA context)")
    do_something_single_stream_baseline() 
    print("Warmup... (running multi stream)")
    do_something_multi_stream()
    print("Warmup done.\n")

    # --- Start Benchmark ---
    print("Benchmarking Single Stream (Baseline)...")
    start_time = time.perf_counter_ns()
    baseline_time = benchmark.benchmark_ns(do_something_single_stream_baseline)
    end_time = time.perf_counter_ns()
    print(f"(Benchmark function took {(end_time - start_time) / 1e9:.4f}s)")


    print("\nBenchmarking Multi Stream...")
    start_time = time.perf_counter_ns()
    multi_stream_time = benchmark.benchmark_ns(do_something_multi_stream)
    end_time = time.perf_counter_ns()
    print(f"(Benchmark function took {(end_time - start_time) / 1e9:.4f}s)")


    # --- Benchmark Results ---
    print(f'\n--- Benchmark Results ---')
    print(f'Single Stream (Baseline) Time: {baseline_time} ns ({(baseline_time / 1e9):.4f} s)')
    print(f'Multi Stream Time            : {multi_stream_time} ns ({(multi_stream_time / 1e9):.4f} s)')

    if multi_stream_time > 0 and baseline_time > 0:
        speedup = (baseline_time / multi_stream_time)
        print(f'Speedup: {speedup:.2f}x')
    else:
        print("Speedup: N/A (time was zero)")
    print(f'----------------------\n')


    # --- Profiling (Multi Stream) ---
    print("Profiling Multi Stream (KERNEL, MEMCPY)...")
    profile = profiler(do_something_multi_stream, ('KERNEL', 'MEMCPY'))
    profile()
    profile.visualize('KERNEL')
    profile_info = profile.spill()

    # Print KERNEL logs
    kernel_logs = profile_info.get('KERNEL', [])
    print(f"\n=== Activity: KERNEL ({len(kernel_logs)} kernels) ===")
    if kernel_logs:
        for kernel_info in kernel_logs[:5]: # Print only 5 (too many)
            print(kernel_info + "\n")
        if len(kernel_logs) > 5:
            print(f"... and {len(kernel_logs) - 5} more kernels ...\n")
    else:
        print("No KERNEL activity captured.\n")

    # Print MEMCPY logs
    memcpy_logs = profile_info.get('MEMCPY', [])
    print(f"\n=== Activity: MEMCPY ({len(memcpy_logs)} copies) ===")
    if memcpy_logs:
        for memcpy_info in memcpy_logs:
            print(memcpy_info + "\n")
    else:
        print("No MEMCPY activity captured.\n")
import torch
from torch import nn
from profiler.profiler import profiler
from profiler import benchmark
import time

# -----------------------------------------------------------------
# 1. Make the model much heavier (BigCNN)
# -----------------------------------------------------------------
class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Greatly increase computation, similar to VGG.
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 32x32 -> 16x16
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 16x16 -> 8x8
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1) # 8x8 -> 1x1
        )
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -----------------------------------------------------------------
# 2. 'Single Stream' (Baseline) function for comparison
# -----------------------------------------------------------------
def do_something_single_stream_baseline():
    """
    Process a total of 4096 batches in a single stream (default stream).
    This is our comparison baseline.
    """
    # Include model creation inside the function for fair benchmarking
    model = BigCNN().to("cuda")
    
    # 4096 = 2048 (S1) + 2048 (S2)
    # Same total workload as multi-stream
    total_batch_size = 4096 
    
    x_cpu = torch.randn(total_batch_size, 3, 32, 32)
    
    # Sequential execution on the default (synchronous) stream
    # (HtoD Copy) -> (Compute) -> (DtoH Copy)
    x_gpu = x_cpu.to("cuda")
    output_gpu = model(x_gpu)
    output_cpu = output_gpu.to("cpu")
    
    # CPU waits for all operations to finish (required)
    torch.cuda.synchronize()
    # print(f"Baseline Result: {output_cpu.device}, Shape: {output_cpu.shape}")
    
    torch.cuda.empty_cache()


# -----------------------------------------------------------------
# 3. 'Multi Stream' function (asynchronous execution)
# -----------------------------------------------------------------
def do_something_multi_stream():
    """
    Process a total of 4096 batches asynchronously in 2 streams of 2048 each.
    """
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    model = BigCNN().to("cuda") # Use BigCNN

    batch_size_per_stream = 2048

    # Use Pinned Memory for asynchronous copy (required)
    x_cpu1 = torch.randn(batch_size_per_stream, 3, 32, 32).pin_memory()
    x_cpu2 = torch.randn(batch_size_per_stream, 3, 32, 32).pin_memory()

    outputs_cpu = [] # List to hold results

    # --- Schedule Stream 1 tasks ---
    with torch.cuda.stream(s1):
        x_gpu1 = x_cpu1.to("cuda", non_blocking=True) # (S1) HtoD
        output_gpu1 = model(x_gpu1) # (S1) Compute
        output_cpu1 = output_gpu1.to("cpu", non_blocking=True) # (S1) DtoH
        outputs_cpu.append(output_cpu1)

    # --- Schedule Stream 2 tasks ---
    # (While S1 is executing, CPU schedules S2 tasks)
    with torch.cuda.stream(s2):
        x_gpu2 = x_cpu2.to("cuda", non_blocking=True) # (S2) HtoD
        output_gpu2 = model(x_gpu2) # (S2) Compute
        output_cpu2 = output_gpu2.to("cpu", non_blocking=True) # (S2) DtoH
        outputs_cpu.append(output_cpu2)

    # 7. CPU waits for all GPU tasks (s1, s2) to complete
    torch.cuda.synchronize()

    # 8. Check results
    # print(f"Result 1 device: {outputs_cpu[0].device}, Shape: {outputs_cpu[0].shape}")
    # print(f"Result 2 device: {outputs_cpu[1].device}, Shape: {outputs_cpu[1].shape}")
    
    torch.cuda.empty_cache()

# -----------------------------------------------------------------
# 4. Main execution and benchmark
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    # --- Warmup ---
    # Run a sufficiently large task beforehand to "warm up" the GPU.
    print("Warmup... (running single stream to setup CUDA context)")
    do_something_single_stream_baseline() 
    print("Warmup... (running multi stream)")
    do_something_multi_stream()
    print("Warmup done.\n")

    # --- Start Benchmark ---
    print("Benchmarking Single Stream (Baseline)...")
    start_time = time.perf_counter_ns()
    baseline_time = benchmark.benchmark_ns(do_something_single_stream_baseline)
    end_time = time.perf_counter_ns()
    print(f"(Benchmark function took {(end_time - start_time) / 1e9:.4f}s)")


    print("\nBenchmarking Multi Stream...")
    start_time = time.perf_counter_ns()
    multi_stream_time = benchmark.benchmark_ns(do_something_multi_stream)
    end_time = time.perf_counter_ns()
    print(f"(Benchmark function took {(end_time - start_time) / 1e9:.4f}s)")


    # --- Benchmark Results ---
    print(f'\n--- Benchmark Results ---')
    print(f'Single Stream (Baseline) Time: {baseline_time} ns ({(baseline_time / 1e9):.4f} s)')
    print(f'Multi Stream Time            : {multi_stream_time} ns ({(multi_stream_time / 1e9):.4f} s)')

    if multi_stream_time > 0 and baseline_time > 0:
        speedup = (baseline_time / multi_stream_time)
        print(f'Speedup: {speedup:.2f}x')
    else:
        print("Speedup: N/A (time was zero)")
    print(f'----------------------\n')


    # --- Profiling (Multi Stream) ---
    print("Profiling Multi Stream (KERNEL, MEMCPY)...")
    profile = profiler(do_something_multi_stream, ('KERNEL', 'MEMCPY'))
    profile()
    profile.visualize('KERNEL')
    profile_info = profile.spill()

    # Print KERNEL logs
    kernel_logs = profile_info.get('KERNEL', [])
    print(f"\n=== Activity: KERNEL ({len(kernel_logs)} kernels) ===")
    if kernel_logs:
        for kernel_info in kernel_logs[:5]: # Print only 5 (too many)
            print(kernel_info + "\n")
        if len(kernel_logs) > 5:
            print(f"... and {len(kernel_logs) - 5} more kernels ...\n")
    else:
        print("No KERNEL activity captured.\n")

    # Print MEMCPY logs
    memcpy_logs = profile_info.get('MEMCPY', [])
    print(f"\n=== Activity: MEMCPY ({len(memcpy_logs)} copies) ===")
    if memcpy_logs:
        for memcpy_info in memcpy_logs:
            print(memcpy_info + "\n")
    else:
        print("No MEMCPY activity captured.\n")
