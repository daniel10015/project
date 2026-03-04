from torch import nn
import torch
from profiler.profiler import profiler


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

        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1) # Flatten from dimension 1 to the end

        return self.fc(x)

def do_something_multi_stream():
    
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    # --- [Modification 1] ---
    # Create 2 model instances.
    # (model.to("cuda") runs on the default stream (Stream 7))
    model1 = BigCNN().to("cuda")
    model2 = BigCNN().to("cuda")
    # --- [Modification 1 End] ---

    batch_size_per_stream = 2048 

    x_cpu1 = torch.randn(batch_size_per_stream, 3, 32, 32).pin_memory()
    x_cpu2 = torch.randn(batch_size_per_stream, 3, 32, 32).pin_memory()

    outputs_cpu = [] 
    output_cpu1 = None
    output_cpu2 = None

    # --- Schedule Stream 1 tasks ---
    with torch.cuda.stream(s1):
        # s1 uses model1.
        # s1 waits for the model1, model2 copy on the default stream (s7) to finish.
        x_gpu1 = x_cpu1.to("cuda", non_blocking=True) 
        output_gpu1 = model1(x_gpu1) # [Modification 2]
        output_cpu1 = output_gpu1.to("cpu", non_blocking=True) 

    # --- Schedule Stream 2 tasks ---
    with torch.cuda.stream(s2):
        # s2 uses model2.
        # s2 also waits for the default stream (s7) to finish.
        x_gpu2 = x_cpu2.to("cuda", non_blocking=True) 
        output_gpu2 = model2(x_gpu2) # [Modification 3]
        output_cpu2 = output_gpu2.to("cpu", non_blocking=True) 


    # 7. CPU waits for all GPU tasks (s1, s2) to complete
    # (At this point, s1 and s2 will run in parallel after s7 finishes)
    torch.cuda.synchronize()

    # 8. Aggregate results
    outputs_cpu.append(output_cpu1)
    outputs_cpu.append(output_cpu2)
    
    torch.cuda.empty_cache()
    
def do_profile():
  profile = profiler(do_something_multi_stream, ('KERNEL','MEMCPY' ))
  profile()


# def do_torchprofile():
#   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
#       with torch.profiler.record_function("do_something_multi_stream"):
#           do_something_multi_stream()

from profiler import benchmark


print("Warming up GPU...")
do_something_multi_stream() 
torch.cuda.synchronize() # Wait for the GPU warmup to finish
print("Warmup complete. Starting benchmarks...")


# warmup
regular_time = benchmark.benchmark_ns(do_something_multi_stream)
profile_time = benchmark.benchmark_ns(do_profile)
# torch_time = benchmark.benchmark_ns(do_torchprofile)


# print(f'regular time took: {regular_time/1e9}s\n'
#       f'profile time took: {profile_time/1e9}s\n'
#       f'torch time took: {torch_time/1e9}s')
    
print(f'regular time took: {regular_time/1e9}s\n'
      f'profile time took: {profile_time/1e9}s\n')
      
profile = profiler(do_something_multi_stream, ('KERNEL','MEMCPY'))

profile()
profile.visualize('KERNEL')

profile_info = profile.spill()



# print(f"\n=== Activity: KERNEL ({len(profile.profile_out['KERNEL'])} kernels) ===")

# for kernel_info in profile.profile_out["KERNEL"]:
#     print(kernel_info)
#     print() 

# for metric_type, metric_out in profile_info.items():
#   info = f'{metric_type} => {metric_out}'
#   print(info)
