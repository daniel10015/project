import torch

a = torch.randn(1024, 1024, device="cuda")

from profiler import profiler

profiler = profiler.profiler(fn = lambda x: x.to("cuda").to("cpu") , metrics=('MEMCPY',) )
profiler(a)

profile_info = profiler.spill()
for metric_type, metric_out in profile_info.items():
  info = f'{metric_type} => {metric_out}'
  print(info)