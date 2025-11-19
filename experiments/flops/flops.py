import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from cupti import cupti

"""
Install fvcore with 'pip install fvcore'

fvcore provides flexible floating point operation analysis
"""

device = torch.device("cuda")

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.input_size = 28*28
    self.output_size = 10

    self.layers = nn.Sequential(
      nn.Linear(self.input_size, 512),
      nn.ReLU(),
      nn.Linear(512,256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 10),
      nn.Softmax(dim=0)
    )

  def forward(self, X):
    return self.layers(X)
  
call_stack = []

def trace_calls(frame, event, arg):
    if event == "call":
        func_name = frame.f_code.co_name
        class_name = None

        # Check if this is a method call and extract class name if possible
        locals_ = frame.f_locals
        if 'self' in locals_:
            # instance method
            class_name = locals_['self'].__class__.__name__
        elif 'cls' in locals_ and isinstance(locals_['cls'], type):
            # class method
            class_name = locals_['cls'].__name__

        # Combine class and function names
        full_name = f"{class_name}.{func_name}" if class_name else func_name

        call_info = {"name": full_name, "frame": frame, "id": id(frame)}
        if not call_stack:
           call_stack = []
        call_stack.append(call_info)

        #print(f"-> [PUSH]: (ID: {call_info['id']}) '{full_name}' called {frame.f_code.co_filename}:{frame.f_lineno}")
        
    elif event == "return":
        func_name = frame.f_code.co_name
        if call_stack:
            popped_call = call_stack.pop()
            #print(f"<- [POP]: (ID: {popped_call['id']}) '{popped_call['name']}'")
        #else:
            #print(f"<- [POP]: empty stack '{func_name}'")
            
    return trace_calls
import sys
from time import time
sys.settrace(trace_calls)

model = MyModel()
model.to(device)
batch_size = 32
X = torch.randn(batch_size, 28, 28, device=device)  # <batch_size> images of 28x28
X = X.view(batch_size, -1)  # flatten image
flops = FlopCountAnalysis(model, X) # does a static analysis of FLOPs
print('it appears since these are just approximiations, layers like relu are negligible so they are 0')

def func_buffer_requested():
  buffer_size = 8 * 1024 * 1024  # 8MB buffer
  max_num_records = 0
  return buffer_size, max_num_records

def func_buffer_completed(activities: list):
  flop_op = flops.by_module()
  print(flop_op)
  for i, activity in enumerate(activities):
    duration_ns = activity.end - activity.start
    layer_name = f'layers.{i}'
    flop_quantity = flop_op.get(layer_name)
    print(layer_name)
    print(flop_quantity)
    if flop_quantity > 0 and duration_ns > 0:
      print(f"kernel name = {activity.name}")
      print(f"kernel duration (ns) = {duration_ns}")
      print(f'TFLOPs: {1e-12*(flop_quantity/(duration_ns*1e-9))}')
      print('')


#Step 1: Register CUPTI callbacks
#cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)

#Step 2: Enable CUPTI Activity Collection
#cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)

start = time()
for i in range(128):
  model(X)
print(time()-start)
#Step 3: Flushing and Disabling CUPTI Activity
#cupti.activity_flush_all(1)
#cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
#print('finished collecting metrics\n----------')