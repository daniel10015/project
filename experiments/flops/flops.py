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
        # On "call" event, a new function is pushed onto the actual call stack.
        # We simulate a "unique ID" for the function call instance by storing 
        # a unique object, like the frame object itself, or simply a representation
        # of the function being called.
        func_name = frame.f_code.co_name
        # Store a representation of the call instance on our custom stack
        call_info = {"name": func_name, "frame": frame, "id": id(frame)}
        call_stack.append(call_info)
        print(f"-> [PUSH]: (ID: {call_info['id']}) '{func_name}' called {frame.f_code.co_filename}:{frame.f_lineno}")
        
    elif event == "return":
        # On "return" event, the current function is popped off the actual stack.
        # We pop from our custom stack to get the unique instance info.
        func_name = frame.f_code.co_name
        if call_stack:
            popped_call = call_stack.pop()
            print(f"<- [POP]: (ID: {popped_call['id']}) '{func_name}'")
        else:
            # This case might happen for tracing at global scope or built-ins
            print(f"<- [POP]: empty stack '{func_name}'")
            
    # The trace function must return itself or another trace function for the new scope
    return trace_calls
import sys
sys.settrace(trace_calls)

model = MyModel()
model.to(device)
X = torch.randn(28, 28).flatten().to(device) # Batch size 1, 1 channels, 28x28 image
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

model(X)

#Step 3: Flushing and Disabling CUPTI Activity
#cupti.activity_flush_all(1)
#cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
#print('finished collecting metrics\n----------')