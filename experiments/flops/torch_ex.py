import torch
import torchvision.models as models
import torch.profiler

import sys

# Prepare model and input
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
inputs = torch.randn(1, 3, 224, 224)
# Warm up CUDA to ensure accurate benchmarking
if torch.cuda.is_available():
    model = model.cuda()
    inputs = inputs.cuda()
    for _ in range(10):
        model(inputs)

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
        print(f"-> [PUSH]: (ID: {call_info['id']}) '{func_name}' from [TODO fillin filename]")
        
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

sys.settrace(trace_calls)

# Run the profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18_forward_trace'),
    with_stack=True # Enable stack tracing
) as prof:
    for i in range(5):
        if i >= 2: # Active steps
            with torch.profiler.record_function("model_forward"):
                model(inputs)
        prof.step()

