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
        func_name = frame.f_code.co_name
        class_name = None

        # Try to get 'self' (for instance methods) or 'cls' (for class methods)
        locals_ = frame.f_locals
        if 'self' in locals_:
            class_name = locals_['self'].__class__.__name__
        elif 'cls' in locals_ and isinstance(locals_['cls'], type):
            class_name = locals_['cls'].__name__

        # Combine class name and function name if available
        full_name = f"{class_name}.{func_name}" if class_name else func_name

        # You can also show the filename
        filename = frame.f_code.co_filename
        call_info = {"name": full_name, "frame": frame, "id": id(frame)}
        call_stack.append(call_info)

        print(f"-> [PUSH]: (ID: {call_info['id']}) '{full_name}' from {filename}")

    elif event == "return":
        func_name = frame.f_code.co_name
        if call_stack:
            popped_call = call_stack.pop()
            print(f"<- [POP]: (ID: {popped_call['id']}) '{popped_call['name']}'")
        else:
            print(f"<- [POP]: empty stack '{func_name}'")
            
    # The trace function must return itself or another trace function for the new scope
    return trace_calls

sys.settrace(trace_calls)

# Run the profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    profile_memory=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18_forward_trace'),
    #with_stack=True # Enable stack tracing
) as prof:
    for i in range(5):
        if i >= 2: # Active steps
            with torch.profiler.record_function("model_forward"):
                model(inputs)
        prof.step()

