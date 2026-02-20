import sys

# example on tracing calls
def trace_calls(frame, event, arg):
    if event == 'call':
        co = frame.f_code
        func_name = co.co_name
        filename = co.co_filename
        lineno = frame.f_lineno
        print(f"Calling function: {func_name} in {filename}:{lineno}")
    return trace_calls

sys.settrace(trace_calls)

def my_function_a():
    print("Inside function A")
    my_function_b()

def my_function_b():
    print("Inside function B")

my_function_a()