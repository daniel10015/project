import torch
import time

print(" Starting GPU stress test! (Check your dashboard)")

# Create two massive matrices to consume a decent amount of GPU memory
size = 15000 
try:
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')

    # Continuously perform matrix multiplication for 3 minutes (180 seconds)
    end_time = time.time() + 180
    count = 0
    while time.time() < end_time:
        c = torch.matmul(a, b)
        count += 1
        if count % 100 == 0:
            print(f"In progress... Completed {count} matrix multiplications")

except Exception as e:
    print(f"Error occurred: {e}")

print("Stress test completed. The GPU is now resting.")