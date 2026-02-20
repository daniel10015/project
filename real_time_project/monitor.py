import pynvml
import time
import json
from datetime import datetime

class GPUMonitor:
    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self.handle = None
        self.is_running = False

    def start(self):
        """Initialize NVML and connect to the target GPU."""
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            print(f"GPU connected: {gpu_name} (Index: {self.gpu_index})")
            self.is_running = True
        except pynvml.NVMLError as e:
            print(f"GPU initialization failed: {e}")
            print("Tip: If there is no NVIDIA GPU in this environment, consider implementing a 'mock data mode'.")
            self.is_running = False

    def stop(self):
        """Shut down NVML and release resources."""
        if self.is_running:
            pynvml.nvmlShutdown()
            print("Monitoring stopped")

    def get_status_stream(self, interval=0.5):
        """
        Generator that yields GPU status as a JSON string every `interval` seconds.
        Designed to be consumed by a web server (e.g., SSE endpoint) or a CLI loop.
        """
        if not self.is_running:
            return

        while True:
            try:
                # 1. Query raw GPU metrics from NVML
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                # 2. Convert and format the raw values
                current_time = time.time()
                gpu_util = util.gpu                    # GPU utilization (%)
                mem_used = mem.used / (1024 ** 2)      # Convert bytes -> MB
                mem_total = mem.total / (1024 ** 2)    # Convert bytes -> MB

                # 3. Bottleneck detection
                # If GPU utilization drops below 20%, flag the sample as a bottleneck
                # candidate. These timestamps can later be queried in SQLite to
                # pinpoint where the data pipeline is starving the GPU.
                event_type = "NORMAL"
                if gpu_util < 20:
                    event_type = "BOTTLENECK_CANDIDATE"

                # 4. Pack everything into a JSON-serializable dict and yield it
                data = {
                    "timestamp": current_time,
                    "datetime": datetime.fromtimestamp(current_time).strftime('%H:%M:%S'),
                    "gpu_util": gpu_util,
                    "mem_used": round(mem_used, 1),
                    "mem_total": round(mem_total, 1),
                    "event": event_type
                }

                yield json.dumps(data)

                time.sleep(interval)

            except pynvml.NVMLError as e:
                print(f"GPU read error: {e}")
                break


# --- Entry point: only runs when this file is executed directly ---
if __name__ == "__main__":
    monitor = GPUMonitor(gpu_index=0)
    monitor.start()

    print("Real-time monitoring started (press Ctrl+C to stop)")
    try:
        for data_json in monitor.get_status_stream(interval=1.0):
            data = json.loads(data_json)

            status = "NORMAL" if data['event'] == "NORMAL" else "WARNING"
            print(
                f"[{data['datetime']}] "
                f"GPU: {data['gpu_util']}%\t| "
                f"MEM: {data['mem_used']}MB / {data['mem_total']}MB\t| "
                f"Status: {data['event']}"
            )

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        monitor.stop()