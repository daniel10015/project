Here is a clean, concise, and professional English version of your `README.md`.

```markdown
# Real-Time GPU Monitor

A lightweight dashboard built with **FastAPI**, **WebSockets**, and **Chart.js** to stream live GPU utilization and memory usage directly to your browser. It automatically flags bottleneck candidates when GPU usage drops below 20%.

## 📂 Project Structure

* `server.py`: FastAPI WebSocket server and endpoint router.
* `monitor.py`: GPU metrics collector using the `pynvml` library.
* `index.html`: Browser dashboard featuring real-time Chart.js graphs.
* `gpu_test.py`: *(Optional)* Script to generate artificial GPU load for testing.

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pynvml

```

### 2. Start the Server

Run the FastAPI server using Uvicorn.

```bash
uvicorn server:app --host 0.0.0.0 --port 8000

```

### 3. Open the Dashboard

Open your web browser and navigate to:

```text
http://<YOUR_INSTANCE_IP>:8000

```


