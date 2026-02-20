from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from monitor import GPUMonitor
import asyncio
import os

app = FastAPI()

# Security Settings (CORS): Allows access from React or other ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify domains in production for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GPU Monitor instance for GPU index 0
# (Global management is recommended for production services)
monitor_system = GPUMonitor(gpu_index=0)

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket Endpoint:
    Sends GPU metrics to the browser every 0.5 seconds upon connection.
    """
    await websocket.accept()
    print(" Web client (Dashboard) connected!")
    
    # Start Monitoring
    monitor_system.start()

    try:
        # Stream data from the generator to the web client
        for data_json in monitor_system.get_status_stream(interval=0.5):
            await websocket.send_text(data_json)
            
            # Critical: Allows other tasks to run in the async loop
            await asyncio.sleep(0) 

    except WebSocketDisconnect:
        print(" Web client disconnected")
    except Exception as e:
        print(f" Server Error: {e}")
    finally:
        # Stop monitoring when connection is lost to save resources
        monitor_system.stop()
        print(" Monitoring suspended")

@app.get("/")
def get_html():
    file_path = "index.html"
    if not os.path.exists(file_path):
        return {"error": "index.html file not found."}
    
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())