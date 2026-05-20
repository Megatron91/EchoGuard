import os
import json
import asyncio
import datetime
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from audio_engine import YAMNetClassifier

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI-Powered Personal Safety Audio Detection Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Instantiate our AI classifier
classifier = YAMNetClassifier()

# Persistent Threat History Logger
THREAT_HISTORY_FILE = os.path.join(BASE_DIR, "assets", "threat_history.json")

def log_threat_event(event_name: str, confidence: float):
    history = []
    if os.path.exists(THREAT_HISTORY_FILE):
        try:
            with open(THREAT_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            pass
            
    new_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_name,
        "confidence": round(confidence, 2)
    }
    history.append(new_record)
    
    try:
        with open(THREAT_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error writing threat history: {e}")
        
    return new_record

# Active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# Background task state for local python mic
local_mic_task = None
local_mic_active = False

@app.get("/")
async def get_dashboard():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        return HTMLResponse(open(index_path).read())
    return HTMLResponse("<h2>index.html not found. Please compile frontend.</h2>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("WebSocket client connected")
    
    # Send handshake confirmation
    await manager.send_json({
        "type": "status",
        "status": "connected",
        "message": "Real-time pipeline connected to FastAPI backend."
    }, websocket)

    # Send threat history upon connection
    if os.path.exists(THREAT_HISTORY_FILE):
        try:
            with open(THREAT_HISTORY_FILE, "r") as f:
                history = json.load(f)
                await manager.send_json({
                    "type": "threat_history",
                    "history": history
                }, websocket)
        except Exception as e:
            print(f"Error loading threat history: {e}")

    try:
        while True:
            # We can receive either binary data (audio chunk from browser) or text data (commands)
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio chunk from browser mic
                audio_bytes = message["bytes"]
                # Convert bytes to float32 numpy array
                # The browser sends 16000Hz mono float32 PCM chunks
                waveform = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Process audio chunk through YAMNet
                result = classifier.classify(waveform)
                
                # Send back classification results
                await websocket.send_json({
                    "type": "inference_result",
                    "source": "browser_mic",
                    **result
                })
                
                # If threat detected, send simulated workflow status and log threat
                if result.get("threat_detected"):
                    threat = result["primary_threat"]
                    log_record = log_threat_event(threat["display_name"], threat["score"])
                    await websocket.send_json({
                        "type": "threat_logged",
                        "record": log_record
                    })
                    await simulate_emergency_workflow(threat["display_name"], threat["score"], websocket)
                    
            elif "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")
                
                if msg_type == "command":
                    command = data.get("command")
                    if command == "start_local_mic":
                        # Start background thread for local mic
                        success = await start_local_mic_stream(websocket)
                        if not success:
                            await websocket.send_json({
                                "type": "status",
                                "status": "error",
                                "message": "Failed to access local microphone (PortAudio/sounddevice). Try browser mic instead."
                            })
                    elif command == "stop_local_mic":
                        await stop_local_mic_stream()
                        await websocket.send_json({
                            "type": "status",
                            "status": "idle",
                            "message": "Local microphone stream stopped."
                        })
                        
                elif msg_type == "wake_phrase":
                    phrase = data.get("phrase", "HELP")
                    print(f"Wake phrase triggered in frontend: {phrase}")
                    log_record = log_threat_event(f"WAKE: {phrase}", 1.0)
                    await websocket.send_json({
                        "type": "threat_logged",
                        "record": log_record
                    })
                    await simulate_emergency_workflow(f"WAKE PHRASE: '{phrase}'", 1.0, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket client disconnected")
        await stop_local_mic_stream()
    except Exception as e:
        print(f"Error in websocket loop: {e}")
        manager.disconnect(websocket)
        await stop_local_mic_stream()

async def simulate_emergency_workflow(event_name: str, confidence: float, websocket: WebSocket):
    """
    Simulates the emergency dispatching workflows as required by the technical brief.
    """
    logs = [
        f"ALERT: Distress Event '{event_name.upper()}' detected (Confidence: {int(confidence*100)}%)",
        "[Simulated] Publishing alert message to MQTT topic 'safety/alerts'...",
        "[Simulated] Dispatching panic SMS and Email to emergency contacts...",
        "[Simulated] Triggering active security webhook escalation...",
        "[Simulated] Alarm workflow complete. Active monitoring continues."
    ]
    
    for log in logs:
        await websocket.send_json({
            "type": "simulated_log",
            "message": log
        })
        await asyncio.sleep(0.3)

# Sounddevice audio streaming for local Python microphone support
async def start_local_mic_stream(websocket: WebSocket) -> bool:
    global local_mic_active, local_mic_task
    if local_mic_active:
        return True
        
    try:
        import sounddevice as sd
        # Test if we can open an input stream (verifies device accessibility)
        sd.check_input_settings()
    except Exception as e:
        print(f"Microphone check failed: {e}")
        return False
        
    local_mic_active = True
    local_mic_task = asyncio.create_task(local_mic_loop(websocket))
    return True

async def stop_local_mic_stream():
    global local_mic_active, local_mic_task
    local_mic_active = False
    if local_mic_task:
        local_mic_task.cancel()
        local_mic_task = None
    print("Local microphone streaming stopped.")

async def local_mic_loop(websocket: WebSocket):
    global local_mic_active
    import sounddevice as sd
    
    # Target settings
    sample_rate = 16000
    # Process audio in 1-second chunks (16000 samples)
    chunk_size = 16000 
    
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    
    def callback(indata, frames, time, status):
        loop.call_soon_threadsafe(queue.put_nowait, indata.copy())
        
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        callback=callback,
        blocksize=chunk_size,
        dtype='float32'
    )
    
    print("Starting sounddevice input stream...")
    try:
        with stream:
            while local_mic_active:
                audio_chunk = await queue.get()
                # Flatten the mono stream
                waveform = audio_chunk.flatten()
                
                # Classify the chunk
                result = classifier.classify(waveform)
                
                # Push results to frontend
                await websocket.send_json({
                    "type": "inference_result",
                    "source": "local_mic",
                    **result
                })
                
                if result.get("threat_detected"):
                    threat = result["primary_threat"]
                    log_record = log_threat_event(threat["display_name"], threat["score"])
                    await websocket.send_json({
                        "type": "threat_logged",
                        "record": log_record
                    })
                    await simulate_emergency_workflow(threat["display_name"], threat["score"], websocket)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error in local mic stream: {e}")
        await websocket.send_json({
            "type": "status",
            "status": "error",
            "message": f"Local mic error: {str(e)}"
        })
    finally:
        local_mic_active = False

@app.get("/api/samples")
async def list_sample_files():
    """
    Returns a list of pre-configured sample files that the user can run in the browser dashboard.
    """
    samples_dir = os.path.join(BASE_DIR, "assets", "samples")
    if not os.path.exists(samples_dir):
        return []
    return [f for f in os.listdir(samples_dir) if f.endswith(".wav") or f.endswith(".mp3")]

@app.get("/api/samples/{filename}")
async def get_sample_file(filename: str):
    samples_dir = os.path.join(BASE_DIR, "assets", "samples")
    file_path = os.path.join(samples_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("File not found", status_code=404)
