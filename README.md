# EchoGuard

EchoGuard is a smart environmental monitoring and threat detection system powered by AI. It continuously analyzes live audio streams to detect emergency events such as screaming, glass shattering, crying, and explosions.

## Architecture
The system is divided into a decoupled frontend and backend for optimal performance:

- **Frontend (Appwrite)**: A lightweight, responsive dashboard that captures live microphone audio, visualizes threat metrics, and simulates audio events.
- **Backend (Render + Python)**: A real-time `FastAPI` server utilizing WebSockets and a `YAMNet` machine learning model to analyze audio chunks 5 times per second and return instantaneous threat classifications.

## Tech Stack
- **AI/ML**: `onnxruntime`, `numpy`
- **Backend**: `FastAPI`, `uvicorn`, `python-multipart`
- **Frontend**: Vanilla HTML/JS/CSS
- **Hosting**: Appwrite (Static Site), Render (Python Server)

## Usage
Simply open the web dashboard, allow microphone permissions, and click "Start Monitor". The system will immediately begin streaming audio to the backend AI engine for real-time classification. You can also trigger built-in simulation sounds to test the system's responsiveness.
