// AEGIS SAFETY DASHBOARD - CLIENT APPLICATION LOGIC
let socket = null;
let audioContext = null;
let microphoneStream = null;
let scriptProcessor = null;

// Audio buffer for sending 1.5-second windows with a 0.75-second hop
let audioBufferQueue = [];
const TARGET_SAMPLE_RATE = 16000;
const WINDOW_SIZE_SECONDS = 1.5;
const HOP_SIZE_SECONDS = 0.75;
const WINDOW_SAMPLES = TARGET_SAMPLE_RATE * WINDOW_SIZE_SECONDS; // 24000
const HOP_SAMPLES = TARGET_SAMPLE_RATE * HOP_SIZE_SECONDS; // 12000

// Oscilloscope visualizer variables
let canvas = document.getElementById("waveform-canvas");
let canvasCtx = canvas.getContext("2d");
let animationFrameId = null;
let visualBuffer = new Float32Array(512);

// Speech Recognition for Wake Word
let speechRecognition = null;
let isMonitoring = false;
let threatHistory = [];

// DOM Elements
const btnBrowserMic = document.getElementById("btn-browser-mic");
const btnLocalMic = document.getElementById("btn-local-mic");
const btnStop = document.getElementById("btn-stop");
const statusOverlay = document.getElementById("status-overlay");
const statusMsg = document.getElementById("status-msg");
const recordingBadge = document.getElementById("recording-badge");
const logStream = document.getElementById("log-stream");
const sysStatusText = document.getElementById("sys-status-text");
const sysStatusDot = document.getElementById("sys-status-dot");
const threatBanner = document.getElementById("threat-alert-banner");
const alertTitle = document.getElementById("alert-title");
const alertSubtitle = document.getElementById("alert-subtitle");
const alarmOverlay = document.getElementById("alarm-overlay");
const btnPlaySample = document.getElementById("btn-play-sample");
const samplesSelect = document.getElementById("samples-select");
const audioPlayer = document.getElementById("demo-audio-player");
const simulatedPhrases = document.querySelectorAll(".simulated-phrase");
const historyTableBody = document.getElementById("history-table-body");
const btnClearHistory = document.getElementById("btn-clear-history");

// Custom Upload elements
const btnUploadTrigger = document.getElementById("btn-upload-trigger");
const audioUpload = document.getElementById("audio-upload");
const uploadFilename = document.getElementById("upload-filename");
const btnAnalyzeUpload = document.getElementById("btn-analyze-upload");

// Metrics Meters
const meters = {
    scream: {
        fill: document.querySelector("#meter-scream .meter-bar-fill"),
        val: document.querySelector("#meter-scream .meter-val")
    },
    glass: {
        fill: document.querySelector("#meter-glass .meter-bar-fill"),
        val: document.querySelector("#meter-glass .meter-val")
    },
    gunshot: {
        fill: document.querySelector("#meter-gunshot .meter-bar-fill"),
        val: document.querySelector("#meter-gunshot .meter-val")
    },
    crying: {
        fill: document.querySelector("#meter-crying .meter-bar-fill"),
        val: document.querySelector("#meter-crying .meter-val")
    }
};

// Initialize Canvas size on load & resize
function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();
drawSilence();

// Initialize WebSocket Connection
function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;
    
    appendLog("System", "Connecting to Aegis AI server...", "system-msg");
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        sysStatusText.innerText = "System Connected";
        sysStatusDot.className = "status-dot green";
        appendLog("System", "WebSocket pipeline connected successfully.", "system-msg");
    };
    
    socket.onclose = () => {
        sysStatusText.innerText = "Disconnected";
        sysStatusDot.className = "status-dot yellow";
        appendLog("System", "Server connection lost. Retrying in 5 seconds...", "system-msg");
        setTimeout(initWebSocket, 5000);
        stopMonitoring();
    };
    
    socket.onerror = (err) => {
        console.error("WebSocket Error:", err);
    };
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };
}

// Start WebSocket on Load
initWebSocket();

// Handle messages received from FastAPI backend
function handleServerMessage(data) {
    if (data.type === "status") {
        appendLog("System", data.message, data.status === "error" ? "threat-msg" : "system-msg");
        if (data.status === "connected") {
            btnStop.disabled = true;
        }
    } 
    
    else if (data.type === "inference_result") {
        updateMeters(data);
        
        if (data.threat_detected && data.primary_threat) {
            triggerAlertUI(data.primary_threat);
        } else {
            // Gradually clear warning state if no threat is detected in active audio
            clearAlertUI();
        }
    } 
    
    else if (data.type === "simulated_log") {
        appendLog("Workflow", data.message, "simulated-msg");
    }
    
    else if (data.type === "threat_history") {
        threatHistory = data.history || [];
        renderThreatHistory();
    }
    
    else if (data.type === "threat_logged") {
        if (data.record) {
            threatHistory.push(data.record);
            renderThreatHistory();
        }
    }
}

// Log utility
function appendLog(sender, message, cssClass = "") {
    const timestamp = new Date().toLocaleTimeString();
    const line = document.createElement("div");
    line.className = `log-line ${cssClass}`;
    line.innerHTML = `<strong>[${timestamp}] ${sender}:</strong> ${message}`;
    logStream.appendChild(line);
    logStream.scrollTop = logStream.scrollHeight;
}

// Update UI meters based on model scores
function updateMeters(result) {
    if (!result.active) {
        // Reset meters to zero if silent
        Object.keys(meters).forEach(key => {
            meters[key].fill.style.width = "0%";
            meters[key].val.innerText = "0%";
        });
        return;
    }
    
    let maxScores = { scream: 0, glass: 0, gunshot: 0, crying: 0 };
    
    // Find the highest score for each class across all prediction frames in this chunk
    result.detections.forEach(frame => {
        frame.predictions.forEach(pred => {
            const cat = pred.threat_category;
            const score = pred.score;
            
            if (cat === "Scream" && score > maxScores.scream) maxScores.scream = score;
            if (cat === "Glass Breaking" && score > maxScores.glass) maxScores.glass = score;
            if (cat === "Gunshot" && score > maxScores.gunshot) maxScores.gunshot = score;
            if (cat === "Distress/Crying" && score > maxScores.crying) maxScores.crying = score;
        });
    });
    
    // Update DOM meters
    Object.keys(meters).forEach(key => {
        const scorePct = Math.round(maxScores[key] * 100);
        meters[key].fill.style.width = `${scorePct}%`;
        meters[key].val.innerText = `${scorePct}%`;
    });
}

// Trigger Red visual alerts in Dashboard
let alertTimeout = null;
function triggerAlertUI(threat) {
    alarmOverlay.classList.add("alarm-active");
    threatBanner.classList.add("banner-active");
    sysStatusDot.className = "status-dot red";
    sysStatusText.innerText = "TARGET THREAT IN PROGRESS";
    
    alertTitle.innerText = `DISTRESS EVENT DETECTED: ${threat.category.toUpperCase()}`;
    alertSubtitle.innerText = `Type: ${threat.display_name.toUpperCase()} | Confidence: ${Math.round(threat.score * 100)}%`;
    
    appendLog("Inference", `Threat event '${threat.display_name}' classified with ${Math.round(threat.score * 100)}% confidence.`, "threat-msg");
    
    // Reset alert clear timer
    if (alertTimeout) clearTimeout(alertTimeout);
    alertTimeout = setTimeout(clearAlertUI, 3000);
}

function clearAlertUI() {
    alarmOverlay.classList.remove("alarm-active");
    threatBanner.classList.remove("banner-active");
    if (isMonitoring) {
        sysStatusDot.className = "status-dot green";
        sysStatusText.innerText = "Monitoring Environment";
    } else {
        sysStatusDot.className = "status-dot green";
        sysStatusText.innerText = "System Ready";
    }
}

// Speech Recognition for local Wake Words
function startSpeechRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        appendLog("Speech", "Speech Recognition API not supported in this browser. Wake words disabled.", "log-line");
        return;
    }
    
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    speechRecognition = new SpeechRec();
    speechRecognition.continuous = true;
    speechRecognition.interimResults = false;
    speechRecognition.lang = "en-US";
    
    speechRecognition.onstart = () => {
        appendLog("Speech", "Vocal Wake Phrase engine active. Listening for 'HELP ME', 'EMERGENCY'...", "system-msg");
    };
    
    speechRecognition.onresult = (event) => {
        const lastIndex = event.resultIndex;
        const transcript = event.results[lastIndex][0].transcript.toUpperCase();
        appendLog("Vocal Engine", `Heard: "${transcript}"`, "system-msg");
        
        // Expanded list of safety threat keywords
        const threatKeywords = ["HELP", "EMERGENCY", "RED", "PANIC", "INTRUDER", "POLICE", "FIRE", "ATTACK", "DANGER", "SOS", "STOP"];
        if (threatKeywords.some(keyword => transcript.includes(keyword))) {
            socket.send(JSON.stringify({
                type: "wake_phrase",
                phrase: transcript.trim()
            }));
        }
    };
    
    speechRecognition.onerror = (e) => {
        console.error("Speech Error:", e);
    };
    
    speechRecognition.onend = () => {
        // Automatically restart speech recognition while monitoring is active
        if (isMonitoring) {
            try { speechRecognition.start(); } catch(e) {}
        }
    };
    
    speechRecognition.start();
}

// Browser Microphone Monitoring Loop
async function startBrowserMic() {
    if (isMonitoring) return;
    
    try {
        // Initialize Web Audio Context at 16kHz
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: TARGET_SAMPLE_RATE
        });
        
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: false
            }
        });
        
        microphoneStream = audioContext.createMediaStreamSource(stream);
        // scriptProcessor nodes take bufferSize, inputChannels, outputChannels
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        microphoneStream.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        
        audioBufferQueue = [];
        
        scriptProcessor.onaudioprocess = (event) => {
            const inputBuffer = event.inputBuffer.getChannelData(0);
            
            // Push frames into Visual buffer for drawOscilloscope
            visualBuffer.set(inputBuffer.subarray(0, 512));
            
            // Queue audio samples for sliding window
            for (let i = 0; i < inputBuffer.length; i++) {
                audioBufferQueue.push(inputBuffer[i]);
            }
            
            // Check if queue has enough samples to send a window (1.5 seconds)
            while (audioBufferQueue.length >= WINDOW_SAMPLES) {
                // Slice window samples
                const windowData = audioBufferQueue.slice(0, WINDOW_SAMPLES);
                const float32Array = new Float32Array(windowData);
                
                // Send raw binary float32 array over socket
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(float32Array.buffer);
                }
                
                // Hop overlap: discard the first hop samples, slide window forward
                audioBufferQueue = audioBufferQueue.slice(HOP_SAMPLES);
            }
        };
        
        isMonitoring = true;
        btnBrowserMic.disabled = true;
        btnLocalMic.disabled = true;
        btnStop.disabled = false;
        statusOverlay.classList.add("hidden");
        recordingBadge.className = "badge active";
        recordingBadge.innerText = "Browser Mic Active";
        sysStatusText.innerText = "Monitoring Environment";
        
        startSpeechRecognition();
        drawOscilloscope();
        appendLog("Mic Monitor", "Browser microphone recording stream started.", "system-msg");
        
    } catch (err) {
        console.error("Microphone Access Error:", err);
        appendLog("Error", `Could not access microphone: ${err.message}`, "threat-msg");
        stopMonitoring();
    }
}

// Start Python sounddevice local microphone capture
function startPythonLocalMic() {
    if (isMonitoring) return;
    
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: "command",
            command: "start_local_mic"
        }));
        
        isMonitoring = true;
        btnBrowserMic.disabled = true;
        btnLocalMic.disabled = true;
        btnStop.disabled = false;
        statusOverlay.classList.add("hidden");
        recordingBadge.className = "badge active";
        recordingBadge.innerText = "Python Local Mic Active";
        sysStatusText.innerText = "Monitoring Environment";
        
        // Render artificial wave since python is capturing locally
        drawSyntheticOscilloscope();
        appendLog("Mic Monitor", "Request sent to python backend to open local microphone...", "system-msg");
    }
}

// Stop Monitoring
function stopMonitoring() {
    isMonitoring = false;
    btnBrowserMic.disabled = false;
    btnLocalMic.disabled = false;
    btnStop.disabled = true;
    statusOverlay.classList.remove("hidden");
    recordingBadge.className = "badge";
    recordingBadge.innerText = "Idle";
    sysStatusText.innerText = "System Ready";
    
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (microphoneStream) {
        microphoneStream.disconnect();
        microphoneStream = null;
    }
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    if (speechRecognition) {
        speechRecognition.stop();
        speechRecognition = null;
    }
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    // Command backend to stop local mic loop
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: "command",
            command: "stop_local_mic"
        }));
    }
    
    drawSilence();
    clearAlertUI();
}

// Visualizer: Draw oscilloscope wave from browser microphone
function drawOscilloscope() {
    if (!isMonitoring) return;
    
    animationFrameId = requestAnimationFrame(drawOscilloscope);
    
    canvasCtx.fillStyle = "rgba(10, 12, 22, 0.4)"; // overlay transparency for trailing effect
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    canvasCtx.lineWidth = 3;
    canvasCtx.strokeStyle = "#3b82f6";
    
    // Glowing gradient line
    const gradient = canvasCtx.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, '#60a5fa');
    gradient.addColorStop(0.5, '#3b82f6');
    gradient.addColorStop(1, '#60a5fa');
    canvasCtx.strokeStyle = gradient;
    
    canvasCtx.beginPath();
    
    const sliceWidth = canvas.width / visualBuffer.length;
    let x = 0;
    
    for (let i = 0; i < visualBuffer.length; i++) {
        // Scale wave ampl
        const v = visualBuffer[i] * 1.5;
        const y = (v + 1) * canvas.height / 2;
        
        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }
        
        x += sliceWidth;
    }
    
    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}

// Draw static wave line when idle
function drawSilence() {
    canvasCtx.fillStyle = "#0a0c16";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "rgba(59, 130, 246, 0.3)";
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, canvas.height / 2);
    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}

// Generate animated artificial wave for local mic display
let syntheticPhase = 0;
function drawSyntheticOscilloscope() {
    if (!isMonitoring) return;
    
    animationFrameId = requestAnimationFrame(drawSyntheticOscilloscope);
    
    canvasCtx.fillStyle = "rgba(10, 12, 22, 0.4)";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    canvasCtx.lineWidth = 3;
    canvasCtx.strokeStyle = "#10b981"; // green for python backend streaming
    
    canvasCtx.beginPath();
    
    const sliceWidth = canvas.width / 100;
    let x = 0;
    
    for (let i = 0; i < 100; i++) {
        // Generate random sine wave shapes to indicate mic activity
        const freq = 0.05;
        const v = 0.15 * Math.sin(x * freq + syntheticPhase) + 0.05 * Math.sin(x * 0.12 + syntheticPhase * 1.7);
        const y = (v + 1) * canvas.height / 2;
        
        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }
        
        x += sliceWidth;
    }
    
    syntheticPhase += 0.08;
    canvasCtx.stroke();
}

// Simulated Vocal phrases
simulatedPhrases.forEach(btn => {
    btn.addEventListener("click", () => {
        const phrase = btn.getAttribute("data-phrase");
        appendLog("Vocal Sim", `Simulated voice command: "${phrase}"`, "system-msg");
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: "wake_phrase",
                phrase: phrase
            }));
        }
    });
});

// Trigger Audio Samples simulation
btnPlaySample.addEventListener("click", async () => {
    const filename = samplesSelect.value;
    if (!filename) return;
    
    appendLog("Simulation", `Triggering audio play sample: ${filename}`, "system-msg");
    
    // Play audio locally through browser speakers so user can hear it
    const url = `/api/samples/${filename}`;
    audioPlayer.src = url;
    audioPlayer.play();
    
    // Load file as arraybuffer and send to backend
    try {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        
        // Decode audio data using browser's AudioContext to get raw PCM values
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        
        // Extract channel data
        let pcmData = audioBuffer.getChannelData(0); // float32 array
        
        // If sample rate doesn't match YAMNet's 16kHz, resample
        if (audioBuffer.sampleRate !== TARGET_SAMPLE_RATE) {
            appendLog("DSP", `Resampling audio sample from ${audioBuffer.sampleRate}Hz to ${TARGET_SAMPLE_RATE}Hz...`, "system-msg");
            pcmData = resampleBuffer(pcmData, audioBuffer.sampleRate, TARGET_SAMPLE_RATE);
        }
        
        // Send PCM binary buffer to backend over WebSocket
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(pcmData.buffer);
        }
        
    } catch (e) {
        console.error("Simulation error:", e);
        appendLog("Error", `Simulation pipeline error: ${e.message}`, "threat-msg");
    }
});

// Resample audio buffer in JavaScript (linear interpolation)
function resampleBuffer(buffer, fromSampleRate, toSampleRate) {
    const ratio = fromSampleRate / toSampleRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
        const index = i * ratio;
        const indexBefore = Math.floor(index);
        const indexAfter = Math.ceil(index);
        const weight = index - indexBefore;
        
        if (indexAfter >= buffer.length) {
            result[i] = buffer[indexBefore];
        } else {
            result[i] = buffer[indexBefore] * (1 - weight) + buffer[indexAfter] * weight;
        }
    }
    return result;
}

// Button Events
btnBrowserMic.addEventListener("click", startBrowserMic);
btnLocalMic.addEventListener("click", startPythonLocalMic);
btnStop.addEventListener("click", stopMonitoring);

// Render threat history items in table UI
function renderThreatHistory() {
    if (!historyTableBody) return;
    
    if (threatHistory.length === 0) {
        historyTableBody.innerHTML = `
            <tr>
                <td colspan="3" style="text-align: center; color: #64748b; padding: 1rem;">No threats recorded yet.</td>
            </tr>
        `;
        return;
    }
    
    // Sort descending by timestamp (newest first)
    const sorted = [...threatHistory].sort((a, b) => new Date(b.timestamp.replace(/-/g, '/')) - new Date(a.timestamp.replace(/-/g, '/')));
    sorted.reverse();
    
    historyTableBody.innerHTML = sorted.map(row => {
        const isWake = row.event.startsWith("WAKE");
        const tagClass = isWake ? "history-threat-tag wake-tag" : "history-threat-tag";
        return `
            <tr>
                <td style="color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">${row.timestamp}</td>
                <td><span class="${tagClass}">${row.event}</span></td>
                <td style="font-weight: 600; color: ${isWake ? '#fbbf24' : '#f87171'}">${Math.round(row.confidence * 100)}%</td>
            </tr>
        `;
    }).join("");
}

// Clear History Button Click Listener
if (btnClearHistory) {
    btnClearHistory.addEventListener("click", () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: "command",
                command: "clear_threat_history"
            }));
        }
        threatHistory = [];
        renderThreatHistory();
        appendLog("System", "Clear request sent. Threat logs cleared.", "system-msg");
    });
}

// File Upload Event Listeners
let uploadedFile = null;

if (btnUploadTrigger && audioUpload) {
    btnUploadTrigger.addEventListener("click", () => {
        audioUpload.click();
    });

    audioUpload.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            uploadedFile = file;
            uploadFilename.innerText = file.name;
            btnAnalyzeUpload.disabled = false;
            appendLog("Upload", `Selected file: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`, "system-msg");
        } else {
            uploadedFile = null;
            uploadFilename.innerText = "No file selected";
            btnAnalyzeUpload.disabled = true;
        }
    });
}

if (btnAnalyzeUpload) {
    btnAnalyzeUpload.addEventListener("click", async () => {
        if (!uploadedFile) return;
        
        appendLog("Simulation", `Triggering analysis for uploaded file: ${uploadedFile.name}`, "system-msg");
        
        // Play local playback so user can hear it
        const fileUrl = URL.createObjectURL(uploadedFile);
        audioPlayer.src = fileUrl;
        audioPlayer.play();
        
        try {
            // Load file as arraybuffer and send to backend
            const response = await fetch(fileUrl);
            const arrayBuffer = await response.arrayBuffer();
            
            // Decode audio data using browser's AudioContext to get raw PCM values
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
            
            // Extract channel data
            let pcmData = audioBuffer.getChannelData(0); // float32 array
            
            // If sample rate doesn't match YAMNet's 16kHz, resample
            if (audioBuffer.sampleRate !== TARGET_SAMPLE_RATE) {
                appendLog("DSP", `Resampling audio sample from ${audioBuffer.sampleRate}Hz to ${TARGET_SAMPLE_RATE}Hz...`, "system-msg");
                pcmData = resampleBuffer(pcmData, audioBuffer.sampleRate, TARGET_SAMPLE_RATE);
            }
            
            // Send PCM binary buffer to backend over WebSocket
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(pcmData.buffer);
            }
            
        } catch (e) {
            console.error("Upload analysis error:", e);
            appendLog("Error", `Upload pipeline error: ${e.message}`, "threat-msg");
        }
    });
}
