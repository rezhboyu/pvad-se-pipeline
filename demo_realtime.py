#!/usr/bin/env python3
"""
pVAD-SE Real-time Demo
=======================
Microphone enrollment + real-time pVAD + GTCRN denoise + waveform display
"""

import io
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from fastapi import FastAPI, WebSocket, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

PROJECT = Path(__file__).resolve().parent
MODELS = PROJECT / "models"

import sys
sys.path.insert(0, str(PROJECT))
from utils.audio import SAMPLE_RATE
from utils.speaker_encoder import SpeakerEncoder, cosine_similarity
from utils.gtcrn_denoiser import GTCRNDenoiser, GTCRN_NFFT, GTCRN_HOP
from utils.gating import SoftGate

# ── Load models ──
print("Loading models...")
encoder = SpeakerEncoder(str(MODELS / "campplus" / "campplus.onnx"))
print(f"  [OK] CAM++ (dim={encoder.embed_dim})")
denoiser = GTCRNDenoiser()
print("  [OK] GTCRN denoiser")

enrollment_embedding = None

app = FastAPI()


# ═══════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pVAD-SE Real-time Demo</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e1a;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif}
.container{max-width:1100px;margin:0 auto;padding:15px}

h1{text-align:center;font-size:1.6em;margin:10px 0;
   background:linear-gradient(135deg,#3b82f6,#8b5cf6);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{text-align:center;color:#64748b;margin-bottom:15px;font-size:0.9em}

.panel{background:#1e293b;border-radius:10px;padding:15px;margin:10px 0;border:1px solid #334155}

button{padding:8px 20px;border:none;border-radius:8px;
       font-size:0.95em;cursor:pointer;font-weight:600;transition:all 0.15s}
button:disabled{opacity:0.4;cursor:not-allowed}
.btn-rec{background:#ef4444;color:white;font-size:1.1em;padding:12px 28px;min-width:140px}
.btn-rec:hover:not(:disabled){background:#dc2626}
.btn-rec.recording{animation:recPulse 1s infinite}
@keyframes recPulse{0%,100%{box-shadow:0 0 0 0 #ef444488}50%{box-shadow:0 0 0 12px #ef444400}}
.btn-start{background:#22c55e;color:white;font-size:1.1em;padding:12px 28px}
.btn-start:hover:not(:disabled){background:#16a34a}
.btn-stop{background:#ef4444;color:white;font-size:1.1em;padding:12px 28px}
.btn-stop:hover:not(:disabled){background:#dc2626}

.status{display:inline-block;padding:4px 12px;border-radius:12px;font-size:0.8em;font-weight:600}
.status-idle{background:#334155;color:#94a3b8}
.status-recording{background:#7f1d1d;color:#fca5a5;animation:pulse 1s infinite}
.status-enrolled{background:#1e3a5f;color:#60a5fa}
.status-live{background:#14532d;color:#4ade80;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.6}}

/* Enrollment section */
.enroll-section{text-align:center;padding:20px}
.prompt-text{font-size:1.15em;color:#f8fafc;background:#0f172a;border-radius:8px;
             padding:15px 20px;margin:12px auto;max-width:700px;line-height:1.6;
             border:1px solid #334155;position:relative}
.prompt-text::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
                     background:#3b82f6;border-radius:3px 0 0 3px}
.prompt-hint{color:#64748b;font-size:0.85em;margin:8px 0}
.enroll-timer{font-size:2.5em;font-weight:700;font-family:monospace;color:#3b82f6;margin:8px 0}
.enroll-wave{margin:10px auto;max-width:600px}
.enroll-wave canvas{width:100%;height:60px;display:block;border-radius:6px;background:#0f172a}

.quality-bar{height:8px;border-radius:4px;background:#1e293b;margin:8px auto;max-width:400px;overflow:hidden}
.quality-fill{height:100%;border-radius:4px;transition:width 0.3s,background 0.3s}

/* Control row */
.ctrl-row{display:flex;gap:10px;align-items:center;justify-content:center;flex-wrap:wrap}
.ctrl-row label{color:#94a3b8;font-size:0.85em}
.ctrl-row select,.ctrl-row input[type="number"]{
    padding:6px 10px;background:#0f172a;border:1px solid #475569;
    border-radius:6px;color:#e2e8f0;width:80px}

/* Waveform area */
.waveform-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}
@media(max-width:768px){.waveform-grid{grid-template-columns:1fr}}
.wave-panel{background:#0f172a;border-radius:8px;padding:10px;position:relative}
.wave-panel h3{font-size:0.9em;color:#94a3b8;margin-bottom:5px}
.wave-panel canvas{width:100%;height:120px;display:block;border-radius:4px}

.sim-panel{background:#0f172a;border-radius:8px;padding:10px}
.sim-panel h3{font-size:0.9em;color:#94a3b8;margin-bottom:5px}
.sim-panel canvas{width:100%;height:100px;display:block;border-radius:4px}

.stats-row{display:flex;gap:15px;justify-content:center;margin:10px 0;flex-wrap:wrap}
.stat{text-align:center;min-width:100px}
.stat-val{font-size:1.8em;font-weight:700;font-family:monospace}
.stat-label{font-size:0.75em;color:#64748b}
.sim-high{color:#4ade80}
.sim-low{color:#f87171}
.sim-mid{color:#fbbf24}

.hidden{display:none!important}
</style>
</head>
<body>
<div class="container">
    <h1>pVAD-SE Real-time Demo</h1>
    <p class="subtitle">CAM++ pVAD + GTCRN Denoise | Microphone Real-time Processing</p>

    <!-- ══════ Step 1: Enrollment ══════ -->
    <div class="panel" id="enrollPanel">
        <div class="enroll-section">
            <h2 style="color:#93c5fd;margin-bottom:8px">Step 1: Voice Enrollment</h2>
            <p class="prompt-hint">Please read the following text aloud (5-10 seconds):</p>
            <div class="prompt-text" id="promptText">
                Today is a beautiful day. I am registering my voice for the personal voice activity detection system.
                This system will learn to recognize my voice and distinguish it from others.
            </div>
            <p class="prompt-hint" id="enrollHint">Click the button below and start reading</p>

            <div class="enroll-wave">
                <canvas id="enrollWave"></canvas>
            </div>

            <div class="enroll-timer" id="enrollTimer">0.0s</div>

            <div class="quality-bar">
                <div class="quality-fill" id="qualityFill" style="width:0%;background:#475569"></div>
            </div>
            <p class="prompt-hint" id="qualityText">-</p>

            <div style="margin-top:12px">
                <button class="btn-rec" id="btnRec" onclick="toggleEnrollRec()">Start Recording</button>
            </div>
            <span class="status status-idle" id="enrollStatus">Not Enrolled</span>
        </div>
    </div>

    <!-- ══════ Step 2: Real-time ══════ -->
    <div class="panel hidden" id="streamPanel">
        <div class="ctrl-row" style="margin-bottom:10px">
            <span class="status status-enrolled" id="liveStatus">Enrolled</span>
            <div style="flex-grow:1"></div>
            <label>Threshold:</label>
            <input type="number" id="threshold" value="0.25" step="0.01" min="0" max="1" />
            <label>Window:</label>
            <select id="windowSec">
                <option value="0.5">0.5s</option>
                <option value="0.75">0.75s</option>
                <option value="1.0" selected>1.0s</option>
            </select>
            <button style="background:#475569;color:white;padding:6px 14px;font-size:0.85em"
                    onclick="reEnroll()">Re-enroll</button>
        </div>

        <div style="text-align:center;margin:10px 0">
            <button class="btn-start" id="btnStart" onclick="startStream()">Start Listening</button>
            <button class="btn-stop hidden" id="btnStop" onclick="stopStream()">Stop</button>
        </div>

        <div class="stats-row">
            <div class="stat"><div class="stat-val" id="simVal">-</div><div class="stat-label">Similarity</div></div>
            <div class="stat"><div class="stat-val" id="targetVal" style="color:#64748b">-</div><div class="stat-label">Target?</div></div>
            <div class="stat"><div class="stat-val" id="rmsIn">-</div><div class="stat-label">Input RMS</div></div>
            <div class="stat"><div class="stat-val" id="rmsOut">-</div><div class="stat-label">Output RMS</div></div>
            <div class="stat"><div class="stat-val" id="latVal">-</div><div class="stat-label">Latency (ms)</div></div>
        </div>

        <div class="waveform-grid">
            <div class="wave-panel">
                <h3>Input (Raw Microphone)</h3>
                <canvas id="waveIn"></canvas>
            </div>
            <div class="wave-panel">
                <h3>Output (Denoised + pVAD Gated)</h3>
                <canvas id="waveOut"></canvas>
            </div>
        </div>

        <div class="panel" style="margin-top:5px">
            <div class="sim-panel">
                <h3>Similarity Curve (real-time)</h3>
                <canvas id="simChart"></canvas>
            </div>
        </div>
    </div>
</div>

<script>
// ═══════════════════════════════════════════════════
// Enrollment via Microphone
// ═══════════════════════════════════════════════════
let enrollStream = null;
let enrollCtx = null;
let enrollNode = null;
let enrollChunks = [];
let enrollRecording = false;
let enrollStartTime = 0;
let enrollTimerInterval = null;
let enrolled = false;

const MIN_ENROLL_SEC = 3;
const MAX_ENROLL_SEC = 15;

function toggleEnrollRec() {
    if (enrollRecording) {
        stopEnrollRec();
    } else {
        startEnrollRec();
    }
}

async function startEnrollRec() {
    try {
        enrollStream = await navigator.mediaDevices.getUserMedia({
            audio: { sampleRate: 16000, channelCount: 1, echoCancellation: false, noiseSuppression: false }
        });
    } catch(e) {
        alert('Microphone access denied: ' + e.message);
        return;
    }

    enrollCtx = new AudioContext({ sampleRate: 16000 });
    const source = enrollCtx.createMediaStreamSource(enrollStream);
    enrollNode = enrollCtx.createScriptProcessor(4096, 1, 1);
    enrollChunks = [];
    enrollRecording = true;
    enrollStartTime = Date.now();

    const btn = document.getElementById('btnRec');
    btn.textContent = 'Stop Recording';
    btn.classList.add('recording');
    document.getElementById('enrollHint').textContent = 'Recording... Please read the text above';
    document.getElementById('qualityText').textContent = 'Recording...';
    document.getElementById('qualityFill').style.width = '0%';
    document.getElementById('qualityFill').style.background = '#fbbf24';

    enrollNode.onaudioprocess = (e) => {
        if (!enrollRecording) return;
        const data = new Float32Array(e.inputBuffer.getChannelData(0));
        enrollChunks.push(data);
        drawEnrollWave(data);

        const elapsed = (Date.now() - enrollStartTime) / 1000;
        document.getElementById('enrollTimer').textContent = elapsed.toFixed(1) + 's';

        // Quality bar
        const pct = Math.min(100, (elapsed / MIN_ENROLL_SEC) * 100);
        const fill = document.getElementById('qualityFill');
        fill.style.width = pct + '%';
        if (elapsed < MIN_ENROLL_SEC) {
            fill.style.background = '#fbbf24';
            document.getElementById('qualityText').textContent = 'Keep reading... (' + MIN_ENROLL_SEC + 's minimum)';
        } else {
            fill.style.background = '#4ade80';
            document.getElementById('qualityText').textContent = 'Good! You can stop now, or keep reading for better accuracy';
        }

        // Auto-stop at max
        if (elapsed >= MAX_ENROLL_SEC) {
            stopEnrollRec();
        }
    };

    source.connect(enrollNode);
    enrollNode.connect(enrollCtx.destination);
}

async function stopEnrollRec() {
    enrollRecording = false;
    if (enrollNode) { enrollNode.disconnect(); enrollNode = null; }
    if (enrollCtx) { enrollCtx.close(); enrollCtx = null; }
    if (enrollStream) { enrollStream.getTracks().forEach(t => t.stop()); enrollStream = null; }

    const btn = document.getElementById('btnRec');
    btn.classList.remove('recording');
    btn.disabled = true;
    btn.textContent = 'Processing...';

    const elapsed = (Date.now() - enrollStartTime) / 1000;
    if (elapsed < MIN_ENROLL_SEC) {
        document.getElementById('qualityText').textContent = 'Too short! Need at least ' + MIN_ENROLL_SEC + 's';
        document.getElementById('qualityFill').style.background = '#ef4444';
        btn.disabled = false;
        btn.textContent = 'Start Recording';
        return;
    }

    // Merge chunks and send to backend
    let totalLen = 0;
    enrollChunks.forEach(c => totalLen += c.length);
    const merged = new Float32Array(totalLen);
    let offset = 0;
    enrollChunks.forEach(c => { merged.set(c, offset); offset += c.length; });

    document.getElementById('enrollHint').textContent = 'Extracting voiceprint...';

    try {
        const res = await fetch('/api/enroll', {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream' },
            body: merged.buffer
        });
        const data = await res.json();

        if (data.success) {
            enrolled = true;
            document.getElementById('enrollStatus').className = 'status status-enrolled';
            document.getElementById('enrollStatus').textContent = 'Enrolled (dim=' + data.embed_dim + ', SNR=' + data.snr_db + 'dB)';
            document.getElementById('enrollHint').textContent = 'Enrollment successful!';
            document.getElementById('qualityText').textContent =
                'Duration: ' + data.duration.toFixed(1) + 's | SNR: ' + data.snr_db + 'dB | RMS: ' + data.rms;
            document.getElementById('qualityFill').style.background = '#4ade80';
            document.getElementById('qualityFill').style.width = '100%';

            // Show stream panel
            setTimeout(() => {
                document.getElementById('streamPanel').classList.remove('hidden');
                document.getElementById('streamPanel').scrollIntoView({ behavior: 'smooth' });
            }, 800);
        } else {
            document.getElementById('enrollHint').textContent = 'Failed: ' + data.message;
            document.getElementById('qualityFill').style.background = '#ef4444';
            btn.disabled = false;
            btn.textContent = 'Try Again';
        }
    } catch(e) {
        document.getElementById('enrollHint').textContent = 'Error: ' + e.message;
        btn.disabled = false;
        btn.textContent = 'Try Again';
    }
}

function reEnroll() {
    enrolled = false;
    document.getElementById('streamPanel').classList.add('hidden');
    document.getElementById('enrollPanel').scrollIntoView({ behavior: 'smooth' });
    const btn = document.getElementById('btnRec');
    btn.disabled = false;
    btn.textContent = 'Start Recording';
    document.getElementById('enrollStatus').className = 'status status-idle';
    document.getElementById('enrollStatus').textContent = 'Not Enrolled';
    document.getElementById('enrollTimer').textContent = '0.0s';
    document.getElementById('qualityFill').style.width = '0%';
    document.getElementById('enrollHint').textContent = 'Click the button below and start reading';
}

function drawEnrollWave(data) {
    const canvas = document.getElementById('enrollWave');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== canvas.clientWidth * dpr) {
        canvas.width = canvas.clientWidth * dpr;
        canvas.height = canvas.clientHeight * dpr;
    }
    const W = canvas.width, H = canvas.height;

    // Shift existing image left
    const shiftPx = Math.ceil(W * 0.05);
    const imgData = ctx.getImageData(shiftPx, 0, W - shiftPx, H);
    ctx.putImageData(imgData, 0, 0);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(W - shiftPx, 0, shiftPx, H);

    // Draw new chunk on right side
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const step = Math.max(1, Math.floor(data.length / shiftPx));
    for (let i = 0; i < shiftPx; i++) {
        const idx = Math.floor(i * step);
        const val = idx < data.length ? data[idx] : 0;
        const x = W - shiftPx + i;
        const y = H/2 - val * H * 0.85;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

// ═══════════════════════════════════════════════════
// Real-time Streaming
// ═══════════════════════════════════════════════════
let ws = null;
let mediaStream = null;
let audioCtx = null;
let scriptNode = null;
let isStreaming = false;

const MAX_WAVE_POINTS = 800;
let waveInBuf = new Float32Array(MAX_WAVE_POINTS);
let waveOutBuf = new Float32Array(MAX_WAVE_POINTS);
const MAX_SIM_POINTS = 200;
let simHistory = [];
let thresholdVal = 0.25;

async function startStream() {
    if (!enrolled) return alert('Please enroll first');
    thresholdVal = parseFloat(document.getElementById('threshold').value) || 0.25;
    const windowSec = document.getElementById('windowSec').value;
    waveInBuf.fill(0);
    waveOutBuf.fill(0);
    simHistory = [];

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: false }
        });
    } catch(e) { alert('Microphone denied: ' + e.message); return; }

    audioCtx = new AudioContext({ sampleRate: 16000 });
    const source = audioCtx.createMediaStreamSource(mediaStream);
    scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/stream?threshold=${thresholdVal}&window=${windowSec}`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        isStreaming = true;
        document.getElementById('btnStart').classList.add('hidden');
        document.getElementById('btnStop').classList.remove('hidden');
        document.getElementById('liveStatus').className = 'status status-live';
        document.getElementById('liveStatus').textContent = 'LIVE';
    };

    ws.onmessage = (evt) => {
        if (typeof evt.data === 'string') {
            updateStats(JSON.parse(evt.data));
        } else {
            appendWave(waveOutBuf, new Float32Array(evt.data));
        }
    };

    ws.onclose = () => stopStream();

    scriptNode.onaudioprocess = (e) => {
        if (!isStreaming || !ws || ws.readyState !== 1) return;
        const input = e.inputBuffer.getChannelData(0);
        const buf = new Float32Array(input.length);
        buf.set(input);
        ws.send(buf.buffer);
        appendWave(waveInBuf, buf);
    };

    source.connect(scriptNode);
    scriptNode.connect(audioCtx.destination);
    requestAnimationFrame(renderLoop);
}

function stopStream() {
    isStreaming = false;
    if (ws) { ws.close(); ws = null; }
    if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    document.getElementById('btnStart').classList.remove('hidden');
    document.getElementById('btnStop').classList.add('hidden');
    document.getElementById('liveStatus').className = 'status status-enrolled';
    document.getElementById('liveStatus').textContent = 'Enrolled';
}

function appendWave(buf, newData) {
    const step = Math.max(1, Math.floor(newData.length / 100));
    const points = [];
    for (let i = 0; i < newData.length; i += step) points.push(newData[i]);
    const shift = points.length;
    buf.copyWithin(0, shift);
    for (let i = 0; i < shift && i < buf.length; i++)
        buf[buf.length - shift + i] = points[i] || 0;
}

function updateStats(msg) {
    const sim = msg.similarity || 0;
    const simEl = document.getElementById('simVal');
    simEl.textContent = sim.toFixed(3);
    simEl.className = 'stat-val ' + (sim > thresholdVal ? 'sim-high' : sim > thresholdVal*0.7 ? 'sim-mid' : 'sim-low');
    const tEl = document.getElementById('targetVal');
    tEl.textContent = msg.is_target ? 'YES' : 'NO';
    tEl.style.color = msg.is_target ? '#4ade80' : '#f87171';
    document.getElementById('rmsIn').textContent = (msg.rms_in||0).toFixed(4);
    document.getElementById('rmsOut').textContent = (msg.rms_out||0).toFixed(4);
    document.getElementById('latVal').textContent = (msg.latency_ms||0).toFixed(0);
    simHistory.push({ sim, is_target: msg.is_target });
    if (simHistory.length > MAX_SIM_POINTS) simHistory.shift();
}

function renderLoop() {
    drawWaveform('waveIn', waveInBuf, '#3b82f6');
    drawWaveform('waveOut', waveOutBuf, '#22c55e');
    drawSimCurve();
    if (isStreaming) requestAnimationFrame(renderLoop);
}

function drawWaveform(id, buf, color) {
    const c = document.getElementById(id);
    const ctx = c.getContext('2d');
    const dpr = window.devicePixelRatio||1;
    if(c.width!==c.clientWidth*dpr){c.width=c.clientWidth*dpr;c.height=c.clientHeight*dpr}
    const W=c.width,H=c.height;
    ctx.fillStyle='#0f172a';ctx.fillRect(0,0,W,H);
    ctx.strokeStyle='#1e293b';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke();
    ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();
    for(let i=0;i<buf.length;i++){
        const x=(i/buf.length)*W,y=H/2-buf[i]*H*0.9;
        if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.strokeStyle=color+'33';ctx.lineWidth=4;ctx.beginPath();
    for(let i=0;i<buf.length;i++){
        const x=(i/buf.length)*W,y=H/2-buf[i]*H*0.9;
        if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }
    ctx.stroke();
}

function drawSimCurve() {
    const c=document.getElementById('simChart');
    const ctx=c.getContext('2d');
    const dpr=window.devicePixelRatio||1;
    if(c.width!==c.clientWidth*dpr){c.width=c.clientWidth*dpr;c.height=c.clientHeight*dpr}
    const W=c.width,H=c.height;
    const p={l:40,r:10,t:10,b:20};
    const pW=W-p.l-p.r,pH=H-p.t-p.b;
    ctx.fillStyle='#0f172a';ctx.fillRect(0,0,W,H);
    ctx.strokeStyle='#1e293b';ctx.lineWidth=1;ctx.fillStyle='#475569';
    ctx.font=(10*dpr)+'px monospace';
    for(let v=0;v<=1;v+=0.25){
        const y=p.t+pH*(1-v);
        ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(W-p.r,y);ctx.stroke();
        ctx.fillText(v.toFixed(2),2,y+4);
    }
    const thY=p.t+pH*(1-thresholdVal);
    ctx.strokeStyle='#ef4444';ctx.lineWidth=1.5;ctx.setLineDash([6,4]);
    ctx.beginPath();ctx.moveTo(p.l,thY);ctx.lineTo(W-p.r,thY);ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#ef4444';ctx.fillText('th='+thresholdVal.toFixed(2),p.l+5,thY-5);
    if(simHistory.length<2)return;
    const n=simHistory.length;
    ctx.fillStyle='#22c55e15';ctx.beginPath();ctx.moveTo(p.l,p.t+pH);
    for(let i=0;i<n;i++){
        const x=p.l+(i/(MAX_SIM_POINTS-1))*pW;
        const s=Math.max(0,Math.min(1,simHistory[i].sim));
        ctx.lineTo(x,s>thresholdVal?p.t+pH*(1-s):p.t+pH);
    }
    ctx.lineTo(p.l+((n-1)/(MAX_SIM_POINTS-1))*pW,p.t+pH);ctx.fill();
    ctx.strokeStyle='#3b82f6';ctx.lineWidth=2;ctx.beginPath();
    for(let i=0;i<n;i++){
        const x=p.l+(i/(MAX_SIM_POINTS-1))*pW;
        const y=p.t+pH*(1-Math.max(0,Math.min(1,simHistory[i].sim)));
        if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }
    ctx.stroke();
    for(let i=0;i<n;i+=3){
        const x=p.l+(i/(MAX_SIM_POINTS-1))*pW;
        const y=p.t+pH*(1-Math.max(0,Math.min(1,simHistory[i].sim)));
        ctx.fillStyle=simHistory[i].is_target?'#4ade80':'#f8717166';
        ctx.beginPath();ctx.arc(x,y,2.5,0,Math.PI*2);ctx.fill();
    }
}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════
# API: Enrollment from raw PCM (microphone)
# ═══════════════════════════════════════════════════
@app.post("/api/enroll")
async def api_enroll(request: Request):
    global enrollment_embedding
    body = await request.body()
    audio = np.frombuffer(body, dtype=np.float32).copy()

    duration = len(audio) / SAMPLE_RATE
    if duration < 3.0:
        return {"success": False, "message": f"Too short: {duration:.1f}s (need >= 3s)"}

    # Quality check
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    if rms < 0.003:
        return {"success": False, "message": f"Too quiet (RMS={rms:.4f}). Speak louder or move closer."}

    # Simple SNR
    frame_len = int(0.025 * SAMPLE_RATE)
    hop = int(0.010 * SAMPLE_RATE)
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    frame_rms = []
    for i in range(n_frames):
        s = i * hop
        fr = audio[s:s+frame_len]
        frame_rms.append(float(np.sqrt(np.mean(fr**2) + 1e-12)))
    sorted_rms = sorted(frame_rms)
    n20 = max(1, len(sorted_rms) // 5)
    noise_rms = np.mean(sorted_rms[:n20])
    speech_rms = np.mean(sorted_rms[-n20:])
    snr_db = round(float(20 * np.log10(speech_rms / (noise_rms + 1e-12))), 1)

    # Truncate to 15s max
    max_samples = int(15 * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    enrollment_embedding = encoder.extract_embedding(audio)

    return {
        "success": True,
        "embed_dim": int(enrollment_embedding.shape[0]),
        "duration": round(duration, 1),
        "rms": round(rms, 4),
        "snr_db": snr_db,
    }


# ═══════════════════════════════════════════════════
# WebSocket: Real-time stream
# ═══════════════════════════════════════════════════
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket, threshold: float = 0.25, window: float = 1.0):
    global enrollment_embedding
    await websocket.accept()

    if enrollment_embedding is None:
        await websocket.send_text(json.dumps({"error": "Not enrolled"}))
        await websocket.close()
        return

    audio_buffer = np.zeros(0, dtype=np.float32)
    window_samples = int(window * SAMPLE_RATE)
    ema_sim = 0.0
    ema_initialized = False
    ema_alpha = 0.5

    try:
        while True:
            data = await websocket.receive_bytes()
            t0 = time.time()

            chunk = np.frombuffer(data, dtype=np.float32).copy()
            rms_in = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))

            audio_buffer = np.concatenate([audio_buffer, chunk])

            # GTCRN denoise
            if len(chunk) >= GTCRN_NFFT:
                denoised_chunk = denoiser.enhance(chunk)
            else:
                padded = np.zeros(GTCRN_NFFT, dtype=np.float32)
                padded[:len(chunk)] = chunk
                denoised_chunk = denoiser.enhance(padded)[:len(chunk)]

            # pVAD
            is_target = False
            current_sim = ema_sim

            if len(audio_buffer) >= window_samples:
                audio_buffer = audio_buffer[-window_samples:]
                emb = encoder.extract_embedding(audio_buffer)
                raw_sim = cosine_similarity(emb, enrollment_embedding)

                if not ema_initialized:
                    ema_sim = raw_sim
                    ema_initialized = True
                else:
                    ema_sim = ema_alpha * raw_sim + (1 - ema_alpha) * ema_sim

                current_sim = ema_sim
                is_target = current_sim > threshold
                audio_buffer = audio_buffer[window_samples // 2:]

            # Gating
            output_chunk = denoised_chunk if is_target else denoised_chunk * 0.05
            rms_out = float(np.sqrt(np.mean(output_chunk ** 2) + 1e-12))
            latency_ms = (time.time() - t0) * 1000

            await websocket.send_text(json.dumps({
                "similarity": round(float(current_sim), 4),
                "is_target": bool(is_target),
                "rms_in": round(rms_in, 5),
                "rms_out": round(rms_out, 5),
                "latency_ms": round(latency_ms, 1),
            }))
            await websocket.send_bytes(output_chunk.astype(np.float32).tobytes())

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket disconnected")


if __name__ == "__main__":
    import socket
    port = 8080
    for p in [8080, 8888, 9000, 9090]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", p))
            s.close()
            port = p
            break
        except OSError:
            continue
    print(f"\n  pVAD-SE Real-time Demo")
    print(f"  http://localhost:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port)
