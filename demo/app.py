"""
pVAD-SE Pipeline Demo — Real-time Streaming (Personal VAD)
===========================================================
1. POST /api/enroll  → 用 CAM++ 提取 enrollment d-vector
2. WebSocket /ws/stream → Personal VAD (frame-level LSTM) + GTCRN 串流降噪
"""

import sys
import io
import numpy as np
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.audio import SAMPLE_RATE
from utils.speaker_encoder import SpeakerEncoder
from utils.gtcrn_denoiser import StreamingGTCRNDenoiser, GTCRN_HOP
from utils.personal_vad import PersonalVAD

print("Loading models...")
MODELS_DIR = PROJECT_DIR / "models"
SPEAKER_ENCODER = SpeakerEncoder(str(MODELS_DIR / "campplus" / "campplus.onnx"))
PVAD_ONNX = str(MODELS_DIR / "personal_vad" / "personal_vad.onnx")
print(f"  CAM++ dim={SPEAKER_ENCODER.embed_dim}")
print(f"  Personal VAD: {PVAD_ONNX}")
print("Models loaded.")

ENROLLMENT_DVECTOR = None

app = FastAPI(title="pVAD-SE Pipeline Demo")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


def decode_audio(data: bytes, filename: str = "") -> tuple:
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        return audio, sr
    except Exception:
        pass
    from pydub import AudioSegment
    ext = Path(filename).suffix.lstrip(".") or "webm"
    seg = AudioSegment.from_file(io.BytesIO(data), format=ext)
    seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    return samples, SAMPLE_RATE


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/enroll")
async def enroll(file: UploadFile = File(...)):
    global ENROLLMENT_DVECTOR
    data = await file.read()
    audio, sr = decode_audio(data, file.filename or "audio.webm")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    ENROLLMENT_DVECTOR = SPEAKER_ENCODER.extract_embedding(audio)

    return JSONResponse({
        "status": "ok",
        "duration": round(len(audio) / SAMPLE_RATE, 2),
        "rms": round(float(np.sqrt(np.mean(audio ** 2))), 4),
    })


@app.websocket("/ws/stream")
async def stream_process(ws: WebSocket):
    """
    即時串流 pipeline (Personal VAD)。
    Client 送: binary (int16 PCM, 16kHz mono)
    Server 回: JSON { output: [float], is_target: bool, confidence: float }

    Personal VAD 是 frame-level LSTM：
    - 每 256 samples (16ms) 就能判定
    - 不需要 0.5s 窗口
    - 冷啟動只需要 25ms (一個 Fbank 窗口)
    """
    await ws.accept()

    if ENROLLMENT_DVECTOR is None:
        await ws.send_json({"error": "尚未註冊"})
        await ws.close()
        return

    # 每個 session 獨立的狀態
    denoiser = StreamingGTCRNDenoiser()
    pvad = PersonalVAD(PVAD_ONNX, ENROLLMENT_DVECTOR)

    residual = np.array([], dtype=np.float32)
    is_target = False
    confidence = 0.0

    # 簡單的 gain smoothing
    current_gain = 0.0
    attack_coeff = 0.3   # 快速開啟
    release_coeff = 0.05  # 慢速關閉

    try:
        while True:
            data = await ws.receive_bytes()

            # int16 PCM → float32
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            residual = np.concatenate([residual, samples])

            output_all = []

            # 每次處理 GTCRN_HOP (256) 個 samples
            while len(residual) >= GTCRN_HOP:
                chunk = residual[:GTCRN_HOP]
                residual = residual[GTCRN_HOP:]

                # GTCRN 串流降噪
                denoised = denoiser.process_shift(chunk)

                # Personal VAD (frame-level, 無需等窗口)
                is_target, confidence = pvad.process_frame(chunk)

                # Gain smoothing
                target_gain = confidence if is_target else 0.0
                if target_gain > current_gain:
                    current_gain += attack_coeff * (target_gain - current_gain)
                else:
                    current_gain += release_coeff * (target_gain - current_gain)

                gated = denoised * current_gain
                output_all.extend(gated.tolist())

            if output_all:
                await ws.send_json({
                    "output": output_all,
                    "is_target": bool(is_target),
                    "confidence": round(float(confidence), 3),
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
