#!/usr/bin/env python3
"""
pVAD-SE Pipeline 展示網站
==========================
FastAPI + 純前端，展示所有模型的效果對比。

功能：
1. 聲紋註冊（上傳音檔 → 品質檢查 → 註冊）
2. 說話人驗證（上傳音檔 → 比對已註冊聲紋）
3. 三模型對比（CAM++ / ECAPA-TDNN / WeSpeaker）
4. pVAD 實時曲線（similarity 隨時間變化）
5. 降噪效果試聽（GTCRN 前後對比）

啟動: python demo_web.py
瀏覽: http://localhost:8000
"""

import io
import json
import base64
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── 專案路徑 ──
PROJECT = Path(__file__).resolve().parent
MODELS = PROJECT / "models"

import sys
sys.path.insert(0, str(PROJECT))
from utils.audio import SAMPLE_RATE, read_audio
from utils.speaker_encoder import SpeakerEncoder, cosine_similarity, _compute_fbank
from utils.gtcrn_denoiser import GTCRNDenoiser
from utils.enrollment import SpeakerEnrollment, QualityReport

import onnxruntime as ort


# ═══════════════════════════════════════════════════════════
# ECAPA-TDNN wrapper (不同輸入格式)
# ═══════════════════════════════════════════════════════════
class EcapaTDNNEncoder:
    def __init__(self, onnx_path: str):
        self.onnx_path = Path(onnx_path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        out_shape = self.session.get_outputs()[0].shape
        self.embed_dim = out_shape[-1] if isinstance(out_shape[-1], int) else 192

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        fbank = _compute_fbank(audio)
        fbank_batch = fbank.T[np.newaxis, :, :]  # (1, 80, T)
        embedding = self.session.run(
            [self.output_name], {self.input_name: fbank_batch}
        )[0].squeeze()
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding.astype(np.float32)


# ═══════════════════════════════════════════════════════════
# 載入模型
# ═══════════════════════════════════════════════════════════
print("Loading models...")
encoders = {}

campp_path = MODELS / "campplus" / "campplus.onnx"
if campp_path.exists():
    encoders["CAM++"] = SpeakerEncoder(str(campp_path))
    print(f"  [OK] CAM++ (dim={encoders['CAM++'].embed_dim})")

ecapa_path = MODELS / "ecapa_tdnn" / "ecapa_tdnn.onnx"
if ecapa_path.exists():
    try:
        encoders["ECAPA-TDNN"] = EcapaTDNNEncoder(str(ecapa_path))
        print(f"  [OK] ECAPA-TDNN (dim={encoders['ECAPA-TDNN'].embed_dim})")
    except Exception as e:
        print(f"  [FAIL] ECAPA-TDNN: {e}")

wespeaker_path = MODELS / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
if wespeaker_path.exists():
    encoders["WeSpeaker"] = SpeakerEncoder(str(wespeaker_path))
    print(f"  [OK] WeSpeaker (dim={encoders['WeSpeaker'].embed_dim})")

# GTCRN 降噪
denoiser = GTCRNDenoiser()
print("  [OK] GTCRN denoiser")

# 註冊器（使用 CAM++）
primary_encoder = encoders.get("CAM++", list(encoders.values())[0])
enrollment = SpeakerEnrollment(primary_encoder, profiles_dir=PROJECT / "profiles")
print(f"Models loaded: {list(encoders.keys())}")


# ═══════════════════════════════════════════════════════════
# FastAPI
# ═══════════════════════════════════════════════════════════
app = FastAPI(title="pVAD-SE Pipeline Demo")


def audio_to_base64_wav(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """將 numpy 音頻轉為 base64 WAV 字串（供前端播放）"""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="FLOAT")
    return base64.b64encode(buf.getvalue()).decode()


async def read_upload_audio(file: UploadFile) -> np.ndarray:
    """讀取上傳的音頻檔"""
    content = await file.read()
    buf = io.BytesIO(content)
    data, file_sr = sf.read(buf, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    if file_sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=file_sr, target_sr=SAMPLE_RATE)
    return data.astype(np.float32)


# ── 頁面 ──
@app.get("/", response_class=HTMLResponse)
async def index():
    profiles = enrollment.list_profiles()
    profiles_json = json.dumps(profiles, ensure_ascii=False)
    models_list = list(encoders.keys())

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>pVAD-SE Pipeline Demo</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}

        h1 {{ text-align: center; font-size: 2em; margin: 20px 0;
              background: linear-gradient(135deg, #3b82f6, #8b5cf6);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        h2 {{ color: #93c5fd; margin: 15px 0 10px; font-size: 1.3em; }}
        h3 {{ color: #a5b4fc; margin: 10px 0 5px; }}

        .card {{ background: #1e293b; border-radius: 12px; padding: 20px;
                 margin: 15px 0; border: 1px solid #334155; }}
        .card:hover {{ border-color: #3b82f6; }}

        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
        @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}

        .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
        @media (max-width: 900px) {{ .grid-3 {{ grid-template-columns: 1fr; }} }}

        label {{ display: block; margin: 8px 0 4px; color: #94a3b8; font-size: 0.9em; }}
        input[type="file"] {{ display: block; width: 100%; padding: 10px;
                              background: #0f172a; border: 1px dashed #475569;
                              border-radius: 8px; color: #e2e8f0; cursor: pointer; }}
        input[type="file"]:hover {{ border-color: #3b82f6; }}
        input[type="text"], input[type="number"], select {{
            width: 100%; padding: 8px 12px; background: #0f172a;
            border: 1px solid #475569; border-radius: 6px; color: #e2e8f0; }}

        button {{ padding: 10px 24px; border: none; border-radius: 8px;
                  font-size: 1em; cursor: pointer; font-weight: 600; transition: all 0.2s; }}
        .btn-primary {{ background: #3b82f6; color: white; }}
        .btn-primary:hover {{ background: #2563eb; transform: translateY(-1px); }}
        .btn-secondary {{ background: #475569; color: white; }}
        .btn-secondary:hover {{ background: #64748b; }}
        .btn-danger {{ background: #ef4444; color: white; }}
        .btn-danger:hover {{ background: #dc2626; }}
        button:disabled {{ opacity: 0.5; cursor: not-allowed; }}

        .result {{ background: #0f172a; border-radius: 8px; padding: 15px;
                   margin: 10px 0; font-family: monospace; white-space: pre-wrap;
                   max-height: 400px; overflow-y: auto; font-size: 0.9em; line-height: 1.5; }}
        .success {{ color: #4ade80; }}
        .error {{ color: #f87171; }}
        .warn {{ color: #fbbf24; }}
        .info {{ color: #60a5fa; }}

        .model-badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
                        font-size: 0.8em; font-weight: 600; margin: 2px; }}
        .badge-campp {{ background: #1e40af; color: #93c5fd; }}
        .badge-ecapa {{ background: #7c2d12; color: #fdba74; }}
        .badge-wespeaker {{ background: #14532d; color: #86efac; }}

        .sim-bar {{ height: 24px; border-radius: 4px; margin: 3px 0;
                    transition: width 0.3s; position: relative; }}
        .sim-bar span {{ position: absolute; right: 8px; top: 2px; font-size: 0.8em; }}

        .profile-list {{ list-style: none; }}
        .profile-list li {{ padding: 8px 12px; background: #0f172a; border-radius: 6px;
                            margin: 4px 0; display: flex; justify-content: space-between;
                            align-items: center; }}

        audio {{ width: 100%; margin: 5px 0; height: 36px; }}

        .loading {{ display: inline-block; width: 16px; height: 16px;
                    border: 2px solid #475569; border-top-color: #3b82f6;
                    border-radius: 50%; animation: spin 0.8s linear infinite; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

        .tab-bar {{ display: flex; gap: 5px; margin-bottom: 15px; }}
        .tab {{ padding: 8px 20px; border-radius: 8px 8px 0 0; cursor: pointer;
                background: #1e293b; color: #94a3b8; border: 1px solid #334155;
                border-bottom: none; }}
        .tab.active {{ background: #1e293b; color: #e2e8f0; border-color: #3b82f6;
                       border-bottom: 2px solid #1e293b; }}

        .chart-container {{ background: #0f172a; border-radius: 8px; padding: 10px;
                            margin: 10px 0; }}
        canvas {{ width: 100% !important; height: 200px !important; }}

        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
        .stat-card {{ text-align: center; padding: 10px; background: #0f172a; border-radius: 8px; }}
        .stat-value {{ font-size: 1.5em; font-weight: 700; }}
        .stat-label {{ font-size: 0.8em; color: #94a3b8; }}

        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; font-weight: 600; font-size: 0.85em; }}
        td {{ font-family: monospace; }}
    </style>
</head>
<body>
<div class="container">
    <h1>pVAD-SE Pipeline Demo</h1>
    <p style="text-align:center; color:#64748b; margin-bottom:20px;">
        個人化語音活動偵測 + 語音增強 | CAM++ · ECAPA-TDNN · WeSpeaker · GTCRN
    </p>

    <!-- ══════ 已註冊聲紋 ══════ -->
    <div class="card">
        <h2>📋 已註冊聲紋</h2>
        <ul class="profile-list" id="profileList"></ul>
        <p id="noProfiles" style="color:#64748b; display:none;">尚無註冊聲紋</p>
    </div>

    <div class="grid">
        <!-- ══════ 聲紋註冊 ══════ -->
        <div class="card">
            <h2>🎤 聲紋註冊</h2>
            <label>名稱</label>
            <input type="text" id="enrollName" placeholder="例如: john" />
            <label>音檔（≥3 秒，16kHz WAV）</label>
            <input type="file" id="enrollFile" accept="audio/*" />
            <div style="margin-top:10px;">
                <button class="btn-primary" onclick="doEnroll()">註冊</button>
            </div>
            <div class="result" id="enrollResult" style="display:none;"></div>
        </div>

        <!-- ══════ 說話人驗證 ══════ -->
        <div class="card">
            <h2>🔐 說話人驗證</h2>
            <label>選擇聲紋</label>
            <select id="verifyProfile"></select>
            <label>測試音檔</label>
            <input type="file" id="verifyFile" accept="audio/*" />
            <label>Threshold</label>
            <input type="number" id="verifyThreshold" value="0.25" step="0.01" min="0" max="1" />
            <div style="margin-top:10px;">
                <button class="btn-primary" onclick="doVerify()">驗證</button>
            </div>
            <div class="result" id="verifyResult" style="display:none;"></div>
        </div>
    </div>

    <!-- ══════ 三模型對比 ══════ -->
    <div class="card">
        <h2>⚖️ 三模型對比</h2>
        <p style="color:#64748b; margin-bottom:10px;">
            上傳 enrollment + 測試音檔，同時用三個模型比較 similarity
        </p>
        <div class="grid">
            <div>
                <label>Enrollment 音檔（目標說話人）</label>
                <input type="file" id="cmpEnroll" accept="audio/*" />
            </div>
            <div>
                <label>測試音檔</label>
                <input type="file" id="cmpTest" accept="audio/*" />
            </div>
        </div>
        <label>窗口大小（秒）</label>
        <select id="cmpWindow">
            <option value="0.5">0.5s (低延遲)</option>
            <option value="0.75">0.75s</option>
            <option value="1.0" selected>1.0s (推薦)</option>
        </select>
        <div style="margin-top:10px;">
            <button class="btn-primary" onclick="doCompare()">開始對比</button>
        </div>
        <div id="compareResult" style="display:none;">
            <div class="grid-3" id="modelCards"></div>
            <div class="chart-container">
                <canvas id="simChart"></canvas>
            </div>
        </div>
    </div>

    <!-- ══════ GTCRN 降噪 ══════ -->
    <div class="card">
        <h2>🔇 GTCRN 降噪試聽</h2>
        <label>音檔</label>
        <input type="file" id="denoiseFile" accept="audio/*" />
        <div style="margin-top:10px;">
            <button class="btn-primary" onclick="doDenoise()">降噪</button>
        </div>
        <div id="denoiseResult" style="display:none;">
            <div class="grid">
                <div>
                    <h3>原始</h3>
                    <audio id="origAudio" controls></audio>
                </div>
                <div>
                    <h3>降噪後</h3>
                    <audio id="denoisedAudio" controls></audio>
                </div>
            </div>
            <div class="stats-grid" id="denoiseStats"></div>
        </div>
    </div>

    <!-- ══════ 完整管線 ══════ -->
    <div class="card">
        <h2>🚀 完整管線（pVAD + GTCRN）</h2>
        <p style="color:#64748b; margin-bottom:10px;">
            上傳 enrollment + 混合音頻 → 降噪 + 說話人過濾 → 只保留目標說話人的語音
        </p>
        <div class="grid">
            <div>
                <label>Enrollment 音檔</label>
                <input type="file" id="pipeEnroll" accept="audio/*" />
            </div>
            <div>
                <label>混合音頻</label>
                <input type="file" id="pipeInput" accept="audio/*" />
            </div>
        </div>
        <div class="grid">
            <div>
                <label>選擇聲紋（或上傳 enrollment）</label>
                <select id="pipeProfile">
                    <option value="">-- 使用上傳的 enrollment --</option>
                </select>
            </div>
            <div>
                <label>Threshold</label>
                <input type="number" id="pipeThreshold" value="0.25" step="0.01" />
            </div>
        </div>
        <div style="margin-top:10px;">
            <button class="btn-primary" onclick="doPipeline()">執行管線</button>
        </div>
        <div id="pipeResult" style="display:none;">
            <div class="grid-3">
                <div>
                    <h3>原始混合</h3>
                    <audio id="pipeOrigAudio" controls></audio>
                </div>
                <div>
                    <h3>僅降噪</h3>
                    <audio id="pipeDenoisedAudio" controls></audio>
                </div>
                <div>
                    <h3>pVAD + 降噪</h3>
                    <audio id="pipeOutputAudio" controls></audio>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="pipeChart"></canvas>
            </div>
        </div>
    </div>
</div>

<script>
// ── 初始化 ──
let profiles = {profiles_json};
const models = {json.dumps(models_list)};

function refreshProfiles() {{
    const list = document.getElementById('profileList');
    const noP = document.getElementById('noProfiles');
    const sel = document.getElementById('verifyProfile');
    const pipeSel = document.getElementById('pipeProfile');

    list.innerHTML = '';
    sel.innerHTML = '';
    // keep first option for pipeSel
    pipeSel.innerHTML = '<option value="">-- 使用上傳的 enrollment --</option>';

    if (profiles.length === 0) {{
        noP.style.display = 'block';
        return;
    }}
    noP.style.display = 'none';

    profiles.forEach(p => {{
        const li = document.createElement('li');
        li.innerHTML = `<span>${{p.name}} <span style="color:#64748b">(dim=${{p.embed_dim}}, model=${{p.encoder_model}})</span></span>
            <button class="btn-danger" style="padding:4px 12px;font-size:0.8em" onclick="doDelete('${{p.name}}')">刪除</button>`;
        list.appendChild(li);

        const opt = document.createElement('option');
        opt.value = p.name;
        opt.textContent = p.name;
        sel.appendChild(opt);

        const opt2 = document.createElement('option');
        opt2.value = p.name;
        opt2.textContent = p.name;
        pipeSel.appendChild(opt2);
    }});
}}
refreshProfiles();

// ── 通用 ──
function showResult(id, text, cls='info') {{
    const el = document.getElementById(id);
    el.style.display = 'block';
    el.innerHTML = `<span class="${{cls}}">${{text}}</span>`;
}}

function showLoading(id) {{
    const el = document.getElementById(id);
    el.style.display = 'block';
    el.innerHTML = '<span class="loading"></span> 處理中...';
}}

// ── 註冊 ──
async function doEnroll() {{
    const name = document.getElementById('enrollName').value.trim();
    const file = document.getElementById('enrollFile').files[0];
    if (!name) return alert('請輸入名稱');
    if (!file) return alert('請選擇音檔');

    showLoading('enrollResult');
    const fd = new FormData();
    fd.append('name', name);
    fd.append('file', file);

    const res = await fetch('/api/enroll', {{ method: 'POST', body: fd }});
    const data = await res.json();

    if (data.success) {{
        showResult('enrollResult', data.message, 'success');
        profiles = data.profiles;
        refreshProfiles();
    }} else {{
        showResult('enrollResult', data.message, 'error');
    }}
}}

// ── 刪除 ──
async function doDelete(name) {{
    if (!confirm(`確定刪除 ${{name}}？`)) return;
    const res = await fetch(`/api/delete/${{name}}`, {{ method: 'DELETE' }});
    const data = await res.json();
    profiles = data.profiles;
    refreshProfiles();
}}

// ── 驗證 ──
async function doVerify() {{
    const profile = document.getElementById('verifyProfile').value;
    const file = document.getElementById('verifyFile').files[0];
    const threshold = document.getElementById('verifyThreshold').value;
    if (!profile) return alert('請選擇聲紋');
    if (!file) return alert('請選擇音檔');

    showLoading('verifyResult');
    const fd = new FormData();
    fd.append('profile', profile);
    fd.append('file', file);
    fd.append('threshold', threshold);

    const res = await fetch('/api/verify', {{ method: 'POST', body: fd }});
    const data = await res.json();

    let html = '';
    if (data.is_match) {{
        html = `<span class="success">✓ 匹配！</span> similarity = ${{data.similarity.toFixed(3)}} (threshold = ${{threshold}})`;
    }} else {{
        html = `<span class="error">✗ 不匹配</span> similarity = ${{data.similarity.toFixed(3)}} (threshold = ${{threshold}})`;
    }}
    html += `\\n\\n品質: duration=${{data.quality.duration_sec}}s, SNR=${{data.quality.snr_db}}dB`;
    document.getElementById('verifyResult').style.display = 'block';
    document.getElementById('verifyResult').innerHTML = html;
}}

// ── 三模型對比 ──
async function doCompare() {{
    const enrollFile = document.getElementById('cmpEnroll').files[0];
    const testFile = document.getElementById('cmpTest').files[0];
    const window = document.getElementById('cmpWindow').value;
    if (!enrollFile || !testFile) return alert('請選擇兩個音檔');

    document.getElementById('compareResult').style.display = 'block';
    document.getElementById('modelCards').innerHTML = '<div class="loading"></div> 三模型對比中...';

    const fd = new FormData();
    fd.append('enroll', enrollFile);
    fd.append('test', testFile);
    fd.append('window', window);

    const res = await fetch('/api/compare', {{ method: 'POST', body: fd }});
    const data = await res.json();

    // 模型卡片
    const colors = {{'CAM++': '#3b82f6', 'ECAPA-TDNN': '#f97316', 'WeSpeaker': '#22c55e'}};
    const badges = {{'CAM++': 'badge-campp', 'ECAPA-TDNN': 'badge-ecapa', 'WeSpeaker': 'badge-wespeaker'}};
    let cardsHtml = '';
    data.models.forEach(m => {{
        const barWidth = Math.max(5, Math.min(100, m.whole_sim * 100));
        const barColor = colors[m.name] || '#666';
        cardsHtml += `
        <div class="card" style="border-color:${{barColor}}44">
            <h3><span class="model-badge ${{badges[m.name] || ''}}">${{m.name}}</span></h3>
            <table>
                <tr><td>整段 sim</td><td style="color:${{barColor}}">${{m.whole_sim.toFixed(3)}}</td></tr>
                <tr><td>窗口平均</td><td>${{m.window_mean.toFixed(3)}}</td></tr>
                <tr><td>窗口 std</td><td>${{m.window_std.toFixed(3)}}</td></tr>
                <tr><td>embed dim</td><td>${{m.embed_dim}}</td></tr>
                <tr><td>推論時間</td><td>${{m.infer_ms.toFixed(0)}} ms</td></tr>
            </table>
            <div class="sim-bar" style="width:${{barWidth}}%;background:${{barColor}}">
                <span>${{m.whole_sim.toFixed(3)}}</span>
            </div>
        </div>`;
    }});
    document.getElementById('modelCards').innerHTML = cardsHtml;

    // 曲線圖 (用簡易 canvas)
    drawSimChart('simChart', data.models, data.window_sec);
}}

function drawSimChart(canvasId, models, winSec) {{
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth - 20;
    canvas.height = 200;
    const W = canvas.width, H = canvas.height;
    const pad = {{left:50, right:20, top:20, bottom:30}};
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, W, H);

    // Y 軸 (0~1)
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    for (let y = 0; y <= 1; y += 0.2) {{
        const py = pad.top + plotH * (1 - y);
        ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(W-pad.right, py); ctx.stroke();
        ctx.fillStyle = '#64748b'; ctx.font = '11px monospace';
        ctx.fillText(y.toFixed(1), 5, py + 4);
    }}

    const colors = {{'CAM++': '#3b82f6', 'ECAPA-TDNN': '#f97316', 'WeSpeaker': '#22c55e'}};

    models.forEach(m => {{
        if (!m.window_sims || m.window_sims.length === 0) return;
        const n = m.window_sims.length;
        ctx.strokeStyle = colors[m.name] || '#888';
        ctx.lineWidth = 2;
        ctx.beginPath();
        m.window_sims.forEach((sim, i) => {{
            const x = pad.left + (i / Math.max(1, n-1)) * plotW;
            const y = pad.top + plotH * (1 - Math.max(0, Math.min(1, sim)));
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();

        // 標籤
        const lastSim = m.window_sims[m.window_sims.length - 1];
        const lx = W - pad.right - 5;
        const ly = pad.top + plotH * (1 - Math.max(0, Math.min(1, lastSim)));
        ctx.fillStyle = colors[m.name] || '#888';
        ctx.font = '11px sans-serif';
        ctx.fillText(m.name, lx - 70, ly - 5);
    }});

    // X 軸
    ctx.fillStyle = '#64748b'; ctx.font = '11px monospace';
    const maxWin = Math.max(...models.map(m => m.window_sims?.length || 0));
    for (let i = 0; i < maxWin; i += Math.max(1, Math.floor(maxWin/5))) {{
        const x = pad.left + (i / Math.max(1, maxWin-1)) * plotW;
        ctx.fillText(`${{(i * winSec).toFixed(1)}}s`, x - 10, H - 5);
    }}
}}

// ── 降噪 ──
async function doDenoise() {{
    const file = document.getElementById('denoiseFile').files[0];
    if (!file) return alert('請選擇音檔');

    document.getElementById('denoiseResult').style.display = 'block';
    document.getElementById('denoiseStats').innerHTML = '<div class="loading"></div>';

    const fd = new FormData();
    fd.append('file', file);

    const res = await fetch('/api/denoise', {{ method: 'POST', body: fd }});
    const data = await res.json();

    document.getElementById('origAudio').src = 'data:audio/wav;base64,' + data.original_wav;
    document.getElementById('denoisedAudio').src = 'data:audio/wav;base64,' + data.denoised_wav;

    document.getElementById('denoiseStats').innerHTML = `
        <div class="stat-card"><div class="stat-value">${{data.duration.toFixed(1)}}s</div><div class="stat-label">長度</div></div>
        <div class="stat-card"><div class="stat-value">${{data.orig_rms.toFixed(4)}}</div><div class="stat-label">原始 RMS</div></div>
        <div class="stat-card"><div class="stat-value">${{data.denoised_rms.toFixed(4)}}</div><div class="stat-label">降噪 RMS</div></div>
        <div class="stat-card"><div class="stat-value">${{data.process_ms.toFixed(0)}}ms</div><div class="stat-label">處理時間</div></div>`;
}}

// ── 完整管線 ──
async function doPipeline() {{
    const enrollFile = document.getElementById('pipeEnroll').files[0];
    const inputFile = document.getElementById('pipeInput').files[0];
    const profile = document.getElementById('pipeProfile').value;
    const threshold = document.getElementById('pipeThreshold').value;

    if (!inputFile) return alert('請選擇混合音頻');
    if (!enrollFile && !profile) return alert('請選擇 enrollment 音檔或已註冊聲紋');

    document.getElementById('pipeResult').style.display = 'block';

    const fd = new FormData();
    if (enrollFile) fd.append('enroll', enrollFile);
    fd.append('input', inputFile);
    fd.append('profile', profile);
    fd.append('threshold', threshold);

    const res = await fetch('/api/pipeline', {{ method: 'POST', body: fd }});
    const data = await res.json();

    document.getElementById('pipeOrigAudio').src = 'data:audio/wav;base64,' + data.original_wav;
    document.getElementById('pipeDenoisedAudio').src = 'data:audio/wav;base64,' + data.denoised_wav;
    document.getElementById('pipeOutputAudio').src = 'data:audio/wav;base64,' + data.output_wav;

    // 畫 similarity 曲線
    drawPipeChart('pipeChart', data.similarities, data.threshold);
}}

function drawPipeChart(canvasId, sims, threshold) {{
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth - 20;
    canvas.height = 200;
    const W = canvas.width, H = canvas.height;
    const pad = {{left:50, right:20, top:20, bottom:30}};
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
    for (let y = 0; y <= 1; y += 0.2) {{
        const py = pad.top + plotH * (1 - y);
        ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(W-pad.right, py); ctx.stroke();
        ctx.fillStyle = '#64748b'; ctx.font = '11px monospace';
        ctx.fillText(y.toFixed(1), 5, py + 4);
    }}

    // Threshold 線
    const thY = pad.top + plotH * (1 - threshold);
    ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 1; ctx.setLineDash([5,3]);
    ctx.beginPath(); ctx.moveTo(pad.left, thY); ctx.lineTo(W-pad.right, thY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ef4444'; ctx.fillText(`th=${{threshold}}`, pad.left + 5, thY - 5);

    // Similarity 曲線
    const n = sims.length;
    ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 2;
    ctx.beginPath();
    sims.forEach((s, i) => {{
        const x = pad.left + (i / Math.max(1, n-1)) * plotW;
        const y = pad.top + plotH * (1 - Math.max(0, Math.min(1, s)));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();

    // 填色（目標區域）
    ctx.fillStyle = '#3b82f622';
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH);
    sims.forEach((s, i) => {{
        const x = pad.left + (i / Math.max(1, n-1)) * plotW;
        const clampedS = Math.max(0, Math.min(1, s));
        const y = clampedS > threshold ? pad.top + plotH * (1 - clampedS) : pad.top + plotH;
        ctx.lineTo(x, y);
    }});
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.fill();
}}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════
# API 端點
# ═══════════════════════════════════════════════════════════

@app.post("/api/enroll")
async def api_enroll(name: str = Form(...), file: UploadFile = File(...)):
    audio = await read_upload_audio(file)
    result = enrollment.enroll_from_audio(audio, name)
    profiles = enrollment.list_profiles()
    return {
        "success": result.success,
        "message": result.message,
        "profiles": profiles,
    }


@app.delete("/api/delete/{name}")
async def api_delete(name: str):
    enrollment.delete_profile(name)
    return {"profiles": enrollment.list_profiles()}


@app.post("/api/verify")
async def api_verify(
    profile: str = Form(...),
    file: UploadFile = File(...),
    threshold: float = Form(0.25),
):
    audio = await read_upload_audio(file)
    quality = enrollment.check_quality(audio)
    is_match, sim = enrollment.verify_speaker(profile, audio, threshold=threshold)
    return {
        "is_match": is_match,
        "similarity": sim,
        "quality": {
            "duration_sec": quality.duration_sec,
            "rms": quality.rms,
            "snr_db": quality.snr_db,
            "silence_ratio": quality.silence_ratio,
        },
    }


@app.post("/api/compare")
async def api_compare(
    enroll: UploadFile = File(...),
    test: UploadFile = File(...),
    window: float = Form(1.0),
):
    enroll_audio = await read_upload_audio(enroll)
    test_audio = await read_upload_audio(test)
    win_samples = int(window * SAMPLE_RATE)

    results = []
    for name, enc in encoders.items():
        t0 = time.time()
        enroll_emb = enc.extract_embedding(enroll_audio)
        whole_emb = enc.extract_embedding(test_audio)
        whole_sim = float(cosine_similarity(whole_emb, enroll_emb))

        # 逐窗口
        n_win = len(test_audio) // win_samples
        win_sims = []
        for i in range(min(n_win, 30)):
            chunk = test_audio[i * win_samples : (i+1) * win_samples]
            emb = enc.extract_embedding(chunk)
            sim = float(cosine_similarity(emb, enroll_emb))
            win_sims.append(round(sim, 4))

        infer_ms = (time.time() - t0) * 1000

        results.append({
            "name": name,
            "whole_sim": round(whole_sim, 4),
            "window_mean": round(float(np.mean(win_sims)) if win_sims else 0, 4),
            "window_std": round(float(np.std(win_sims)) if win_sims else 0, 4),
            "window_sims": win_sims,
            "embed_dim": int(enc.embed_dim),
            "infer_ms": round(infer_ms, 1),
        })

    return {"models": results, "window_sec": window}


@app.post("/api/denoise")
async def api_denoise(file: UploadFile = File(...)):
    audio = await read_upload_audio(file)

    t0 = time.time()
    denoised = denoiser.enhance(audio)
    process_ms = (time.time() - t0) * 1000

    return {
        "original_wav": audio_to_base64_wav(audio),
        "denoised_wav": audio_to_base64_wav(denoised),
        "duration": len(audio) / SAMPLE_RATE,
        "orig_rms": round(float(np.sqrt(np.mean(audio**2))), 4),
        "denoised_rms": round(float(np.sqrt(np.mean(denoised**2))), 4),
        "process_ms": round(process_ms, 1),
    }


@app.post("/api/pipeline")
async def api_pipeline(
    input: UploadFile = File(...),
    threshold: float = Form(0.25),
    profile: str = Form(""),
    enroll: UploadFile = File(None),
):
    input_audio = await read_upload_audio(input)

    # 取得 enrollment embedding
    if profile and profile.strip():
        enroll_emb = enrollment.load_profile(profile)
    elif enroll:
        enroll_audio = await read_upload_audio(enroll)
        enroll_emb = primary_encoder.extract_embedding(enroll_audio)
    else:
        return JSONResponse({"error": "需要 enrollment"}, status_code=400)

    # GTCRN 降噪
    denoised = denoiser.enhance(input_audio)

    # pVAD (CAM++ 1.0s 窗口)
    from utils.gating import SoftGate
    from utils.gtcrn_denoiser import GTCRN_NFFT, GTCRN_HOP
    from utils.audio import frame_signal, overlap_add
    from pipeline_parallel import ParallelPVAD

    pvad = ParallelPVAD(
        speaker_encoder=primary_encoder,
        enrollment_dvector=enroll_emb,
        extract_interval=63,
        window_sec=1.0,
        threshold=threshold,
        ema_alpha=0.5,
    )
    gate = SoftGate(gain_floor=0.05, attack_ms=5.0, release_ms=50.0, hop=GTCRN_HOP)

    denoised_frames = frame_signal(denoised, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    n_frames = denoised_frames.shape[0]
    enhanced_frames = np.empty((n_frames, GTCRN_NFFT), dtype=np.float32)
    sims = []

    for i in range(n_frames):
        start = i * GTCRN_HOP
        end = min(start + GTCRN_HOP, len(input_audio))
        raw_chunk = input_audio[start:end]
        is_target, sim = pvad.process_frame(raw_chunk)
        sims.append(round(float(sim), 4))
        enhanced_frames[i] = gate.process(denoised_frames[i], is_target, confidence=sim)

    output = overlap_add(enhanced_frames, block_len=GTCRN_NFFT, block_shift=GTCRN_HOP)
    output = output[:len(input_audio)]
    peak = np.max(np.abs(output))
    if peak > 0.99:
        output = output * 0.99 / peak

    # 降採樣 similarity（前端不需要每幀的值）
    step = max(1, len(sims) // 200)
    sims_down = [sims[i] for i in range(0, len(sims), step)]

    return {
        "original_wav": audio_to_base64_wav(input_audio),
        "denoised_wav": audio_to_base64_wav(denoised),
        "output_wav": audio_to_base64_wav(output),
        "similarities": sims_down,
        "threshold": threshold,
    }


# ═══════════════════════════════════════════════════════════
# 啟動
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  pVAD-SE Pipeline Demo")
    print(f"  http://localhost:<port>")
    print("=" * 50 + "\n")
    import socket
    port = 8000
    for p in [8000, 8080, 8888, 9000]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", p))
            s.close()
            port = p
            break
        except OSError:
            continue
    print(f"  使用端口: {port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
