#!/usr/bin/env python3
"""Quick eval: 2 speakers x 3 ratios x 1 mode x 1 SNR = 6 cases."""
import sys, os, json, time, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path

REPO_DIR      = Path(__file__).resolve().parent
PVAD_DIR      = REPO_DIR.parent
ALLVOICE_BASE = PVAD_DIR / "all_voice"
NEWVAL_BASE   = PVAD_DIR / "newval_v3"
OUTPUT_DIR    = REPO_DIR / "eval_output"
OUTPUT_DIR.mkdir(exist_ok=True)

from utils.audio import (SAMPLE_RATE, BLOCK_LEN, BLOCK_SHIFT,
    read_audio, write_audio, stft_frame, istft_frame,
    magnitude_phase, reconstruct_complex, frame_signal, overlap_add)
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder, SimplePVAD
from pipeline import DTLNDenoiser

THRESHOLD = 0.25

def log(msg):
    print(msg, flush=True)

def load_models():
    ecapa = REPO_DIR / "models" / "campplus" / "campplus.onnx"
    if not ecapa.exists():
        ecapa = REPO_DIR / "models" / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    if not ecapa.exists():
        ecapa = REPO_DIR / "models" / "ecapa_tdnn" / "ecapa_tdnn.onnx"
    return SpeakerEncoder(str(ecapa)), DTLNDenoiser(), SoftGate()

def run_one(encoder, denoiser, gate, enroll_path, mixed_path, out_path):
    enroll = read_audio(str(enroll_path))
    dvec   = encoder.extract_embedding(enroll)
    pvad   = SimplePVAD(encoder, dvec, threshold=THRESHOLD)
    frames = frame_signal(read_audio(str(mixed_path)))
    enhanced = np.empty_like(frames)
    s1 = s2 = None
    sims = []
    for i, fr in enumerate(frames):
        den, s1, s2 = denoiser.process_frame(fr, s1, s2)
        is_t, sim = pvad.process_frame(den)
        sims.append(float(sim))
        enhanced[i] = gate.process(den, is_t, confidence=sim)
    out = overlap_add(enhanced)
    peak = np.max(np.abs(out))
    if peak > 0.99: out = out * 0.99 / peak
    write_audio(str(out_path), out)
    return np.array(sims)

def metrics(sims, json_path):
    with open(json_path) as f:
        segs = json.load(f)["segments"]
    n = len(sims)
    fd = BLOCK_SHIFT / SAMPLE_RATE
    gt = np.zeros(n, dtype=int)
    for s in segs:
        if s["label"] == 1:
            gt[int(s["start"]/fd):min(int(s["end"]/fd), n)] = 1
    pred = (sims > THRESHOLD).astype(int)
    tp = int(np.sum((pred==1)&(gt==1)))
    fp = int(np.sum((pred==1)&(gt==0)))
    fn = int(np.sum((pred==0)&(gt==1)))
    P  = tp/(tp+fp) if tp+fp else 0.0
    R  = tp/(tp+fn) if tp+fn else 0.0
    F1 = 2*P*R/(P+R) if P+R else 0.0
    return dict(precision=round(P,4), recall=round(R,4), f1=round(F1,4),
                mean_sim=round(float(np.mean(sims)),4),
                target_ratio=round(float(np.mean(pred)),4),
                gt_ratio=round(float(np.mean(gt)),4), n_frames=n)

def main():
    log("Loading models...")
    encoder, denoiser, gate = load_models()
    log("Models ready.\n")

    with open(NEWVAL_BASE / "registration_manifest.json") as f:
        manifest = json.load(f)

    configs = [
        (4, "0.3", "interleaved", "classroom", "0.2"),
        (4, "1",   "interleaved", "classroom", "0.6"),
        (4, "1.5", "interleaved", "classroom", "1"),
        (7, "0.3", "three_segment", "classroom", "0.2"),
        (7, "1",   "three_segment", "classroom", "0.6"),
        (7, "1.5", "three_segment", "classroom", "1"),
    ]

    results = []
    for i, (spk, ratio, mode, noise, snr) in enumerate(configs, 1):
        enroll_file = ALLVOICE_BASE / "all_voice" / str(spk) / manifest[str(spk)][0]
        cond_dir = NEWVAL_BASE / mode / f"ratio_{ratio}" / f"spk{spk}" / f"noise_{noise}" / f"snr_{snr}"
        wavs = sorted(cond_dir.glob("*_mixed.wav"))
        if not wavs:
            log(f"  [{i}/6] SKIP: no files in {cond_dir}")
            continue
        wav = wavs[0]
        out_name = f"spk{spk}_r{ratio}_{mode[:3]}_snr{snr}.wav"
        out_path = OUTPUT_DIR / out_name

        t0 = time.time()
        log(f"  [{i}/6] spk{spk} ratio={ratio} {mode[:3]} snr={snr} ...")
        sims = run_one(encoder, denoiser, gate, enroll_file, wav, out_path)
        elapsed = time.time() - t0
        dur = len(sims) * BLOCK_SHIFT / SAMPLE_RATE
        m = metrics(sims, wav.with_suffix(".json"))
        m["rtf"] = round(elapsed/dur, 3)
        log(f"       F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} RTF={m['rtf']:.2f}")
        results.append({"speaker":spk,"ratio":ratio,"mode":mode,"noise":noise,"snr":snr,
                        "output_file":out_name, **m})

    # Save
    (REPO_DIR/"eval_results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    log(f"\nSaved eval_results.json")
    generate_html(results)
    log(f"Generated eval_report.html")

def color_f1(v):
    if v >= 0.7: return "#4caf50"
    if v >= 0.4: return "#ff9800"
    return "#f44336"

def generate_html(results):
    rows = ""
    for r in results:
        c = color_f1(r["f1"])
        rows += f"""<tr>
  <td>spk{r['speaker']}</td><td>{r['ratio']}</td><td>{r['mode']}</td><td>{r['snr']}</td>
  <td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>
  <td style="color:{c};font-weight:700">{r['f1']:.3f}</td>
  <td>{r['mean_sim']:.3f}</td><td>{r['target_ratio']:.1%}</td><td>{r['gt_ratio']:.1%}</td>
  <td>{r['rtf']:.2f}</td>
  <td><a href="eval_output/{r['output_file']}">▶</a></td>
</tr>"""

    ratio_vals = {}
    for r in results:
        ratio_vals.setdefault(r["ratio"], []).append(r["f1"])
    r_labels = sorted(ratio_vals)
    r_data   = [round(sum(ratio_vals[k])/len(ratio_vals[k]),3) for k in r_labels]

    snr_vals = {}
    for r in results:
        snr_vals.setdefault(r["snr"], []).append(r["f1"])
    s_labels = sorted(snr_vals)
    s_data   = [round(sum(snr_vals[k])/len(snr_vals[k]),3) for k in s_labels]

    mode_vals = {}
    for r in results:
        mode_vals.setdefault(r["mode"], []).append(r["f1"])
    m_labels = sorted(mode_vals)
    m_data   = [round(sum(mode_vals[k])/len(mode_vals[k]),3) for k in m_labels]

    overall_f1 = round(sum(r["f1"] for r in results)/len(results),3)
    overall_p  = round(sum(r["precision"] for r in results)/len(results),3)
    overall_r  = round(sum(r["recall"] for r in results)/len(results),3)
    overall_rtf= round(sum(r["rtf"] for r in results)/len(results),3)

    def bar_color(vals):
        return json.dumps(['rgba(76,175,80,.8)' if v>=0.7 else 'rgba(255,152,0,.8)' if v>=0.4 else 'rgba(244,67,54,.8)' for v in vals])

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PVAD newval_v3 評估報告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Microsoft JhengHei",sans-serif;background:#f0f2f5;color:#222}}
header{{background:#1a237e;color:#fff;padding:1.5rem 2rem}}
header h1{{font-size:1.6rem}}
header p{{color:#9fa8da;font-size:.9rem;margin-top:.3rem}}
.wrap{{max-width:1200px;margin:0 auto;padding:1.5rem 1rem}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem}}
.card{{background:#fff;border-radius:10px;padding:1.2rem 1.5rem;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
.kpi .card{{text-align:center}}
.kpi .val{{font-size:2.4rem;font-weight:700;color:#1a237e}}
.kpi .lbl{{color:#888;font-size:.85rem;margin-top:.2rem}}
.charts{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem}}
.chart-card{{background:#fff;border-radius:10px;padding:1.2rem;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
.chart-card h3{{font-size:.9rem;color:#555;margin-bottom:.8rem}}
.table-card{{background:#fff;border-radius:10px;padding:1.5rem;box-shadow:0 1px 4px rgba(0,0,0,.1);overflow-x:auto;margin-bottom:1.5rem}}
.table-card h2{{font-size:1.1rem;margin-bottom:1rem;color:#333}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}}
th{{background:#e8eaf6;color:#3949ab;padding:.6rem .8rem;text-align:left;white-space:nowrap}}
td{{padding:.5rem .8rem;border-bottom:1px solid #f0f0f0}}
tr:hover td{{background:#f5f5f5}}
.note{{color:#888;font-size:.8rem;margin-top:1rem}}
footer{{text-align:center;padding:2rem;color:#aaa;font-size:.8rem}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem;color:#fff;background:#3949ab}}
</style>
</head>
<body>
<header>
  <h1>PVAD newval_v3 評估報告</h1>
  <p>Speakers 4 &amp; 7 · noise: classroom · threshold: {THRESHOLD} · {time.strftime('%Y-%m-%d %H:%M')}</p>
</header>
<div class="wrap">

<div class="kpi">
  <div class="card"><div class="val">{overall_f1}</div><div class="lbl">平均 F1</div></div>
  <div class="card"><div class="val">{overall_p}</div><div class="lbl">平均 Precision</div></div>
  <div class="card"><div class="val">{overall_r}</div><div class="lbl">平均 Recall</div></div>
  <div class="card"><div class="val">{overall_rtf}×</div><div class="lbl">平均 RTF</div></div>
</div>

<div class="charts">
  <div class="chart-card"><h3>F1 vs 混音比例 (ratio)</h3><canvas id="cRatio"></canvas></div>
  <div class="chart-card"><h3>F1 vs 環境噪音 SNR</h3><canvas id="cSNR"></canvas></div>
  <div class="chart-card"><h3>F1 vs 混音模式</h3><canvas id="cMode"></canvas></div>
</div>

<div class="table-card">
  <h2>詳細結果 <span class="badge">{len(results)} cases</span></h2>
  <table>
    <thead><tr>
      <th>Speaker</th><th>Ratio</th><th>Mode</th><th>SNR</th>
      <th>Precision</th><th>Recall</th><th>F1</th>
      <th>Mean Sim</th><th>Pred%</th><th>GT%</th><th>RTF</th><th>Audio</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p class="note">
    Registration: spk4 → session-1 files | spk7 → session-2 files (no overlap with test set)<br>
    Labels: 0=silence, 1=target speaker, 2=same-gender noise, 3=diff-gender noise
  </p>
</div>
</div>
<footer>PVAD-SE Pipeline · pvad-se-pipeline / eval_report.html</footer>

<script>
const opts = label => ({{
  responsive:true,
  plugins:{{legend:{{display:false}}}},
  scales:{{y:{{min:0,max:1,title:{{display:true,text:'F1'}}}},x:{{title:{{display:true,text:label}}}}}}
}});
new Chart(document.getElementById('cRatio'),{{type:'bar',data:{{
  labels:{json.dumps(r_labels)},
  datasets:[{{data:{json.dumps(r_data)},backgroundColor:{bar_color(r_data)},borderRadius:4}}]
}},options:opts('Ratio')}});
new Chart(document.getElementById('cSNR'),{{type:'bar',data:{{
  labels:{json.dumps(s_labels)},
  datasets:[{{data:{json.dumps(s_data)},backgroundColor:{bar_color(s_data)},borderRadius:4}}]
}},options:opts('SNR')}});
new Chart(document.getElementById('cMode'),{{type:'bar',data:{{
  labels:{json.dumps(m_labels)},
  datasets:[{{data:{json.dumps(m_data)},backgroundColor:{bar_color(m_data)},borderRadius:4}}]
}},options:opts('Mode')}});
</script>
</body></html>"""
    (REPO_DIR / "eval_report.html").write_text(html, encoding="utf-8")

if __name__ == "__main__":
    main()
