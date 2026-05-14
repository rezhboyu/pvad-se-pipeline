#!/usr/bin/env python3
"""Generate eval_report.html from hardcoded 4-case results."""
import json, time
from pathlib import Path

REPO = Path(__file__).resolve().parent
THRESHOLD = 0.25

results = [
    {"speaker":4,"ratio":"0.3","mode":"interleaved",   "noise":"classroom","snr":"0.2",
     "precision":0.750,"recall":0.003,"f1":0.005,"mean_sim":0.123,"target_ratio":0.004,"gt_ratio":0.381,"rtf":7.11,"output_file":"spk4_r0.3_int_snr0.2.wav"},
    {"speaker":4,"ratio":"1",  "mode":"interleaved",   "noise":"classroom","snr":"0.6",
     "precision":1.000,"recall":0.001,"f1":0.002,"mean_sim":0.099,"target_ratio":0.001,"gt_ratio":0.381,"rtf":8.22,"output_file":"spk4_r1_int_snr0.6.wav"},
    {"speaker":4,"ratio":"1.5","mode":"interleaved",   "noise":"classroom","snr":"1",
     "precision":0.000,"recall":0.000,"f1":0.000,"mean_sim":0.091,"target_ratio":0.000,"gt_ratio":0.381,"rtf":11.40,"output_file":"spk4_r1.5_int_snr1.wav"},
    {"speaker":7,"ratio":"0.3","mode":"three_segment", "noise":"classroom","snr":"0.2",
     "precision":0.000,"recall":0.000,"f1":0.000,"mean_sim":0.097,"target_ratio":0.000,"gt_ratio":0.442,"rtf":15.57,"output_file":"spk7_r0.3_thr_snr0.2.wav"},
]

def color(v):
    return "#4caf50" if v >= 0.7 else "#ff9800" if v >= 0.4 else "#f44336"

rows = ""
for r in results:
    c = color(r["f1"])
    rows += (
        f'<tr><td>spk{r["speaker"]}</td><td>{r["ratio"]}</td>'
        f'<td>{r["mode"]}</td><td>{r["snr"]}</td>'
        f'<td>{r["precision"]:.3f}</td><td>{r["recall"]:.3f}</td>'
        f'<td style="color:{c};font-weight:700">{r["f1"]:.3f}</td>'
        f'<td>{r["mean_sim"]:.3f}</td><td>{r["target_ratio"]:.1%}</td>'
        f'<td>{r["gt_ratio"]:.1%}</td><td>{r["rtf"]:.1f}x</td>'
        f'<td><a href="eval_output/{r["output_file"]}" target="_blank">▶</a></td></tr>\n'
    )

r_vals, s_vals, m_vals = {}, {}, {}
for r in results:
    r_vals.setdefault(r["ratio"], []).append(r["f1"])
    s_vals.setdefault(r["snr"],   []).append(r["f1"])
    m_vals.setdefault(r["mode"],  []).append(r["f1"])

def avg_dict(d):
    return {k: round(sum(v)/len(v), 3) for k, v in d.items()}

r_avg = avg_dict(r_vals); s_avg = avg_dict(s_vals); m_avg = avg_dict(m_vals)
r_l, r_d = sorted(r_avg), [r_avg[k] for k in sorted(r_avg)]
s_l, s_d = sorted(s_avg), [s_avg[k] for k in sorted(s_avg)]
m_l, m_d = sorted(m_avg), [m_avg[k] for k in sorted(m_avg)]
sim_vals  = [r["mean_sim"] for r in results]
sim_lbls  = [f'spk{r["speaker"]} r={r["ratio"]} snr={r["snr"]}' for r in results]

overall = {k: round(sum(r[k] for r in results)/len(results), 3)
           for k in ("f1", "precision", "recall")}
overall_rtf = round(sum(r["rtf"] for r in results)/len(results), 1)

def bc(vals):
    return json.dumps(["rgba(76,175,80,.8)" if v >= 0.7 else
                       "rgba(255,152,0,.8)"  if v >= 0.4 else
                       "rgba(244,67,54,.8)"  for v in vals])

html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PVAD-SE newval_v3 評估報告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box }}
body {{ font-family:-apple-system,"Microsoft JhengHei",sans-serif; background:#f0f2f5; color:#222 }}
header {{ background:linear-gradient(135deg,#1a237e,#283593); color:#fff; padding:2rem }}
header h1 {{ font-size:1.8rem; font-weight:700 }}
header p  {{ color:#9fa8da; font-size:.9rem; margin-top:.4rem }}
.wrap {{ max-width:1200px; margin:0 auto; padding:1.5rem 1rem }}
.kpi {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem }}
.card {{ background:#fff; border-radius:12px; padding:1.3rem 1.5rem; box-shadow:0 2px 8px rgba(0,0,0,.08) }}
.kpi .card {{ text-align:center }}
.kpi .val {{ font-size:2.4rem; font-weight:800; color:#1a237e }}
.kpi .lbl {{ color:#888; font-size:.82rem; margin-top:.3rem }}
.charts {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem }}
.chart-card {{ background:#fff; border-radius:12px; padding:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,.08) }}
.chart-card h3 {{ font-size:.88rem; color:#555; margin-bottom:.8rem; font-weight:600 }}
.obs {{ background:#fff3e0; border-left:4px solid #ff9800; padding:1rem 1.5rem;
         border-radius:0 8px 8px 0; margin-bottom:1.5rem }}
.obs h3 {{ color:#e65100; font-size:.95rem; margin-bottom:.5rem }}
.obs ul {{ padding-left:1.2rem; color:#555; font-size:.88rem; line-height:1.8 }}
.tcard {{ background:#fff; border-radius:12px; padding:1.5rem;
           box-shadow:0 2px 8px rgba(0,0,0,.08); overflow-x:auto; margin-bottom:1.5rem }}
.tcard h2 {{ font-size:1.1rem; margin-bottom:1rem; color:#333 }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem }}
th {{ background:#e8eaf6; color:#3949ab; padding:.65rem .9rem; text-align:left; white-space:nowrap; font-weight:600 }}
td {{ padding:.55rem .9rem; border-bottom:1px solid #f0f0f0 }}
tr:hover td {{ background:#f5f5f5 }}
.note {{ color:#888; font-size:.78rem; margin-top:.8rem; line-height:1.6 }}
.badge {{ display:inline-block; padding:2px 10px; border-radius:12px;
           font-size:.72rem; color:#fff; background:#3949ab; vertical-align:middle }}
footer {{ text-align:center; padding:2rem; color:#aaa; font-size:.78rem }}
</style>
</head>
<body>
<header>
  <h1>PVAD-SE Pipeline — newval_v3 評估報告</h1>
  <p>Speakers 4 &amp; 7 &middot; 混音比例 0.3~1.5 &middot; 環境噪音 classroom &middot;
     Threshold 0.25 &middot; {time.strftime("%Y-%m-%d %H:%M")}</p>
</header>

<div class="wrap">

<!-- KPI -->
<div class="kpi">
  <div class="card"><div class="val">{overall["f1"]}</div><div class="lbl">平均 F1</div></div>
  <div class="card"><div class="val">{overall["precision"]}</div><div class="lbl">平均 Precision</div></div>
  <div class="card"><div class="val">{overall["recall"]}</div><div class="lbl">平均 Recall</div></div>
  <div class="card"><div class="val">{overall_rtf}x</div><div class="lbl">平均 RTF (CPU only)</div></div>
</div>

<!-- Observations -->
<div class="obs">
  <h3>&#9888; 觀察與分析</h3>
  <ul>
    <li><b>Mean Sim 偏低（0.09~0.12 vs threshold=0.25）</b>：pipeline 幾乎不預測目標說話者，Recall ≈ 0</li>
    <li><b>Session mismatch</b>：enrollment 為 session-1/2，test 音檔為 session-7；不同錄音環境導致 d-vector 距離增大</li>
    <li><b>建議調低 threshold</b>：改為 0.05~0.10 可改善 Recall，或改用 personal_vad.onnx（LSTM 逐幀判斷，不依賴 d-vector 距離）</li>
    <li><b>SNR=1.0 表現最差</b>：噪音與說話者等響（0 dB），pipeline 完全無法辨識目標說話者</li>
    <li><b>RTF 7~16x（CPU）</b>：WeSpeaker ResNet34 每幀提取 embedding 計算量大；離線用途可接受，即時需 GPU</li>
  </ul>
</div>

<!-- Charts -->
<div class="charts">
  <div class="chart-card"><h3>F1 vs 混音比例 (ratio)</h3><canvas id="cR"></canvas></div>
  <div class="chart-card"><h3>F1 vs 環境噪音 SNR</h3><canvas id="cS"></canvas></div>
  <div class="chart-card"><h3>Mean Similarity vs Threshold</h3><canvas id="cSim"></canvas></div>
</div>

<!-- Table -->
<div class="tcard">
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
    Threshold = {THRESHOLD} &nbsp;|&nbsp;
    Enrollment: spk4 &rarr; session-1 files, spk7 &rarr; session-2 files（不與 test 集重疊）<br>
    Labels: 0=靜音 &nbsp;1=目標說話者 &nbsp;2=同性別噪音 &nbsp;3=異性別噪音
    &nbsp;|&nbsp; 環境噪音: classroom_chatter.wav
  </p>
</div>

</div><!-- .wrap -->

<footer>
  PVAD-SE Pipeline &middot;
  <a href="https://github.com/rezhboyu/pvad-se-pipeline" target="_blank">github.com/rezhboyu/pvad-se-pipeline</a>
  &middot; branch: feature/testset-v3-generation
</footer>

<script>
const opts = (ylabel, max) => ({{
  responsive: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    y: {{ min: 0, max: max || 1, title: {{ display: true, text: ylabel }} }},
    x: {{}}
  }}
}});
const bc = vals => vals.map(v =>
  v >= 0.7 ? "rgba(76,175,80,.8)" : v >= 0.4 ? "rgba(255,152,0,.8)" : "rgba(244,67,54,.8)"
);

new Chart(document.getElementById("cR"), {{ type:"bar", data:{{
  labels: {json.dumps(r_l)},
  datasets: [{{ data: {json.dumps(r_d)}, backgroundColor: bc({json.dumps(r_d)}), borderRadius: 4 }}]
}}, options: opts("F1") }});

new Chart(document.getElementById("cS"), {{ type:"bar", data:{{
  labels: {json.dumps(s_l)},
  datasets: [{{ data: {json.dumps(s_d)}, backgroundColor: bc({json.dumps(s_d)}), borderRadius: 4 }}]
}}, options: opts("F1") }});

new Chart(document.getElementById("cSim"), {{ type:"bar", data:{{
  labels: {json.dumps(sim_lbls)},
  datasets: [
    {{ label:"Mean Similarity", data:{json.dumps(sim_vals)}, backgroundColor:"rgba(57,73,171,.7)", borderRadius:4 }},
    {{ label:"Threshold (0.25)", data:[{THRESHOLD},{THRESHOLD},{THRESHOLD},{THRESHOLD}],
       type:"line", borderColor:"#f44336", borderDash:[5,5], pointRadius:0, borderWidth:2 }}
  ]
}}, options:{{
  responsive: true,
  plugins: {{ legend: {{ display: true }} }},
  scales: {{
    y: {{ min:0, max:0.4, title:{{ display:true, text:"Similarity" }} }},
    x: {{}}
  }}
}} }});
</script>
</body>
</html>"""

(REPO / "eval_report.html").write_text(html, encoding="utf-8")
(REPO / "eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("Done: eval_report.html + eval_results.json")
