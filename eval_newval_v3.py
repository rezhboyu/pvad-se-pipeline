#!/usr/bin/env python3
"""
Evaluate PVAD pipeline on the newval_v3 test set.

Samples representative test cases from newval_v3/:
  - 2 speakers (4, 7)
  - 5 ratios (0.3, 0.5, 1.0, 1.2, 1.5)
  - 2 modes (interleaved, three_segment)
  - 1 noise type (classroom)
  - 3 SNR levels (0.2, 0.6, 1.0)

Uses the corresponding registration file as enrollment audio.
Computes frame-level precision / recall / F1 against JSON ground-truth labels.

Outputs:
  - eval_output/  — enhanced WAV files
  - eval_results.json — all metrics
  - eval_report.html  — static HTML report
"""

import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path

# ─── paths ─────────────────────────────────────────────────────────────────────
REPO_DIR      = Path(__file__).resolve().parent
PVAD_DIR      = REPO_DIR.parent                     # C:\...\PVAD
ALLVOICE_BASE = PVAD_DIR / "all_voice"
NEWVAL_BASE   = PVAD_DIR / "newval_v3"
OUTPUT_DIR    = REPO_DIR / "eval_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── pipeline imports ──────────────────────────────────────────────────────────
from utils.audio import (
    SAMPLE_RATE, BLOCK_LEN, BLOCK_SHIFT,
    read_audio, write_audio,
    stft_frame, istft_frame, magnitude_phase, reconstruct_complex,
    frame_signal, overlap_add,
)
from utils.gating import SoftGate
from utils.speaker_encoder import SpeakerEncoder, SimplePVAD
from pipeline import DTLNDenoiser

MODELS_DIR = REPO_DIR / "models"

# ─── evaluation config ─────────────────────────────────────────────────────────
SPEAKERS     = [4, 7]
RATIOS       = ["0.3", "0.5", "1", "1.2", "1.5"]
MODES        = ["interleaved", "three_segment"]
NOISE        = "classroom"
SNRS         = ["0.2", "0.6", "1"]
THRESHOLD    = 0.25


def load_models():
    ecapa_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    if not ecapa_path.exists():
        ecapa_path = MODELS_DIR / "ecapa_tdnn" / "ecapa_tdnn.onnx"
    encoder  = SpeakerEncoder(str(ecapa_path))
    denoiser = DTLNDenoiser()
    gate     = SoftGate(gain_floor=0.05, attack_ms=5.0, release_ms=50.0)
    return encoder, denoiser, gate


def run_one(encoder, denoiser, gate, enrollment_path, mixed_path, out_path):
    """Run pipeline, return (similarities, output_audio)."""
    enroll_audio = read_audio(str(enrollment_path))
    dvector      = encoder.extract_embedding(enroll_audio)

    pvad   = SimplePVAD(encoder, dvector, threshold=THRESHOLD)
    frames = frame_signal(read_audio(str(mixed_path)))

    enhanced_frames = np.empty_like(frames)
    states_1 = states_2 = None
    similarities = []

    for i, frame in enumerate(frames):
        denoised, states_1, states_2 = denoiser.process_frame(frame, states_1, states_2)
        is_target, sim = pvad.process_frame(denoised)
        similarities.append(float(sim))
        enhanced_frames[i] = gate.process(denoised, is_target, confidence=sim)

    output_audio = overlap_add(enhanced_frames)
    peak = np.max(np.abs(output_audio))
    if peak > 0.99:
        output_audio = output_audio * 0.99 / peak

    write_audio(str(out_path), output_audio)
    return np.array(similarities)


def compute_metrics(similarities, json_path, audio_duration_s):
    """
    Compare frame-level predictions vs ground-truth labels.
    Label 1 = target speaker → positive class.
    """
    with open(json_path) as f:
        segs = json.load(f)["segments"]

    n_frames = len(similarities)
    frame_dur = BLOCK_SHIFT / SAMPLE_RATE      # seconds per frame hop

    gt = np.zeros(n_frames, dtype=int)
    for seg in segs:
        if seg["label"] == 1:
            start_f = int(seg["start"] / frame_dur)
            end_f   = int(seg["end"]   / frame_dur)
            gt[start_f:min(end_f, n_frames)] = 1

    pred = (similarities > THRESHOLD).astype(int)

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision":    round(precision, 4),
        "recall":       round(recall, 4),
        "f1":           round(f1, 4),
        "mean_sim":     round(float(np.mean(similarities)), 4),
        "std_sim":      round(float(np.std(similarities)), 4),
        "target_ratio": round(float(np.mean(pred)), 4),
        "gt_ratio":     round(float(np.mean(gt)), 4),
        "n_frames":     n_frames,
        "tp": tp, "fp": fp, "fn": fn,
    }


def get_registration_file(speaker_id):
    """Return first registration file for this speaker."""
    manifest_path = NEWVAL_BASE / "registration_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    reg_files = manifest[str(speaker_id)]
    if not reg_files:
        raise FileNotFoundError(f"No registration files for speaker {speaker_id}")
    wav_dir = ALLVOICE_BASE / "all_voice" / str(speaker_id)
    return wav_dir / reg_files[0]


def get_first_test_file(speaker_id, ratio, mode, noise, snr):
    """Return first WAV+JSON pair in the given condition directory."""
    cond_dir = NEWVAL_BASE / mode / f"ratio_{ratio}" / f"spk{speaker_id}" / f"noise_{noise}" / f"snr_{snr}"
    wavs = sorted(cond_dir.glob("*_mixed.wav"))
    if not wavs:
        return None, None
    wav = wavs[0]
    return wav, wav.with_suffix(".json")


def main():
    print("Loading models…")
    encoder, denoiser, gate = load_models()
    print("Models ready.\n")

    results = []
    total = len(SPEAKERS) * len(RATIOS) * len(MODES) * len(SNRS)
    done = 0

    for spk_id in SPEAKERS:
        enrollment = get_registration_file(spk_id)
        print(f"=== Speaker {spk_id}  enrollment: {enrollment.name} ===")

        for ratio in RATIOS:
            for mode in MODES:
                for snr in SNRS:
                    wav_path, json_path = get_first_test_file(spk_id, ratio, mode, NOISE, snr)
                    if wav_path is None:
                        print(f"  skip spk{spk_id} r={ratio} {mode} snr={snr}")
                        continue

                    out_name = f"spk{spk_id}_r{ratio}_{mode}_snr{snr}.wav"
                    out_path = OUTPUT_DIR / out_name

                    t0 = time.time()
                    sims = run_one(encoder, denoiser, gate, enrollment, wav_path, out_path)
                    elapsed = time.time() - t0

                    audio_dur = len(sims) * BLOCK_SHIFT / SAMPLE_RATE
                    metrics = compute_metrics(sims, json_path, audio_dur)
                    metrics["rtf"] = round(elapsed / audio_dur, 3)

                    record = {
                        "speaker":     spk_id,
                        "ratio":       ratio,
                        "mode":        mode,
                        "noise":       NOISE,
                        "snr":         snr,
                        "enrollment":  enrollment.name,
                        "mixed_file":  wav_path.name,
                        "output_file": out_name,
                        **metrics,
                    }
                    results.append(record)

                    done += 1
                    print(f"  [{done:>3}/{total}] r={ratio} {mode[:3]} snr={snr} "
                          f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                          f"RTF={metrics['rtf']:.2f}")

    # Save JSON
    results_path = REPO_DIR / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # Generate HTML report
    generate_html(results)
    print(f"Report generated: {REPO_DIR / 'eval_report.html'}")


# ─── HTML generation ───────────────────────────────────────────────────────────
def color_f1(v):
    if v >= 0.7:  return "#4caf50"
    if v >= 0.4:  return "#ff9800"
    return "#f44336"

def generate_html(results):
    # aggregate by ratio
    ratio_f1 = {}
    for r in results:
        ratio_f1.setdefault(r["ratio"], []).append(r["f1"])
    avg_f1_by_ratio = {k: round(sum(v)/len(v), 3) for k, v in ratio_f1.items()}

    snr_f1 = {}
    for r in results:
        snr_f1.setdefault(r["snr"], []).append(r["f1"])
    avg_f1_by_snr = {k: round(sum(v)/len(v), 3) for k, v in snr_f1.items()}

    mode_f1 = {}
    for r in results:
        mode_f1.setdefault(r["mode"], []).append(r["f1"])
    avg_f1_by_mode = {k: round(sum(v)/len(v), 3) for k, v in mode_f1.items()}

    # build table rows
    rows_html = ""
    for r in results:
        f1c = color_f1(r["f1"])
        rows_html += f"""
<tr>
  <td>spk{r['speaker']}</td>
  <td>{r['ratio']}</td>
  <td>{r['mode']}</td>
  <td>{r['snr']}</td>
  <td>{r['precision']:.3f}</td>
  <td>{r['recall']:.3f}</td>
  <td style="color:{f1c};font-weight:700">{r['f1']:.3f}</td>
  <td>{r['mean_sim']:.3f}</td>
  <td>{r['target_ratio']:.2%}</td>
  <td>{r['gt_ratio']:.2%}</td>
  <td>{r['rtf']:.2f}</td>
  <td><a href="eval_output/{r['output_file']}" target="_blank">▶</a></td>
</tr>"""

    # chart data
    ratio_labels  = sorted(avg_f1_by_ratio)
    ratio_values  = [avg_f1_by_ratio[k] for k in ratio_labels]
    snr_labels    = sorted(avg_f1_by_snr)
    snr_values    = [avg_f1_by_snr[k] for k in snr_labels]
    mode_labels   = sorted(avg_f1_by_mode)
    mode_values   = [avg_f1_by_mode[k] for k in mode_labels]

    overall_f1   = round(sum(r["f1"] for r in results) / len(results), 3)
    overall_prec = round(sum(r["precision"] for r in results) / len(results), 3)
    overall_rec  = round(sum(r["recall"] for r in results) / len(results), 3)
    overall_rtf  = round(sum(r["rtf"] for r in results) / len(results), 3)

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PVAD newval_v3 評估報告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Microsoft JhengHei", sans-serif; background: #f0f2f5; color: #222; }}
  header {{ background: #1a237e; color: #fff; padding: 1.5rem 2rem; }}
  header h1 {{ font-size: 1.6rem; }}
  header p  {{ color: #9fa8da; font-size: .9rem; margin-top: .3rem; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 1.5rem 1rem; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
  .kpi {{ background: #fff; border-radius: 10px; padding: 1.2rem 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.1); text-align: center; }}
  .kpi .val {{ font-size: 2.4rem; font-weight: 700; color: #1a237e; }}
  .kpi .lbl {{ color: #888; font-size: .85rem; margin-top: .2rem; }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
  .chart-card {{ background: #fff; border-radius: 10px; padding: 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  .chart-card h3 {{ font-size: .95rem; color: #555; margin-bottom: .8rem; }}
  .table-card {{ background: #fff; border-radius: 10px; padding: 1.2rem 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.1); overflow-x: auto; }}
  .table-card h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: #333; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .85rem; }}
  th {{ background: #e8eaf6; color: #3949ab; text-align: left; padding: .6rem .8rem; white-space: nowrap; }}
  td {{ padding: .5rem .8rem; border-bottom: 1px solid #f0f0f0; }}
  tr:hover td {{ background: #f5f5f5; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: .75rem; color: #fff; }}
  .note {{ color: #888; font-size: .8rem; margin-top: 1rem; }}
  footer {{ text-align: center; padding: 2rem; color: #aaa; font-size: .8rem; }}
</style>
</head>
<body>

<header>
  <h1>PVAD newval_v3 評估報告</h1>
  <p>Speaker 4 &amp; 7 × 5 ratios × 2 modes × 3 SNR levels — noise: {NOISE}</p>
</header>

<div class="container">

<!-- KPI cards -->
<div class="kpi-grid">
  <div class="kpi"><div class="val">{overall_f1}</div><div class="lbl">平均 F1 Score</div></div>
  <div class="kpi"><div class="val">{overall_prec}</div><div class="lbl">平均 Precision</div></div>
  <div class="kpi"><div class="val">{overall_rec}</div><div class="lbl">平均 Recall</div></div>
  <div class="kpi"><div class="val">{overall_rtf}×</div><div class="lbl">平均 RTF</div></div>
</div>

<!-- Charts -->
<div class="charts-grid">
  <div class="chart-card">
    <h3>F1 vs 混音比例 (ratio)</h3>
    <canvas id="chartRatio"></canvas>
  </div>
  <div class="chart-card">
    <h3>F1 vs 環境噪音 SNR</h3>
    <canvas id="chartSNR"></canvas>
  </div>
  <div class="chart-card">
    <h3>F1 vs 混音模式</h3>
    <canvas id="chartMode"></canvas>
  </div>
</div>

<!-- Detail table -->
<div class="table-card">
  <h2>詳細結果</h2>
  <table>
    <thead>
      <tr>
        <th>Speaker</th><th>Ratio</th><th>Mode</th><th>SNR</th>
        <th>Precision</th><th>Recall</th><th>F1</th>
        <th>Mean Sim</th><th>Pred Ratio</th><th>GT Ratio</th>
        <th>RTF</th><th>Audio</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <p class="note">
    Threshold = {THRESHOLD} | Noise source: {NOISE}_chatter.wav |
    Registration: per-speaker session-1 (spk4) / session-2 (spk7) files
  </p>
</div>

</div><!-- .container -->

<footer>PVAD-SE Pipeline · newval_v3 evaluation · {time.strftime('%Y-%m-%d %H:%M')}</footer>

<script>
const chartOpts = (label) => ({{
  responsive: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    y: {{ min: 0, max: 1, title: {{ display: true, text: 'F1 Score' }} }},
    x: {{ title: {{ display: true, text: label }} }},
  }}
}});

const barColor = (vals) => vals.map(v =>
  v >= 0.7 ? 'rgba(76,175,80,.8)' : v >= 0.4 ? 'rgba(255,152,0,.8)' : 'rgba(244,67,54,.8)'
);

new Chart(document.getElementById('chartRatio'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(ratio_labels)},
    datasets: [{{ data: {json.dumps(ratio_values)}, backgroundColor: barColor({json.dumps(ratio_values)}), borderRadius: 4 }}]
  }},
  options: chartOpts('混音比例')
}});

new Chart(document.getElementById('chartSNR'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(snr_labels)},
    datasets: [{{ data: {json.dumps(snr_values)}, backgroundColor: barColor({json.dumps(snr_values)}), borderRadius: 4 }}]
  }},
  options: chartOpts('SNR')
}});

new Chart(document.getElementById('chartMode'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(mode_labels)},
    datasets: [{{ data: {json.dumps(mode_values)}, backgroundColor: barColor({json.dumps(mode_values)}), borderRadius: 4 }}]
  }},
  options: chartOpts('模式')
}});
</script>
</body>
</html>"""

    out = REPO_DIR / "eval_report.html"
    out.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
