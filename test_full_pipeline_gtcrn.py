#!/usr/bin/env python3
"""
完整管線測試：GTCRN + CachedPVAD
==================================
用 MAT 真實語音 + 白噪音（SNR 10dB、0dB）跑完整管線：
1. GTCRN 離線降噪
2. GTCRN 串流降噪 + CachedPVAD + Soft Gating
3. 測量 RTF、延遲、降噪品質 (SNR improvement, PESQ, STOI)
4. 儲存所有音檔供試聽
"""

import os
import sys
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path

# 加入 project root
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.audio import SAMPLE_RATE, read_audio, write_audio
from utils.gtcrn_denoiser import GTCRNDenoiser, StreamingGTCRNDenoiser, GTCRN_HOP, GTCRN_NFFT
from utils.speaker_encoder import SpeakerEncoder, CachedPVAD
from utils.gating import SoftGate

MAT_DIR = Path("/sessions/sharp-stoic-albattani/mnt/MAT")
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "test_results_gtcrn"

SR = 16000


def add_noise(clean, snr_db):
    noise = np.random.randn(len(clean)).astype(np.float32)
    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    scale = np.sqrt(clean_power / (noise_power * 10**(snr_db/10)))
    noise = noise * scale
    return clean + noise, noise


def compute_snr(signal, noise):
    sig_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-20:
        return 100.0
    return 10 * np.log10(sig_power / noise_power)


def main():
    np.random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR / "audio", exist_ok=True)

    # ── 找語者 ──
    wav_files = sorted([f for f in os.listdir(MAT_DIR) if f.endswith('.wav')])
    speakers = {}
    for f in wav_files:
        if 'hsuan_7' in f and 'hsuan' not in speakers:
            speakers['hsuan'] = f
        elif '0911636193' in f and '0911636193' not in speakers:
            speakers['0911636193'] = f
        elif 'FEMH' in f and 'FEMH' not in speakers:
            speakers['FEMH'] = f

    # 找 enrollment 用的不同錄音（hsuan_2）
    enrollment_files = {}
    for f in wav_files:
        if 'hsuan_2' in f and 'hsuan' not in enrollment_files:
            enrollment_files['hsuan'] = f

    print(f"測試語者: {list(speakers.keys())}")
    print(f"測試檔案: {list(speakers.values())}")

    # ── 載入模型 ──
    print("\n載入模型...")
    gtcrn_denoiser = GTCRNDenoiser()
    ecapa_path = MODELS_DIR / "ecapa_tdnn" / "wespeaker_resnet34.onnx"
    speaker_encoder = SpeakerEncoder(str(ecapa_path))
    print("模型載入完成")

    snr_conditions = [10, 0]
    results = {}

    for spk_name, wav_file in speakers.items():
        print(f"\n{'='*60}")
        print(f"語者: {spk_name} - {wav_file}")

        clean = read_audio(str(MAT_DIR / wav_file))
        max_samples = 15 * SR
        if len(clean) > max_samples:
            clean = clean[:max_samples]

        duration = len(clean) / SR
        print(f"  長度: {duration:.2f}s")

        # Enrollment: 用同語者不同錄音，或用 clean 的前 3 秒
        if spk_name in enrollment_files:
            enroll_audio = read_audio(str(MAT_DIR / enrollment_files[spk_name]))
        else:
            enroll_audio = clean[:3 * SR]  # 用前 3 秒

        enrollment_dvector = speaker_encoder.extract_embedding(enroll_audio)
        print(f"  enrollment d-vector: {enrollment_dvector.shape}")

        results[spk_name] = {}

        for snr_db in snr_conditions:
            print(f"\n  --- SNR = {snr_db}dB ---")
            noisy, noise = add_noise(clean, snr_db)

            # 儲存 clean + noisy
            sf.write(str(OUTPUT_DIR / "audio" / f"{spk_name}_clean.wav"), clean, SR)
            sf.write(str(OUTPUT_DIR / "audio" / f"{spk_name}_noisy_{snr_db}dB.wav"), noisy, SR)

            # ── GTCRN 離線降噪 ──
            print("    GTCRN 離線降噪...")
            t0 = time.time()
            gtcrn_enhanced = gtcrn_denoiser.enhance(noisy)
            t_denoise = time.time() - t0
            denoise_rtf = t_denoise / duration
            sf.write(str(OUTPUT_DIR / "audio" / f"{spk_name}_gtcrn_{snr_db}dB.wav"), gtcrn_enhanced, SR)
            print(f"    降噪 RTF: {denoise_rtf:.4f}")

            # 計算降噪品質
            min_len = min(len(clean), len(gtcrn_enhanced), len(noisy))
            c, e, n = clean[:min_len], gtcrn_enhanced[:min_len], noisy[:min_len]
            snr_before = compute_snr(c, n - c)
            snr_after = compute_snr(c, e - c)
            snr_imp = snr_after - snr_before

            # PESQ + STOI
            try:
                from pesq import pesq
                pesq_noisy = pesq(SR, c, n, 'wb')
                pesq_enh = pesq(SR, c, e, 'wb')
            except Exception as ex:
                print(f"    PESQ 計算失敗: {ex}")
                pesq_noisy = pesq_enh = float('nan')

            try:
                from pystoi import stoi
                stoi_noisy = stoi(c, n, SR)
                stoi_enh = stoi(c, e, SR)
            except Exception as ex:
                print(f"    STOI 計算失敗: {ex}")
                stoi_noisy = stoi_enh = float('nan')

            print(f"    SNR improvement: {snr_imp:.1f}dB")
            print(f"    PESQ: {pesq_noisy:.3f} -> {pesq_enh:.3f}")
            print(f"    STOI: {stoi_noisy:.3f} -> {stoi_enh:.3f}")

            # ── 串流管線（GTCRN + CachedPVAD + Gating）──
            print("    串流管線（GTCRN + CachedPVAD + Gating）...")
            streaming_denoiser = StreamingGTCRNDenoiser()
            pvad = CachedPVAD(
                speaker_encoder=speaker_encoder,
                enrollment_dvector=enrollment_dvector,
                extract_interval=32,
                threshold=0.25,
            )
            gate = SoftGate(gain_floor=0.05, attack_ms=5.0, release_ms=50.0, hop=GTCRN_HOP)

            total_shifts = len(noisy) // GTCRN_HOP
            padded = np.zeros(total_shifts * GTCRN_HOP, dtype=np.float32)
            padded[:min(len(noisy), len(padded))] = noisy[:len(padded)]

            output_chunks = []
            frame_times = []
            similarities = []

            for i in range(total_shifts):
                t0 = time.perf_counter()
                chunk = padded[i * GTCRN_HOP:(i+1) * GTCRN_HOP]
                denoised = streaming_denoiser.process_shift(chunk)
                is_target, sim = pvad.process_frame(denoised)
                gated = gate.process(denoised, is_target, confidence=sim)
                output_chunks.append(gated)
                t1 = time.perf_counter()
                frame_times.append(t1 - t0)
                similarities.append(sim)

            pipeline_output = np.concatenate(output_chunks)[:len(noisy)]
            peak = np.max(np.abs(pipeline_output))
            if peak > 0.99:
                pipeline_output *= 0.99 / peak

            sf.write(str(OUTPUT_DIR / "audio" / f"{spk_name}_pipeline_{snr_db}dB.wav"), pipeline_output, SR)

            frame_ms = np.array(frame_times) * 1000
            shift_ms = GTCRN_HOP / SR * 1000
            pipeline_rtf = frame_ms.mean() / shift_ms
            sims = np.array(similarities)

            print(f"    管線 RTF: {pipeline_rtf:.4f}")
            print(f"    每幀平均: {frame_ms.mean():.2f}ms, P95: {np.percentile(frame_ms, 95):.2f}ms")
            print(f"    pVAD sim 平均: {sims.mean():.3f}, 目標活躍: {(sims > 0.25).mean():.1%}")

            results[spk_name][f"snr_{snr_db}dB"] = {
                'denoise_rtf': float(denoise_rtf),
                'snr_improvement': float(snr_imp),
                'pesq_noisy': float(pesq_noisy),
                'pesq_enhanced': float(pesq_enh),
                'stoi_noisy': float(stoi_noisy),
                'stoi_enhanced': float(stoi_enh),
                'pipeline_rtf': float(pipeline_rtf),
                'pipeline_avg_frame_ms': float(frame_ms.mean()),
                'pipeline_p95_frame_ms': float(np.percentile(frame_ms, 95)),
                'pvad_sim_mean': float(sims.mean()),
                'pvad_target_ratio': float((sims > 0.25).mean()),
            }

    # ── 儲存結果 ──
    with open(str(OUTPUT_DIR / "results.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 總結 ──
    print(f"\n{'='*80}")
    print("總結")
    print(f"{'='*80}")
    print(f"{'語者':<12} {'SNR':<10} {'SNR_imp':<10} {'PESQ':<12} {'STOI':<12} {'RTF':<10}")
    print("-" * 66)
    for spk in results:
        for cond in results[spk]:
            r = results[spk][cond]
            print(f"{spk:<12} {cond:<10} "
                  f"{r['snr_improvement']:>+7.1f}dB  "
                  f"{r['pesq_enhanced']:>5.3f}      "
                  f"{r['stoi_enhanced']:>5.3f}      "
                  f"{r['pipeline_rtf']:>6.4f}")

    print(f"\n所有音檔和結果: {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    results = main()
