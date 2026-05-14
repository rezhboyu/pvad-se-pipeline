import os
import sys
import json
import random
import math
import time
from pydub import AudioSegment

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

# ─── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ALLVOICE_BASE = os.path.join(SCRIPT_DIR, "all_voice")
NOISE_DIR     = os.path.join(SCRIPT_DIR, "0429")
OUTPUT_BASE   = os.path.join(SCRIPT_DIR, "newval_v3")

# FFmpeg: try sibling of all_voice first, then PVAD root
for _ffdir in [os.path.join(ALLVOICE_BASE, "FFmpeg"),
               os.path.join(SCRIPT_DIR, "FFmpeg")]:
    if os.path.exists(_ffdir):
        AudioSegment.converter = os.path.join(_ffdir, "ffmpeg")
        os.environ["PATH"] += os.pathsep + _ffdir
        break

# ─── config ────────────────────────────────────────────────────────────────────
MAIN_SPEAKERS = {
    4: {
        "reg_session":  "1",          # session 1 → registration (excluded from test)
        "test_session": "7",          # session 7 → used as main audio in mixing
        "same_gender":  [5, 10, 13, 14],   # label 2
        "diff_gender":  [2, 3, 6],         # label 3
    },
    7: {
        "reg_session":  "2",
        "test_session": "7",
        "same_gender":  [2, 3, 6, 15],    # label 2
        "diff_gender":  [1, 5, 10],        # label 3
    },
}

RATIOS     = [0.3, 0.5, 1.0, 1.2, 1.5]
SNR_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]

NOISE_FILES = {
    "classroom": "classroom_chatter.wav",
    "fan":       "fan_noise.wav",
    "cafeteria": "cafeteria_babble.wav",
    "crowd":     "crowd_laughter.wav",
}

# ─── helpers ───────────────────────────────────────────────────────────────────
def get_session(filename):
    """Extract session number from filename: {UUID}_{name}_{session}.wav"""
    return os.path.splitext(filename)[0].split("_")[-1]


def loop_to_length(audio, target_ms):
    """Tile audio until it reaches target_ms, then trim."""
    if len(audio) == 0 or target_ms <= 0:
        return audio
    result = audio
    while len(result) < target_ms:
        result += audio
    return result[:target_ms]


def snr_scale(noise_audio, signal_audio, snr_ratio):
    """Scale noise so its RMS level = snr_ratio × signal RMS level."""
    if noise_audio.rms == 0 or signal_audio.rms == 0 or snr_ratio <= 0:
        return noise_audio
    target_dbfs = signal_audio.dBFS + 20 * math.log10(snr_ratio)
    if not math.isfinite(target_dbfs) or not math.isfinite(noise_audio.dBFS):
        return noise_audio
    return noise_audio.apply_gain(target_dbfs - noise_audio.dBFS)


def build_secondary_pool(cfg, m_voice_dur, ratio):
    """
    Build a shuffled pool of secondary audio clips.

    Splits target duration equally between same-gender (label 2) and
    diff-gender (label 3) groups, then further evenly across speakers.
    """
    groups = [
        (2, cfg["same_gender"]),
        (3, cfg["diff_gender"]),
    ]
    target_per_group = (m_voice_dur * ratio) / 2
    all_selected = []

    for label, spk_ids in groups:
        target_per_spk = target_per_group / max(len(spk_ids), 1)
        group_clips = []

        for s_id in spk_ids:
            v_dir = os.path.join(ALLVOICE_BASE, "all_voice", str(s_id))
            j_dir = os.path.join(ALLVOICE_BASE, "all_voice_dia", str(s_id))
            if not os.path.exists(v_dir):
                continue

            wav_files = [f for f in os.listdir(v_dir) if f.lower().endswith(".wav")]
            segs = []
            for w in wav_files:
                j_path = os.path.join(j_dir, os.path.splitext(w)[0] + ".json")
                if not os.path.exists(j_path):
                    continue
                try:
                    audio = AudioSegment.from_file(os.path.join(v_dir, w))
                    with open(j_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for seg in data.get("segments", []):
                        s, e = int(seg["start"] * 1000), int(seg["end"] * 1000)
                        if e > s:
                            segs.append({
                                "audio": audio[s:e],
                                "label": label,
                                "source_file": w,
                                "source_folder": s_id,
                            })
                except Exception:
                    continue

            random.shuffle(segs)
            tmp_ms = 0
            for seg in segs:
                if tmp_ms >= target_per_spk:
                    break
                group_clips.append(seg)
                tmp_ms += len(seg["audio"])

        all_selected.extend(group_clips)

    random.shuffle(all_selected)
    return all_selected


# ─── mixing mode A: probabilistic interleaving ─────────────────────────────────
def mix_interleaved(main_audio, m_segs, secondary_pool, ref_dbfs):
    final = AudioSegment.empty()
    segs_out = []

    def add_clip(clip, label):
        nonlocal final
        if label >= 2 and clip.rms > 0:
            clip = clip.apply_gain(ref_dbfs - clip.dBFS)
        t0 = round(len(final) / 1000.0, 3)
        final += clip
        t1 = round(len(final) / 1000.0, 3)
        if t1 > t0:
            segs_out.append({"start": t0, "end": t1, "label": label})

    last_ms = 0
    sec_idx = 0
    num_main = len(m_segs)
    num_sec  = len(secondary_pool)

    for i, seg in enumerate(m_segs):
        s, e = int(seg["start"] * 1000), int(seg["end"] * 1000)
        if s > last_ms:
            add_clip(main_audio[last_ms:s], 0)
        add_clip(main_audio[s:e], 1)
        last_ms = e

        rem_main = num_main - i
        rem_sec  = num_sec - sec_idx
        if rem_main > 0 and rem_sec > 0:
            prob = rem_sec / rem_main
            while random.random() < prob and sec_idx < num_sec:
                p = secondary_pool[sec_idx]
                add_clip(p["audio"], p["label"])
                sec_idx += 1
                prob *= 0.5

    while sec_idx < num_sec:
        p = secondary_pool[sec_idx]
        add_clip(p["audio"], p["label"])
        sec_idx += 1

    if last_ms < len(main_audio):
        add_clip(main_audio[last_ms:], 0)

    return final.normalize(), segs_out


# ─── mixing mode B: three-part random segment insertion ────────────────────────
def mix_three_segment(main_audio, m_segs, secondary_pool, ref_dbfs):
    # SNR-normalize secondary clips
    normed = []
    for p in secondary_pool:
        clip = p["audio"]
        if clip.rms > 0:
            clip = clip.apply_gain(ref_dbfs - clip.dBFS)
        normed.append((clip, p["label"]))

    # Build main audio timeline as list of (clip, label)
    main_timeline = []
    last_ms = 0
    for seg in m_segs:
        s, e = int(seg["start"] * 1000), int(seg["end"] * 1000)
        if s > last_ms:
            main_timeline.append((main_audio[last_ms:s], 0))
        main_timeline.append((main_audio[s:e], 1))
        last_ms = e
    if last_ms < len(main_audio):
        main_timeline.append((main_audio[last_ms:], 0))

    # If no secondary clips, return main as-is
    if not normed:
        final = AudioSegment.empty()
        segs_out = []
        for clip, label in main_timeline:
            t0 = round(len(final) / 1000.0, 3)
            final += clip
            t1 = round(len(final) / 1000.0, 3)
            if t1 > t0:
                segs_out.append({"start": t0, "end": t1, "label": label})
        return final.normalize(), segs_out

    # Concatenate all secondary clips; track cumulative boundaries for label assignment
    sec_audio = AudioSegment.empty()
    boundaries = []  # (start_ms, end_ms, label)
    for clip, lbl in normed:
        t0 = len(sec_audio)
        sec_audio += clip
        boundaries.append((t0, len(sec_audio), lbl))

    total_ms = len(sec_audio)

    def dominant_label(start, end):
        """Return the label that covers the most time in [start, end]."""
        counts = {}
        for seg_s, seg_e, lbl in boundaries:
            overlap = max(0, min(end, seg_e) - max(start, seg_s))
            counts[lbl] = counts.get(lbl, 0) + overlap
        return max(counts, key=counts.get) if counts else 2

    # Split into 3 parts at two random cut points
    if total_ms >= 3:
        cut1 = random.randint(1, total_ms - 2)
        cut2 = random.randint(cut1 + 1, total_ms - 1)
    else:
        cut1, cut2 = total_ms // 3, 2 * total_ms // 3

    sec_parts = [
        (sec_audio[:cut1],        dominant_label(0, cut1)),
        (sec_audio[cut1:cut2],    dominant_label(cut1, cut2)),
        (sec_audio[cut2:],        dominant_label(cut2, total_ms)),
    ]
    sec_parts = [(clip, lbl) for clip, lbl in sec_parts if len(clip) > 0]

    # Choose insertion slots (0 = before slot 0, 1 = before slot 1, ...)
    n_slots = len(main_timeline) + 1
    n_parts = len(sec_parts)
    if n_slots >= n_parts:
        insert_at = sorted(random.sample(range(n_slots), n_parts))
    else:
        insert_at = sorted(random.choices(range(n_slots), k=n_parts))

    # Map slot → list of (clip, label)
    insertions = {}
    for idx, slot in enumerate(insert_at):
        insertions.setdefault(slot, []).append(sec_parts[idx])

    # Assemble final audio
    final = AudioSegment.empty()
    segs_out = []

    def append_clip(clip, label):
        nonlocal final
        t0 = round(len(final) / 1000.0, 3)
        final += clip
        t1 = round(len(final) / 1000.0, 3)
        if t1 > t0:
            segs_out.append({"start": t0, "end": t1, "label": label})

    for idx in range(len(main_timeline) + 1):
        for ins_clip, ins_lbl in insertions.get(idx, []):
            append_clip(ins_clip, ins_lbl)
        if idx < len(main_timeline):
            append_clip(*main_timeline[idx])

    return final.normalize(), segs_out


# ─── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Pre-load noise files
    noise_cache = {}
    for n_name, n_file in NOISE_FILES.items():
        n_path = os.path.join(NOISE_DIR, n_file)
        if os.path.exists(n_path):
            noise_cache[n_name] = AudioSegment.from_file(n_path)
            print(f"✔ 噪音載入: {n_name} ({len(noise_cache[n_name])/1000:.1f}s)")
        else:
            print(f"⚠ 找不到噪音: {n_path}")

    manifest = {}
    total_exported = 0
    start_time = time.time()

    for main_id, cfg in MAIN_SPEAKERS.items():
        v_dir = os.path.join(ALLVOICE_BASE, "all_voice", str(main_id))
        j_dir = os.path.join(ALLVOICE_BASE, "all_voice_dia", str(main_id))

        all_wavs = sorted([f for f in os.listdir(v_dir) if f.lower().endswith(".wav")])
        reg_files  = [f for f in all_wavs if get_session(f) == cfg["reg_session"]]
        test_files = [f for f in all_wavs if get_session(f) == cfg["test_session"]]

        manifest[str(main_id)] = reg_files
        print(f"\n=== Speaker {main_id} ===")
        print(f"  Registration ({len(reg_files)}): {reg_files}")
        print(f"  Test ({len(test_files)}): {test_files}")

        for wav_name in test_files:
            wav_path  = os.path.join(v_dir, wav_name)
            json_path = os.path.join(j_dir, os.path.splitext(wav_name)[0] + ".json")
            if not os.path.exists(json_path):
                print(f"  ⚠ 缺少標註 JSON: {wav_name}")
                continue

            main_audio = AudioSegment.from_file(wav_path)
            with open(json_path, "r", encoding="utf-8") as f:
                m_segs = json.load(f).get("segments", [])

            # Reference dBFS from the first speech segment
            ref_dbfs = main_audio.dBFS
            if m_segs:
                first = m_segs[0]
                ref_clip = main_audio[int(first["start"] * 1000):int(first["end"] * 1000)]
                if ref_clip.rms > 0:
                    ref_dbfs = ref_clip.dBFS

            m_voice_dur = sum(int((s["end"] - s["start"]) * 1000) for s in m_segs)
            stem = os.path.splitext(wav_name)[0]

            for ratio in RATIOS:
                ratio_str = f"{ratio:.1f}".rstrip("0").rstrip(".")
                secondary_pool = build_secondary_pool(cfg, m_voice_dur, ratio)

                for mode_name, mix_fn in [("interleaved",   mix_interleaved),
                                           ("three_segment", mix_three_segment)]:
                    pool_copy = list(secondary_pool)
                    random.shuffle(pool_copy)
                    mixed_audio, mixed_segs = mix_fn(main_audio, m_segs, pool_copy, ref_dbfs)

                    for n_name, n_audio in noise_cache.items():
                        for snr_ratio in SNR_RATIOS:
                            snr_str = f"{snr_ratio:.1f}".rstrip("0").rstrip(".")

                            out_dir = os.path.join(
                                OUTPUT_BASE, mode_name,
                                f"ratio_{ratio_str}", f"spk{main_id}",
                                f"noise_{n_name}", f"snr_{snr_str}",
                            )
                            os.makedirs(out_dir, exist_ok=True)

                            looped_noise = loop_to_length(n_audio, len(mixed_audio))
                            scaled_noise = snr_scale(looped_noise, mixed_audio, snr_ratio)
                            final_audio  = mixed_audio.overlay(scaled_noise).normalize()

                            out_wav  = os.path.join(out_dir, f"{stem}_mixed.wav")
                            out_json = os.path.join(out_dir, f"{stem}_mixed.json")

                            final_audio.export(out_wav, format="wav")
                            with open(out_json, "w", encoding="utf-8") as jf:
                                json.dump({"segments": mixed_segs}, jf, indent=2)

                            total_exported += 1

                print(f"  [{total_exported:>4}/2000] spk{main_id} | {stem[-8:]}… "
                      f"| r={ratio_str} | {time.time()-start_time:.0f}s elapsed")

    # Save registration manifest
    manifest_path = os.path.join(OUTPUT_BASE, "registration_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n✅ 完成！共匯出 {total_exported} 個音檔，耗時 {elapsed/60:.1f} 分鐘。")
    print(f"   輸出目錄: {OUTPUT_BASE}")
    print(f"   Registration manifest: {manifest_path}")


if __name__ == "__main__":
    main()
