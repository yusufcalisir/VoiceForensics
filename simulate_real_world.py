"""
Real-world audio degradation simulator.
Produces multiple variants of each clean audio file:
  1. Additive white noise (+20dB SNR)
  2. Bitrate degradation (MP3 64kbps round-trip)
  3. Resampling to 8kHz and back to 16kHz
  4. Resampling to 44.1kHz and back to 16kHz
  5. Mild reverberation (simple convolution with synthetic IR)
  6. Combined: noise + reverb + MP3 compression
"""
import os
import numpy as np
import soundfile as sf

def add_white_noise(audio, snr_db=20):
    """Add white noise at a given SNR in dB."""
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, len(audio))
    return audio + noise

def add_background_noise(audio, snr_db=15):
    """Add simulated background noise (brownian noise) at SNR."""
    rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-10
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.cumsum(np.random.normal(0, 1, len(audio)))
    noise = noise - np.mean(noise)
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-10
    noise = noise * (rms_noise / noise_rms)
    return audio + noise

def mp3_roundtrip(audio, sr, bitrate="64k"):
    """Compress to MP3 and decode back to WAV using pydub (requires ffmpeg)."""
    try:
        from pydub import AudioSegment
        import io, tempfile

        # Write to a temporary WAV
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, audio, sr)
        tmp_wav.close()

        seg = AudioSegment.from_wav(tmp_wav.name)
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_mp3.close()
        seg.export(tmp_mp3.name, format="mp3", bitrate=bitrate)

        decoded = AudioSegment.from_mp3(tmp_mp3.name)
        decoded = decoded.set_frame_rate(sr).set_channels(1)

        samples = np.array(decoded.get_array_of_samples(), dtype=np.float64)
        samples = samples / (2**15)  # normalize int16 -> float

        os.unlink(tmp_wav.name)
        os.unlink(tmp_mp3.name)
        return samples
    except Exception as e:
        print(f"  MP3 roundtrip skipped (ffmpeg missing?): {e}")
        return None

def resample_roundtrip(audio, orig_sr, target_sr):
    """Resample to target_sr then back to orig_sr using scipy."""
    from scipy.signal import resample
    n_target = int(len(audio) * target_sr / orig_sr)
    downsampled = resample(audio, n_target)
    n_back = int(len(downsampled) * orig_sr / target_sr)
    return resample(downsampled, n_back)

def add_reverb(audio, sr, rt60_ms=300):
    """Apply mild reverberation using a synthetic exponential decay impulse response."""
    ir_len = int(sr * rt60_ms / 1000)
    t = np.arange(ir_len) / sr
    # Exponential decay impulse response
    ir = np.random.randn(ir_len) * np.exp(-6.9 * t / (rt60_ms / 1000))
    ir[0] = 1.0  # direct path
    ir = ir / np.sqrt(np.sum(ir ** 2))  # energy normalize

    convolved = np.convolve(audio, ir, mode='full')[:len(audio)]
    # Normalize to original peak
    peak_orig = np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 1.0
    peak_conv = np.max(np.abs(convolved)) if np.max(np.abs(convolved)) > 0 else 1.0
    return convolved * (peak_orig / peak_conv)

def simulate_real_world(input_folder, output_folder):
    """Walk input_folder, apply all degradations, write results to output_folder."""
    os.makedirs(output_folder, exist_ok=True)
    count = 0

    for root, _, files in os.walk(input_folder):
        for file in files:
            if not file.endswith(('.flac', '.wav')):
                continue

            in_path = os.path.join(root, file)
            audio, sr = sf.read(in_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            rel_base = os.path.relpath(root, input_folder)
            out_dir = os.path.join(output_folder, rel_base)
            os.makedirs(out_dir, exist_ok=True)

            base = os.path.splitext(file)[0]

            # 0. Clean reference
            sf.write(os.path.join(out_dir, f"{base}_CLEAN.wav"), audio, sr)

            # 1. White noise
            noisy = add_white_noise(audio, snr_db=20)
            sf.write(os.path.join(out_dir, f"{base}_NOISE20dB.wav"), noisy, sr)

            # 1b. Background noise
            bg_noisy = add_background_noise(audio, snr_db=15)
            sf.write(os.path.join(out_dir, f"{base}_BGNOISE15dB.wav"), bg_noisy, sr)

            # 2. MP3 64kbps roundtrip
            mp3_out = mp3_roundtrip(audio, sr, bitrate="64k")
            if mp3_out is not None:
                sf.write(os.path.join(out_dir, f"{base}_MP3_64k.wav"), mp3_out, sr)

            # 3. Resample 8kHz roundtrip
            rs8k = resample_roundtrip(audio, sr, 8000)
            sf.write(os.path.join(out_dir, f"{base}_RESAMPLE_8k.wav"), rs8k, sr)

            # 4. Resample 44.1kHz roundtrip
            rs44k = resample_roundtrip(audio, sr, 44100)
            sf.write(os.path.join(out_dir, f"{base}_RESAMPLE_44k.wav"), rs44k, sr)

            # 5. Mild reverb
            reverbed = add_reverb(audio, sr, rt60_ms=300)
            sf.write(os.path.join(out_dir, f"{base}_REVERB.wav"), reverbed, sr)

            # 6. Combined: noise + reverb + MP3
            combined = add_reverb(add_white_noise(audio, snr_db=25), sr, rt60_ms=200)
            combined_mp3 = mp3_roundtrip(combined, sr, bitrate="64k")
            if combined_mp3 is not None:
                sf.write(os.path.join(out_dir, f"{base}_COMBINED.wav"), combined_mp3, sr)
            else:
                sf.write(os.path.join(out_dir, f"{base}_COMBINED.wav"), combined, sr)

            count += 1

    print(f"Processed {count} files into {output_folder}/")

if __name__ == "__main__":
    if os.path.exists("sample_dataset"):
        simulate_real_world("sample_dataset", "mixed_quality_dataset")
    else:
        print("No sample_dataset/ found. Run extract_libri.py first.")


# ---------------------------------------------------------------------------
# Real-world validation suite
# ---------------------------------------------------------------------------

def run_validation_suite(input_folder, core_module, n_repeats=3):
    """
    For each speaker folder, process CLEAN vs each degradation variant.
    Repeats n_repeats× with different random seeds for noise-based
    degradations.  Returns a structured results dict and prints a
    consistency report.
    """
    import json

    CONDITIONS = [
        ("NOISE20dB",    lambda a, sr: add_white_noise(a, snr_db=20)),
        ("BGNOISE15dB",  lambda a, sr: add_background_noise(a, snr_db=15)),
        ("RESAMPLE_8k",  lambda a, sr: resample_roundtrip(a, sr, 8000)),
        ("RESAMPLE_44k", lambda a, sr: resample_roundtrip(a, sr, 44100)),
        ("REVERB",       lambda a, sr: add_reverb(a, sr, rt60_ms=300)),
        ("MP3_64k",      lambda a, sr: mp3_roundtrip(a, sr, bitrate="64k")),
    ]

    results = {}  # condition -> list of similarity scores across repeats

    for cond_name, _ in CONDITIONS:
        results[cond_name] = []

    source_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.flac', '.wav')):
                source_files.append(os.path.join(root, file))

    if not source_files:
        print("No source files found.")
        return results

    import tempfile

    for repeat in range(n_repeats):
        np.random.seed(42 + repeat)
        for src_path in source_files:
            audio, sr = sf.read(src_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Get clean embedding
            tmp_clean = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp_clean.name, audio, sr)
            tmp_clean.close()
            clean_res = core_module.process_audio(tmp_clean.name)
            clean_emb = clean_res.get("embeddings")
            os.unlink(tmp_clean.name)

            if clean_emb is None:
                continue

            for cond_name, transform_fn in CONDITIONS:
                degraded = transform_fn(audio, sr)
                if degraded is None:
                    continue

                tmp_deg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_deg.name, degraded, sr)
                tmp_deg.close()
                deg_res = core_module.process_audio(tmp_deg.name)
                deg_emb = deg_res.get("embeddings")
                os.unlink(tmp_deg.name)

                if deg_emb is not None:
                    sim = core_module.compute_similarity(clean_emb, deg_emb)
                    results[cond_name].append(sim)

    # Print consistency report
    print("\n" + "=" * 60)
    print("REAL-WORLD VALIDATION REPORT")
    print(f"Source files: {len(source_files)}, Repeats: {n_repeats}")
    print("=" * 60)
    print(f"{'Condition':<18} {'Mean Sim':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'N':>5}")
    print("-" * 63)
    for cond_name, scores in results.items():
        if scores:
            arr = np.array(scores)
            print(f"{cond_name:<18} {arr.mean():>10.4f} {arr.std():>10.4f} {arr.min():>10.4f} {arr.max():>10.4f} {len(scores):>5}")
        else:
            print(f"{cond_name:<18} {'N/A':>10}")
    print("=" * 60)

    return results
