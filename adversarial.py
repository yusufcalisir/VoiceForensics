"""
Adversarial Testing & Robustness Evaluation Suite
==================================================
Exposes failure modes, stability issues, and decision inconsistencies
under controlled perturbations and challenging conditions.

Uses the existing core pipeline (ECAPA-TDNN, VAD, embeddings).
All perturbations are deterministic (seeded) for reproducibility.
"""

import numpy as np
import torch
import warnings
from itertools import combinations
import core
from translations import t


# ---------------------------------------------------------------------------
# 1 & 3. Per-file adversarial perturbation battery
# ---------------------------------------------------------------------------

def _apply_perturbations(waveform, sr=16000):
    """
    Return list of (label, perturbed_waveform) tuples.
    Covers gain, noise, reverb, crop, time-shift.
    """
    results = [("original", waveform)]

    # Gain +3 dB / -3 dB / +6 dB
    results.append(("gain_+3dB", waveform * (10 ** (3 / 20))))
    results.append(("gain_-3dB", waveform * (10 ** (-3 / 20))))
    results.append(("gain_+6dB", waveform * (10 ** (6 / 20))))

    # Additive white noise at different SNRs
    for snr in [30, 20, 15]:
        rms = torch.sqrt(torch.mean(waveform ** 2)) + 1e-10
        noise_rms = rms / (10 ** (snr / 20))
        torch.manual_seed(42)
        noise = torch.randn_like(waveform) * noise_rms
        results.append((f"noise_{snr}dB", waveform + noise))

    # Slight echo (simple delay-and-add)
    delay_samples = int(0.03 * sr)  # 30ms echo
    if waveform.shape[-1] > delay_samples:
        echo = torch.zeros_like(waveform)
        echo[:, delay_samples:] = waveform[:, :-delay_samples] * 0.3
        results.append(("echo_30ms", waveform + echo))

    # Crop: first 5%, last 5%, first 10%
    n_samples = waveform.shape[-1]
    crop5 = max(int(n_samples * 0.05), 1)
    crop10 = max(int(n_samples * 0.10), 1)
    results.append(("crop_start_5%", waveform[:, crop5:]))
    results.append(("crop_end_5%", waveform[:, :-crop5]))
    results.append(("crop_start_10%", waveform[:, crop10:]))

    # Time shift (circular shift by 50ms)
    shift = int(0.05 * sr)
    if n_samples > shift:
        shifted = torch.cat([waveform[:, shift:], waveform[:, :shift]], dim=-1)
        results.append(("time_shift_50ms", shifted))

    # Minor time shift (10ms)
    shift_small = int(0.01 * sr)
    if n_samples > shift_small:
        shifted_s = torch.cat([waveform[:, shift_small:], waveform[:, :shift_small]], dim=-1)
        results.append(("time_shift_10ms", shifted_s))

    return results


def run_perturbation_battery(file_path):
    """
    Run full perturbation battery on a single file.
    Returns dict with per-perturbation similarity to original, plus
    aggregate stability metrics.
    """
    waveform, _ = core.get_audio_tensor(file_path)
    base_emb = core.extract_embeddings(waveform)
    if base_emb is None:
        return None

    perturbations = _apply_perturbations(waveform)
    results = []
    for label, wav_p in perturbations:
        if label == "original":
            continue
        emb = core.extract_embeddings(wav_p)
        if emb is not None:
            sim = core.compute_similarity(base_emb, emb)
            results.append({"perturbation": label, "similarity": round(sim, 5)})

    if not results:
        return None

    sims = [r["similarity"] for r in results]
    return {
        "file": file_path,
        "details": results,
        "mean": float(np.mean(sims)),
        "variance": float(np.var(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "n_perturbations": len(results),
    }


# ---------------------------------------------------------------------------
# 4. Segment robustness (begin / middle / end)
# ---------------------------------------------------------------------------

def run_segment_test(file_path):
    """
    Split audio into 3 equal segments, compute embeddings, measure
    pairwise consistency.  Returns per-segment sims and flags.
    """
    waveform, _ = core.get_audio_tensor(file_path)
    n = waveform.shape[-1]
    third = n // 3
    if third < int(16000 * 0.5):
        return None  # segments too short

    segments = {
        "begin":  waveform[:, :third],
        "middle": waveform[:, third:2*third],
        "end":    waveform[:, 2*third:],
    }
    embs = {}
    for name, seg in segments.items():
        e = core.extract_embeddings(seg)
        if e is None:
            return None
        embs[name] = e

    pairs = [("begin", "middle"), ("begin", "end"), ("middle", "end")]
    sims = {}
    for a, b in pairs:
        sims[f"{a}-{b}"] = round(core.compute_similarity(embs[a], embs[b]), 5)

    vals = list(sims.values())
    flag = max(vals) - min(vals) > 0.15  # significant inconsistency
    return {
        "file": file_path,
        "segment_similarities": sims,
        "mean": float(np.mean(vals)),
        "spread": float(max(vals) - min(vals)),
        "flagged": flag,
    }


# ---------------------------------------------------------------------------
# 6. Identical vs slightly modified
# ---------------------------------------------------------------------------

def run_identity_test(file_path):
    """
    Compare original with very minor modifications (tiny noise, 10ms shift).
    Expect near-identical similarity (>0.99).  Flag if not.
    """
    waveform, _ = core.get_audio_tensor(file_path)
    base_emb = core.extract_embeddings(waveform)
    if base_emb is None:
        return None

    results = []

    # Tiny noise (50 dB SNR — nearly inaudible)
    rms = torch.sqrt(torch.mean(waveform ** 2)) + 1e-10
    noise_rms = rms / (10 ** (50 / 20))
    torch.manual_seed(99)
    noisy = waveform + torch.randn_like(waveform) * noise_rms
    e = core.extract_embeddings(noisy)
    if e is not None:
        sim = core.compute_similarity(base_emb, e)
        results.append({"modification": "noise_50dB", "similarity": round(sim, 5), "expected": ">0.99", "ok": sim > 0.99})

    # 10ms shift
    shift = int(0.01 * 16000)
    if waveform.shape[-1] > shift:
        shifted = torch.cat([waveform[:, shift:], waveform[:, :shift]], dim=-1)
        e = core.extract_embeddings(shifted)
        if e is not None:
            sim = core.compute_similarity(base_emb, e)
            results.append({"modification": "shift_10ms", "similarity": round(sim, 5), "expected": ">0.99", "ok": sim > 0.99})

    # Gain +0.5 dB (barely noticeable)
    slight_gain = waveform * (10 ** (0.5 / 20))
    e = core.extract_embeddings(slight_gain)
    if e is not None:
        sim = core.compute_similarity(base_emb, e)
        results.append({"modification": "gain_+0.5dB", "similarity": round(sim, 5), "expected": ">0.99", "ok": sim > 0.99})

    n_failed = sum(1 for r in results if not r["ok"])
    return {
        "file": file_path,
        "tests": results,
        "all_passed": n_failed == 0,
        "n_failed": n_failed,
    }


# ---------------------------------------------------------------------------
# 2. Hard negative finder
# ---------------------------------------------------------------------------

def find_hard_negatives(file_names, sim_matrix, labels_mapped, top_k=5):
    """
    Among different-speaker pairs, find those with the highest similarity
    (i.e., most likely to fool the system).
    """
    candidates = []
    n = len(file_names)
    for i in range(n):
        for j in range(i + 1, n):
            la = labels_mapped.get(file_names[i], "Unknown")
            lb = labels_mapped.get(file_names[j], "Unknown")
            if la == "Unknown" or lb == "Unknown":
                continue
            if la != lb:
                candidates.append({
                    "file_a": file_names[i],
                    "file_b": file_names[j],
                    "label_a": la,
                    "label_b": lb,
                    "similarity": round(float(sim_matrix[i][j]), 5),
                })
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# 7. Confusion set evaluator
# ---------------------------------------------------------------------------

def evaluate_confusion_sets(file_names, sim_matrix, labels_mapped, threshold):
    """
    Group files by speaker.  For each speaker group, measure:
    - intra-group mean similarity (should be high)
    - inter-group mean similarity with each other group (should be low)
    Highlight ambiguous groups where separation is weak.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for i, fn in enumerate(file_names):
        label = labels_mapped.get(fn, "Unknown")
        if label != "Unknown":
            groups[label].append(i)

    if len(groups) < 2:
        return None

    report = []
    group_labels = sorted(groups.keys())
    for gl in group_labels:
        idxs = groups[gl]
        # Intra-group
        intra_sims = []
        for a, b in combinations(idxs, 2):
            intra_sims.append(float(sim_matrix[a][b]))
        intra_mean = float(np.mean(intra_sims)) if intra_sims else 1.0

        # Inter-group: mean sim with each other group
        inter = {}
        for other_gl in group_labels:
            if other_gl == gl:
                continue
            other_idxs = groups[other_gl]
            cross = [float(sim_matrix[a][b]) for a in idxs for b in other_idxs]
            inter[other_gl] = float(np.mean(cross)) if cross else 0.0

        worst_inter_label = max(inter, key=inter.get) if inter else None
        worst_inter_val = inter[worst_inter_label] if worst_inter_label else 0.0
        separation = intra_mean - worst_inter_val

        report.append({
            "speaker": gl,
            "n_files": len(idxs),
            "intra_mean": round(intra_mean, 4),
            "worst_inter_speaker": worst_inter_label,
            "worst_inter_sim": round(worst_inter_val, 4),
            "separation": round(separation, 4),
            "ambiguous": separation < 0.15,
        })

    return report


# ---------------------------------------------------------------------------
# 5. Cross-comparison inconsistency scorer
# ---------------------------------------------------------------------------

def compute_inconsistency_score(sim_matrix, file_names, threshold):
    """
    Quantify overall matrix consistency.  Count transitivity violations
    and compute a 0-1 inconsistency score.
    """
    n = len(file_names)
    total_triples = 0
    violation_count = 0
    low_bound = max(0.3, threshold - 0.2)

    for i, j, k in combinations(range(n), 3):
        total_triples += 1
        for p1, p2, p3 in [(i, j, k), (j, k, i), (k, i, j)]:
            if (sim_matrix[p1][p2] >= threshold
                    and sim_matrix[p2][p3] >= threshold
                    and sim_matrix[p1][p3] < low_bound):
                violation_count += 1
                break  # count each triple at most once

    score = violation_count / max(total_triples, 1)
    return {
        "inconsistency_score": round(score, 4),
        "violation_count": violation_count,
        "total_triples": total_triples,
    }


# ---------------------------------------------------------------------------
# 8 & 10. Full adversarial report generator
# ---------------------------------------------------------------------------

def generate_full_report(file_names, file_paths, sim_matrix, file_metrics,
                         labels_mapped, threshold, embeddings, lang="en"):
    """
    Run all adversarial tests and return a structured report dict.
    """
    report = {
        "perturbation_tests": [],
        "segment_tests": [],
        "identity_tests": [],
        "hard_negatives": [],
        "confusion_sets": None,
        "inconsistency": None,
        "borderline_cases": [],
        "failure_log": [],
    }

    # Per-file tests
    for fn in file_names:
        fp = file_paths.get(fn)
        if fp is None:
            continue

        # Perturbation battery
        pt = run_perturbation_battery(fp)
        if pt:
            pt["file"] = fn
            report["perturbation_tests"].append(pt)
            if pt["variance"] > 1e-3:
                report["failure_log"].append({
                    "test": "perturbation",
                    "file": fn,
                    "issue": t("adv_issue_var", lang, var=pt["variance"], min_sim=pt["min"]),
                })

        # Segment test
        st_res = run_segment_test(fp)
        if st_res:
            st_res["file"] = fn
            report["segment_tests"].append(st_res)
            if st_res["flagged"]:
                report["failure_log"].append({
                    "test": "segment_consistency",
                    "file": fn,
                    "issue": t("adv_issue_spread", lang, spread=st_res["spread"]),
                })

        # Identity test
        id_res = run_identity_test(fp)
        if id_res:
            id_res["file"] = fn
            report["identity_tests"].append(id_res)
            if not id_res["all_passed"]:
                report["failure_log"].append({
                    "test": "identity_stability",
                    "file": fn,
                    "issue": t("adv_issue_ident", lang, n=id_res["n_failed"]),
                })

    # Cross-file tests (require labels)
    has_labels = any(v != "Unknown" for v in labels_mapped.values())
    if has_labels:
        report["hard_negatives"] = find_hard_negatives(
            file_names, sim_matrix, labels_mapped, top_k=10)
        report["confusion_sets"] = evaluate_confusion_sets(
            file_names, sim_matrix, labels_mapped, threshold)

    # Inconsistency
    report["inconsistency"] = compute_inconsistency_score(
        sim_matrix, file_names, threshold)

    # Borderline cases (scores within ±0.05 of threshold)
    n = len(file_names)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i][j])
            if abs(s - threshold) <= 0.05:
                report["borderline_cases"].append({
                    "file_a": file_names[i],
                    "file_b": file_names[j],
                    "score": round(s, 4),
                    "distance_from_threshold": round(abs(s - threshold), 4),
                })

    report["borderline_cases"].sort(key=lambda x: x["distance_from_threshold"])

    return report
