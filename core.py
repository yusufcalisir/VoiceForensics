import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

import numpy as np
import warnings
from translations import t
from itertools import combinations
import threading

# ---------------------------------------------------------------------------
# Model loading (cached natively to decouple from Streamlit)
# ---------------------------------------------------------------------------

_SILERO_MODEL = None
_SILERO_UTILS = None
_ECAPA_MODEL = None
_MODEL_LOCK = threading.Lock()

def load_silero_vad():
    global _SILERO_MODEL, _SILERO_UTILS
    if _SILERO_MODEL is None:
        with _MODEL_LOCK:
            if _SILERO_MODEL is None:
                _SILERO_MODEL, _SILERO_UTILS = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True,
                )
    return _SILERO_MODEL, _SILERO_UTILS


def load_ecapa_model():
    global _ECAPA_MODEL
    if _ECAPA_MODEL is None:
        with _MODEL_LOCK:
            if _ECAPA_MODEL is None:
                from speechbrain.inference.speaker import EncoderClassifier
                _ECAPA_MODEL = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                )
    return _ECAPA_MODEL

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def extract_embeddings(waveform):
    """Return 1-D numpy embedding or None if audio is too short."""
    if waveform.shape[-1] < int(16000 * 0.5):
        return None
    classifier = load_ecapa_model()
    with torch.no_grad():
        emb = classifier.encode_batch(waveform)
        return emb.squeeze().cpu().numpy()


def get_audio_tensor(file_path):
    """Load any audio file, convert to mono float32 @ 16 kHz."""
    import soundfile as sf
    wav_np, sample_rate = sf.read(file_path)

    if wav_np.ndim > 1:
        wav_np = np.mean(wav_np, axis=1)

    if sample_rate != 16000:
        import torchaudio
        wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
        wav_t = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wav_t)
        wav_np = wav_t.squeeze(0).numpy()

    waveform = torch.from_numpy(wav_np).float().unsqueeze(0)  # [1, T]
    return waveform, wav_np

# ---------------------------------------------------------------------------
# Full file processing
# ---------------------------------------------------------------------------

def process_audio(file_path):
    try:
        waveform, wav_np = get_audio_tensor(file_path)
    except Exception as e:
        return {"error": f"Failed to load audio: {e}"}

    silero_model, utils = load_silero_vad()
    get_speech_timestamps = utils[0]

    wav_tensor = waveform.squeeze(0)
    total_duration = wav_tensor.shape[0] / 16000.0

    rms = float(np.sqrt(np.mean(np.square(wav_np))))
    has_clipping = (np.sum(np.abs(wav_np) >= 0.99) / len(wav_np)) > 0.01
    low_volume = rms < 0.005

    # Estimate noise floor via lowest-energy 10% of frames
    frame_len = int(0.025 * 16000)  # 25ms frames
    n_frames = max(1, len(wav_np) // frame_len)
    frame_energies = []
    for fi in range(n_frames):
        frame = wav_np[fi * frame_len:(fi + 1) * frame_len]
        frame_energies.append(float(np.sqrt(np.mean(np.square(frame)) + 1e-12)))
    frame_energies.sort()
    noise_floor = float(np.mean(frame_energies[:max(1, len(frame_energies) // 10)]))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            speech_timestamps = get_speech_timestamps(wav_tensor, silero_model, sampling_rate=16000)
        effective_speech_duration = sum((ts['end'] - ts['start']) for ts in speech_timestamps) / 16000.0
    except Exception:
        effective_speech_duration = total_duration
        speech_timestamps = []

    speech_ratio = effective_speech_duration / total_duration if total_duration > 0 else 0.0
    short_audio_warning = effective_speech_duration < 3.0

    emb = extract_embeddings(waveform)

    return {
        "error": None,
        "embeddings": emb,
        "total_duration": total_duration,
        "effective_speech_duration": effective_speech_duration,
        "speech_ratio": speech_ratio,
        "rms": rms,
        "noise_floor": noise_floor,
        "has_clipping": has_clipping,
        "low_volume": bool(low_volume),
        "short_audio_warning": bool(short_audio_warning),
        "speech_timestamps": speech_timestamps,
    }

# ---------------------------------------------------------------------------
# Multi-segment robustness  (>=3 segments, each >=1.5 s speech)
# ---------------------------------------------------------------------------

def process_robust_segmentation(file_path):
    """
    Extract the 3 longest VAD-validated speech segments (each >= 1.5 s).
    Compute pairwise similarity among them.
    Return mean similarity or None if fewer than 3 valid segments.
    """
    try:
        waveform, _ = get_audio_tensor(file_path)
        silero_model, utils = load_silero_vad()
        get_speech_timestamps = utils[0]
        wav_tensor = waveform.squeeze(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stamps = get_speech_timestamps(wav_tensor, silero_model, sampling_rate=16000)

        MIN_SAMPLES = int(16000 * 1.5)
        valid = [s for s in stamps if (s['end'] - s['start']) >= MIN_SAMPLES]
        valid.sort(key=lambda s: s['end'] - s['start'], reverse=True)

        if len(valid) < 3:
            return None  # not enough valid segments

        chunks = [wav_tensor[s['start']:s['end']].unsqueeze(0) for s in valid]
        embs = [extract_embeddings(c) for c in chunks]

        if any(e is None for e in embs):
            return None

        sims = []
        n_chunks = len(embs)
        for i in range(n_chunks):
            for j in range(i + 1, n_chunks):
                sims.append(compute_similarity(embs[i], embs[j]))
        return float(np.mean(sims))
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Perturbation stability scoring
# ---------------------------------------------------------------------------

def compute_stability_score(file_path):
    """
    Apply small perturbations (gain +/-3 dB, slight crop) and measure
    how much the embedding changes.  Return a 0-1 stability score
    (mean cosine similarity across perturbations).
    """
    try:
        waveform, _ = get_audio_tensor(file_path)
        base_emb = extract_embeddings(waveform)
        if base_emb is None:
            return None

        perturbed_embs = []

        # Gain +3 dB
        gain_up = waveform * (10 ** (3 / 20))  # ~1.41x
        e = extract_embeddings(gain_up)
        if e is not None:
            perturbed_embs.append(e)

        # Gain -3 dB
        gain_down = waveform * (10 ** (-3 / 20))  # ~0.71x
        e = extract_embeddings(gain_down)
        if e is not None:
            perturbed_embs.append(e)

        # Crop first 5 %
        crop_len = max(int(waveform.shape[1] * 0.05), 1)
        e = extract_embeddings(waveform[:, crop_len:])
        if e is not None:
            perturbed_embs.append(e)

        # Crop last 5 %
        e = extract_embeddings(waveform[:, :-crop_len])
        if e is not None:
            perturbed_embs.append(e)

        # Additive noise (30 dB SNR)
        rms_val = torch.sqrt(torch.mean(waveform ** 2)) + 1e-10
        noise_rms = rms_val / (10 ** (30 / 20))
        torch.manual_seed(42)
        noisy = waveform + torch.randn_like(waveform) * noise_rms
        e = extract_embeddings(noisy)
        if e is not None:
            perturbed_embs.append(e)

        if not perturbed_embs:
            return None

        sims = [compute_similarity(base_emb, pe) for pe in perturbed_embs]
        return {
            "score": float(np.mean(sims)),
            "variance": float(np.var(sims))
        }
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Similarity / transitivity
# ---------------------------------------------------------------------------

def compute_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def calculate_transitivity(matrix, files, ui_strong_sim_threshold=0.70):
    n = len(files)
    penalties = {i: 0.0 for i in range(n)}
    violations = []
    low_bound = max(0.4, ui_strong_sim_threshold - 0.2)

    for i, j, k in combinations(range(n), 3):
        for p1, p2, p3 in [(i, j, k), (j, k, i), (k, i, j)]:
            if (matrix[p1][p2] >= ui_strong_sim_threshold
                    and matrix[p2][p3] >= ui_strong_sim_threshold
                    and matrix[p1][p3] < low_bound):
                penalties[p1] += 0.1
                penalties[p2] += 0.1
                penalties[p3] += 0.1
                violations.append(
                    f"`{files[p1]}` ≈ `{files[p2]}` and `{files[p2]}` ≈ `{files[p3]}`, "
                    f"BUT `{files[p1]}` ≠ `{files[p3]}`"
                )

    for i in penalties:
        penalties[i] = min(penalties[i], 0.3)
    return penalties, violations


def get_interpretation(score, high, low):
    if score >= high:
        return "High Similarity"
    elif score <= low:
        return "Low Similarity"
    return "Medium / Uncertain"


# ---------------------------------------------------------------------------
# Data-derived EER (thresholds = sorted unique scores, no fixed grid)
# ---------------------------------------------------------------------------

def compute_eer(same_scores, diff_scores):
    """
    Sweep thresholds derived from the actual similarity scores.
    Returns (eer_value, best_threshold, tpr_at_thresh, fpr_at_thresh).
    """
    all_scores = sorted(set(same_scores + diff_scores))

    # Insert midpoints for finer granularity
    thresholds = []
    for i in range(len(all_scores)):
        thresholds.append(all_scores[i])
        if i + 1 < len(all_scores):
            thresholds.append((all_scores[i] + all_scores[i + 1]) / 2.0)

    if len(thresholds) < 200:
        lo = min(all_scores) - 0.05
        hi = max(all_scores) + 0.05
        thresholds = list(set(thresholds) | set(np.linspace(lo, hi, 200).tolist()))
    thresholds = sorted(thresholds)

    best_gap = float('inf')
    best_t = thresholds[0]

    for t in thresholds:
        tpr = sum(1 for s in same_scores if s >= t) / len(same_scores)
        fpr = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = 1.0 - tpr
        gap = abs(frr - fpr)
        if gap < best_gap:
            best_gap = gap
            best_t = t

    final_tpr = sum(1 for s in same_scores if s >= best_t) / len(same_scores)
    final_fpr = sum(1 for s in diff_scores if s >= best_t) / len(diff_scores)
    eer = ((1.0 - final_tpr) + final_fpr) / 2.0

    return eer, best_t, final_tpr, final_fpr


def compute_roc_data(same_scores, diff_scores, n_points=100):
    """
    Generate TPR and FPR points for an ROC curve, plus AUC calculation.
    """
    all_scores = sorted(same_scores + diff_scores)
    thresholds = np.linspace(min(all_scores)-0.01, max(all_scores)+0.01, n_points)
    
    tpr_list, fpr_list = [], []
    for t in thresholds:
        tpr = sum(1 for s in same_scores if s >= t) / len(same_scores)
        fpr = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    # Calculate AUC using trapezoidal rule
    # Sort by FPR for integration
    points = sorted(zip(fpr_list, tpr_list))
    fpr_sorted, tpr_sorted = zip(*points)
    auc = np.trapz(tpr_sorted, fpr_sorted)
    
    return fpr_list, tpr_list, float(auc)


# ===========================================================================
# SCORE CALIBRATION  (Platt scaling + Isotonic + Sigmoid fallback)
# ===========================================================================

class ScoreCalibrator:
    """Map raw cosine similarity scores to calibrated probabilities.
    
    When trained with labeled data, uses Platt scaling (logistic regression)
    or isotonic regression.  When no labels are available, falls back to a
    sigmoid approximation centered on the cosine score distribution.
    """

    def __init__(self):
        self._platt = None
        self._isotonic = None
        self._fitted = False
        # Sigmoid fallback parameters
        self._sigmoid_center = 0.50
        self._sigmoid_scale = 8.0  # steepness

    def fit(self, same_scores, diff_scores):
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        X = np.array(same_scores + diff_scores).reshape(-1, 1)
        y = np.array([1] * len(same_scores) + [0] * len(diff_scores))

        # Platt scaling
        self._platt = LogisticRegression(solver="lbfgs", max_iter=1000)
        self._platt.fit(X, y)

        # Isotonic regression
        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(X.ravel(), y.astype(float))

        self._fitted = True

    def fit_sigmoid_fallback(self, all_scores):
        """Estimate sigmoid parameters from unlabeled score distribution."""
        if len(all_scores) > 0:
            self._sigmoid_center = float(np.median(all_scores))
            score_range = float(np.percentile(all_scores, 90) - np.percentile(all_scores, 10))
            if score_range > 0:
                self._sigmoid_scale = 4.0 / score_range  # maps 10th-90th pct → ~0.12 to ~0.88

    def _sigmoid(self, score):
        """Logistic sigmoid fallback."""
        z = self._sigmoid_scale * (score - self._sigmoid_center)
        return float(1.0 / (1.0 + np.exp(-z)))

    @property
    def is_fitted(self):
        return self._fitted

    def predict_proba(self, score, method="auto"):
        """Return calibrated probability for a single score.
        
        method: 'platt', 'isotonic', 'sigmoid', or 'auto'
          'auto' uses platt if fitted, else sigmoid fallback
        """
        if method == "auto":
            method = "platt" if self._fitted else "sigmoid"

        if method == "sigmoid" or not self._fitted:
            return self._sigmoid(score)

        x = np.array([[score]])
        if method == "isotonic":
            return float(self._isotonic.predict(x.ravel())[0])
        return float(self._platt.predict_proba(x)[0, 1])

    def predict_proba_batch(self, scores, method="auto"):
        """Return calibrated probabilities for a list of scores."""
        if method == "auto":
            method = "platt" if self._fitted else "sigmoid"

        if method == "sigmoid" or not self._fitted:
            return [self._sigmoid(s) for s in scores]

        x = np.array(scores).reshape(-1, 1)
        if method == "isotonic":
            return [float(v) for v in self._isotonic.predict(x.ravel())]
        return [float(v) for v in self._platt.predict_proba(x)[:, 1]]

    def ece(self, same_scores, diff_scores, n_bins=10):
        """Expected Calibration Error."""
        all_scores = same_scores + diff_scores
        true_labels = [1] * len(same_scores) + [0] * len(diff_scores)
        probs = self.predict_proba_batch(all_scores, method="platt")

        bins = np.linspace(0, 1, n_bins + 1)
        total_ece = 0.0
        for b in range(n_bins):
            mask = [(p >= bins[b] and p < bins[b + 1]) for p in probs]
            n_bin = sum(mask)
            if n_bin == 0:
                continue
            avg_conf = np.mean([p for p, m in zip(probs, mask) if m])
            avg_acc = np.mean([t for t, m in zip(true_labels, mask) if m])
            total_ece += (n_bin / len(probs)) * abs(avg_acc - avg_conf)
        return float(total_ece)

    def reliability_data(self, same_scores, diff_scores, n_bins=10):
        """Return (bin_midpoints, observed_freq, bin_counts) for reliability diagram."""
        all_scores = same_scores + diff_scores
        true_labels = [1] * len(same_scores) + [0] * len(diff_scores)
        probs = self.predict_proba_batch(all_scores, method="platt")

        bins = np.linspace(0, 1, n_bins + 1)
        midpoints, observed, counts = [], [], []
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            mask = [(p >= lo and p < hi) for p in probs]
            n_bin = sum(mask)
            midpoints.append((lo + hi) / 2)
            counts.append(n_bin)
            if n_bin > 0:
                observed.append(float(np.mean([t for t, m in zip(true_labels, mask) if m])))
            else:
                observed.append(0.0)
        return midpoints, observed, counts


# ===========================================================================
# DECISION BOUNDARY LEARNING  (logistic regression on scores)
# ===========================================================================

class DecisionBoundaryLearner:
    """Simple logistic regression on similarity scores as an alternative to fixed threshold."""

    def __init__(self):
        self._model = None
        self._fitted = False

    def fit(self, same_scores, diff_scores):
        from sklearn.linear_model import LogisticRegression
        X = np.array(same_scores + diff_scores).reshape(-1, 1)
        y = np.array([1] * len(same_scores) + [0] * len(diff_scores))
        self._model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, score):
        if not self._fitted:
            return None
        return int(self._model.predict(np.array([[score]]))[0])

    def predict_proba(self, score):
        if not self._fitted:
            return None
        return float(self._model.predict_proba(np.array([[score]]))[0, 1])

    def learned_threshold(self):
        """The implicit threshold where P(same)=0.5."""
        if not self._fitted:
            return None
        return float(-self._model.intercept_[0] / self._model.coef_[0, 0])

    def compare_with_fixed(self, same_scores, diff_scores, fixed_thresh):
        """Return dict with accuracy & F1 for both fixed and learned."""
        all_scores = same_scores + diff_scores
        y_true = [1] * len(same_scores) + [0] * len(diff_scores)

        fixed_preds = [1 if s >= fixed_thresh else 0 for s in all_scores]
        learned_preds = [self.predict(s) for s in all_scores] if self._fitted else fixed_preds

        from sklearn.metrics import accuracy_score, f1_score
        return {
            "fixed": {
                "accuracy": float(accuracy_score(y_true, fixed_preds)),
                "f1": float(f1_score(y_true, fixed_preds, zero_division=0)),
                "threshold": float(fixed_thresh),
            },
            "learned": {
                "accuracy": float(accuracy_score(y_true, learned_preds)),
                "f1": float(f1_score(y_true, learned_preds, zero_division=0)),
                "threshold": self.learned_threshold(),
            },
        }


# ===========================================================================
# DECISION ENGINE — tri-state logic with stability penalties
# ===========================================================================

class DecisionEngine:
    """Produces structured tri-state decisions from calibrated probabilities.
    
    Decision boundaries:
        p < 0.60            → "different"
        0.60 ≤ p ≤ 0.80    → "uncertain"
        p > 0.80            → "same"
    
    When stability is low the uncertain zone widens, preventing the system
    from forcing confident decisions on shaky evidence.
    """

    DIFF_BOUNDARY = 0.60
    SAME_BOUNDARY = 0.80

    def decide(self, calibrated_prob, confidence=1.0, stability_score=1.0):
        """Return structured decision dict."""
        # Compute effective boundaries — widen uncertain zone if unstable
        stability_penalty = max(0.0, 0.90 - stability_score) * 0.5 if stability_score < 0.90 else 0.0
        eff_diff = self.DIFF_BOUNDARY + stability_penalty   # push up → harder to say "different"
        eff_same = self.SAME_BOUNDARY - stability_penalty    # push down → harder to say "same" (but never below eff_diff)
        eff_same = max(eff_same, eff_diff + 0.05)            # ensure there's always an uncertain band

        # Low confidence also widens uncertain zone
        if confidence < 0.50:
            conf_penalty = (0.50 - confidence) * 0.10
            eff_diff += conf_penalty
            eff_same -= conf_penalty
            eff_same = max(eff_same, eff_diff + 0.05)

        if calibrated_prob > eff_same:
            decision = "same"
        elif calibrated_prob < eff_diff:
            decision = "different"
        else:
            decision = "uncertain"

        return {
            "decision": decision,
            "calibrated_probability": round(calibrated_prob, 4),
            "effective_boundaries": {
                "different_below": round(eff_diff, 4),
                "same_above": round(eff_same, 4),
            },
            "boundary_adjustments": {
                "stability_penalty": round(stability_penalty, 4),
                "stability_score_used": round(stability_score, 4),
                "confidence_used": round(confidence, 4),
            },
        }


# ===========================================================================
# CONFIDENCE MODEL — multi-factor, deterministic
# ===========================================================================

class ConfidenceModel:
    """Compute comparison confidence from multiple measurable factors.
    
    Factors and weights:
        duration       (0.30) — effective speech duration of shorter file
        speech_ratio   (0.15) — speech-to-total ratio of worse file
        stability      (0.30) — perturbation stability score
        inconsistency  (0.15) — transitivity violation penalty
        quality        (0.10) — audio quality (clipping, volume, noise)
    
    Confidence strictly decreases when:
        - audio is short
        - stability is low
        - inconsistency is high
    """

    WEIGHTS = {
        "duration": 0.30,
        "speech_ratio": 0.15,
        "stability": 0.30,
        "inconsistency": 0.15,
        "quality": 0.10,
    }

    def compute(self, metrics_a, metrics_b,
                stability_score=None, stability_variance=None,
                inconsistency_penalty=0.0):
        """Return confidence value in [0, 1] and per-factor breakdown."""

        # --- Duration factor ---
        eff_dur = min(metrics_a['effective_speech_duration'],
                      metrics_b['effective_speech_duration'])
        if eff_dur >= 10.0:
            dur_f = 1.0
        elif eff_dur <= 1.0:
            dur_f = 0.30
        else:
            dur_f = 0.30 + 0.70 * ((eff_dur - 1.0) / 9.0)

        # --- Speech ratio factor ---
        s_ratio = min(metrics_a['speech_ratio'], metrics_b['speech_ratio'])
        ratio_f = min(1.0, s_ratio / 0.50)

        # --- Stability factor ---
        if stability_score is not None:
            if stability_score >= 0.95:
                stab_f = 1.0
            elif stability_score <= 0.80:
                stab_f = 0.30
            else:
                stab_f = 0.30 + 0.70 * ((stability_score - 0.80) / 0.15)
            # Extra penalty for high variance
            if stability_variance is not None and stability_variance > 1e-3:
                stab_f *= 0.80
        else:
            stab_f = 0.70  # unknown stability → moderate penalty

        # --- Inconsistency factor ---
        incon_f = max(0.30, 1.0 - min(inconsistency_penalty, 0.70))

        # --- Quality factor ---
        quality_penalties = 0.0
        if metrics_a.get('low_volume') or metrics_b.get('low_volume'):
            quality_penalties += 0.25
        if metrics_a.get('has_clipping') or metrics_b.get('has_clipping'):
            quality_penalties += 0.25
        if metrics_a.get('short_audio_warning') or metrics_b.get('short_audio_warning'):
            quality_penalties += 0.20
        # Noise floor penalty
        noise_a = metrics_a.get('noise_floor', 0.0)
        noise_b = metrics_b.get('noise_floor', 0.0)
        max_noise = max(noise_a, noise_b)
        if max_noise > 0.01:
            quality_penalties += min(0.20, (max_noise - 0.01) * 10.0)
        qual_f = max(0.20, 1.0 - quality_penalties)

        # --- Weighted geometric mean ---
        factors = {
            "duration": dur_f,
            "speech_ratio": ratio_f,
            "stability": stab_f,
            "inconsistency": incon_f,
            "quality": qual_f,
        }

        log_sum = sum(self.WEIGHTS[k] * np.log(max(factors[k], 1e-6))
                       for k in self.WEIGHTS)
        confidence = float(np.exp(log_sum))
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        return {
            "confidence": confidence,
            "factors": {k: round(v, 4) for k, v in factors.items()},
            "min_duration": round(eff_dur, 2),
            "min_speech_ratio": round(s_ratio, 3),
        }


# ===========================================================================
# RISK SCORER
# ===========================================================================

class RiskScorer:
    """Compute risk score from confidence, stability, duration, inconsistency.
    
    Risk levels:
        < 0.30  → "low"
        0.30–0.60 → "medium"
        > 0.60  → "high"
    """

    def compute(self, confidence, stability_score=None, stability_variance=None,
                min_duration=10.0, inconsistency_penalty=0.0, lang="en"):
        """Return risk level, score, and contributing factors."""
        factors = []
        risk_terms = []

        # Low confidence
        if confidence < 0.80:
            penalty = (0.80 - confidence) * 0.50
            risk_terms.append(penalty)
            if confidence < 0.50:
                factors.append(t("very_low_conf", lang, conf=confidence))
            else:
                factors.append(t("mod_conf", lang, conf=confidence))

        # High instability
        if stability_score is not None and stability_score < 0.90:
            penalty = (0.90 - stability_score) * 0.80
            risk_terms.append(penalty)
            factors.append(t("low_pert_stab", lang, stab=stability_score))

        if stability_variance is not None and stability_variance > 5e-4:
            penalty = min(0.20, stability_variance * 200.0)
            risk_terms.append(penalty)
            factors.append(t("high_stab_var", lang, var=stability_variance))

        # Short duration
        if min_duration < 3.0:
            penalty = (3.0 - min_duration) / 3.0 * 0.30
            risk_terms.append(penalty)
            factors.append(t("short_audio_fact", lang, dur=min_duration))

        # Inconsistency
        if inconsistency_penalty > 0.05:
            penalty = min(0.25, inconsistency_penalty * 0.50)
            risk_terms.append(penalty)
            factors.append(t("trans_incon", lang, pen=inconsistency_penalty))

        risk_score = float(min(1.0, sum(risk_terms)))

        if risk_score < 0.30:
            level = "low"
        elif risk_score <= 0.60:
            level = "medium"
        else:
            level = "high"

        return {
            "level": level,
            "score": round(risk_score, 4),
            "factors": factors if factors else [t("no_risk_fact", lang)],
        }


# ===========================================================================
# EXPLANATION ENGINE
# ===========================================================================

class ExplanationEngine:
    """Generate human-readable explanations for speaker comparison decisions."""

    def explain(self, decision_result, confidence_result, risk_result,
                metrics_a, metrics_b, stability_info,
                raw_score, calibrated_prob, file_a="File A", file_b="File B", lang="en"):
        """Return structured explanation dict with natural language text."""

        decision = decision_result["decision"]
        prob = decision_result["calibrated_probability"]
        boundaries = decision_result["effective_boundaries"]
        conf = confidence_result["confidence"]
        conf_factors = confidence_result["factors"]
        risk_level = risk_result["level"]
        risk_factors = risk_result["factors"]

        # --- Decision reason ---
        if decision == "same":
            decision_reason = t("dec_reason_same", lang, prob=prob, same_above=boundaries["same_above"], file_a=file_a, file_b=file_b)
        elif decision == "different":
            decision_reason = t("dec_reason_diff", lang, prob=prob, different_below=boundaries["different_below"], file_a=file_a, file_b=file_b)
        else:
            decision_reason = t("dec_reason_uncert", lang, prob=prob, different_below=boundaries["different_below"], same_above=boundaries["same_above"], file_a=file_a, file_b=file_b)

        # --- Contributing factors ---
        contributing = []
        # Sort factors by value to identify strongest/weakest
        sorted_factors = sorted(conf_factors.items(), key=lambda x: x[1])
        weakest = sorted_factors[0]
        strongest = sorted_factors[-1]

        contributing.append(t("strongest_fact", lang, fact=strongest[0], val=strongest[1]))
        contributing.append(t("weakest_fact", lang, fact=weakest[0], val=weakest[1]))

        if conf_factors["duration"] < 0.60:
            contributing.append(t("short_audio_lim", lang, min_dur=confidence_result["min_duration"]))
        if conf_factors["stability"] < 0.70:
            contributing.append(t("pert_stab_low", lang))
        if conf_factors["inconsistency"] < 0.80:
            contributing.append(t("trans_viol", lang))

        # --- Uncertainty reasons ---
        uncertainty_reasons = []
        if decision == "uncertain":
            uncertainty_reasons.append(t("prob_ambig", lang, prob=prob))
        if conf < 0.50:
            uncertainty_reasons.append(t("overall_conf_low", lang, conf=conf))
        if risk_level == "high":
            uncertainty_reasons.append(t("risk_high_mult", lang))

        adj = decision_result.get("boundary_adjustments", {})
        if adj.get("stability_penalty", 0) > 0.01:
            uncertainty_reasons.append(t("bound_widened", lang, pen=adj["stability_penalty"]))

        # --- Warnings ---
        warn_list = []
        dur_a = metrics_a.get("effective_speech_duration", 0)
        dur_b = metrics_b.get("effective_speech_duration", 0)
        if dur_a < 3.0:
            warn_list.append(t("file_short_speech", lang, file=file_a, dur=dur_a))
        if dur_b < 3.0:
            warn_list.append(t("file_short_speech", lang, file=file_b, dur=dur_b))
        if metrics_a.get("has_clipping"):
            warn_list.append(t("file_clip", lang, file=file_a))
        if metrics_b.get("has_clipping"):
            warn_list.append(t("file_clip", lang, file=file_b))
        if metrics_a.get("low_volume"):
            warn_list.append(t("file_low_vol", lang, file=file_a))
        if metrics_b.get("low_volume"):
            warn_list.append(t("file_low_vol", lang, file=file_b))

        if stability_info:
            if stability_info.get("variance", 0) > 1e-3:
                warn_list.append(t("high_pert_var", lang, var=stability_info["variance"]))

        # --- Potential failure reasons ---
        failure_reasons = []
        if raw_score > 0.40 and raw_score < 0.70:
            failure_reasons.append(
                t("raw_sim_ambig", lang))
        if conf_factors["quality"] < 0.60:
            failure_reasons.append(
                t("poor_audio", lang))
        if conf_factors["duration"] < 0.50:
            failure_reasons.append(
                t("insuf_speech", lang))

        return {
            "decision_reason": decision_reason,
            "contributing_factors": contributing,
            "uncertainty_reasons": uncertainty_reasons if uncertainty_reasons else [t("no_uncert_fact", lang)],
            "warnings": warn_list if warn_list else [t("no_warn", lang)],
            "potential_failure_reasons": failure_reasons if failure_reasons else [t("no_fail_risks", lang)],
            "summary": self._build_summary(decision, prob, conf, risk_level, lang),
        }

    def _build_summary(self, decision, prob, conf, risk_level, lang="en"):
        """One-sentence executive summary."""
        risk_desc = {"low": t("low_risk", lang), "medium": t("mod_risk", lang), "high": t("high_risk", lang)}
        if decision == "same":
            return t("summ_same", lang, prob=prob, conf=conf, risk=risk_desc[risk_level])
        elif decision == "different":
            return t("summ_diff", lang, prob=prob, conf=conf, risk=risk_desc[risk_level])
        else:
            return t("summ_uncert", lang, prob=prob, conf=conf, risk=risk_desc[risk_level])


# ===========================================================================
# ERROR ANALYSIS (enhanced with warnings & failure reasons)
# ===========================================================================

def build_error_report(file_names, sim_matrix, labels_mapped, file_metrics,
                       threshold, calibrator=None):
    """
    Identify false positives and false negatives at the given threshold.
    Enhanced with diagnostic warnings and potential failure reasons.
    """
    confidence_model = ConfidenceModel()
    errors = []
    n = len(file_names)
    for i in range(n):
        for j in range(i + 1, n):
            fa = file_names[i]
            fb = file_names[j]
            la = labels_mapped.get(fa, "Unknown")
            lb = labels_mapped.get(fb, "Unknown")
            if la == "Unknown" or lb == "Unknown":
                continue

            score = float(sim_matrix[i][j])
            actually_same = la == lb
            predicted_same = score >= threshold

            if actually_same == predicted_same:
                continue  # correct

            ma = file_metrics[fa]
            mb = file_metrics[fb]
            conf_result = confidence_model.compute(ma, mb)

            # Calibrated probability if available
            if calibrator and calibrator.is_fitted:
                cal_prob = calibrator.predict_proba(score, method="platt")
            else:
                cal_prob = None

            error_type = "False Positive" if predicted_same else "False Negative"

            # Diagnostic warnings
            warn = []
            min_dur = min(ma["effective_speech_duration"], mb["effective_speech_duration"])
            if min_dur < 3.0:
                warn.append(f"Short audio ({min_dur:.1f}s)")
            if ma.get("low_volume") or mb.get("low_volume"):
                warn.append("Low volume detected")
            if ma.get("has_clipping") or mb.get("has_clipping"):
                warn.append("Audio clipping detected")
            if min(ma["speech_ratio"], mb["speech_ratio"]) < 0.30:
                warn.append("Low speech ratio")

            # Potential failure reasons
            failure = []
            if error_type == "False Positive":
                failure.append("Similar acoustic conditions or speaking style may inflate cosine similarity")
                if min_dur < 5.0:
                    failure.append("Short utterances reduce discriminative power")
            else:
                failure.append("Channel mismatch or recording quality differences may deflate similarity")
                if ma.get("has_clipping") or mb.get("has_clipping"):
                    failure.append("Clipping distortion corrupts embedding fidelity")

            errors.append({
                "type": error_type,
                "file_a": fa,
                "file_b": fb,
                "label_a": la,
                "label_b": lb,
                "score": round(score, 4),
                "calibrated_prob": round(cal_prob, 4) if cal_prob is not None else "N/A",
                "confidence": round(conf_result["confidence"], 3),
                "min_duration": round(min_dur, 2),
                "min_speech_ratio": round(min(ma["speech_ratio"], mb["speech_ratio"]), 3),
                "warnings": "; ".join(warn) if warn else "None",
                "potential_causes": "; ".join(failure),
            })
    return errors


# ===========================================================================
# PAIRWISE PERTURBATION ROBUSTNESS
# ===========================================================================

def compute_pairwise_perturbation_stability(file_path_a, file_path_b):
    """
    Apply gain ±3 dB and mild noise to both files, compute similarity
    for each perturbation combo.  Return dict with mean, variance, scores.
    """
    try:
        wav_a, _ = get_audio_tensor(file_path_a)
        wav_b, _ = get_audio_tensor(file_path_b)
        base_emb_a = extract_embeddings(wav_a)
        base_emb_b = extract_embeddings(wav_b)
        if base_emb_a is None or base_emb_b is None:
            return None

        def _perturb(w):
            variants = [w]
            variants.append(w * (10 ** (3 / 20)))   # +3 dB
            variants.append(w * (10 ** (-3 / 20)))   # -3 dB
            # mild noise injection
            torch.manual_seed(42)
            noise = torch.randn_like(w) * 0.002
            variants.append(w + noise)
            return variants

        vars_a = _perturb(wav_a)
        vars_b = _perturb(wav_b)

        embs_a = [extract_embeddings(v) for v in vars_a]
        embs_b = [extract_embeddings(v) for v in vars_b]

        scores = []
        for ea in embs_a:
            for eb in embs_b:
                if ea is not None and eb is not None:
                    scores.append(compute_similarity(ea, eb))

        if not scores:
            return None
        return {
            "mean": float(np.mean(scores)),
            "variance": float(np.var(scores)),
            "scores": scores,
        }
    except Exception:
        return None


# ===========================================================================
# MASTER COMPARISON FUNCTION
# ===========================================================================

def compare_speakers(metrics_a, metrics_b, embeddings_a, embeddings_b,
                     stability_a=None, stability_b=None,
                     pairwise_stability=None,
                     inconsistency_penalty=0.0,
                     calibrator=None,
                     file_a="File A", file_b="File B", lang="en"):
    """
    Full decision pipeline for a single speaker pair.
    
    Returns a structured result with decision, probability, confidence,
    risk, explanation, and all metadata.
    """
    # 1. Raw similarity
    raw_sim = compute_similarity(embeddings_a, embeddings_b)

    # 2. Calibrated probability
    if calibrator is None:
        calibrator = ScoreCalibrator()
    cal_prob = calibrator.predict_proba(raw_sim, method="auto")

    # 3. Stability (use pairwise if available, else average per-file)
    if pairwise_stability is not None:
        stab_score = pairwise_stability.get("mean", 0.95)
        stab_var = pairwise_stability.get("variance", 0.0)
    elif stability_a is not None and stability_b is not None:
        stab_score = min(stability_a.get("score", 0.95),
                         stability_b.get("score", 0.95))
        stab_var = max(stability_a.get("variance", 0.0),
                       stability_b.get("variance", 0.0))
    elif stability_a is not None:
        stab_score = stability_a.get("score", 0.95)
        stab_var = stability_a.get("variance", 0.0)
    elif stability_b is not None:
        stab_score = stability_b.get("score", 0.95)
        stab_var = stability_b.get("variance", 0.0)
    else:
        stab_score = None
        stab_var = None

    stability_info = {"score": stab_score, "variance": stab_var} if stab_score is not None else None

    # 4. Confidence
    conf_model = ConfidenceModel()
    conf_result = conf_model.compute(
        metrics_a, metrics_b,
        stability_score=stab_score,
        stability_variance=stab_var,
        inconsistency_penalty=inconsistency_penalty,
    )

    # 5. Decision
    engine = DecisionEngine()
    decision_result = engine.decide(
        cal_prob,
        confidence=conf_result["confidence"],
        stability_score=stab_score if stab_score is not None else 0.95,
    )

    # 6. Risk
    risk_scorer = RiskScorer()
    min_dur = min(metrics_a.get("effective_speech_duration", 0),
                  metrics_b.get("effective_speech_duration", 0))
    risk_result = risk_scorer.compute(
        confidence=conf_result["confidence"],
        stability_score=stab_score,
        stability_variance=stab_var,
        min_duration=min_dur,
        inconsistency_penalty=inconsistency_penalty,
        lang=lang
    )

    # 7. Explanation
    explainer = ExplanationEngine()
    explanation = explainer.explain(
        decision_result, conf_result, risk_result,
        metrics_a, metrics_b, stability_info,
        raw_sim, cal_prob, file_a, file_b,
        lang=lang
    )

    return {
        "raw_similarity": round(raw_sim, 5),
        "calibrated_probability": round(cal_prob, 4),
        "decision": decision_result["decision"],
        "decision_detail": decision_result,
        "confidence": conf_result["confidence"],
        "confidence_detail": conf_result,
        "risk": risk_result,
        "explanation": explanation,
        "stability": stability_info,
        "metadata": {
            "file_a": file_a,
            "file_b": file_b,
            "duration_a": round(metrics_a.get("effective_speech_duration", 0), 2),
            "duration_b": round(metrics_b.get("effective_speech_duration", 0), 2),
            "speech_ratio_a": round(metrics_a.get("speech_ratio", 0), 3),
            "speech_ratio_b": round(metrics_b.get("speech_ratio", 0), 3),
        },
    }
