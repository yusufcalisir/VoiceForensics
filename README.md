# VoiceForensics 🎙️🔍

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20SpeechBrain-orange.svg)](https://speechbrain.github.io/)
[![UI](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**🔗 Live Demo Application:** [https://voiceforensics.streamlit.app/](https://voiceforensics.streamlit.app/)

**VoiceForensics** is an enterprise-grade, forensic speaker identification and biometric intelligence engine. Far beyond a simple script calculating cosine similarities, this application functions as a **complete decision, calibration, and deterministic uncertainty engine**. 

Designed for professional audio analysis, threat intelligence, and precise identity verification, VoiceForensics explicitly models neural volatility, highlights adversarial risks, and computes mathematically robust "Trust Scores" using multi-factor penalty regressions instead of relying solely on opaque neural network outputs.

---

## 📌 Executive Summary & The Problem

Traditional speaker identification systems (including commercial APIs) typically yield a raw, uninterpretable multi-dimensional distance score (e.g., `0.72`). When this score hovers near a binary decision threshold, human operators are forced to guess, leading to catastrophic False Positives (FP) or False Negatives (FN). Furthermore, traditional systems fail to explain *why* two voices match or *how confident* they are in the underlying audio quality.

**VoiceForensics fixes this structural flaw by introducing:**

1.  **Explicit Tri-State Boundaries**: Rather than forcing a True/False decision on weak evidence, the engine classifies entities definitively as `Same Speaker`, `Different Speaker`, or casts them into an explicitly mapped `Uncertain` zone.

2.  **Calibrated Probabilities**: Raw neural vector similarity is wrapped in empirical statistical probability frameworks. An output of "85%" natively correlates to an 85% real-world likelihood of being the same human.

3.  **Explainable AI (XAI)**: A transparent "Forensic Rationale" interface generates human-readable text detailing the exact telemetry and algorithmic behavior that forced the final decision.

4.  **Adversarial Robustness Simulation**: Natively tests audio clips under deterministic permutations (background noise simulation, matrix segment slicing) and artificially restricts the core confidence score if the neural network flickers under stress.

---

## 🧠 Deep Dive: The Algorithmic Pipeline

VoiceForensics is built upon a strictly decoupled architecture. The `core.py` engine handles purely mathematical regressions and tensor execution, safely separated from the `app.py` presentation and UI logic.

### 1. Ingestion & Pre-Processing

User-uploaded audio (`WAV`, `FLAC`, `MP3`) is dynamically read into memory caches. Files are cryptographically hashed using SHA-256 to ensure data integrity during cross-comparison. Audio tensors are automatically resampled to standard biometrics boundaries (16kHz).

### 2. Voice Activity Detection (VAD)

VoiceForensics utilizes **Silero VAD** sequentially. Feeding silence, background static, or room noise into a neural network permanently corrupts speaker embeddings. All incoming audio is swept for non-speech noise. The system precisely crops the tensor map to measure the intrinsic mathematical fingerprint of the *active vocal cord oscillation* alone.

### 3. Neural Embeddings Matrix (ECAPA-TDNN)

The isolated speech segments are passed into SpeechBrain's **ECAPA-TDNN** (Emphasized Channel Attention, Propagation, and Aggregation - Time Delay Neural Network). This architecture generates an ultra-dense, 192-dimensional numerical vector representation (embedding) that is heavily invariant to pitch, spoken language, and emotion.

### 4. Z-Score Normalization & Matrix Transitivity

When utilizing large arrays of files ($N \ge 6$):

*   **Z-Scoring**: Removes baseline environmental similarities across mass arrays, eliminating the risk of two people appearing "similar" merely because they were recorded in the same echoey room.
*   **Transitivity Consistency Engine**: Validates the logical loop. If `Speaker A` matches `Speaker B`, and `Speaker B` matches `Speaker C`, but `Speaker A` wildly diverges from `Speaker C`, the system detects a catastrophic matrix violation and cascades a "Trust Penalty".

### 5. Multi-Factor Risk Assessment (Trust Score)

Raw similarity is not enough. VoiceForensics constructs a **Confidence Ratio** by aggregating telemetry:

*   **Duration Penalty**: Aggressively drops confidence if a continuous speech block falls under a 3.0-second critical mass threshold.
*   **Acoustic Clipping**: Flags RMS (Root Mean Square) energy spikes and volume decay that distort neural reads.

---

## ⚙️ The Calibration Engine (Moving from "Score" to "Probability")

When raw comparisons are rendered without historical ground truth, the system defaults to a **Dynamic Sigmoid Fallback** mapping curve. However, VoiceForensics unlocks its true potential when trained *on the fly* via the "Identify Labels" system:

1.  **Assign Ground Truth**: By explicitly labeling known audio samples in the Sidebar settings ("Known Target 1", "Suspect A"), you activate the Empirical Engine.

2.  **Platt Strategy (Linear Logistic)**: The system assesses your labeled data and overlays a regression curve mapping the neural embeddings of "Known Same" vs "Known Different" clips to expected percentages.

3.  **Isotonic (Non-parametric)**: A step-wise mathematical mapping that avoids assuming a normal distribution pipeline. This is highly effective if your audio data is recorded across wildly varying, unsynchronized heterogeneous microphones.

---

## 🚀 Setup & Installation (Local Execution)

This project relies on strict tensor bindings and is tailored for **Python 3.9 through 3.11**. *Please avoid using Python 3.12 or 3.13, as PyTorch compiling modules and C++ dependencies are often unstable globally on absolute bleeding-edge versions.*

### Step 1: Clone & Isolate Environment

```bash
git clone https://github.com/yusufcalisir/VoiceForensics.git
cd VoiceForensics

# Initiate Virtual Environment (Prevents dependency bleeding)
python -m venv venv
```

**Activate your Pipeline:**

*   **Windows**: `.\venv\Scripts\activate`
*   **Mac/Linux**: `source venv/bin/activate`

### Step 2: GPU Acceleration vs CPU Fallback (Crucial)

If you **do not** have an NVIDIA Dedicated GPU (or are using a standard Mac), the engine will safely default to CPU execution. It operates highly efficiently, albeit slower on massive arrays.

If you **DO** have an NVIDIA GPU, you MUST install the specific CUDA-Torch wheels *before* running the general requirements, or pip will download the CPU baseline:

```powershell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Engine Core Requirements

```bash
pip install -r requirements.txt
```

---

## 💻 Launching VoiceForensics (Avoiding PyTorch Overlaps)

Modern Machine Learning loops (Torch Dynamo JIT compiling) and HuggingFace Transformers logic often aggressively scan file structures. This natively **conflicts** with Streamlit's live-reload functionality, leading to endless compiling logs or fatal module scanning crashes (e.g., `ModuleNotFoundError: k2`).

**To guarantee a stable, pristine deployment, launch the application using this exact configuration:**

#### Windows (PowerShell)

```powershell
$env:TRANSFORMERS_NO_TORCHDYNAMO="1"
streamlit run app.py --server.runOnSave false --server.fileWatcherType=none
```

#### Mac / Linux (Bash)

```bash
export TRANSFORMERS_NO_TORCHDYNAMO=1
streamlit run app.py --server.runOnSave false --server.fileWatcherType=none
```

> **Architecture Warning**: The flag `--server.fileWatcherType=none` stops Streamlit from attempting to "peer" into HuggingFace library integrations, preventing catastrophic lazy-load crashes and ensuring 0 memory leaks during analysis.

---

## 🔬 Operator's Manual: Conducting an Investigation

### Phase 1: Ingestion & Telemetry Verification

1.  Open the Sidebar (Left) and upload your audio files. 
2.  Navigate directly to the **📂 Specimen Data** tab.
3.  Validate your target recordings. Look at the `Speech (s)` ratio versus the `Total Duration`. If an audio file is 10 minutes long, but the VAD engine detects only `0.8 seconds` of distinct, uninterrupted human speech, the system applies a `Short Audio Risk` flag. The risk thresholds tighten safely.

### Phase 2: Pairwise Analysis & The Dashboard

Select two distinct specimens in the Dropdown menus. Read the Premium Glassmorphism UI:

*   **Verdict**: The absolute bottom line (Same / Different / Uncertain).
*   **Probability**: Converted via the chosen calibrator (Defaults to Sigmoid).
*   **Confidence**: The internal mathematical Trust Ratio (out of 100%).
*   **Risk Profile**: Flags high acoustic instability. Look immediately to the **🔬 Analysis Rationale** tab to read the human-readable explanation of *why* the Risk Profile is high.

### Phase 3: The Similarity Array (Matrix)

Review the **📊 Performance Matrix** tab. The system charts a 2D Heatmap across every single uploaded file. Colors denote similarity blocks. 

### Phase 4: Enabling ROC Curves (Evaluation Mode)

To view the **Equal Error Rate (EER)** and **ROC Curves**:

1.  Assign labels in the sidebar (e.g., label File 1 and File 2 as "Speaker A", and File 3 as "Speaker B").
2.  The UI natively unlocks standard biometric visualizations. The **Red Operating Point** actively traces along the ROC Curve to signify the physical probability position of your selected Pairwise comparison relative to the known matrix.

### Phase 5: The Adversarial Stress Suite

Clicking the red **Run Full Adversarial Robustness Check** button subjects the loaded files to dynamic simulated destruction. 
VoiceForensics will segment the audio into isolated temporal blocks (Beginning, Middle, End), inject mathematical jitter, and re-compute the identity metrics against themselves. If the target speaker "changes identity" in the middle of a clip due to erratic noise or multi-speaker contamination, this suite registers an irrecoverable Identity Flag limit failure.

---

## ⚠️ Known Triggers & Troubleshooting

*   **UI Artifacts / Overlaps**: If text visually overlaps on 1080p monitors, ensure your browser zoom level is exactly 100%. The UI uses static pixel glassmorphism blocks that scale heavily with zoom profiles.
*   **Transducer Loss Deprecation Warnings**: These non-fatal warnings trigger natively when `speechbrain` loads and dynamically binds (missing `numba` paths). These are entirely harmless and will not impact matrix rendering.
*   **Containerized Memory Constraints (OOM)**: Local desktop deployments are totally RAM-safe. However, if deploying to a limited Docker instance or Streamlit Community Cloud (which imposes strict 1GB limits), cross-compiling Matrix visualizations for $N \ge 15$ long files may force the hypervisor to crash due to `matplotlib` rendering payloads stacking atop PyTorch Tensors.
*   **Local SSD Bloat**: Over protracted investigation lifetimes, the `temp_audio/` folder will cache data payload extracts to ensure instantaneous matrix loading. Manually delete the contents of this folder upon concluding an entire investigation case securely.

---

## ⚖️ License & Integrity Agreement

This software is released openly under the **MIT License**. It leverages advanced ecosystem capabilities natively governed by SpeechBrain (Apache 2.0) and Silero VAD (MIT). 

*VoiceForensics is fundamentally built to prevent deterministic black-box guesswork. Every probabilistic percentage and confidence score metric outputs an auditable mathematical rationale.*