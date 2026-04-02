import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import hashlib
from scipy.stats import zscore
import core

# Premium UI Configuration
st.set_page_config(
    page_title="Speaker Identity Engine | Premium",
    page_icon="🎙️",
    layout="wide"
)

# Modern Premium Styling
st.markdown("""
<style>
    /* Main Layout */
    .stApp {
        background: radial-gradient(circle at top right, #1a1b26, #16161e);
        color: #a9b1d6;
    }
    
    /* Header Styling */
    h1 {
        background: linear-gradient(90deg, #7aa2f7, #bb9af7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* Card / Container Styling */
    .stMetric {
        background: rgba(36, 40, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(122, 162, 247, 0.1);
        border-radius: 16px;
        padding: 1.5rem !important;
        transition: transform 0.2s ease, border 0.2s ease;
    }
    .stMetric:hover {
        border: 1px solid rgba(122, 162, 247, 0.3);
        transform: translateY(-2px);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1b26 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Global Spacing */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 5rem !important;
    }
    .stAlert {
        border-radius: 12px;
        border: none;
        background: rgba(65, 72, 104, 0.5);
    }
    
    /* Typography */
    .stMarkdown p, .stMarkdown li {
        color: #9aa5ce;
        line-height: 1.6;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent !important;
        border: none !important;
        color: #565f89 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #7aa2f7 !important;
        border-bottom: 2px solid #7aa2f7 !important;
    }
    
    /* Removing the "forensic" cramped margins */
    .stPlotlyChart, .stImage, .stPyplot {
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Caching ---
@st.cache_data(show_spinner=False)
def cached_audio_process(file_bytes_hash, file_name, file_bytes):
    os.makedirs("temp_audio", exist_ok=True)
    temp_path = os.path.join("temp_audio", file_name)
    with open(temp_path, "wb") as f_out:
        f_out.write(file_bytes)
    return core.process_audio(temp_path)

@st.cache_data(show_spinner=False)
def cached_stability_score(file_bytes_hash, file_name, file_bytes):
    os.makedirs("temp_audio", exist_ok=True)
    temp_path = os.path.join("temp_audio", file_name)
    if not os.path.exists(temp_path):
        with open(temp_path, "wb") as f_out:
            f_out.write(file_bytes)
    return core.compute_stability_score(temp_path)

# ==========================================
# SIDEBAR: Control Panel
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/microphone.png", width=60)
    st.title("Settings")
    
    st.markdown("##### 📥 Data Source")
    uploaded_files = st.file_uploader(
        "Drop audio files",
        accept_multiple_files=True,
        type=['wav', 'mp3', 'flac'],
        help="Supports broadcast-quality WAV, FLAC, and common MP3s."
    )
    n = len(uploaded_files) if uploaded_files else 0
    
    st.markdown("##### ⚙️ Calibration")
    calib_method = st.selectbox("Model", ["Auto", "Platt", "Isotonic", "Sigmoid"], index=0)
    method_map = {"Auto": "auto", "Platt": "platt", "Isotonic": "isotonic", "Sigmoid": "sigmoid"}
    
    if n > 1:
        st.markdown("##### 📐 Thresholds")
        trans_boundary = st.slider("Transitivity Focus", 0.50, 0.95, 0.70, 0.01)
        apply_zscore = st.checkbox("Z-Score (N>=6)", value=False, disabled=(n < 6))
        
        st.markdown("---")
        st.markdown("##### 🏷️ Identity Labels")
        if "labels" not in st.session_state:
            st.session_state.labels = {f.name: "Unknown" for f in uploaded_files}
        
        # Keep label editor in sidebar but make it clean
        label_df = pd.DataFrame([
            {"File": f.name, "Speaker": st.session_state.labels.get(f.name, "Unknown")} 
            for f in uploaded_files
        ])
        edited = st.data_editor(label_df, use_container_width=True, hide_index=True, num_rows="fixed")
        for _, row in edited.iterrows():
            st.session_state.labels[row["File"]] = row["Speaker"]
        labels_mapped = dict(zip(edited["File"], edited["Speaker"]))
    else:
        labels_mapped = {}

# ==========================================
# MAIN: Results Dashboard
# ==========================================
st.title("Speaker Identity Intelligence")

if n > 1:
    # 1. Background Loading
    file_metrics, embeddings, file_names, file_stabilities = {}, {}, [], {}
    
    with st.status("Analyzing specimens...", expanded=False) as status:
        for f in uploaded_files:
            st.write(f"Extracting features from {f.name}...")
            file_bytes = f.read()
            fb_hash = hashlib.sha256(file_bytes).hexdigest()
            res = cached_audio_process(fb_hash, f.name, file_bytes)
            if not res.get("error"):
                file_metrics[f.name] = res
                embeddings[f.name] = res["embeddings"]
                file_names.append(f.name)
                file_stabilities[f.name] = cached_stability_score(fb_hash, f.name, file_bytes)
            f.seek(0)
        status.update(label="Analysis Complete", state="complete", expanded=False)

    n_proc = len(file_names)
    if n_proc < 2:
        st.info("Upload at least two valid audio files to begin comparison.")
        st.stop()

    # 2. Similarity Matrix Logic
    sim_matrix = np.zeros((n_proc, n_proc))
    for i in range(n_proc):
        for j in range(n_proc):
            sim_matrix[i][j] = core.compute_similarity(embeddings[file_names[i]], embeddings[file_names[j]])
            
    if apply_zscore and n_proc >= 6:
        off_diag = [sim_matrix[i][j] for i in range(n_proc) for j in range(n_proc) if i != j]
        if off_diag:
            m, s = np.mean(off_diag), np.std(off_diag) if np.std(off_diag) > 0 else 1.0
            for i in range(n_proc):
                for j in range(n_proc):
                    if i != j: sim_matrix[i][j] = (sim_matrix[i][j] - m) / s

    # Calibrator fitting
    same_scores, diff_scores = [], []
    for i in range(n_proc):
        for j in range(i+1, n_proc):
            la, lb = labels_mapped.get(file_names[i]), labels_mapped.get(file_names[j])
            if la and lb and la != "Unknown" and lb != "Unknown":
                if la == lb: same_scores.append(sim_matrix[i][j])
                else: diff_scores.append(sim_matrix[i][j])
    
    calibrator = core.ScoreCalibrator()
    has_labels = len(same_scores) > 0 and len(diff_scores) > 0
    if has_labels:
        calibrator.fit(same_scores, diff_scores)
    else:
        calibrator.fit_sigmoid_fallback([sim_matrix[i][j] for i in range(n_proc) for j in range(n_proc) if i != j])

    # 3. Selection & Prime Results
    st.markdown("### 🧪 Pairwise Comparison")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1: fa = st.selectbox("File A", file_names, index=0)
    with col_sel2: fb = st.selectbox("File B", file_names, index=1 if n_proc > 1 else 0)
    
    path_a = os.path.join("temp_audio", fa)
    path_b = os.path.join("temp_audio", fb)
    pair_stab = core.compute_pairwise_perturbation_stability(path_a, path_b)
    
    idx1, idx2 = file_names.index(fa), file_names.index(fb)
    penalties, _ = core.calculate_transitivity(sim_matrix, file_names, ui_strong_sim_threshold=trans_boundary)
    
    comp = core.compare_speakers(
        file_metrics[fa], file_metrics[fb], embeddings[fa], embeddings[fb],
        file_stabilities.get(fa), file_stabilities.get(fb), pair_stab,
        max(penalties[idx1], penalties[idx2]) * 2, calibrator, fa, fb
    )

    # Big Premium Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        d = comp["decision"].upper()
        d_color = "#76ff03" if d == "SAME" else "#ff1744" if d == "DIFFERENT" else "#ffc107"
        st.markdown(f"**Verdict**  \n<p style='font-size:32px; font-weight:900; color:{d_color}; margin:0;'>{d}</p>", unsafe_allow_html=True)
    with m2: st.metric("Probability", f"{comp['calibrated_probability']:.1%}")
    with m3: st.metric("Confidence", f"{comp['confidence']:.1%}")
    with m4:
        r = comp["risk"]
        r_color = "#4caf50" if r['level'] == 'low' else "#ff9800" if r['level'] == 'medium' else "#f44336"
        st.markdown(f"**Risk Profile**  \n<p style='font-size:32px; font-weight:900; color:{r_color}; margin:0;'>{r['level'].upper()}</p>", unsafe_allow_html=True)

    # 4. Insight Sections
    st.markdown("---")
    res_tab1, res_tab2, res_tab3 = st.tabs(["📊 Performance Matrix", "🔬 Analysis Rationale", "📂 Specimen Data"])
    
    with res_tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Similarity Heatmap")
            fig_h, ax_h = plt.subplots(figsize=(6, 5), facecolor='none')
            sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="vlag", xticklabels=file_names, yticklabels=file_names, 
                        ax=ax_h, cbar=False)
            plt.setp(ax_h.get_xticklabels(), rotation=45, ha="right", color="#565f89")
            plt.setp(ax_h.get_yticklabels(), color="#565f89")
            ax_h.set_title("Pairwise Scores", color="#a9b1d6")
            st.pyplot(fig_h)
            
        with c2:
            st.markdown("#### Probability Gauge")
            bd = comp["decision_detail"]["effective_boundaries"]
            gauge_fig, g_ax = plt.subplots(figsize=(8, 1))
            g_ax.axhspan(0, 1, xmin=0, xmax=1, color="#16161e")
            g_ax.axvspan(0, bd["different_below"], color="#ff1744", alpha=0.3)
            g_ax.axvspan(bd["different_below"], bd["same_above"], color="#ffc107", alpha=0.3)
            g_ax.axvspan(bd["same_above"], 1.0, color="#76ff03", alpha=0.3)
            current_p = comp["calibrated_probability"]
            g_ax.axvline(current_p, color="#7aa2f7", linewidth=4)
            g_ax.set_xlim(0, 1)
            g_ax.axis('off')
            st.pyplot(gauge_fig)
            
            if has_labels:
                st.markdown("#### ROC Curve")
                fpr, tpr, auc = core.compute_roc_data(same_scores, diff_scores)
                fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                ax_roc.plot(fpr, tpr, color="#bb9af7", lw=2, label=f"AUC: {auc:.3f}")
                ax_roc.plot([0, 1], [0, 1], 'w--', alpha=0.2)
                ax_roc.set_facecolor('#1a1b26')
                ax_roc.tick_params(colors='#565f89')
                ax_roc.legend(facecolor='#16161e', labelcolor='#a9b1d6')
                st.pyplot(fig_roc)

    with res_tab2:
        st.markdown("#### System Explanation")
        st.info(f"**Interpretation:** {comp['explanation']['decision_reason']}")
        
        expl_col1, expl_col2 = st.columns(2)
        with expl_col1:
            st.markdown("**Evidence Highlights**")
            for f in comp['explanation']['contributing_factors']:
                st.write(f"✅ {f}")
        with expl_col2:
            st.markdown("**Potential Risks**")
            for f in comp['risk']['factors']:
                st.write(f"⚠️ {f}")
            if comp['explanation']['uncertainty_reasons']:
                for r in comp['explanation']['uncertainty_reasons']:
                    st.write(f"❓ {r}")

    with res_tab3:
        st.markdown("#### File Telemetry")
        m_data = []
        for fn in file_names:
            met = file_metrics[fn]
            m_data.append({
                "Specimen": fn,
                "Speech (s)": f"{met['effective_speech_duration']:.2f}s",
                "Ratio": f"{met['speech_ratio']:.1%}",
                "RMS": f"{met['rms']:.4f}",
                "Stability": f"{file_stabilities.get(fn, {}).get('score', 0):.3f}",
                "Quality Tags": ", ".join(["Short" if met['short_audio_warning'] else "Good"])
            })
        st.table(pd.DataFrame(m_data))

    # 5. Global Actions
    st.markdown("---")
    if st.button("🚀 Run Full Adversarial Robustness Check", use_container_width=True):
        with st.status("Performing stress tests...", expanded=True) as adv_status:
            import adversarial
            file_paths = {fn: os.path.join("temp_audio", fn) for fn in file_names}
            adv_rep = adversarial.generate_full_report(file_names, file_paths, sim_matrix, file_metrics, labels_mapped, 0.70, embeddings)
            st.write("Adversarial battery complete.")
            adv_status.update(label="Stress Tests Complete", state="complete")
        
        # Display results in columns
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.markdown("##### Perturbation Results")
            for p_test in adv_rep["perturbation_tests"]:
                st.write(f"**{p_test['file']}**: Variance {p_test['variance']:.2e}")
        with a_col2:
            st.markdown("##### Consistency Exceptions")
            if not adv_rep["failure_log"]:
                st.success("No critical identity failures detected.")
            else:
                for fail in adv_rep["failure_log"]:
                    st.error(f"Failure in {fail['file']} during {fail['perturbation']}")

else:
    # Empty State - Beautifully Designed
    st.markdown("""
    <div style='text-align: center; padding: 100px;'>
        <h2 style='color: #565f89;'>Welcome to the Intelligence Engine</h2>
        <p style='color: #414868;'>Upload samples in the sidebar to begin biometric analysis.</p>
    </div>
    """, unsafe_allow_html=True)
