import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import hashlib
from scipy.stats import zscore
import core
from translations import t

# Handle Lang
if "lang" not in st.session_state:
    st.session_state.lang = "en"
l = st.session_state.lang

# Dynamic JS Translator for hardcoded Streamlit elements
import streamlit.components.v1 as components
target_btn = 'Dosya Seç' if l == 'tr' else 'Browse files'
target_desc = 'Dosya başı limit: 200MB | WAV, MP3, FLAC' if l == 'tr' else 'Limit 200MB per file \u2022 WAV, MP3, FLAC'
components.html(f'''
<script>
const targetBtn = "{target_btn}";
const targetDesc = "{target_desc}";
const translateStreamlit = () => {{
    const parentDoc = window.parent.document;
    if (!parentDoc) return;
    const uploaderBtns = parentDoc.querySelectorAll('[data-testid="stFileUploader"] button');
    uploaderBtns.forEach(btn => {{
        if (btn.innerHTML.includes('Upload') || btn.innerHTML.includes('Browse files') || btn.innerHTML.includes('Dosya Seç')) {{
            if (!btn.innerHTML.includes(targetBtn)) {{
                btn.innerHTML = btn.innerHTML.replace(/Upload|Browse files|Dosya Seç/g, targetBtn);
            }}
        }}
    }});
    const instructions = parentDoc.querySelectorAll('[data-testid="stFileUploaderDropzoneInstructions"]');
    instructions.forEach(inst => {{
        if (inst.innerHTML.includes('200MB per file') || inst.innerHTML.includes('Dosya başı limit: 200MB')) {{
            if (!inst.innerHTML.includes(targetDesc)) {{
                inst.innerHTML = inst.innerHTML.replace(/(200MB per file[^<]*|Dosya başı limit: 200MB[^<]*)/g, targetDesc);
            }}
        }}
    }});
}};
if (window.parent.document.streamlitTranslateObserver) {{
    window.parent.document.streamlitTranslateObserver.disconnect();
}}
window.parent.document.streamlitTranslateObserver = new MutationObserver(() => translateStreamlit());
window.parent.document.streamlitTranslateObserver.observe(window.parent.document.body, {{ childList: true, subtree: true }});
translateStreamlit();
</script>
''', height=0, width=0)

# Premium UI Configuration
st.set_page_config(
    page_title=t("page_title", l),
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
    
    /* Removed existing margins */
    .stPlotlyChart, .stImage, .stPyplot {
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    
    /* MOBILE RESPONSIVENESS */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        h1 {
            font-size: 1.8rem !important;
        }
        .stMetric {
            padding: 1rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem !important;
            padding: 5px !important;
        }
    }
</style>
""", unsafe_allow_html=True)




# --- Caching ---
@st.cache_data(show_spinner=False)
def cached_audio_process(file_bytes_hash, file_name, file_bytes):
    os.makedirs("temp_audio", exist_ok=True)
    safe_name = f"{file_bytes_hash}_{file_name}"
    temp_path = os.path.join("temp_audio", safe_name)
    with open(temp_path, "wb") as f_out:
        f_out.write(file_bytes)
    res = core.process_audio(temp_path)
    if isinstance(res, dict):
        res["safe_path"] = temp_path
    return res

@st.cache_data(show_spinner=False)
def cached_stability_score(safe_path):
    if os.path.exists(safe_path):
        return core.compute_stability_score(safe_path)
    return {"score": 0.0, "details": "File missing"}

# ==========================================
# SIDEBAR: Control Panel
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/microphone.png", width=60)
    
    colTR, colEN = st.columns(2)
    if colEN.button("🇬🇧 EN", use_container_width=True): 
        st.session_state.lang = "en"
        st.rerun()
    if colTR.button("🇹🇷 TR", use_container_width=True): 
        st.session_state.lang = "tr"
        st.rerun()
    l = st.session_state.lang

    st.title(t("settings", l))
    
    st.markdown(f"##### {t('data_source', l)}")
    uploaded_files = st.file_uploader(
        t("drop_audio", l),
        accept_multiple_files=True,
        type=['wav', 'mp3', 'flac'],
        help=t("supports_audio", l)
    )
    n = len(uploaded_files) if uploaded_files else 0
    
    st.markdown(f"##### {t('calibration', l)}")
    calib_method = st.selectbox(t("model", l), [t("auto_model", l), "Platt", "Isotonic", "Sigmoid"], index=0)
    method_map = {t("auto_model", l): "auto", "Platt": "platt", "Isotonic": "isotonic", "Sigmoid": "sigmoid"}
    
    if n > 1:
        st.markdown(f"##### {t('thresholds', l)}")
        trans_boundary = st.slider(t("transitivity_focus", l), 0.50, 0.95, 0.70, 0.01)
        apply_zscore = st.checkbox(t("zscore", l), value=False, disabled=(n < 6))
        
        st.markdown("---")
        st.markdown(f"##### {t('identity_labels', l)}")
        if "labels" not in st.session_state:
            st.session_state.labels = {f.name: "unknown" for f in uploaded_files}
        
        # Retroactively fix any leaked TR strings from previous bugs
        for k, v in st.session_state.labels.items():
            if v == "Belirsiz" or v == "Unknown":
                st.session_state.labels[k] = "unknown"
                
        # Keep label editor in sidebar but make it clean
        label_df = pd.DataFrame([
            {t("col_file", l): f.name, t("col_speaker", l): t("unknown", l) if st.session_state.labels.get(f.name) == "unknown" else st.session_state.labels.get(f.name)} 
            for f in uploaded_files
        ])
        edited = st.data_editor(label_df, use_container_width=True, hide_index=True, num_rows="fixed")
        for _, row in edited.iterrows():
            val = row[t("col_speaker", l)]
            st.session_state.labels[row[t("col_file", l)]] = "unknown" if val == t("unknown", l) else val
        labels_mapped = dict(zip(edited[t("col_file", l)], edited[t("col_speaker", l)]))
    else:
        labels_mapped = {}

# ==========================================
# MAIN: Results Dashboard
# ==========================================
st.title(t("speaker_id_intel", l))

if n > 1:
    # 1. Background Loading
    file_metrics, embeddings, file_names, file_paths = {}, {}, [], {}
    
    with st.status(t("analyzing_specimens", l), expanded=False) as status:
        for f in uploaded_files:
            file_bytes = f.read()
            fb_hash = hashlib.sha256(file_bytes).hexdigest()
            res = cached_audio_process(fb_hash, f.name, file_bytes)
            if not res.get("error"):
                file_metrics[f.name] = res
                embeddings[f.name] = res["embeddings"]
                file_names.append(f.name)
                file_paths[f.name] = res.get("safe_path", os.path.join("temp_audio", f"{fb_hash}_{f.name}"))
            f.seek(0)
        status.update(label=t("initial_ingestion_complete", l), state="complete", expanded=False)

    n_proc = len(file_names)
    if n_proc < 2:
        st.info(t("upload_at_least_two", l))
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
            if la and lb and la != t("unknown", l) and lb != t("unknown", l):
                if la == lb: same_scores.append(sim_matrix[i][j])
                else: diff_scores.append(sim_matrix[i][j])
    
    calibrator = core.ScoreCalibrator()
    has_labels = len(same_scores) > 0 and len(diff_scores) > 0
    if has_labels:
        calibrator.fit(same_scores, diff_scores)
    else:
        calibrator.fit_sigmoid_fallback([sim_matrix[i][j] for i in range(n_proc) for j in range(n_proc) if i != j])

    # 3. Selection & Prime Results
    st.markdown(f"### {t('pairwise_comparison', l)}")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1: fa = st.selectbox(t("file_a", l), file_names, index=0)
    with col_sel2: fb = st.selectbox(t("file_b", l), file_names, index=1 if n_proc > 1 else 0)
    
    path_a = file_paths[fa]
    path_b = file_paths[fb]
    
    with st.spinner(t("computing_precision", l)):
        pair_stab = core.compute_pairwise_perturbation_stability(path_a, path_b)
        stab_a = cached_stability_score(path_a)
        stab_b = cached_stability_score(path_b)
    
    idx1, idx2 = file_names.index(fa), file_names.index(fb)
    penalties, _ = core.calculate_transitivity(sim_matrix, file_names, ui_strong_sim_threshold=trans_boundary)
    
    comp = core.compare_speakers(
        file_metrics[fa], file_metrics[fb], embeddings[fa], embeddings[fb],
        stab_a, stab_b, pair_stab,
        max(penalties[idx1], penalties[idx2]) * 2, calibrator, fa, fb, lang=l
    )

    # Big Premium Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        d_val = comp["decision"].upper()
        d_color = "#76ff03" if d_val == "SAME" else "#ff1744" if d_val == "DIFFERENT" else "#ffc107"
        d_disp = t(d_val, l)
        st.markdown(f"{t('verdict', l)}  \n<p style='font-size:32px; font-weight:900; color:{d_color}; margin:0;'>{d_disp}</p>", unsafe_allow_html=True)
    with m2: st.metric(t("probability", l), f"{comp['calibrated_probability']:.1%}")
    with m3: st.metric(t("confidence", l), f"{comp['confidence']:.1%}")
    with m4:
        r_level = comp["risk"]['level'].upper()
        r_color = "#4caf50" if r_level == 'LOW' else "#ff9800" if r_level == 'MEDIUM' else "#f44336"
        r_disp = t(r_level, l)
        st.markdown(f"{t('risk_profile', l)}  \n<p style='font-size:32px; font-weight:900; color:{r_color}; margin:0;'>{r_disp}</p>", unsafe_allow_html=True)

    # 4. Insight Sections
    st.markdown("---")
    res_tab1, res_tab2, res_tab3 = st.tabs([t("perf_matrix", l), t("analysis_rationale", l), t("specimen_data", l)])
    
    with res_tab1:
        st.markdown(f"### {t('prob_gauge', l)}")
        bd = comp["decision_detail"]["effective_boundaries"]
        diff_bz = bd["different_below"] * 100
        same_bz = bd["same_above"] * 100
        current_p = comp["calibrated_probability"] * 100
        
        gauge_color = "#76ff03" if comp['decision'] == 'same' else "#ff1744" if comp['decision'] == 'different' else "#ffc107"
        glow_box = f"box-shadow: 0 0 15px {gauge_color}, inset 0 0 5px {gauge_color};" 
        
        html_gauge = f'''
        <div style="width: 100%; max-width: 800px; margin: 0 auto; padding: 20px 0; font-family: 'Inter', sans-serif;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; color: #a9b1d6; font-size: clamp(12px, 3vw, 14px); font-weight: 600;">
                <span>0%</span>
                <span style="color: {gauge_color}; font-size: clamp(16px, 5vw, 20px); font-weight: 800; text-shadow: 0 0 10px {gauge_color}55;">{current_p:.1f}%</span>
                <span>100%</span>
            </div>
            <div style="position: relative; width: 100%; height: clamp(20px, 5vw, 32px); background-color: #1a1b26; border-radius: 16px; overflow: hidden; display: flex; border: 1px solid #24283b; box-shadow: inset 0 0 15px rgba(0,0,0,0.8);">
                <div style="position: absolute; left: 0; width: {diff_bz}%; background: linear-gradient(90deg, #4f101d, #ff1744); height: 100%; opacity: 0.85;"></div>
                <div style="position: absolute; left: {diff_bz}%; width: {same_bz - diff_bz}%; background: linear-gradient(90deg, #e67e22, #f1c40f); height: 100%; opacity: 0.85;"></div>
                <div style="position: absolute; left: {same_bz}%; width: {100 - same_bz}%; background: linear-gradient(90deg, #27ae60, #76ff03); height: 100%; opacity: 0.85;"></div>
                <div style="position: absolute; top: -3px; left: {current_p}%; width: 4px; height: 120%; background-color: #ffffff; border-radius: 3px; {glow_box} transform: translateX(-50%); z-index: 10; transition: left 1s ease-out;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; width: 100%; margin-top: 15px; font-size: clamp(9px, 2.5vw, 12px); font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">
                <div style="color: #ff5252; text-align: left;">{t('DIFFERENT', l)}</div>
                <div style="color: #ffd740; text-align: center;">{t('UNCERTAIN', l)}</div>
                <div style="color: #b2ff59; text-align: right;">{t('SAME', l)}</div>
            </div>
        </div>
        '''
        st.markdown(html_gauge, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if has_labels:
            cols = st.columns(2)
            c_heat = cols[0]
            c_roc = cols[1]
        else:
            c_heat = st.container()
            c_roc = None
            
        with c_heat:
            st.markdown(f"#### {t('sim_heatmap', l)}")
            fig_h, ax_h = plt.subplots(figsize=(8, 6), facecolor='none')
            sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="mako", xticklabels=file_names, yticklabels=file_names, 
                        ax=ax_h, cbar=True, cbar_kws={'shrink': 0.8})
            plt.setp(ax_h.get_xticklabels(), rotation=45, ha="right", color="#c0caf5", weight='bold')
            plt.setp(ax_h.get_yticklabels(), color="#c0caf5", weight='bold')
            ax_h.set_title(t('pairwise_scores', l), color="#c0caf5", pad=15, weight='bold')
            ax_h.tick_params(colors='#a9b1d6')
            st.pyplot(fig_h)
            
        if c_roc:
            with c_roc:
                st.markdown(f"#### {t('roc_curve', l)}")
                fpr, tpr, auc = core.compute_roc_data(same_scores, diff_scores)
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6), facecolor='none')
                ax_roc.plot(fpr, tpr, color="#bb9af7", lw=3, label=f"AUC: {auc:.3f}")
                ax_roc.fill_between(fpr, tpr, alpha=0.1, color="#bb9af7")
                ax_roc.plot([0, 1], [0, 1], 'w--', alpha=0.2)
                ax_roc.set_facecolor('#1a1b26')
                ax_roc.spines['bottom'].set_color('#565f89')
                ax_roc.spines['top'].set_visible(False) 
                ax_roc.spines['right'].set_visible(False)
                ax_roc.spines['left'].set_color('#565f89')
                ax_roc.tick_params(colors='#a9b1d6')
                ax_roc.legend(facecolor='#16161e', labelcolor='#a9b1d6', edgecolor='#565f89')
                st.pyplot(fig_roc)

    with res_tab2:
        st.markdown(f"#### {t('system_explanation', l)}")
        st.info(t("interpretation", l, reason=comp['explanation']['decision_reason']))
        
        expl_col1, expl_col2 = st.columns(2)
        with expl_col1:
            st.markdown(t("evidence_highlights", l))
            for f in comp['explanation']['contributing_factors']:
                st.write(f"✅ {f}")
        with expl_col2:
            st.markdown(t("potential_risks", l))
            for f in comp['risk']['factors']:
                st.write(f"⚠️ {f}")
            if comp['explanation']['uncertainty_reasons']:
                for r in comp['explanation']['uncertainty_reasons']:
                    st.write(f"❓ {r}")

    with res_tab3:
        st.markdown(f"#### {t('file_telemetry', l)}")
        m_data = []
        for fn in file_names:
            met = file_metrics[fn]
            sq_tag = t("short", l) if met['short_audio_warning'] else t("good", l)
            stab_lazy = f"{cached_stability_score(file_paths[fn]).get('score', 0):.3f}" if fn in [fa, fb] else t("pend", l)
            m_data.append({
                t("specimen", l): fn,
                t("speech_s", l): f"{met['effective_speech_duration']:.2f}s",
                t("ratio", l): f"{met['speech_ratio']:.1%}",
                t("rms", l): f"{met['rms']:.4f}",
                t("stability_lazy", l): stab_lazy,
                t("quality_tags", l): sq_tag
            })
        st.dataframe(pd.DataFrame(m_data), use_container_width=True, hide_index=True)

    # 5. Global Actions
    st.markdown("---")
    if st.button(t("run_full_adv", l), use_container_width=True):
        with st.status(t("performing_stress", l), expanded=True) as adv_status:
            import adversarial
            adv_rep = adversarial.generate_full_report(file_names, file_paths, sim_matrix, file_metrics, labels_mapped, 0.70, embeddings, lang=l)
            st.write(t("adv_battery_complete", l))
            adv_status.update(label=t("stress_tests_complete", l), state="complete")
        
        # Display results in columns
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.markdown(f"#### {t('perturbation_re', l)}")
            for p_test in adv_rep["perturbation_tests"]:
                st.write(f"**{p_test['file']}**: {t('variance_lbl', l)} {p_test['variance']:.2e}")
        with a_col2:
            st.markdown(f"#### {t('consistency_ex', l)}")
            if not adv_rep["failure_log"]:
                st.success(t("no_critical_fail", l))
            else:
                for fail in adv_rep["failure_log"]:
                    st.error(t("failure_in", l, file=fail['file'], test=fail['test'], issue=fail['issue']))

else:
    # Empty State - Beautifully Designed
    st.markdown(f"""
    <div style='text-align: center; padding: 100px;'>
        <h2 style='color: #565f89;'>{t("welcome", l)}</h2>
        <p style='color: #414868;'>{t("upload_samples_side", l)}</p>
    </div>
    """, unsafe_allow_html=True)
