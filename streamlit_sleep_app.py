"""
Streamlit sleep EEG dashboard — cyberpunk UI, YASA detection, cognitive outlook.
Run: streamlit run streamlit_sleep_app.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yasa

EPOCH_SEC = 30
N2_STAGE = 2
N3_STAGE = 3
DEFAULT_CHANNEL = "Fpz-Cz"

CYBER_ACCENT = "#00f5ff"
CYBER_MAGENTA = "#ff00ea"
CYBER_VIOLET = "#bf5fff"
CYBER_WARN = "#ffcc00"
CYBER_GRID = "#1f1f35"
CYBER_BG = "#0a0a12"
CYBER_PANEL = "#12121f"

CYBERPUNK_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">
<style>
  .stApp {
    background: radial-gradient(ellipse 120% 80% at 50% -20%, #1a0a2e 0%, #0a0a12 45%, #050508 100%) !important;
    font-family: system-ui, "Segoe UI", sans-serif;
  }
  h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.06em !important;
    text-shadow: 0 0 12px rgba(0, 245, 255, 0.35);
  }
  /*
   * Never apply mono to bare `span` — Streamlit icons use Material Symbols ligatures
   * (text like "arrow_right"); wrong font shows literal names and overlaps labels.
   */
  div[data-testid="stMarkdownContainer"] p,
  div[data-testid="stMarkdownContainer"] li,
  div[data-testid="stMarkdownContainer"] td,
  div[data-testid="stMarkdownContainer"] th,
  [data-testid="stCaption"],
  [data-testid="stText"] {
    font-family: 'Share Tech Mono', monospace !important;
  }
  .material-symbols-outlined,
  .material-symbols-rounded {
    font-family: "Material Symbols Outlined", sans-serif !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 1.35rem !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    font-feature-settings: "liga" !important;
    -webkit-font-smoothing: antialiased;
  }
  /* Baseweb / Streamlit controls: icon spans next to labels */
  [data-testid="stExpander"] details summary {
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    font-family: 'Share Tech Mono', monospace !important;
  }
  [data-testid="stExpander"] details summary > span:first-child {
    font-family: "Material Symbols Outlined", sans-serif !important;
    font-size: 1.35rem !important;
    flex-shrink: 0 !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    width: 1.35rem !important;
    text-align: center !important;
    overflow: hidden !important;
  }
  .stTabs [data-baseweb="tab-list"] button span,
  .stTabs [data-baseweb="tab-list"] ~ button span {
    font-family: "Material Symbols Outlined", sans-serif !important;
    font-size: 1.25rem !important;
    letter-spacing: normal !important;
    line-height: 1 !important;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12121f 0%, #0d0d18 100%) !important;
    border-right: 1px solid rgba(0, 245, 255, 0.25);
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.45);
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #00f5ff !important;
    font-size: 1rem !important;
  }
  div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #16162a 0%, #0f0f1a 100%);
    border: 1px solid rgba(191, 95, 255, 0.35);
    border-radius: 4px;
    padding: 0.75rem 1rem;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.08), inset 0 1px 0 rgba(255,255,255,0.04);
  }
  div[data-testid="stMetric"] label {
    color: #9aa4c4 !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.7rem !important;
  }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #00f5ff !important;
    font-family: 'Orbitron', sans-serif !important;
    text-shadow: 0 0 8px rgba(0, 245, 255, 0.4);
  }
  .stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: transparent;
    border-bottom: 1px solid rgba(0, 245, 255, 0.2);
  }
  /* Keep system UI font on tab row so scroll arrows / chevrons render (Orbitron lacks those glyphs). */
  .stTabs [data-baseweb="tab"] {
    font-family: system-ui, "Segoe UI", sans-serif !important;
    letter-spacing: 0.04em;
    font-size: 0.8125rem !important;
    color: #7a8199;
    border-radius: 4px 4px 0 0;
    border: 1px solid transparent;
  }
  .stTabs [aria-selected="true"] {
    color: #00f5ff !important;
    background: rgba(0, 245, 255, 0.08) !important;
    border: 1px solid rgba(0, 245, 255, 0.35) !important;
    border-bottom: none !important;
    box-shadow: 0 -4px 16px rgba(0, 245, 255, 0.12);
  }
  .cyber-hero {
    border: 1px solid rgba(0, 245, 255, 0.35);
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    background: linear-gradient(125deg, rgba(18, 18, 31, 0.95) 0%, rgba(26, 10, 46, 0.5) 100%);
    box-shadow: 0 0 40px rgba(255, 0, 234, 0.06), inset 0 1px 0 rgba(255,255,255,0.05);
  }
  .cyber-hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.65rem !important;
    color: #e8e8f0 !important;
    -webkit-text-fill-color: #e8e8f0 !important;
    background: none !important;
    background-clip: unset !important;
    -webkit-background-clip: unset !important;
  }
  /* Streamlit anchor / copy-link next to markdown headings */
  [data-testid="stHeaderActionElements"],
  [data-testid="stHeadingWithActionElements"] button,
  [data-testid="stMarkdownContainer"] h1 a,
  [data-testid="stMarkdownContainer"] h2 a,
  [data-testid="stMarkdownContainer"] h3 a,
  [data-testid="stMarkdownContainer"] h4 a,
  [data-testid="stMarkdownContainer"] h5 a,
  [data-testid="stMarkdownContainer"] h6 a {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
  }
  [data-testid="stDeployButton"],
  [data-testid="stToolbarDeployButton"] {
    display: none !important;
  }
  .cyber-section-title {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.06em;
    color: #00f5ff !important;
    margin: 0.5rem 0 0.75rem 0;
    border-bottom: 1px solid rgba(0, 245, 255, 0.2);
    padding-bottom: 0.35rem;
  }
  .cyber-panel {
    border-left: 3px solid #ff00ea;
    padding: 0.85rem 1rem;
    margin: 0.75rem 0;
    background: rgba(18, 18, 31, 0.65);
    border-radius: 0 6px 6px 0;
  }
  .cyber-sidebar-title {
    font-family: 'Orbitron', sans-serif !important;
    color: #00f5ff !important;
    font-size: 1rem !important;
    margin: 0 0 0.75rem 0 !important;
    letter-spacing: 0.05em;
  }
  .cyber-tag {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    border: 1px solid #00f5ff;
    color: #00f5ff;
    border-radius: 2px;
    margin-right: 0.5rem;
    font-family: 'Share Tech Mono', monospace;
  }
  div[data-testid="stExpander"] {
    border: 1px solid rgba(191, 95, 255, 0.25);
    border-radius: 6px;
    background: rgba(15, 15, 26, 0.6);
  }
  .stDownloadButton button {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.1em;
    border: 1px solid #00f5ff !important;
    color: #0a0a12 !important;
    background: linear-gradient(90deg, #00f5ff, #bf5fff) !important;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.35);
  }
</style>
"""


def inject_cyberpunk_theme() -> None:
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)
# Ensure the CSS is defined as a string first
CYBERPUNK_CSS = """
<style>
/* Force all interpretation and body text to be off-white/bright */
.stMarkdown p, .stMarkdown span, .stMarkdown div {
    color: #595d76 !important;
    line-height: 1.6;
}

/* Specifically target the 'Memory / consolidation read' headers */
h1, h2, h3, h4 {
    color: #6fdee3 !important; 
}
</style>
"""

def inject_cyberpunk_theme() -> None:
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)

def _cyber_plotly_base(fig: go.Figure) -> None:
    fig.update_layout(
        template=None,
        paper_bgcolor=CYBER_PANEL,
        plot_bgcolor=CYBER_BG,
        # Updated to the brighter #595d76 color and slightly larger size
        font=dict(color="#595d76", family="Share Tech Mono, monospace", size=12),
        title_font=dict(color=CYBER_ACCENT, size=14, family="Orbitron, sans-serif"),
        xaxis=dict(gridcolor=CYBER_GRID, zerolinecolor=CYBER_GRID),
        yaxis=dict(gridcolor=CYBER_GRID, zerolinecolor=CYBER_GRID),
    )


def load_hypnogram(hyp_path: Path) -> np.ndarray:
    """Specialized loader for hypnogram files with headers."""
    try:
        # Read the file as raw bytes first to handle special characters
        content = hyp_path.read_bytes()
        
        # The 'actual' stage data usually starts after the 'Hypnogram' keyword
        # and a long run of empty space or headers.
        # We look for the common integer markers (0, 1, 2, 3, 4, 5)
        # For your specific file format (Sleep-EDF), the stages are often at the end.
        
        # We'll try to find the start of the numeric sequence.
        # Most of these files have a predictable structure where the stages 
        # start after a specific byte offset.
        
        # NEW STRATEGY: Find the last few hundred bytes which contain the stages
        # Stages are usually represented as integers (0=W, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM)
        
        stages = []
        # We start looking for the stages after the header info
        header_end_signal = b"None"
        start_index = content.find(header_end_signal)
        
        if start_index != -1:
            # Skip past "None" and the extra spacing
            raw_stages = content[start_index + 4:].strip()
            # Convert bytes to integers, ignoring non-numeric junk
            for b in raw_stages:
                if 0 <= b <= 6: # Standard sleep stages are 0-6
                    stages.append(int(b))
        
        if not stages:
            # Fallback: if the logic above fails, try to just grab everything 
            # that looks like a stage byte from the whole file
            stages = [int(b) for b in content if 0 <= b <= 6]

        hyp = np.array(stages)
        
        # If the hypnogram is massive (e.g., thousands of entries), 
        # it's likely sampled at 1-second intervals. 
        # We need it at 30-second intervals for the rest of the app logic.
        if len(hyp) > 5000: 
             hyp = hyp[::EPOCH_SEC]
             
        return hyp

    except Exception as e:
        st.error(f"Error parsing .hyp file: {e}")
        return np.array([])


def hyp_to_annotations(hyp: np.ndarray, max_duration: float) -> mne.Annotations:
    onset = np.arange(len(hyp), dtype=float) * EPOCH_SEC
    duration = np.full(len(hyp), EPOCH_SEC, dtype=float)
    
    # Safety: Clip any annotation that goes past the end of the recording
    mask = onset < max_duration
    onset = onset[mask]
    duration = duration[mask]
    # Ensure the last annotation doesn't overstep
    if len(onset) > 0 and (onset[-1] + duration[-1] > max_duration):
        duration[-1] = max_duration - onset[-1]
        
    desc = [f"Stage {int(hyp[i])}" for i in range(len(onset))]
    return mne.Annotations(onset=onset, duration=duration, description=desc)


def find_channel(raw: mne.io.BaseRaw, requested: str) -> str:
    if requested in raw.ch_names:
        return requested
    lowered = requested.lower()
    for ch in raw.ch_names:
        if lowered in ch.lower():
            return ch
    raise ValueError(f"Channel '{requested}' not found.")


def read_raw_from_uploaded_rec(rec_path: Path) -> mne.io.BaseRaw:
    if rec_path.suffix.lower() == ".rec":
        td = tempfile.TemporaryDirectory()
        edf_copy = Path(td.name) / f"{rec_path.stem}.edf"
        shutil.copyfile(rec_path, edf_copy)
        raw = mne.io.read_raw_edf(edf_copy, preload=True, verbose=False)
        td.cleanup()
        return raw
    return mne.io.read_raw_edf(rec_path, preload=True, verbose=False)


@st.cache_data(show_spinner="Loading and analyzing EEG…")
def analyze_sleep(
    rec_bytes: bytes,
    hyp_bytes: bytes,
    _rec_suffix: str,
) -> dict:
    import io
    
    # We use a temporary directory to handle the .rec extension trick
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Create a file with an .edf suffix so MNE knows how to read it
        rec_path = td_path / "data.edf" 
        hyp_path = td_path / "upload.hyp"
        
        rec_path.write_bytes(rec_bytes)
        hyp_path.write_bytes(hyp_bytes)

        # Load the EEG
        raw = mne.io.read_raw_edf(rec_path, preload=True, verbose=False)
        sf = float(raw.info["sfreq"])
        
        # Load the Hypnogram
        hyp_epoch = load_hypnogram(hyp_path)
        
        # Ensure the hypnogram and EEG match in length
        # YASA needs these to be aligned perfectly
# Get actual duration of the raw data
        max_dur = raw.n_times / sf
        raw.set_annotations(hyp_to_annotations(hyp_epoch, max_dur))
      
# --- START OF TRIMMING LOGIC ---
        eeg_dur = raw.n_times / sf
        hyp_dur = len(hyp_epoch) * EPOCH_SEC
        
        # 1. Align the objects
        if hyp_dur > eeg_dur:
            n_epochs_to_keep = int(eeg_dur // EPOCH_SEC)
            hyp_epoch = hyp_epoch[:n_epochs_to_keep]
        elif eeg_dur > hyp_dur:
            raw.crop(0, hyp_dur)

        # 2. Extract the data
        ch_name = find_channel(raw, DEFAULT_CHANNEL)
        data = raw.get_data(picks=[ch_name]).squeeze() * 1e6
        times = raw.times.copy()

        # 3. Final matching (The "Safety Valve")
        # Ensure hyp_epoch doesn't represent more time than 'data' has
        expected_hyp_len = int(len(data) / (sf * EPOCH_SEC))
        if len(hyp_epoch) > expected_hyp_len:
            hyp_epoch = hyp_epoch[:expected_hyp_len]
        # --- END OF TRIMMING LOGIC ---
        # Now proceed with upsampling
        hyp_sample = yasa.hypno_upsample_to_data(
            hypno=hyp_epoch, sf_hypno=1 / EPOCH_SEC, data=data, sf_data=sf
        )
        # --- END OF TRIMMING LOGIC ---

        # Now proceed with upsampling
        hyp_sample = yasa.hypno_upsample_to_data(
            hypno=hyp_epoch, sf_hypno=1 / EPOCH_SEC, data=data, sf_data=sf
        )
        ch_name = find_channel(raw, DEFAULT_CHANNEL)
        data = raw.get_data(picks=[ch_name]).squeeze() * 1e6
        times = raw.times.copy()

        # Resample hypnogram to match EEG data points
        hyp_sample = yasa.hypno_upsample_to_data(
            hypno=hyp_epoch, sf_hypno=1 / EPOCH_SEC, data=data, sf_data=sf
        )
        
        # Filter for only N2 and N3 sleep stages
        hyp_for_detect = hyp_sample.copy()
        hyp_for_detect[~np.isin(hyp_for_detect, [N2_STAGE, N3_STAGE])] = -99

        # Run YASA detections
        sp = yasa.spindles_detect(
            data=data, sf=sf, ch_names=[ch_name],
            hypno=hyp_for_detect, include=(N2_STAGE, N3_STAGE), verbose=False,
        )
        sw = yasa.sw_detect(
            data=data, sf=sf, ch_names=[ch_name],
            hypno=hyp_for_detect, include=(N2_STAGE, N3_STAGE),
            coupling=True, verbose=False,
        )

        sp_df = sp.summary() if sp is not None else pd.DataFrame()
        sw_df = sw.summary() if sw is not None else pd.DataFrame()

        n2n3_epochs = int(np.isin(hyp_epoch, [N2_STAGE, N3_STAGE]).sum())
        n2n3_minutes = n2n3_epochs * EPOCH_SEC / 60.0

        phase = None
        sigma_peak = None
        if not sw_df.empty and "PhaseAtSigmaPeak" in sw_df.columns:
            mask = sw_df["PhaseAtSigmaPeak"].notna()
            if mask.any():
                phase = sw_df.loc[mask, "PhaseAtSigmaPeak"].to_numpy(dtype=float)
                if "SigmaPeak" in sw_df.columns:
                    sigma_peak = sw_df.loc[mask, "SigmaPeak"].to_numpy(dtype=float)

        return {
            "ch_name": ch_name,
            "sf": sf,
            "times": times,
            "data": data,
            "hyp_epoch": hyp_epoch,
            "sp_df": sp_df,
            "sw_df": sw_df,
            "n2n3_minutes": n2n3_minutes,
            "n_spindles": int(len(sp_df)),
            "n_slow_waves": int(len(sw_df)),
            "phase": phase,
            "sigma_peak": sigma_peak,
            "duration_sec": float(times[-1]) if len(times) else 0.0,
        }


def default_window_start(
    hyp_epoch: np.ndarray,
    sp_df: pd.DataFrame,
    duration: float,
) -> float:
    max_start = max(0.0, duration - EPOCH_SEC)
    n2_epochs = np.where(hyp_epoch == N2_STAGE)[0]
    if len(n2_epochs) == 0:
        return 0.0
    if not sp_df.empty and {"Start", "End"}.issubset(sp_df.columns):
        for ep in n2_epochs:
            s = float(ep * EPOCH_SEC)
            e = s + EPOCH_SEC
            if ((sp_df["End"] > s) & (sp_df["Start"] < e)).any():
                return min(s, max_start)
    return min(float(n2_epochs[0] * EPOCH_SEC), max_start)


def build_report_row(result: dict) -> pd.DataFrame:
    n2n3_min = result["n2n3_minutes"]
    n_sp = result["n_spindles"]
    density = (n_sp / n2n3_min) if n2n3_min > 0 else np.nan
    coupling_strength = np.nan
    if result["phase"] is not None and len(result["phase"]) > 0:
        rad = np.deg2rad(result["phase"])
        coupling_strength = float(np.abs(np.mean(np.exp(1j * rad))))
    return pd.DataFrame(
        [
            {
                "channel": result["ch_name"],
                "duration_sec": result["duration_sec"],
                "n2n3_total_min": n2n3_min,
                "total_spindles_n2n3": n_sp,
                "total_slow_waves_n2n3": result["n_slow_waves"],
                "spindle_density_per_min": density,
                "so_spindle_coupling_strength": coupling_strength,
            }
        ]
    )


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute_cognitive_outlook(report: pd.Series) -> dict:
    """
    Heuristic composite index (0–100) and narrative tiers — not a clinical diagnosis.
    Reference bands are illustrative for single-night lab PSG context.
    """
    density = float(report["spindle_density_per_min"])
    if np.isnan(density):
        density = 0.0
    coupling = report["so_spindle_coupling_strength"]
    if pd.isna(coupling):
        coupling = 0.35
    n2n3_min = float(report["n2n3_total_min"])
    n_sw = int(report["total_slow_waves_n2n3"])
    sw_per_min = (n_sw / n2n3_min) if n2n3_min > 0 else 0.0

    spindle_score = _clamp01((density - 0.25) / (4.5 - 0.25)) * 100
    coupling_score = _clamp01((float(coupling) - 0.12) / (0.72 - 0.12)) * 100
    sw_score = _clamp01((sw_per_min - 1.5) / (14.0 - 1.5)) * 100

    composite = 0.38 * spindle_score + 0.34 * coupling_score + 0.28 * sw_score
    composite = float(np.clip(composite, 0, 100))

    if composite >= 78:
        tier = "NEURAL PRIME"
        tier_note = "Markers align with strong NREM oscillatory scaffolding."
        memory = (
            "Higher spindle throughput and tighter SO–spindle phase locking are "
            "often linked in research to **overnight consolidation** of declarative memories "
            "(hippocampal–neocortical dialogue models)."
        )
        quality = "Estimated sleep-quality signal: **high** for oscillatory richness in N2/N3."
    elif composite >= 58:
        tier = "SYNAPTIC STABLE"
        tier_note = "Physiology sits in a typical functional band for healthy adults."
        memory = (
            "Oscillatory metrics suggest **adequate** windows for memory replay–like dynamics; "
            "inter-individual baselines vary widely with age, medication, and sleep debt."
        )
        quality = "Estimated sleep-quality signal: **moderate–strong**."
    elif composite >= 38:
        tier = "BUFFER DRIFT"
        tier_note = "Some oscillatory features run below an optimal research band."
        memory = (
            "Reduced spindle density or weaker coupling can co-occur with **fragmented deep sleep** "
            "or prior sleep restriction; cognitive impact is context-dependent."
        )
        quality = "Estimated sleep-quality signal: **moderate** — worth contextual review."
    else:
        tier = "LOW-BANDWIDTH NREM"
        tier_note = "Several markers fall in a lower range for this heuristic model."
        memory = (
            "Low oscillatory markers may align with **sleep loss, aging, or pathology** in some cohorts, "
            "but a single night and one EEG derivation cannot infer diagnosis."
        )
        quality = "Estimated sleep-quality signal: **lower** on this composite — interpret cautiously."

    return {
        "composite": composite,
        "spindle_score": spindle_score,
        "coupling_score": coupling_score,
        "sw_score": sw_score,
        "sw_per_min": sw_per_min,
        "tier": tier,
        "tier_note": tier_note,
        "memory_narrative": memory,
        "quality_narrative": quality,
    }


def plotly_so_spindle_coupling(phase_deg: np.ndarray, sigma: np.ndarray) -> go.Figure:
    phase_deg = np.asarray(phase_deg, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    s_min, s_max = float(sigma.min()), float(sigma.max())
    r = (sigma - s_min) / (s_max - s_min + 1e-12) * 0.85 + 0.15

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=phase_deg,
            mode="markers",
            marker=dict(
                size=10,
                color=sigma,
                colorscale=[[0, "#1a0a2e"], [0.5, CYBER_ACCENT], [1, CYBER_MAGENTA]],
                showscale=True,
                colorbar=dict(
                    title=dict(text="σ peak", font=dict(color="#b8c0d8")),
                    tickfont=dict(color="#8890a8"),
                ),
                opacity=0.9,
                line=dict(width=0.5, color="rgba(0,245,255,0.4)"),
            ),
            name="SO–spindle events",
        )
    )

    mean_rad = np.angle(np.mean(np.exp(1j * np.deg2rad(phase_deg))))
    mean_deg = float(np.rad2deg(mean_rad))
    fig.add_trace(
        go.Scatterpolar(
            r=[0.0, 1.0],
            theta=[mean_deg, mean_deg],
            mode="lines",
            line=dict(color=CYBER_MAGENTA, width=3),
            name="Mean phase vector",
        )
    )

    fig.update_layout(
        title=dict(
            text="Slow oscillation phase at spindle (sigma) peak",
            font=dict(size=15, color=CYBER_ACCENT, family="Orbitron, sans-serif"),
        ),
        polar=dict(
            bgcolor=CYBER_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=CYBER_GRID,
                linecolor=CYBER_GRID,
                title=dict(text="norm σ", font=dict(color="#8890a8", size=10)),
            ),
            angularaxis=dict(
                thetaunit="degrees",
                rotation=90,
                direction="counterclockwise",
                gridcolor=CYBER_GRID,
                linecolor=CYBER_GRID,
            ),
        ),
        showlegend=True,
        height=520,
        margin=dict(l=40, r=50, t=56, b=40),
        paper_bgcolor=CYBER_PANEL,
        font=dict(color="#c5cee0", family="Share Tech Mono, monospace"),
     legend=dict(
            font=dict(color="#b8c0d8", size=10),
            bgcolor="rgba(10,10,18,0.85)",
            bordercolor=CYBER_ACCENT,
            borderwidth=1,
            # --- ADD THESE THREE LINES BELOW ---
            orientation="h",       # Makes the legend horizontal
            yanchor="bottom",      # Anchors the legend to its bottom
            y=-0.2,                # Moves it below the plot (out of the way)
            xanchor="center",
            x=0.5
        ),
    )
    return fig


def plotly_eeg_window(
    times: np.ndarray,
    data: np.ndarray,
    sp_df: pd.DataFrame,
    hyp_sample: np.ndarray,  # Add this
    window_start: float,
    window_sec: float,
    ch_name: str,
    sf: float,               # Add this
) -> go.Figure:
    t_end = window_start + window_sec
    mask = (times >= window_start) & (times < t_end)
    x = times[mask] - window_start
    y = data[mask]
    
    # Extract the hypnogram slice for this window
    idx_start = int(window_start * sf)
    idx_end = int(t_end * sf)
    win_hyp = hyp_sample[idx_start:idx_end]

    fig = go.Figure()

    # --- ADD N2/N3 STAGE HIGHLIGHTS ---
    # This colors the background based on the sleep stage
    for i in range(0, len(win_hyp), int(sf)): # Check every 1 second to save performance
        stage = win_hyp[i]
        if stage in [2, 3]: # N2 or N3
            color = "rgba(100, 149, 237, 0.1)" if stage == 2 else "rgba(0, 0, 139, 0.15)"
            fig.add_vrect(
                x0=i/sf, x1=(i + int(sf))/sf,
                fillcolor=color, layer="below", line_width=0
            )

    # --- EXISTING SPINDLE HIGHLIGHTS ---
    if not sp_df.empty and {"Start", "End"}.issubset(sp_df.columns):
        for _, row in sp_df.iterrows():
            s = max(float(row["Start"]), window_start)
            e = min(float(row["End"]), t_end)
            if e > s:
                fig.add_vrect(
                    x0=s - window_start,
                    x1=e - window_start,
                    fillcolor="rgba(255, 0, 234, 0.22)", # Pink for spindles
                    layer="below",
                    line_width=0,
                )

    # The EEG Line
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name=ch_name,
            line=dict(color=CYBER_ACCENT, width=1.1),
        )
    )
    
    _cyber_plotly_base(fig) # Apply your theme
    return fig

    fig.update_layout(
        title=dict(
            text=f"EEG ({ch_name}) — {window_sec:.0f} s window",
            font=dict(size=14, color=CYBER_ACCENT, family="Orbitron, sans-serif"),
        ),
        xaxis_title="time (s)",
        yaxis_title="µV",
        height=440,
        margin=dict(l=56, r=16, t=52, b=48),
        hovermode="x unified",
        dragmode="zoom",
    )
    _cyber_plotly_base(fig)
    fig.update_xaxes(range=[0, window_sec])
    return fig


def main():
    # --- UI STYLING ---
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, #3D4DC2 0%, #1a1a2e 100%);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }}
        @keyframes gradient {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        </style>
    """, unsafe_allow_html=True)

    # ... [Your existing data loading/analysis code here] ...

    # After your analysis is done and report_df is created:
    report_df = pd.DataFrame([stats]) # Assuming 'stats' is your results dict

    tab_raw, tab_outlook, tab_stats, tab_research = st.tabs(
        ["EEG view", "Interpretation", "Summary table", "Coupling plot"]
    )

    # ... [Tab logic goes here] ...

    # --- FIX FOR NAMEERROR ---
    # Move any logic that uses report_df INSIDE main()
    st.sidebar.header("Export Data")
    if st.sidebar.button("Prepare Export"):
        export = report_df.iloc[0].to_dict()
        st.sidebar.json(export)
        st.sidebar.success("Metadata extracted successfully.")

# This ensures main() runs and all variables are contained within it
if __name__ == "__main__":
    main()
    inject_cyberpunk_theme()

    st.markdown(
        """
        <div class="cyber-hero">
          <h1>Sleep analysis</h1>
          <p style="color:#9aa4c4; margin:0; font-size:0.95rem;">
            NREM oscillations (N2 and N3), Fpz-Cz channel, detected with YASA.
            For research or education only; not a medical diagnosis.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            '<p class="cyber-sidebar-title">Data upload</p>',
            unsafe_allow_html=True,
        )
        rec_f = st.file_uploader("EEG (.rec / .edf)", type=["rec", "edf"])
        hyp_f = st.file_uploader("Hypnogram (.hyp / .edf)", type=["hyp", "edf"])
        st.markdown(
            "<p style='color:#6a7088;font-size:0.75rem;margin-top:1.5rem;'>"
            "This tool does not replace clinical sleep scoring.</p>",
            unsafe_allow_html=True,
        )

    if rec_f is None or hyp_f is None:
        st.markdown(
            "<div class='cyber-panel'><b style='color:#ffcc00;'>No files yet</b> — "
            "Upload an EEG file and a hypnogram in the sidebar to run the analysis.</div>",
            unsafe_allow_html=True,
        )
        return

    rec_suffix = Path(rec_f.name).suffix.lower() or ".rec"
    rec_bytes = rec_f.getvalue()
    hyp_bytes = hyp_f.getvalue()

    try:
        result = analyze_sleep(rec_bytes, hyp_bytes, rec_suffix)
    except Exception as e:
        st.error(f"Could not load or analyze the files: {e}")
        return

    times = result["times"]
    data = result["data"]
    sp_df = result["sp_df"]
    duration = result["duration_sec"]
    max_start = max(0.0, duration - EPOCH_SEC)
    report_df = build_report_row(result)
    outlook = compute_cognitive_outlook(report_df.iloc[0])

    upload_key = f"{rec_f.name}:{len(rec_bytes)}_{hyp_f.name}:{len(hyp_bytes)}"
    if st.session_state.get("_sleep_upload_key") != upload_key:
        st.session_state["_sleep_upload_key"] = upload_key
        st.session_state.pop("win_slider", None)

    if "win_slider" not in st.session_state:
        st.session_state["win_slider"] = float(
            default_window_start(result["hyp_epoch"], sp_df, duration)
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("OSCILLATORY INDEX", f"{outlook['composite']:.0f}", help="Heuristic 0–100 blend of spindle density, SW rate, coupling.")
    with c2:
        st.metric("SPINDLES (N2+N3)", f"{result['n_spindles']:,}")
    with c3:
        st.metric("SLOW WAVES", f"{result['n_slow_waves']:,}")
    with c4:
        d = report_df.iloc[0]["spindle_density_per_min"]
        st.metric("SPINDLE DENSITY", f"{d:.2f}/min" if pd.notna(d) else "—")
    with c5:
        st.metric("COUPLING (R)", f"{report_df.iloc[0]['so_spindle_coupling_strength']:.3f}" if pd.notna(report_df.iloc[0]["so_spindle_coupling_strength"]) else "—")

    tab_raw, tab_outlook, tab_stats, tab_research = st.tabs(
        ["EEG view", "Interpretation", "Summary table", "Coupling plot"]
    )

    # These variables must be extracted from the 'result' dictionary inside main()
    # and the logic must be properly indented to stay inside the function.
    sf = result['sf'] 
    hyp_sample = result.get('hyp_sample') # Use .get if it might be missing

    with tab_raw:
        st.markdown(
            '<p class="cyber-section-title">EEG trace (30 s)</p>',
            unsafe_allow_html=True,
        )
        st.caption("Spindle detections are shaded in magenta.")
        
        win_start = st.slider(
            "Window offset (s)",
            min_value=0.0,
            max_value=float(max_start),
            step=1.0,
            key="win_slider",
            help="30 s viewport. Use Plotly controls to zoom.",
        )
        
        # Fixed: passed hyp_sample variable instead of the string ['hyp_sample']
        fig_eeg = plotly_eeg_window(times, data, sp_df, hyp_sample, win_start, 30.0, result["ch_name"], sf)
        st.plotly_chart(fig_eeg, use_container_width=True)

    with tab_outlook:
        st.markdown(
            '<p class="cyber-section-title">Cognitive outlook (heuristic)</p>',
            unsafe_allow_html=True,
        )
      
        st.markdown(
            f"<div class='cyber-panel'><span class='cyber-tag'>{outlook['tier']}</span>"
            f"<span style='color:#e8e8f0'>{outlook['tier_note']}</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(outlook["composite"] / 100.0)
        st.caption(
            f"Composite index {outlook['composite']:.1f}/100 "
            f"(spindle {outlook['spindle_score']:.0f} · coupling {outlook['coupling_score']:.0f} · SW/min {outlook['sw_per_min']:.1f} → SW score {outlook['sw_score']:.0f})"
        )
        st.markdown("**Memory / consolidation read (research framing)**")
        st.write(outlook["memory_narrative"])
        st.markdown("**Sleep-quality signal (non-clinical)**")
        st.write(outlook["quality_narrative"])

    with tab_stats:
        st.markdown('<p class="cyber-section-title">Metrics Summary</p>', unsafe_allow_html=True)
        st.dataframe(report_df, use_container_width=True)

    with tab_research:
        if result["phase"] is not None:
            fig_polar = plotly_so_spindle_coupling(result["phase"], result["sigma_peak"])
            st.plotly_chart(fig_polar, use_container_width=True)
        else:
            st.info("No coupling data available for this recording.")

if __name__ == "__main__":
    main()

    st.divider()
    export = report_df.iloc[0].to_dict()
    export["oscillatory_index_0_100"] = round(outlook["composite"], 2)
    export["outlook_tier"] = outlook["tier"]
    report_csv = pd.DataFrame([export]).to_csv(index=False)
    cols = st.columns([3, 1])
    with cols[0]:
        st.caption("Download the summary metrics as a CSV file.")
    with cols[1]:
        st.download_button(
            label="DOWNLOAD CSV",
            data=report_csv,
            file_name="sleep_eeg_report.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
