"""
Campus Pedestrian Detection & Counting  —  Presentation Dashboard (v3)
=======================================================================
Polished, minimal research presentation dashboard for faculty demo.

Run:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config  ·  must be the first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAV Multi-Object Detection Framework",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  —  minimal academic theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Globals ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    .stApp { background: #f8fafc; }

    /* ── Hide sidebar completely ── */
    [data-testid="stSidebar"]       { display: none !important; }
    [data-testid="collapsedControl"]{ display: none !important; }

    /* ── Main container — full width, generous padding ── */
    .main .block-container {
        padding: 0 3rem 5rem !important;
        max-width: 1380px !important;
    }

    /* ── Base font size ── */
    html { font-size: 17px; }

    /* ── Hero banner ── */
    .hero {
        background: #0f172a;
        color: #f8fafc;
        margin: 0 -3rem 0;
        padding: 3.5rem 3.2rem 3rem;
    }
    .hero-eyebrow {
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.85rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 300;
        color: #f1f5f9;
        letter-spacing: -0.5px;
        line-height: 1.25;
        margin-bottom: 0.6rem;
    }
    .hero-title strong { font-weight: 700; }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.7rem;
        font-weight: 400;
    }
    .badge {
        display: inline-block;
        border: 1px solid #334155;
        border-radius: 99px;
        padding: 0.32rem 1rem;
        font-size: 0.88rem;
        font-weight: 500;
        color: #cbd5e1;
        margin-right: 0.45rem;
        letter-spacing: 0.02em;
    }

    /* ── Section label ── */
    .sec-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 2.8rem 0 1.1rem;
        padding-bottom: 0.55rem;
        border-bottom: 1px solid #e2e8f0;
    }

    /* ── Metric card ── */
    .m-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.6rem 1rem 1.4rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    }
    .m-card .val {
        font-size: 2.3rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .m-card .lbl {
        font-size: 0.82rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
    }
    .m-card-sm .val {
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: -0.3px;
    }

    /* ── Video label ── */
    .vid-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.55rem;
    }

    /* ── Finding card ── */
    .f-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.7rem 1.6rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    }
    .f-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-bottom: 1.05rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #f1f5f9;
    }
    .f-item {
        font-size: 1rem;
        color: #334155;
        padding: 0.55rem 0 0.55rem 1.2rem;
        border-bottom: 1px solid #f8fafc;
        position: relative;
        line-height: 1.5;
    }
    .f-item:last-child { border-bottom: none; }
    .f-item::before {
        content: '–';
        position: absolute;
        left: 0;
        color: #cbd5e1;
        font-weight: 400;
    }

    /* ── Method card ── */
    .meth-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.6rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    }
    .meth-tag {
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #2563eb;
        margin-bottom: 0.4rem;
    }
    .meth-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.55rem;
    }
    .meth-body {
        font-size: 1.02rem;
        color: #475569;
        line-height: 1.65;
    }

    /* ── Placeholder ── */
    .ph-box {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 5rem 2rem;
        text-align: center;
    }
    .ph-title { font-size: 1.2rem; font-weight: 600; color: #334155; margin-bottom: 0.55rem; }
    .ph-sub   { font-size: 1rem;   color: #94a3b8; }
    .ph-path  { font-size: 0.92rem; font-family: monospace; color: #64748b;
                background: #f1f5f9; padding: 0.25rem 0.55rem; border-radius: 4px; margin: 0.25rem 0; display: inline-block; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: #0f172a !important;
        padding: 0 3rem !important;
        margin: 0 -3rem 2.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #64748b !important;
        padding: 1.1rem 1.5rem !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
        letter-spacing: 0.01em;
    }
    .stTabs [aria-selected="true"] {
        color: #f1f5f9 !important;
        border-bottom-color: #2563eb !important;
        font-weight: 600 !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: #0f172a !important;
        color: #f1f5f9 !important;
        border: none !important;
        border-radius: 7px !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.4rem !important;
        letter-spacing: 0.02em;
    }
    .stDownloadButton > button:hover { background: #1e293b !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #e2e8f0; font-size: 0.95rem; }

    /* ── Streamlit default overrides ── */
    h1, h2, h3, h4, h5 { font-family: 'Inter', sans-serif !important; }
    .stCheckbox label { font-size: 1rem !important; color: #475569 !important; }
    p, li { font-size: 1rem !important; line-height: 1.6 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Location registry  —  add new sites by extending this dict only
# ─────────────────────────────────────────────────────────────────────────────
LOCATIONS: dict[str, dict] = {
    "University Park": {
        "input_video":   "data/university_park/input.mp4",
        "output_video":  "data/university_park/output.mp4",
        "csv":           "data/university_park/results.csv",
        "model":         "YOLOv8m",
        "tracker":       "ByteTrack",
        "count_method":  "Multi-Line · 3 Vertical Lines",
        "line_pos":      "25% · 50% · 75%",
        "conf":          "0.35",
        "smooth_alpha":  "0.6",
        "track_buffer":  "10 frames",
        "ready": True,
    },
    "Railroad Crossing": {
        "input_video":   "data/railroad_crossing/input.mp4",
        "output_video":  "data/railroad_crossing/output.mp4",
        "csv":           "data/railroad_crossing/results.csv",
        "model":         "YOLOv8l",
        "tracker":       "BOTSort",
        "count_method":  "Multi-Line · 3 Vertical Lines",
        "line_pos":      "25% · 50% · 75%",
        "conf":          "0.10",
        "smooth_alpha":  "0.75",
        "track_buffer":  "45 frames",
        "ready": True,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce")
    df["counted"]       = df["counted"].astype(str).str.lower() == "true"
    return df


@st.cache_data(show_spinner=False)
def load_video_bytes(path: str) -> bytes | None:
    p = Path(path)
    return p.read_bytes() if p.exists() else None


def get_metrics(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Compute summary metrics.

    Note on track counting:
      Raw `track_id.nunique()` is misleading because trackers create a new
      ID every time the detector momentarily fires (even on noise like a
      bird or leaf). To get a meaningful "people seen" number we filter
      to tracks that lasted ≥ 30 frames — roughly 1 second at 30 fps,
      which matches a real person walking through the frame.
    """
    dur   = df["timestamp_sec"].max()
    m, s  = divmod(int(dur), 60)
    fps   = round(df["frame_no"].max() / dur, 1) if dur > 0 else 0.0

    track_lengths     = df.groupby("track_id").size()
    confirmed_tracks  = int((track_lengths >= 30).sum())   # ≥ 1 second
    raw_tracks        = int(df["track_id"].nunique())

    return {
        "counted":      int(df["counted"].sum()),
        "tracks":       confirmed_tracks,
        "raw_tracks":   raw_tracks,
        "duration":     f"{m}:{s:02d}",
        "fps":          str(fps),
        "model":        cfg["model"],
        "tracker":      cfg["tracker"],
        "method":       cfg["count_method"],
        "confidence":   str(round(df["confidence"].mean(), 2)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers  —  consistent clean Plotly theme
# ─────────────────────────────────────────────────────────────────────────────
ACCENT   = "#2563eb"
MUTED    = "#94a3b8"
NAVY     = "#0f172a"
SLATE    = "#334155"


def _style(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title_text=title,
        title_font=dict(size=12, color=NAVY, family="Inter"),
        font=dict(family="Inter, sans-serif", size=11, color=SLATE),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=48, r=16, t=44, b=40),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0", tickfont_size=10)
    fig.update_yaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont_size=10)
    return fig


def chart_detections_per_second(df: pd.DataFrame) -> go.Figure:
    agg = (
        df.assign(sec=df["timestamp_sec"].astype(int))
        .groupby("sec").size().reset_index(name="n")
    )
    fig = go.Figure(go.Bar(
        x=agg["sec"], y=agg["n"],
        marker_color=ACCENT, marker_line_width=0,
    ))
    return _style(fig, "Detections per Second")


def chart_cumulative_count(df: pd.DataFrame) -> go.Figure:
    events = df[df["counted"]].sort_values("timestamp_sec").copy()
    fig = go.Figure()
    if not events.empty:
        events["cum"] = range(1, len(events) + 1)
        xs = [0.0] + events["timestamp_sec"].tolist()
        ys = [0]   + events["cum"].tolist()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=ACCENT, width=2.5, shape="hv"),
            name="Cumulative count",
        ))
        fig.add_trace(go.Scatter(
            x=events["timestamp_sec"], y=events["cum"],
            mode="markers",
            marker=dict(size=7, color="#dc2626", symbol="circle"),
            name="Count event",
        ))
    return _style(fig, "Cumulative People Counted")


def chart_track_frequency(df: pd.DataFrame, top: int = 20) -> go.Figure:
    counted_set = set(df[df["counted"]]["track_id"])
    freq = (
        df.groupby("track_id").size()
        .reset_index(name="frames")
        .sort_values("frames", ascending=False)
        .head(top)
    )
    freq["color"] = freq["track_id"].apply(
        lambda t: ACCENT if t in counted_set else "#cbd5e1"
    )
    fig = go.Figure(go.Bar(
        x=freq["track_id"].astype(str),
        y=freq["frames"],
        marker_color=freq["color"],
        marker_line_width=0,
    ))
    fig.update_xaxes(type="category")
    return _style(fig, f"Top {top} Tracks — Frames Detected")


def chart_confidence_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df["confidence"], nbinsx=30,
        marker_color=ACCENT, marker_line_width=0,
        opacity=0.85,
    ))
    mean_c = df["confidence"].mean()
    fig.add_vline(
        x=mean_c, line_dash="dot", line_color="#dc2626", line_width=1.5,
        annotation_text=f"mean {mean_c:.2f}",
        annotation_font=dict(size=10, color="#dc2626"),
        annotation_position="top right",
    )
    return _style(fig, "Detection Confidence Distribution")


def chart_class_breakdown(df: pd.DataFrame) -> go.Figure:
    """Bar chart of counted events grouped by class (multi-class CSVs)."""
    counts = (
        df[df["counted"]]
        .groupby("class_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )
    palette = {
        "person": "#22c55e", "bicycle": "#06b6d4", "car": "#2563eb",
        "motorcycle": "#f97316", "bus": "#a855f7", "truck": "#ef4444",
    }
    colors = [palette.get(c, "#94a3b8") for c in counts["class_name"]]
    fig = go.Figure(go.Bar(
        y=counts["class_name"], x=counts["count"],
        orientation="h",
        marker_color=colors, marker_line_width=0,
        text=counts["count"], textposition="outside",
        textfont=dict(size=12, color=NAVY),
    ))
    return _style(fig, "Counted Events by Class")


def chart_spatial_heatmap(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(6000, len(df)), random_state=42)
    fig = go.Figure(go.Histogram2dContour(
        x=sample["center_x"],
        y=sample["center_y"],
        colorscale="Blues",
        reversescale=False,
        ncontours=12,
        showscale=True,
    ))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Pedestrian Position Density")


# ─────────────────────────────────────────────────────────────────────────────
# UI components
# ─────────────────────────────────────────────────────────────────────────────
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Bridgewater State University &nbsp;·&nbsp; Directed Study &nbsp;·&nbsp; Spring 2026</div>
            <div class="hero-title">A Robust Computer Vision Framework<br>for <strong>Multi-Object and Vehicle Detection in UAV Imagery</strong></div>
            <div class="hero-subtitle">Faculty Advisor: Dr. Uma Shama</div>
            <span class="badge">YOLOv8m</span>
            <span class="badge">ByteTrack</span>
            <span class="badge">Multi-Line Counting</span>
            <span class="badge">EMA Smoothing</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(m: dict) -> None:
    st.markdown('<div class="sec-label">Key Metrics</div>', unsafe_allow_html=True)

    # Row 1 — primary numbers
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    primary = [
        (m["counted"],    "People Counted"),
        (m["tracks"],     "Confirmed Tracks (≥1s)"),
        (m["duration"],   "Video Duration"),
        (m["fps"],        "Source FPS"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], primary):
        with col:
            st.markdown(
                f'<div class="m-card"><div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # Row 2 — configuration
    c5, c6, c7, c8 = st.columns(4, gap="medium")
    secondary = [
        (m["model"],      "Model"),
        (m["tracker"],    "Tracker"),
        (m["method"],     "Counting Method"),
        (m["confidence"], "Avg. Confidence"),
    ]
    for col, (val, lbl) in zip([c5, c6, c7, c8], secondary):
        with col:
            st.markdown(
                f'<div class="m-card m-card-sm"><div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )


def render_videos(cfg: dict) -> None:
    st.markdown('<div class="sec-label">Video Comparison</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="vid-label">Original Footage</div>', unsafe_allow_html=True)
        raw = load_video_bytes(cfg["input_video"])
        if raw:
            st.video(raw)
        else:
            st.warning(f"Not found: `{cfg['input_video']}`")

    with col_b:
        st.markdown('<div class="vid-label">Processed Output</div>', unsafe_allow_html=True)
        out = load_video_bytes(cfg["output_video"])
        if out:
            st.video(out)
        else:
            st.warning(f"Not found: `{cfg['output_video']}`")


def render_charts(df: pd.DataFrame, key_prefix: str) -> None:
    st.markdown('<div class="sec-label">Analytics</div>', unsafe_allow_html=True)

    multi_class = df["class_name"].nunique() > 1

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        if multi_class:
            st.plotly_chart(chart_class_breakdown(df), width="stretch",
                            key=f"{key_prefix}_class")
        else:
            st.plotly_chart(chart_detections_per_second(df), width="stretch",
                            key=f"{key_prefix}_det_sec")
    with col2:
        st.plotly_chart(chart_cumulative_count(df), width="stretch",
                        key=f"{key_prefix}_cum")

    if multi_class:
        st.plotly_chart(chart_detections_per_second(df), width="stretch",
                        key=f"{key_prefix}_det_sec_full")

    col3, col4 = st.columns(2, gap="medium")
    with col3:
        st.plotly_chart(chart_track_frequency(df), width="stretch",
                        key=f"{key_prefix}_freq")
    with col4:
        st.plotly_chart(chart_confidence_histogram(df), width="stretch",
                        key=f"{key_prefix}_conf")

    st.plotly_chart(chart_spatial_heatmap(df), width="stretch",
                    key=f"{key_prefix}_heat")


def render_csv_section(df: pd.DataFrame, loc_slug: str) -> None:
    st.markdown('<div class="sec-label">Detection Log</div>', unsafe_allow_html=True)

    col_toggle, _ = st.columns([2, 6])
    with col_toggle:
        events_only = st.checkbox("Count events only", key=f"ev_{loc_slug}")

    display = df[df["counted"]] if events_only else df
    st.dataframe(display.reset_index(drop=True), height=300, width="stretch")

    st.download_button(
        label="Download CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name=f"{loc_slug}_results.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full location demo  (reused for every site tab)
# ─────────────────────────────────────────────────────────────────────────────
def render_location(loc_name: str, cfg: dict) -> None:
    slug = loc_name.lower().replace(" ", "_")

    if not cfg["ready"]:
        slug_path = slug
        st.markdown(
            f"""
            <div class="ph-box">
                <div class="ph-title">Data not yet available for {loc_name}</div>
                <div class="ph-sub">Add the following files to activate this view:</div>
                <br>
                <span class="ph-path">data/{slug_path}/input.mp4</span><br>
                <span class="ph-path">data/{slug_path}/output.mp4</span><br>
                <span class="ph-path">data/{slug_path}/results.csv</span><br>
                <br>
                <div class="ph-sub">Then set <code>"ready": True</code> in the LOCATIONS dict in app.py</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = load_csv(cfg["csv"])
    if df is None:
        st.error(f"CSV not found at `{cfg['csv']}`")
        return

    metrics = get_metrics(df, cfg)

    render_videos(cfg)
    render_metrics(metrics)
    render_charts(df, key_prefix=slug)
    render_csv_section(df, slug)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis tab content
# ─────────────────────────────────────────────────────────────────────────────
def render_analysis() -> None:
    st.markdown('<div class="sec-label">Key Findings</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    findings = [
        (
            col1, "What Worked",
            [
                "YOLOv8m detected pedestrians reliably across variable lighting conditions",
                "ByteTrack maintained stable IDs through most of the 59-second clip",
                "Three-line coverage eliminated missed counts from single-line gaps",
                "EMA smoothing visibly reduced bounding-box jitter between frames",
                "Track buffer filled short occlusion gaps without false counts",
            ],
        ),
        (
            col2, "Challenges",
            [
                "mp4v codec in OpenCV output is not browser-playable without re-encoding",
                "ID switches occur when two people walk very close together",
                "People near frame edges are partially clipped, reducing detection confidence",
                "Shadows cause false-positive detections at conf < 0.3",
                "Long absences from frame assign a new ID on re-entry, risking double-count",
            ],
        ),
        (
            col3, "Future Improvements",
            [
                "Fine-tune YOLOv8 on campus-specific annotated data for better precision",
                "Adopt BOTSort with Re-ID for denser, occluded crowd scenes",
                "Add zone-based counting for entrance/exit monitoring",
                "Deploy on Apple M-series device for real-time live stream analysis",
                "Aggregate hourly traffic data into a persistent dashboard for planners",
            ],
        ),
    ]

    for col, title, items in findings:
        with col:
            items_html = "".join(f'<div class="f-item">{i}</div>' for i in items)
            st.markdown(
                f'<div class="f-card">'
                f'<div class="f-card-title">{title}</div>'
                f'{items_html}'
                f'</div>',
                unsafe_allow_html=True,
            )



# ─────────────────────────────────────────────────────────────────────────────
# Methodology tab content
# ─────────────────────────────────────────────────────────────────────────────
def render_methodology() -> None:
    st.markdown('<div class="sec-label">Pipeline Overview</div>', unsafe_allow_html=True)

    methods = [
        (
            "Detection",
            "YOLOv8 — Person Detection",
            "YOLOv8m (medium) runs inference on every decoded frame and returns bounding boxes "
            "for class 0 (person) only. All other COCO classes are discarded before tracking. "
            "Confidence threshold: 0.35 · IoU threshold: 0.45.",
        ),
        (
            "Tracking",
            "ByteTrack — Multi-Frame Association",
            "ByteTrack maintains a persistent unique ID for each person across frames using a "
            "Kalman filter for motion prediction and the Hungarian algorithm for assignment. "
            "It leverages every detection — including low-confidence ones — to reduce ID switches "
            "in crowds. The persist=True flag passes tracker state between consecutive frames.",
        ),
        (
            "Counting",
            "Multi-Line Crossing Detection",
            "Three vertical counting lines at 25%, 50%, and 75% of the frame width ensure every "
            "pedestrian path is covered. A person is counted exactly once when their bounding-box "
            "centre crosses any line for the first time. A global counted_ids set prevents "
            "duplicate counting across multiple lines.",
        ),
        (
            "Smoothing",
            "Exponential Moving Average (EMA) on Bounding Boxes",
            "Raw detections jitter between frames due to small pose or lighting changes. "
            "Each box coordinate is smoothed using: smooth_t = 0.6 × smooth_{t−1} + 0.4 × raw_t. "
            "This eliminates high-frequency noise while preserving real movement.",
        ),
        (
            "Memory",
            "Track Buffer — Ghost Box Rendering",
            "When the detector misses a person for up to 10 consecutive frames (brief occlusion, "
            "frame edge), the last confirmed smoothed box is displayed as a grey ghost. "
            "Ghost boxes are visual only — they never influence counting logic.",
        ),
    ]

    for tag, title, body in methods:
        st.markdown(
            f"""
            <div class="meth-card">
                <div class="meth-tag">{tag}</div>
                <div class="meth-title">{title}</div>
                <div class="meth-body">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────
render_hero()

tab_up, tab_rr, tab_analysis, tab_method = st.tabs([
    "University Park",
    "Railroad Crossing",
    "Analysis",
    "Methodology",
])

with tab_up:
    render_location("University Park", LOCATIONS["University Park"])

with tab_rr:
    render_location("Railroad Crossing", LOCATIONS["Railroad Crossing"])

with tab_analysis:
    render_analysis()

with tab_method:
    render_methodology()
