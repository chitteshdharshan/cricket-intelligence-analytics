"""
Cricket Intelligence Design System & Reusable UI Components.
Professional sports analytics styling inspired by ESPNcricinfo & Cricbuzz.
Palette:
    Background: #080D18
    Sidebar:    #101827
    Cards:      #151F30
    Borders:    #263449
    Accent 1:   #16C784 (Emerald sports green)
    Accent 2:   #3B82F6 (Electric royal blue)
    Main text:  #F5F7FA
    Muted text: #94A3B8
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Color tokens
COLOR_BG = "#080D18"
COLOR_SIDEBAR = "#101827"
COLOR_CARD = "#151F30"
COLOR_BORDER = "#263449"
COLOR_PRIMARY = "#16C784"
COLOR_SECONDARY = "#3B82F6"
COLOR_TEXT = "#F5F7FA"
COLOR_MUTED = "#94A3B8"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#EF4444"

_CRICKET_LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "cricket_logo.svg"

# Embedded bat-and-ball SVG fallback in emerald & clean white
FALLBACK_CRICKET_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Cricket Logo">
  <title>Cricket Intelligence Logo</title>
  <path d="M34 19 43 7l5 4-9 13-5-5Z" fill="#16C784"/>
  <path d="m11 47 23-28 9 8-23 28a6 6 0 0 1-8-8Z" fill="#F5F7FA"/>
  <path d="m34 19 9 8-4 5-9-8 4-5Z" fill="#94A3B8"/>
  <circle cx="47" cy="47" r="9" fill="#16C784"/>
  <path d="M43 40c2 4 6 8 9 11m-11-8c3-2 7-3 11-2" fill="none" stroke="#080D18" stroke-linecap="round" stroke-width="1.5"/>
</svg>"""

FALLBACK_TEXT_LOGO = '<span class="ci-brand-fallback" role="img" aria-label="Cricket">🏏</span>'


def _find_cricket_logo_path() -> Optional[Path]:
    """Finds the local cricket logo SVG across common project directory structures."""
    candidate_paths = [
        Path(__file__).resolve().parents[1] / "assets" / "cricket_logo.svg",
        Path(__file__).resolve().parent.parent / "assets" / "cricket_logo.svg",
        Path(__file__).resolve().parent / "assets" / "cricket_logo.svg",
        Path.cwd() / "assets" / "cricket_logo.svg",
        Path.cwd() / "cricket_intelligence" / "assets" / "cricket_logo.svg",
    ]
    for p in candidate_paths:
        try:
            if p.is_file():
                return p
        except Exception:
            continue
    return None


def _cricket_logo_markup(as_text: bool = False) -> str:
    """Loads the local cricket logo SVG, with reliable local fallback and text fallback."""
    if as_text:
        return FALLBACK_TEXT_LOGO

    logo_path = _find_cricket_logo_path()
    if logo_path is not None:
        try:
            content = logo_path.read_text(encoding="utf-8").strip()
            if content.startswith("<svg") and content.endswith("</svg>"):
                return content
        except Exception:
            pass

    return FALLBACK_CRICKET_SVG


def apply_custom_theme():
    """Injects high-end sports analytics styling matching the professional dark theme."""
    st.markdown("""
        <style>
        /* Base page background & typography */
        .stApp {
            background-color: #080D18 !important;
            color: #F5F7FA !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }

        /* Sidebar container */
        section[data-testid="stSidebar"] {
            background-color: #101827 !important;
            border-right: 1px solid #263449 !important;
        }
        section[data-testid="stSidebar"] > div {
            background-color: #101827 !important;
        }

        /* Sidebar Navigation links */
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
            background-color: #101827 !important;
            padding-top: 4px;
        }
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a {
            border-radius: 6px !important;
            margin-bottom: 2px !important;
            color: #94A3B8 !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            padding: 8px 12px !important;
            transition: all 0.15s ease-in-out !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a:hover {
            background-color: #151F30 !important;
            color: #F5F7FA !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[aria-current="page"] {
            background-color: rgba(22, 199, 132, 0.12) !important;
            color: #16C784 !important;
            font-weight: 600 !important;
            border-left: 3px solid #16C784 !important;
        }

        /* Card & Container styling */
        .ci-card {
            background-color: #151F30;
            border: 1px solid #263449;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }
        .ci-metric-card {
            background-color: #151F30;
            border: 1px solid #263449;
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            transition: border-color 0.2s ease, transform 0.15s ease;
        }
        .ci-metric-card:hover {
            border-color: #16C784;
            transform: translateY(-1px);
        }
        .ci-metric-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .ci-metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #F5F7FA;
            line-height: 1.1;
            font-feature-settings: "tnum";
        }
        .ci-metric-sub {
            font-size: 0.75rem;
            color: #16C784;
            margin-top: 6px;
            font-weight: 500;
        }
        .ci-metric-sub.secondary {
            color: #3B82F6;
        }
        .ci-metric-sub.muted {
            color: #94A3B8;
        }

        /* Banner Header */
        .ci-banner {
            background-color: #151F30;
            border: 1px solid #263449;
            border-left: 4px solid #16C784;
            padding: 16px 22px;
            border-radius: 8px;
            margin-bottom: 22px;
            display: flex;
            align-items: center;
            gap: 18px;
        }
        .ci-banner-logo {
            flex: 0 0 48px;
            width: 48px;
            height: 48px;
        }
        .ci-banner-logo svg,
        .ci-brand-logo svg {
            display: block;
            width: 100%;
            height: 100%;
        }
        .ci-brand-logo {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 6px;
        }
        .ci-brand-fallback {
            font-size: 2rem;
            line-height: 1;
        }
        .ci-banner-content h2 {
            margin: 0;
            color: #F5F7FA;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        .ci-banner-content p {
            margin: 4px 0 0 0;
            color: #94A3B8;
            font-size: 0.85rem;
        }

        /* Section Headings */
        .ci-section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 24px;
            margin-bottom: 14px;
            padding-bottom: 8px;
            border-bottom: 1px solid #263449;
        }
        .ci-section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #F5F7FA;
            letter-spacing: -0.01em;
            margin: 0;
        }
        .ci-section-badge {
            background-color: rgba(22, 199, 132, 0.12);
            color: #16C784;
            border: 1px solid rgba(22, 199, 132, 0.3);
            border-radius: 4px;
            padding: 2px 7px;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        /* Streamlit DataFrames */
        div[data-testid="stDataFrame"] {
            border: 1px solid #263449 !important;
            border-radius: 8px !important;
            background-color: #151F30 !important;
            overflow: hidden;
        }

        /* Tabs styling */
        div[data-testid="stTabs"] button[role="tab"] {
            color: #94A3B8 !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            border-bottom: 2px solid transparent !important;
            padding: 8px 16px !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #16C784 !important;
            border-bottom-color: #16C784 !important;
        }

        /* Streamlit Inputs and Selectboxes */
        div[data-testid="stSelectbox"] > div > div {
            background-color: #151F30 !important;
            border-color: #263449 !important;
            color: #F5F7FA !important;
            border-radius: 6px !important;
        }
        div[data-testid="stTextInput"] input {
            background-color: #151F30 !important;
            border-color: #263449 !important;
            color: #F5F7FA !important;
            border-radius: 6px !important;
        }
        div[data-testid="stDateInput"] input {
            background-color: #151F30 !important;
            border-color: #263449 !important;
            color: #F5F7FA !important;
            border-radius: 6px !important;
        }

        /* Buttons */
        div[data-testid="stButton"] button {
            background-color: #151F30 !important;
            border: 1px solid #263449 !important;
            color: #F5F7FA !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            transition: all 0.15s ease !important;
        }
        div[data-testid="stButton"] button:hover {
            border-color: #16C784 !important;
            color: #16C784 !important;
        }
        div[data-testid="stButton"] button:active {
            background-color: rgba(22, 199, 132, 0.15) !important;
        }
        </style>
    """, unsafe_allow_html=True)


def render_brand_logo(container=None, size: int = 48):
    """Renders the shared local cricket logo in the selected Streamlit container."""
    target = container if container is not None else st
    target.markdown(
        f'<div class="ci-brand-logo" style="width:{size}px;height:{size}px">'
        f'{_cricket_logo_markup()}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_branding(container=None):
    """
    Renders consistent sidebar branding across all pages of the Cricket Intelligence platform.
    Features the vector cricket logo and professional platform title.
    """
    target = container if container is not None else st.sidebar
    render_brand_logo(target, size=48)
    target.markdown(
        """
        <div style="text-align: center; margin-top: -4px; margin-bottom: 12px;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #F5F7FA; letter-spacing: -0.01em;">Cricket Intelligence</div>
            <div style="font-size: 0.74rem; font-weight: 600; color: #16C784; text-transform: uppercase; letter-spacing: 0.06em;">Professional Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    target.divider()


def render_banner(title: str, subtitle: str, badge: Optional[str] = None):
    """Renders a styled header banner with optional category badge."""
    badge_html = f'<span class="ci-section-badge" style="margin-left: 10px;">{badge}</span>' if badge else ''
    st.markdown(f"""
        <div class="ci-banner">
            <div class="ci-banner-logo">{_cricket_logo_markup()}</div>
            <div class="ci-banner-content">
                <h2>{title}{badge_html}</h2>
                <p>{subtitle}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_kpi(title: str, value: Any, subtext: str = "", tone: str = "primary"):
    """
    Renders a modern sports analytics KPI card.
    tone: 'primary' (green), 'secondary' (blue), or 'muted' (slate).
    """
    sub_class = "secondary" if tone == "secondary" else ("muted" if tone == "muted" else "")
    st.markdown(f"""
        <div class="ci-metric-card">
            <div class="ci-metric-title">{title}</div>
            <div class="ci-metric-value">{value}</div>
            {f'<div class="ci-metric-sub {sub_class}">{subtext}</div>' if subtext else ''}
        </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, badge: Optional[str] = None):
    """Renders a clean section header with optional category badge."""
    badge_html = f'<span class="ci-section-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
        <div class="ci-section-header">
            <h3 class="ci-section-title">{title}</h3>
            {badge_html}
        </div>
    """, unsafe_allow_html=True)


def get_plotly_layout_defaults() -> Dict[str, Any]:
    """Returns standardized layout dictionary matching the dark sports analytics theme."""
    return dict(
        plot_bgcolor="#080D18",
        paper_bgcolor="#151F30",
        font=dict(color="#F5F7FA", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12),
        margin=dict(l=35, r=25, t=50, b=45),
        xaxis=dict(gridcolor="#263449", showline=False, zerolinecolor="#263449"),
        yaxis=dict(gridcolor="#263449", showline=False, zerolinecolor="#263449"),
        legend=dict(font=dict(color="#94A3B8")),
        colorway=["#16C784", "#3B82F6", "#F59E0B", "#EF4444", "#A855F7", "#06B6D4"]
    )


def create_styled_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_col: Optional[str] = None) -> go.Figure:
    """Generates standard styled Plotly bar chart matching the dark palette."""
    fig = px.bar(
        df, x=x_col, y=y_col,
        color=color_col if color_col else y_col,
        color_continuous_scale=["#101827", "#16C784", "#3B82F6"],
        title=title
    )
    defaults = get_plotly_layout_defaults()
    fig.update_layout(**defaults, coloraxis_showscale=False)
    return fig
