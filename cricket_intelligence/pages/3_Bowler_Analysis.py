"""
CRICKET INTELLIGENCE • INDIVIDUAL BOWLER PROFILE & ANALYSIS
Dedicated bowler profile page with phase analysis, handedness split, and delivery effectiveness.
"""

import sys
from pathlib import Path
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import load_all_data, get_lookup_maps
from src.bowling_analysis import (
    get_bowler_kpis,
    get_bowler_phase_stats,
    get_bowler_handedness_split,
    get_bowler_dismissal_types,
    get_bowler_delivery_styles
)
from src.ui_components import apply_custom_theme, render_banner, render_kpi, render_sidebar_branding, render_section_header, get_plotly_layout_defaults

st.set_page_config(page_title="Bowler Analysis | Cricket Intelligence", page_icon="🎯", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
bbb = data["bbb"]
matches = data["matches"]
comps = data["comps"]
teams = data["teams"]
players_master = data["players_master"]
lookups = get_lookup_maps(data)

# Persistent bowler selection
if "selected_bowler_id" not in st.session_state:
    # Default to Arshdeep Singh (ID: 6464) or Rashid Khan
    st.session_state["selected_bowler_id"] = 6464

# Filter players who have bowled
bowlers_master = players_master[players_master["legal_balls"] > 0]
bowler_options = dict(zip(
    bowlers_master["player_id"],
    bowlers_master["player_name"] + " (ID: " + bowlers_master["player_id"].astype(str) + ")"
))

st.sidebar.markdown("### Select Bowler")
current_bid = st.session_state.get("selected_bowler_id", 6464)
if current_bid not in bowler_options:
    current_bid = list(bowler_options.keys())[0]

chosen_bid = st.sidebar.selectbox(
    "Choose Bowler",
    options=list(bowler_options.keys()),
    index=list(bowler_options.keys()).index(current_bid),
    format_func=lambda x: bowler_options[x]
)
st.session_state["selected_bowler_id"] = chosen_bid
bowler_id = chosen_bid

# Bowler Info
b_info = players_master[players_master["player_id"] == bowler_id].iloc[0]
bowler_name = b_info["player_name"]

bowler_bbb = bbb[bbb["bowler_id"] == bowler_id]
bowler_comps = ["All"] + sorted(bowler_bbb["comp_name"].dropna().unique().tolist())
bowler_seasons = ["All"] + sorted(bowler_bbb["season"].dropna().unique().tolist())

st.sidebar.markdown("### Granular Filters")
f_comp = st.sidebar.selectbox("Filter Competition", bowler_comps)
f_season = st.sidebar.selectbox("Filter Season", bowler_seasons)

filters = {}
if f_comp != "All":
    c_sub = comps[comps["comp_name"] == f_comp]
    if len(c_sub) > 0:
        filters["comp_id"] = c_sub["comp_id"].iloc[0]
if f_season != "All":
    filters["season"] = f_season

# Header Banner
render_banner(
    f"{bowler_name} • Bowling Dossier",
    f"Bowler ID: {bowler_id} | Bowling Style: {b_info['bowling_style']} | Category: {bowler_bbb['pace_or_spin'].iloc[0] if len(bowler_bbb)>0 else 'Unknown'}",
    badge=f"{b_info['wickets']} career wickets"
)

# Bowler Bio
b_col1, b_col2, b_col3, b_col4 = st.columns(4)
with b_col1:
    st.markdown(f"**Full Name:** `{bowler_name}`")
    st.markdown(f"**Bowler ID:** `{bowler_id}`")
with b_col2:
    st.markdown(f"**Squads:** {b_info['teams']}")
    st.markdown(f"**Bowling Style:** `{b_info['bowling_style']}`")
with b_col3:
    st.markdown(f"**Pace or Spin:** `{bowler_bbb['pace_or_spin'].iloc[0] if len(bowler_bbb)>0 else 'N/A'}`")
    st.markdown(f"**Matches Played:** `{b_info['matches_played']}`")
with b_col4:
    st.markdown(f"**Innings Bowled:** `{b_info['bowl_innings']}`")
    st.markdown(f"**Career Wickets:** `{b_info['wickets']}`")

st.divider()

# Bowler KPIs
render_section_header("Key Bowling Performance Indicators")
b_kpis = get_bowler_kpis(bbb, bowler_id, filters)

k_row1 = st.columns(6)
with k_row1[0]:
    render_kpi("Wickets", f"{b_kpis['wickets']}", f"Best: {b_kpis['best_figures']}", tone='primary')
with k_row1[1]:
    render_kpi("Economy Rate", f"{b_kpis['economy']}", "Runs per 6 balls", tone='secondary')
with k_row1[2]:
    render_kpi("Bowling Average", f"{b_kpis['average']}", "Runs per wicket", tone='muted')
with k_row1[3]:
    render_kpi("Strike Rate", f"{b_kpis['strike_rate']}", "Balls per wicket", tone='secondary')
with k_row1[4]:
    render_kpi("Overs Bowled", f"{b_kpis['overs']}", f"{b_kpis['legal_balls']} legal balls", tone='muted')
with k_row1[5]:
    render_kpi("Dot Ball %", f"{b_kpis['dot_pct']}%", f"{b_kpis['dot_balls']} dot balls", tone='primary')

k_row2 = st.columns(6)
with k_row2[0]:
    render_kpi("Runs Conceded", f"{b_kpis['runs_conceded']:,}", "Bowler charges", tone='primary')
with k_row2[1]:
    render_kpi("Fours Conceded", f"{b_kpis['fours_conceded']}", "Boundaries allowed", tone='muted')
with k_row2[2]:
    render_kpi("Sixes Conceded", f"{b_kpis['sixes_conceded']}", "Maximums allowed", tone='muted')
with k_row2[3]:
    render_kpi("Matches Bowled", f"{b_kpis['matches']}", "Appearances", tone='muted')
with k_row2[4]:
    render_kpi("Innings Bowled", f"{b_kpis['innings']}", "Spells", tone='muted')
with k_row2[5]:
    boundary_runs = b_kpis['fours_conceded']*4 + b_kpis['sixes_conceded']*6
    b_pct = round(boundary_runs / b_kpis['runs_conceded'] * 100, 1) if b_kpis['runs_conceded']>0 else 0
    render_kpi("Boundary Runs %", f"{b_pct}%", "In boundaries", tone='muted')

st.markdown("<br>", unsafe_allow_html=True)

# Phase & Handedness Splits
render_section_header("Phase Intelligence & Handedness Splits")
tab_phase, tab_hand, tab_dismissals = st.tabs(["Innings Phase Split", "Performance vs RHB / LHB", "Dismissal Breakdown"])

with tab_phase:
    ph_df = get_bowler_phase_stats(bbb, bowler_id, filters)
    if len(ph_df) > 0:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.dataframe(ph_df, width="stretch", height=180)
        with c2:
            fig_p = px.bar(
                ph_df, x="Phase", y="Economy", color="Phase",
                title="Economy Rate by Phase",
                color_discrete_map={"Powerplay": "#16C784", "Middle": "#3B82F6", "Death": "#EF4444"}
            )
            _ph_layout = get_plotly_layout_defaults()
            fig_p.update_layout(**_ph_layout)
            st.plotly_chart(fig_p, width="stretch")

with tab_hand:
    hand_df = get_bowler_handedness_split(bbb, bowler_id, filters)
    if len(hand_df) > 0:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.dataframe(hand_df, width="stretch", height=160)
        with c2:
            fig_h = px.bar(
                hand_df, x="Batsman Hand", y="Average", color="Batsman Hand",
                title="Bowling Average vs Handedness"
            )
            fig_h.update_layout(**get_plotly_layout_defaults())
            st.plotly_chart(fig_h, width="stretch")

with tab_dismissals:
    d_df = get_bowler_dismissal_types(bbb, bowler_id)
    if len(d_df) > 0:
        c1, c2 = st.columns([1.2, 1.8])
        with c1:
            st.dataframe(d_df, width="stretch", height=220)
        with c2:
            fig_d = px.pie(
                d_df, names="Dismissal Type", values="Wickets",
                title="Wickets by Dismissal Mode",
                hole=0.4
            )
            fig_d.update_layout(**get_plotly_layout_defaults())
            st.plotly_chart(fig_d, width="stretch")

st.markdown("<br>", unsafe_allow_html=True)

# Delivery Style Effectiveness
render_section_header("Delivery Style Effectiveness")
styles_df = get_bowler_delivery_styles(bbb, bowler_id)
if len(styles_df) > 0:
    st.dataframe(styles_df, width="stretch", height=280)

st.divider()

# Performance Against Individual Batsmen Faced
render_section_header("Performance Against Individual Batsmen Faced")
b_vs_bat = bowler_bbb.groupby(["striker_id", "striker_name"]).agg(
    balls=("is_legal_ball_bowler", "sum"),
    runs=("bowler_conceded_runs", "sum"),
    wickets=("is_bowler_wicket", "sum"),
    dots=("is_dot_ball_bowler", "sum"),
    fours=("is_four", "sum"),
    sixes=("is_six", "sum")
).reset_index()

b_vs_bat["Economy"] = np.where(b_vs_bat["balls"] > 0, (b_vs_bat["runs"] / b_vs_bat["balls"] * 6.0).round(2), 0.0)
b_vs_bat["SR"] = np.where(b_vs_bat["balls"] > 0, (b_vs_bat["runs"] / b_vs_bat["balls"] * 100).round(2), 0.0)
b_vs_bat = b_vs_bat.sort_values(by="balls", ascending=False).rename(columns={
    "striker_id": "Batsman ID", "striker_name": "Batsman", "balls": "Balls", "runs": "Runs", "wickets": "Wickets",
    "dots": "Dots", "fours": "4s", "sixes": "6s"
})

st.dataframe(b_vs_bat, width="stretch", height=360)

