"""
COMPETITION & TOURNAMENT INTELLIGENCE
Tournament standings, leading run scorers, leading wicket takers, and boundary metrics.
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

from src.data_loader import load_all_data
from src.ui_components import apply_custom_theme, render_banner, render_kpi, render_sidebar_branding, render_section_header

st.set_page_config(page_title="Competition Analysis | Cricket Intelligence", page_icon="🏆", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
bbb = data["bbb"]
matches = data["matches"]
comps = data["comps"]

render_banner(
    "COMPETITION & TOURNAMENT INTELLIGENCE",
    "Tournament leaderboards, match outcomes, strike rate leaders, and boundary statistics"
)

# Filters
c1, c2 = st.columns([2.5, 1.5])
with c1:
    comp_names = sorted(comps["comp_name"].unique().tolist())
    sel_comp_name = st.selectbox("Select Competition", comp_names)
with c2:
    comp_row = comps[comps["comp_name"] == sel_comp_name].iloc[0]
    st.markdown(f"**Format:** `{comp_row['match_format']}` | **Season:** `{comp_row['season']}` | **ID:** `{comp_row['comp_id']}`")

comp_id = comp_row["comp_id"]
c_matches = matches[matches["comp_id"] == comp_id]
c_bbb = bbb[bbb["comp_id"] == comp_id]

# Tournament KPIs
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi("Matches", f"{len(c_matches)}", "Total matches", tone='muted')
with k2:
    total_runs = int(c_bbb["total_delivery_runs"].sum())
    render_kpi("Tournament Runs", f"{total_runs:,}", "All deliveries", tone='primary')
with k3:
    total_wkts = int(c_bbb["is_wicket"].sum())
    render_kpi("Total Wickets", f"{total_wkts}", "Dismissals", tone='primary')
with k4:
    fours = int(c_bbb["is_four"].sum())
    sixes = int(c_bbb["is_six"].sum())
    render_kpi("Boundaries", f"{fours + sixes:,}", f"{fours}x4 | {sixes}x6", tone='secondary')
with k5:
    balls = int(c_bbb["is_legal_ball_batter"].sum())
    comp_sr = round(total_runs / balls * 100, 2) if balls > 0 else 0
    render_kpi("Overall Run Rate", f"{round(comp_sr/100*6, 2)} rpo", f"SR: {comp_sr}", tone='secondary')

st.markdown("<br>", unsafe_allow_html=True)

# Leaderboards
render_section_header("Tournament Leaderboards")
l_col1, l_col2 = st.columns(2)

with l_col1:
    render_section_header("Top Run Scorers")
    top_bat = c_bbb.groupby(["striker_id", "striker_name"]).agg(
        runs=("batter_runs", "sum"),
        balls=("is_legal_ball_batter", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        innings=("match_id", "nunique")
    ).reset_index()
    top_bat["SR"] = np.where(top_bat["balls"] > 0, (top_bat["runs"] / top_bat["balls"] * 100).round(2), 0.0)
    top_bat = top_bat.sort_values(by="runs", ascending=False).head(10).rename(columns={
        "striker_id": "ID", "striker_name": "Player", "runs": "Runs", "balls": "Balls", "fours": "4s", "sixes": "6s", "innings": "Inns"
    })
    st.dataframe(top_bat, width="stretch", height=360)

with l_col2:
    render_section_header("Top Wicket Takers")
    top_bowl = c_bbb.groupby(["bowler_id", "bowler_name"]).agg(
        wickets=("is_bowler_wicket", "sum"),
        balls=("is_legal_ball_bowler", "sum"),
        runs=("bowler_conceded_runs", "sum"),
        innings=("match_id", "nunique")
    ).reset_index()
    top_bowl["Econ"] = np.where(top_bowl["balls"] > 0, (top_bowl["runs"] / top_bowl["balls"] * 6.0).round(2), 0.0)
    top_bowl["Overs"] = (top_bowl["balls"] // 6).astype(str) + "." + (top_bowl["balls"] % 6).astype(str)
    top_bowl = top_bowl.sort_values(by="wickets", ascending=False).head(10).rename(columns={
        "bowler_id": "ID", "bowler_name": "Bowler", "wickets": "Wkts", "runs": "Runs", "innings": "Inns"
    })
    st.dataframe(top_bowl[["ID", "Bowler", "Wkts", "Overs", "Runs", "Econ", "Inns"]], width="stretch", height=360)

st.markdown("<br>", unsafe_allow_html=True)

# Matches in Competition
render_section_header("Match Results in this Competition")
st.dataframe(
    c_matches[["match_id", "match_number", "match_date", "home_team", "away_team", "venue_country", "result"]].rename(columns={
        "match_id": "Match ID", "match_number": "Match #", "match_date": "Date", "home_team": "Home",
        "away_team": "Away", "venue_country": "Venue", "result": "Result"
    }),
    width="stretch"
)

