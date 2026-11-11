"""
TEAM PERFORMANCE & FRANCHISE INTELLIGENCE
Team-level analytics, win ratios, powerplay/death run-rates, and squad contributions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import load_all_data
from src.team_analysis import get_team_kpis, get_team_player_contributions
from src.ui_components import apply_custom_theme, render_banner, render_kpi, render_sidebar_branding, render_section_header

st.set_page_config(page_title="Team Analysis | Cricket Intelligence", page_icon="🛡️", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
bbb = data["bbb"]
matches = data["matches"]
teams = data["teams"]

render_banner(
    "TEAM PERFORMANCE & FRANCHISE INTELLIGENCE",
    "Win records, scoring rates, phase efficiency, and intra-squad player contributions"
)

# Team Selector
team_list = sorted(teams["team_name"].unique().tolist())
sel_team_name = st.selectbox("Select Team", team_list)

team_id = teams[teams["team_name"] == sel_team_name]["team_id"].iloc[0]

# KPIs
kpis = get_team_kpis(matches, bbb, team_id)

if kpis:
    render_section_header(f"{sel_team_name} Overview (Team ID: {team_id})")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        render_kpi("Matches", f"{kpis['matches']}", f"{kpis['wins']}W - {kpis['losses']}L", tone='muted')
    with k2:
        render_kpi("Win Rate", f"{kpis['win_pct']}%", "Win ratio", tone='primary')
    with k3:
        render_kpi("Average Score", f"{kpis['avg_score']}", "Per innings", tone='secondary')
    with k4:
        render_kpi("Highest Score", f"{kpis['highest_score']}", f"Lowest: {kpis['lowest_score']}", tone='primary')
    with k5:
        render_kpi("Batting Strike Rate", f"{kpis['bat_sr']}", "Team batting SR", tone='secondary')
    with k6:
        render_kpi("Bowling Economy", f"{kpis['bowl_econ']}", f"{kpis['wickets_taken']} wickets", tone='muted')

    st.markdown("<br>", unsafe_allow_html=True)

    # Phase Run Rates
    p1, p2 = st.columns(2)
    with p1:
        render_kpi("Powerplay Run Rate", f"{kpis['pp_run_rate']} rpo", "Overs 1-6 team scoring")
    with p2:
        render_kpi("Death Overs Run Rate", f"{kpis['death_run_rate']} rpo", "Overs 16-20 team finishing")

    st.markdown("<br>", unsafe_allow_html=True)

    # Player Contributions
    render_section_header("Squad Player Contributions")
    contribs = get_team_player_contributions(bbb, team_id)
    
    t_bat, t_bowl = st.tabs(["🏏 Leading Batsmen for Team", "🎯 Leading Bowlers for Team"])
    with t_bat:
        st.dataframe(contribs["batsmen"], use_container_width=True, height=340)
    with t_bowl:
        st.dataframe(contribs["bowlers"], use_container_width=True, height=340)
else:
    st.info("No match records found for this team.")

