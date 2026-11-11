"""
Player Database & Scouting Directory
Searchable directory of all players with career statistics, filters, and quick profile inspection.
"""

import streamlit as st
import pandas as pd
from src.data_loader import load_all_data, get_lookup_maps
from src.ui_components import (
    apply_custom_theme,
    render_banner,
    render_kpi,
    render_sidebar_branding,
    render_section_header
)

st.set_page_config(page_title="Player Database | Cricket Intelligence", page_icon="👥", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
players_master = data["players_master"]
teams = data["teams"]
comps = data["comps"]

# Filters Bar
col_s1, col_s2, col_s3, col_s4 = st.columns([2, 1.5, 1, 1])

with col_s1:
    search_query = st.text_input("Search by Name or Player ID", placeholder="Search player name, squad or ID...")

with col_s2:
    team_filter = st.selectbox("Filter Team", ["All Teams"] + sorted(teams["team_name"].unique().tolist()))

with col_s3:
    bat_style_filter = st.selectbox("Batting Style", ["All", "RHB", "LHB"])

with col_s4:
    bowl_style_filter = st.selectbox("Bowling Style", ["All"] + sorted([b for b in players_master["bowling_style"].unique() if b != "Unknown"]))

# Filter DataFrame
df_display = players_master.copy()

if search_query:
    q = search_query.lower().strip()
    df_display = df_display[
        df_display["player_name"].str.lower().str.contains(q) |
        df_display["player_id"].astype(str).str.contains(q) |
        df_display["teams"].str.lower().str.contains(q)
    ]

if team_filter != "All Teams":
    df_display = df_display[df_display["teams"].str.contains(team_filter, case=False, na=False)]

if bat_style_filter != "All":
    df_display = df_display[df_display["batting_style"] == bat_style_filter]

if bowl_style_filter != "All":
    df_display = df_display[df_display["bowling_style"] == bowl_style_filter]

render_banner(
    "Player Database & Scouting Directory",
    "Comprehensive directory of all 1,217 players with career records, multi-skill classification, and direct dossier inspection.",
    badge=f"{len(df_display):,} Athletes Found"
)

# Summary KPI Cards
c1, c2, c3, c4 = st.columns(4)
avg_sr = round(df_display[df_display["balls_faced"] > 50]["bat_strike_rate"].mean(), 1) if len(df_display[df_display["balls_faced"] > 50]) > 0 else 0.0
with c1:
    render_kpi("Players Found", f"{len(df_display):,}", "Matching active filters", tone="primary")
with c2:
    render_kpi("Career Runs Scored", f"{df_display['runs'].sum():,}", "Cumulative across matches", tone="primary")
with c3:
    render_kpi("Career Wickets", f"{df_display['wickets'].sum():,}", "All bowling dismissals", tone="secondary")
with c4:
    render_kpi("Average Strike Rate", f"{avg_sr}", "Min 50 legal balls faced", tone="muted")

st.markdown("<br>", unsafe_allow_html=True)

# Main interactive table
render_section_header("Scouting Directory Table", badge="Full Career Records")
st.dataframe(
    df_display[[
        "player_id", "player_name", "teams", "batting_style", "bowling_style",
        "matches_played", "bat_innings", "runs", "balls_faced", "bat_strike_rate",
        "wickets", "legal_balls", "bowl_economy"
    ]].rename(columns={
        "player_id": "Player ID",
        "player_name": "Player Name",
        "teams": "Teams",
        "batting_style": "Bat Style",
        "bowling_style": "Bowl Style",
        "matches_played": "Matches",
        "bat_innings": "Bat Inns",
        "runs": "Runs",
        "balls_faced": "Balls Faced",
        "bat_strike_rate": "Strike Rate",
        "wickets": "Wickets",
        "legal_balls": "Balls Bowled",
        "bowl_economy": "Economy"
    }),
    column_config={
        "Player ID": st.column_config.NumberColumn("ID", format="%d"),
        "Player Name": st.column_config.TextColumn("Player Name", width="medium"),
        "Teams": st.column_config.TextColumn("Teams", width="large"),
        "Bat Style": st.column_config.TextColumn("Bat Style"),
        "Bowl Style": st.column_config.TextColumn("Bowl Style"),
        "Matches": st.column_config.NumberColumn("Matches", format="%d"),
        "Bat Inns": st.column_config.NumberColumn("Innings", format="%d"),
        "Runs": st.column_config.NumberColumn("Runs", format="%d"),
        "Balls Faced": st.column_config.NumberColumn("Balls", format="%d"),
        "Strike Rate": st.column_config.NumberColumn("Strike Rate", format="%.2f"),
        "Wickets": st.column_config.NumberColumn("Wickets", format="%d"),
        "Balls Bowled": st.column_config.NumberColumn("Balls Bowled", format="%d"),
        "Economy": st.column_config.NumberColumn("Economy", format="%.2f")
    },
    use_container_width=True,
    height=420
)

# Deep Profile Inspector Card
st.markdown("<br>", unsafe_allow_html=True)
render_section_header("Open Detailed Player Profile", badge="Deep Dive")

player_map = dict(zip(
    df_display["player_id"],
    df_display["player_name"] + " (ID: " + df_display["player_id"].astype(str) + ")"
))

if len(player_map) > 0:
    col_sel, col_btn1, col_btn2 = st.columns([2, 1, 1])
    with col_sel:
        target_pid = st.selectbox(
            "Select Player for Deep Dive",
            options=list(player_map.keys()),
            format_func=lambda x: player_map[x],
            key="db_player_select"
        )
    with col_btn1:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Open Batsman Profile", use_container_width=True):
            st.session_state["selected_player_id"] = target_pid
            st.switch_page("pages/2_Batsman_Analysis.py")
    with col_btn2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Open Bowler Profile", use_container_width=True):
            st.session_state["selected_bowler_id"] = target_pid
            st.switch_page("pages/3_Bowler_Analysis.py")
