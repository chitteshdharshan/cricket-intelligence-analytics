"""
CRICKET INTELLIGENCE: Premium Cricket Analytics Platform
Main Application & Home Dashboard.
"""

import sys
from pathlib import Path

# Ensure cricket_intelligence and repository root are in sys.path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))
if str(_current_dir.parent) not in sys.path:
    sys.path.insert(1, str(_current_dir.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_all_data, get_lookup_maps
from src.ui_components import (
    apply_custom_theme,
    render_banner,
    render_sidebar_branding,
    render_kpi,
    render_section_header,
    get_plotly_layout_defaults,
)

st.set_page_config(
    page_title="Cricket Intelligence | Premium Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()

# Load cached analytical datasets
data = load_all_data()
lookups = get_lookup_maps(data)

bbb = data["bbb"]
matches = data["matches"]
comps = data["comps"]
teams = data["teams"]
players_master = data["players_master"]

# Sidebar branding & filters
render_sidebar_branding(st.sidebar)

st.sidebar.markdown("### Global Filters")
selected_comp = st.sidebar.selectbox("Filter Competition", ["All"] + sorted(comps["comp_name"].unique().tolist()))
selected_season = st.sidebar.selectbox("Filter Season", ["All"] + sorted(comps["season"].unique().tolist()))
selected_team = st.sidebar.selectbox("Filter Team", ["All"] + sorted(teams["team_name"].unique().tolist()))

# Filtered slices
df_filtered = bbb
m_filtered = matches

if selected_comp != "All":
    cid = comps[comps["comp_name"] == selected_comp]["comp_id"].iloc[0]
    df_filtered = df_filtered[df_filtered["comp_id"] == cid]
    m_filtered = m_filtered[m_filtered["comp_id"] == cid]

if selected_season != "All":
    df_filtered = df_filtered[df_filtered["season"] == str(selected_season)]
    comp_ids_season = comps[comps["season"] == str(selected_season)]["comp_id"].unique()
    m_filtered = m_filtered[m_filtered["comp_id"].isin(comp_ids_season)]

if selected_team != "All":
    t_row = teams[teams["team_name"] == selected_team]
    if len(t_row) > 0:
        tid = t_row["team_id"].iloc[0]
        df_filtered = df_filtered[(df_filtered["batting_team_id"] == tid) | (df_filtered["bowling_team_id"] == tid)]
        m_filtered = m_filtered[(m_filtered["home_team_id"] == tid) | (m_filtered["away_team_id"] == tid)]

# Header Banner
active_filter_label = []
if selected_comp != "All": active_filter_label.append(selected_comp)
if selected_season != "All": active_filter_label.append(f"Season {selected_season}")
if selected_team != "All": active_filter_label.append(selected_team)
badge_str = " • ".join(active_filter_label) if active_filter_label else "All Competitions & Seasons"

render_banner(
    "Cricket Analytics Dashboard",
    "Comprehensive T20 & Multi-Format Intelligence Engine powered by 200,000+ ball-by-ball event records.",
    badge=badge_str
)

# 1. EXECUTIVE KPI CARDS (Matches, Deliveries, Players, Teams, Competitions, Innings)
k_col1, k_col2, k_col3, k_col4, k_col5, k_col6 = st.columns(6)
total_matches = m_filtered["match_id"].nunique()
total_deliveries = len(df_filtered)
total_players = len(set(df_filtered["striker_id"]).union(set(df_filtered["bowler_id"])))
total_teams = len(set(m_filtered["home_team_id"]).union(set(m_filtered["away_team_id"])))
total_comps = m_filtered["comp_id"].nunique()
total_innings = df_filtered.groupby(["match_id", "innings"]).ngroups

with k_col1:
    render_kpi("Total Matches", f"{total_matches:,}", "Completed fixtures", tone="primary")
with k_col2:
    render_kpi("Total Deliveries", f"{total_deliveries:,}", "Legal & extra balls", tone="primary")
with k_col3:
    render_kpi("Total Players", f"{total_players:,}", "Active athletes", tone="secondary")
with k_col4:
    render_kpi("Total Teams", f"{total_teams:,}", "Squads represented", tone="secondary")
with k_col5:
    render_kpi("Total Competitions", f"{total_comps:,}", "Tournaments", tone="muted")
with k_col6:
    render_kpi("Total Innings", f"{total_innings:,}", "Match innings", tone="muted")

st.markdown("<br>", unsafe_allow_html=True)

# 2. TOURNAMENT BENCHMARK RECORDS
render_section_header("Tournament Benchmark Records", badge="All-Time Leaders")
rec1, rec2, rec3, rec4, rec5 = st.columns(5)

# Highest Individual Score
inn_bat = df_filtered.groupby(["match_id", "innings", "striker_id", "striker_name"]).agg(
    runs=("batter_runs", "sum"),
    balls=("is_legal_ball_batter", "sum"),
    is_out=("is_wicket", lambda x: any(x))
).reset_index().sort_values(by="runs", ascending=False)

if len(inn_bat) > 0:
    top_inn = inn_bat.iloc[0]
    rec1.markdown(f"""
        <div class="ci-metric-card">
            <div class="ci-metric-title">Highest Individual Score</div>
            <div class="ci-metric-value">{top_inn['runs']}{'*' if not top_inn['is_out'] else ''}</div>
            <div class="ci-metric-sub">{top_inn['striker_name']} ({top_inn['balls']}b)</div>
        </div>
    """, unsafe_allow_html=True)

# Best Bowling in an Innings
inn_bowl = df_filtered.groupby(["match_id", "innings", "bowler_id", "bowler_name"]).agg(
    wkts=("is_bowler_wicket", "sum"),
    runs=("bowler_conceded_runs", "sum"),
    balls=("is_legal_ball_bowler", "sum")
).reset_index().sort_values(by=["wkts", "runs"], ascending=[False, True])

if len(inn_bowl) > 0:
    top_bowl = inn_bowl.iloc[0]
    ov_str = f"{top_bowl['balls'] // 6}.{top_bowl['balls'] % 6}"
    rec2.markdown(f"""
        <div class="ci-metric-card">
            <div class="ci-metric-title">Best Bowling Figures</div>
            <div class="ci-metric-value">{top_bowl['wkts']}/{top_bowl['runs']}</div>
            <div class="ci-metric-sub">{top_bowl['bowler_name']} ({ov_str} ov)</div>
        </div>
    """, unsafe_allow_html=True)

# Most Runs Scorer
most_runs_df = df_filtered.groupby(["striker_id", "striker_name"]).agg(
    runs=("batter_runs", "sum"),
    balls=("is_legal_ball_batter", "sum")
).reset_index().sort_values(by="runs", ascending=False)

if len(most_runs_df) > 0:
    top_run_scorer = most_runs_df.iloc[0]
    rec3.markdown(f"""
        <div class="ci-metric-card">
            <div class="ci-metric-title">Leading Run Scorer</div>
            <div class="ci-metric-value">{top_run_scorer['runs']:,}</div>
            <div class="ci-metric-sub">{top_run_scorer['striker_name']} ({top_run_scorer['balls']}b)</div>
        </div>
    """, unsafe_allow_html=True)

# Most Wickets
most_wkts_df = df_filtered.groupby(["bowler_id", "bowler_name"]).agg(
    wkts=("is_bowler_wicket", "sum"),
    runs=("bowler_conceded_runs", "sum")
).reset_index().sort_values(by="wkts", ascending=False)

if len(most_wkts_df) > 0:
    top_wkt_taker = most_wkts_df.iloc[0]
    rec4.markdown(f"""
        <div class="ci-metric-card">
            <div class="ci-metric-title">Leading Wicket Taker</div>
            <div class="ci-metric-value">{top_wkt_taker['wkts']}</div>
            <div class="ci-metric-sub">{top_wkt_taker['bowler_name']} ({top_wkt_taker['runs']}r)</div>
        </div>
    """, unsafe_allow_html=True)

# Highest Strike Rate (qualifying min 120 legal balls)
sr_qual = most_runs_df[most_runs_df["balls"] >= 120].copy()
if len(sr_qual) > 0:
    sr_qual["sr"] = (sr_qual["runs"] / sr_qual["balls"] * 100).round(2)
    top_sr = sr_qual.sort_values(by="sr", ascending=False).iloc[0]
    rec5.markdown(f"""
        <div class="ci-metric-card">
            <div class="ci-metric-title">Highest Strike Rate</div>
            <div class="ci-metric-value">{top_sr['sr']}</div>
            <div class="ci-metric-sub">{top_sr['striker_name']} (min 120b)</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 3. INTERACTIVE ANALYTICS TABS
render_section_header("In-Depth Analytical Visualizations", badge="Interactive Analytics")
tab_leaders, tab_phase, tab_teams, tab_matches = st.tabs([
    "🏆 Top Run Scorers & Wicket Takers",
    "⏱️ Runs by Innings Phase",
    "🛡️ Most Successful Teams",
    "📅 Recent Match Results"
])

layout_defaults = get_plotly_layout_defaults()

with tab_leaders:
    lead_c1, lead_c2 = st.columns(2)
    with lead_c1:
        st.markdown("##### Leading Run Scorers")
        top_batsmen = most_runs_df.head(10).sort_values(by="runs", ascending=True)
        fig_batsmen = px.bar(
            top_batsmen,
            x="runs",
            y="striker_name",
            orientation="h",
            color="runs",
            color_continuous_scale=["#151F30", "#16C784"],
            labels={"runs": "Runs Scored", "striker_name": "Batsman"}
        )
        fig_batsmen.update_layout(
            **layout_defaults,
            title="Top 10 Batsmen by Aggregate Runs",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_batsmen, width="stretch")

    with lead_c2:
        st.markdown("##### Leading Wicket Takers")
        top_bowlers = most_wkts_df.head(10).sort_values(by="wkts", ascending=True)
        fig_bowlers = px.bar(
            top_bowlers,
            x="wkts",
            y="bowler_name",
            orientation="h",
            color="wkts",
            color_continuous_scale=["#151F30", "#3B82F6"],
            labels={"wkts": "Wickets Taken", "bowler_name": "Bowler"}
        )
        fig_bowlers.update_layout(
            **layout_defaults,
            title="Top 10 Bowlers by Wickets Taken",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bowlers, width="stretch")

with tab_phase:
    phase_stats = df_filtered.groupby("phase").agg(
        runs=("batter_runs", "sum"),
        balls=("is_legal_ball_batter", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        wickets=("is_wicket", "sum")
    ).reset_index()

    # Ensure standard order
    order_map = {"Powerplay": 0, "Middle": 1, "Death": 2}
    phase_stats["order"] = phase_stats["phase"].map(order_map).fillna(3)
    phase_stats = phase_stats.sort_values(by="order").reset_index(drop=True)
    phase_stats["sr"] = np.where(phase_stats["balls"] > 0, (phase_stats["runs"] / phase_stats["balls"] * 100).round(1), 0.0)

    ph_col1, ph_col2 = st.columns([1.6, 1])
    with ph_col1:
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Bar(
            x=phase_stats["phase"],
            y=phase_stats["runs"],
            name="Runs Scored",
            marker_color="#16C784",
            yaxis="y"
        ))
        fig_phase.add_trace(go.Scatter(
            x=phase_stats["phase"],
            y=phase_stats["sr"],
            name="Strike Rate",
            line=dict(color="#3B82F6", width=3),
            mode="lines+markers+text",
            text=[f"{v} SR" for v in phase_stats["sr"]],
            textposition="top center",
            yaxis="y2"
        ))
        d_phase = layout_defaults.copy()
        d_phase.pop("yaxis", None)
        fig_phase.update_layout(
            **d_phase,
            title="Runs Scored & Strike Rate Across Match Phases",
            yaxis=dict(title="Runs Scored", gridcolor="#263449"),
            yaxis2=dict(title="Strike Rate", overlaying="y", side="right", showgrid=False)
        )
        st.plotly_chart(fig_phase, width="stretch")

    with ph_col2:
        st.markdown("##### Phase Breakdown Metrics")
        st.dataframe(
            phase_stats[["phase", "runs", "balls", "sr", "fours", "sixes", "wickets"]].rename(columns={
                "phase": "Match Phase",
                "runs": "Runs",
                "balls": "Balls",
                "sr": "Strike Rate",
                "fours": "4s",
                "sixes": "6s",
                "wickets": "Wickets"
            }),
            column_config={
                "Match Phase": st.column_config.TextColumn("Phase"),
                "Runs": st.column_config.NumberColumn("Runs", format="%d"),
                "Balls": st.column_config.NumberColumn("Balls", format="%d"),
                "Strike Rate": st.column_config.NumberColumn("Strike Rate", format="%.1f"),
                "4s": st.column_config.NumberColumn("4s", format="%d"),
                "6s": st.column_config.NumberColumn("6s", format="%d"),
                "Wickets": st.column_config.NumberColumn("Wickets", format="%d")
            },
            width="stretch",
            height=240
        )

with tab_teams:
    team_dict = teams.groupby("team_id")["team_name"].first().to_dict()
    valid_wins = m_filtered["win_team_id"].dropna().map(team_dict).value_counts().head(10).reset_index()
    valid_wins.columns = ["Team", "Total Wins"]

    t_col1, t_col2 = st.columns([1.5, 1])
    with t_col1:
        fig_teams = px.bar(
            valid_wins.sort_values(by="Total Wins", ascending=True),
            x="Total Wins",
            y="Team",
            orientation="h",
            color="Total Wins",
            color_continuous_scale=["#151F30", "#16C784"],
            labels={"Total Wins": "Matches Won", "Team": "Franchise / Team"}
        )
        fig_teams.update_layout(
            **layout_defaults,
            title="Most Successful Teams by Total Wins",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_teams, width="stretch")

    with t_col2:
        st.markdown("##### Team Win Standings")
        st.dataframe(
            valid_wins,
            column_config={
                "Team": st.column_config.TextColumn("Team"),
                "Total Wins": st.column_config.NumberColumn("Wins", format="%d")
            },
            width="stretch",
            height=320
        )

with tab_matches:
    recent_matches = m_filtered.sort_values(by="match_date", ascending=False).head(15).copy()
    recent_matches["date_str"] = pd.to_datetime(recent_matches["match_date"]).dt.strftime("%Y-%m-%d")
    recent_matches["fixture"] = recent_matches["home_team"] + " vs " + recent_matches["away_team"]

    st.markdown("##### Recent Match Results & Outcomes")
    st.dataframe(
        recent_matches[[
            "match_id", "date_str", "fixture", "venue_country", "stage", "result"
        ]].rename(columns={
            "match_id": "Match ID",
            "date_str": "Date",
            "fixture": "Fixture",
            "venue_country": "Country",
            "stage": "Stage",
            "result": "Result"
        }),
        column_config={
            "Match ID": st.column_config.NumberColumn("Match ID", format="%d"),
            "Date": st.column_config.TextColumn("Date"),
            "Fixture": st.column_config.TextColumn("Fixture", width="large"),
            "Country": st.column_config.TextColumn("Country"),
            "Stage": st.column_config.TextColumn("Stage"),
            "Result": st.column_config.TextColumn("Result", width="large")
        },
        width="stretch",
        height=320
    )

st.markdown("<br>", unsafe_allow_html=True)

# 4. SEARCHABLE PLAYER SCOUTING DIRECTORY
render_section_header("Searchable Player Intelligence Directory", badge="Scouting Hub")
st.markdown("Search any athlete by name, ID, or squad to open in-depth career dossier and delivery analytics.")

p_search = st.text_input("Search Player by Name, ID, or Squad", placeholder="e.g. Suryakumar Yadav, Rashid Khan, Virat Kohli, 1699...")

players_display = players_master[[
    "player_id", "player_name", "teams", "batting_style", "bowling_style",
    "matches_played", "runs", "balls_faced", "bat_strike_rate", "wickets", "bowl_economy"
]].copy()

if p_search:
    search_str = p_search.lower().strip()
    players_display = players_display[
        players_display["player_name"].str.lower().str.contains(search_str) |
        players_display["player_id"].astype(str).str.contains(search_str) |
        players_display["teams"].str.lower().str.contains(search_str)
    ]

st.dataframe(
    players_display.rename(columns={
        "player_id": "Player ID",
        "player_name": "Player Name",
        "teams": "Teams",
        "batting_style": "Batting Style",
        "bowling_style": "Bowling Style",
        "matches_played": "Matches",
        "runs": "Runs",
        "balls_faced": "Balls Faced",
        "bat_strike_rate": "Bat SR",
        "wickets": "Wickets",
        "bowl_economy": "Econ"
    }),
    column_config={
        "Player ID": st.column_config.NumberColumn("ID", format="%d"),
        "Player Name": st.column_config.TextColumn("Player Name", width="medium"),
        "Teams": st.column_config.TextColumn("Teams", width="large"),
        "Batting Style": st.column_config.TextColumn("Bat Style"),
        "Bowling Style": st.column_config.TextColumn("Bowl Style"),
        "Matches": st.column_config.NumberColumn("Matches", format="%d"),
        "Runs": st.column_config.NumberColumn("Runs", format="%d"),
        "Balls Faced": st.column_config.NumberColumn("Balls", format="%d"),
        "Bat SR": st.column_config.NumberColumn("Bat SR", format="%.2f"),
        "Wickets": st.column_config.NumberColumn("Wickets", format="%d"),
        "Econ": st.column_config.NumberColumn("Econ", format="%.2f")
    },
    width="stretch",
    height=320
)

# Quick Player Profile Launcher
st.markdown("#### Quick Launch Dossier")
launch_col1, launch_col2, launch_col3 = st.columns([2, 1, 1])

player_choices = dict(zip(
    players_master["player_id"],
    players_master["player_name"] + " (ID: " + players_master["player_id"].astype(str) + ")"
))

with launch_col1:
    selected_player_id = st.selectbox(
        "Select Player to Inspect",
        options=list(player_choices.keys()),
        format_func=lambda x: player_choices[x],
        key="quick_launch_player_select"
    )

with launch_col2:
    if st.button("Open Batsman Profile", width="stretch"):
        st.session_state["selected_player_id"] = selected_player_id
        st.switch_page("pages/2_Batsman_Analysis.py")

with launch_col3:
    if st.button("Open Bowler Profile", width="stretch"):
        st.session_state["selected_bowler_id"] = selected_player_id
        st.switch_page("pages/3_Bowler_Analysis.py")
