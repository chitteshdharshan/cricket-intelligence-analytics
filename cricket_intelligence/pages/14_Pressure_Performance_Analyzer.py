"""
PRESSURE PERFORMANCE ANALYZER
Evaluates historical last-over pressure situations (e.g. 15 runs required in the 20th over).
Calculates Top 10 Batters (Chasing) and Top 10 Bowlers (Defending), provides transparent scoring,
player drill-down, match-by-match delivery timelines, and CSV export.
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
from src.ui_components import (
    apply_custom_theme,
    render_banner,
    render_sidebar_branding,
    render_kpi,
    render_section_header,
    get_plotly_layout_defaults
)
from src.pressure_analysis_engine import (
    get_qualifying_pressure_matches,
    calculate_batter_pressure_rankings,
    calculate_bowler_pressure_rankings,
    get_player_pressure_drilldown
)

# Page Setup
st.set_page_config(
    page_title="Pressure Performance Analyzer | Cricket Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()
render_sidebar_branding()

# Header Banner
render_banner(
    title="⚡ Pressure Performance Analyzer",
    subtitle="Analytical evaluation of historical final-over clutch scenarios — 15 runs required in the 20th over",
    badge="WicketIQ Decision Engine"
)

# Load Datasets
with st.spinner("Loading ball-by-ball and match records..."):
    data = load_all_data()
    bbb = data["bbb"]
    matches = data["matches"]
    comps = data["comps"]
    pm = data["players_master"]
    teams_df = data["teams"]

# Sidebar Search Interface
st.sidebar.markdown("### 🎯 Pressure Search Controls")

analysis_type = st.sidebar.radio(
    "Analysis Mode",
    ["Both Batters & Bowlers", "Top Batters Only", "Top Bowlers Only"],
    index=0
)

# Match situation preset
situation_preset = st.sidebar.selectbox(
    "Match Situation",
    [
        "15 runs required in the last over (Standard)",
        "12 runs required in the last over",
        "10 runs required in the last over",
        "18 runs required in the last over",
        "20 runs required in the last over",
        "Custom runs required in the last over"
    ],
    index=0
)

if "Custom" in situation_preset:
    target_runs = st.sidebar.number_input("Custom Runs Required", min_value=1, max_value=40, value=15, step=1)
else:
    target_runs = int(situation_preset.split(" ")[0])

# Match Format Filter
format_options = ["All Formats"] + sorted(list(comps["match_format"].dropna().unique()))
selected_format = st.sidebar.selectbox("Match Format", format_options, index=0)

# Team Filter
all_team_names = ["All Teams"] + sorted(list(teams_df["team_name"].dropna().unique()))
selected_team = st.sidebar.selectbox("Team Filter", all_team_names, index=0)

# Season Filter
season_options = ["All Seasons"] + sorted(list(comps["season"].dropna().unique()), reverse=True)
selected_season = st.sidebar.selectbox("Season", season_options, index=0)

# Minimum Opportunities Filter
min_opps = st.sidebar.slider("Minimum Opportunities", min_value=1, max_value=5, value=1, step=1)

# Player Search Dropdown
all_player_names = ["-- Select a Player to Inspect --"] + sorted(list(pm["player_name"].dropna().unique()))
selected_player_search = st.sidebar.selectbox("Direct Player Search", all_player_names, index=0)

# Action button
st.sidebar.markdown("<br>", unsafe_allow_html=True)
run_analysis = st.sidebar.button("⚡ Analyse Pressure Situation", width="stretch")

# Calculation Execution
q_matches = get_qualifying_pressure_matches(
    bbb=bbb,
    matches=matches,
    comps=comps,
    target_runs=target_runs,
    match_format=selected_format if selected_format != "All Formats" else None,
    team_name=selected_team if selected_team != "All Teams" else None,
    season=selected_season if selected_season != "All Seasons" else None,
    min_over=20
)

df_batters = calculate_batter_pressure_rankings(
    qualifying_matches=q_matches,
    bbb=bbb,
    target_runs=target_runs,
    min_opportunities=min_opps,
    over_id=20
)

df_bowlers = calculate_bowler_pressure_rankings(
    qualifying_matches=q_matches,
    bbb=bbb,
    target_runs=target_runs,
    min_opportunities=min_opps,
    over_id=20
)

# Overview KPIs
st.markdown("### 📊 Scenario Verification & Intelligence Overview")
k1, k2, k3, k4 = st.columns(4)

with k1:
    render_kpi("Qualifying Matches", f"{len(q_matches)}", f"Needing {target_runs} runs in 20th over", tone="primary")
with k2:
    chase_wins = int(q_matches["chase_won"].sum()) if not q_matches.empty else 0
    render_kpi("Chases Won", f"{chase_wins}", f"{round(chase_wins/max(1, len(q_matches))*100, 1)}% win rate", tone="success")
with k3:
    defences = int(q_matches["defence_won"].sum()) if not q_matches.empty else 0
    render_kpi("Defences Successful", f"{defences}", f"{round(defences/max(1, len(q_matches))*100, 1)}% defence rate", tone="primary")
with k4:
    tied_or_nr = len(q_matches) - (chase_wins + defences)
    render_kpi("Tied / Super Over", f"{tied_or_nr}", "Unsettled in regulation", tone="secondary")

# Transparent Methodology Notice
with st.expander("ℹ️ Transparent Scoring Formulas & Data Accuracy Disclosure", expanded=False):
    st.markdown(f"""
    **Data Integrity & Verification Rules:**
    1. **Target Score Derivation:** Target is derived directly from actual 1st innings scores (`1st innings runs + 1`) or verified `revised_target` for rain-affected encounters.
    2. **Point-in-Time Measurement:** The score and required runs are measured strictly at the start of over 20 (after exactly 19.0 overs), before any final-over delivery is bowled.
    3. **Legal Ball & Conceded Run Calculations:** Excludes run-outs from bowler wickets. Wides are charged to bowler economy but excluded from batter balls faced. No-balls count as batter balls faced.
    4. **Incomplete Matches:** Abandoned or No-Result fixtures are never treated as successful chases or defences.
    
    **Batter Pressure Rating Formula (0–100 Scale):**
    $$\\text{{Pressure Rating}} = \\left( 0.35 \\times \\text{{Chase Win \\%}} + 0.30 \\times \\min\\left(\\frac{{\\text{{SR}}}}{{250}}, 1.0\\right) + 0.15 \\times \\text{{Boundary \\%}} + 0.20 \\times \\min\\left(\\frac{{\\text{{Runs/Opp}}}}{{{target_runs}}}, 1.0\\right) \\right) \\times \\text{{SampleFactor}}$$
    
    **Bowler Pressure Rating Formula (0–100 Scale):**
    $$\\text{{Pressure Rating}} = \\left( 0.40 \\times \\text{{Defence \\%}} + 0.30 \\times \\max\\left(0, \\frac{{{target_runs} - \\text{{Econ}}}}{{{target_runs}}}\\right) + 0.20 \\times \\min\\left(\\frac{{\\text{{Wkts/Opp}}}}{{2}}, 1.0\\right) + 0.10 \\times \\text{{Dot \\%}} \\right) \\times \\text{{SampleFactor}}$$
    """)

if q_matches.empty:
    st.warning(f"⚠️ No matches in the historical database met the exact criteria of needing **{target_runs} runs** at the start of the 20th over under the selected filters. Please adjust the filters or select a different scenario preset.")
    st.stop()

# State management for player drill-down
if "selected_pressure_player_id" not in st.session_state:
    st.session_state.selected_pressure_player_id = None
if "selected_pressure_player_type" not in st.session_state:
    st.session_state.selected_pressure_player_type = "Batter"

# Main Tables Section
tab_bat, tab_bowl, tab_drill = st.tabs([
    "🏏 Top Batters (Chasing)",
    "🎯 Top Bowlers (Defending)",
    "🔍 Detailed Player Drill-Down"
])

# ----------------- TAB 1: BATTERS -----------------
with tab_bat:
    st.markdown(f"### 🏏 Top Batters — Chasing {target_runs} Runs in the Final Over")
    st.caption(f"Showing batters who faced at least 1 legal delivery in qualifying 20th overs (Min Opportunities: {min_opps}).")

    if df_batters.empty:
        st.info("No batters met the opportunity threshold under these filters.")
    else:
        top10_batters = df_batters.head(10).copy()

        # Display Table
        display_bat = top10_batters[[
            "rank", "player_name", "opportunities", "runs", "balls",
            "strike_rate", "fours", "sixes", "chase_wins", "pressure_rating", "sample_confidence"
        ]].rename(columns={
            "rank": "Rank",
            "player_name": "Batter",
            "opportunities": "Opps",
            "runs": "Runs",
            "balls": "Balls",
            "strike_rate": "Strike Rate",
            "fours": "4s",
            "sixes": "6s",
            "chase_wins": "Chases Won",
            "pressure_rating": "Pressure Rating",
            "sample_confidence": "Reliability"
        })

        st.dataframe(display_bat, width="stretch", hide_index=True)

        # Plotly Comparison Chart
        st.markdown("#### 📈 Batter Pressure Rating Comparison")
        fig_bat = px.bar(
            top10_batters,
            x="pressure_rating",
            y="player_name",
            orientation="h",
            text="pressure_rating",
            color="chase_wins",
            color_continuous_scale=["#151F30", "#3B82F6", "#16C784"],
            labels={"pressure_rating": "Pressure Rating (0-100)", "player_name": "Batter", "chase_wins": "Chases Won"},
            title=f"Top 10 Batters by Pressure Rating (Chasing {target_runs} in 20th Over)"
        )
        layout_defs = get_plotly_layout_defaults()
        fig_bat.update_layout(**layout_defs)
        fig_bat.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bat, width="stretch")

        # Interactive Player Selector from Top 10
        st.markdown("#### 🔎 Select a Batter from Rankings for In-Depth Drill-Down:")
        col_sel1, col_sel2 = st.columns([3, 1])
        with col_sel1:
            bat_choice = st.selectbox(
                "Click to Inspect Batter Performance Log:",
                top10_batters["player_name"].tolist(),
                key="bat_sel_box"
            )
        with col_sel2:
            if st.button("Open Batter Drill-Down ➡️", key="btn_open_bat"):
                chosen_row = top10_batters[top10_batters["player_name"] == bat_choice].iloc[0]
                st.session_state.selected_pressure_player_id = int(chosen_row["player_id"])
                st.session_state.selected_pressure_player_type = "Batter"
                st.rerun()

        # CSV Download
        csv_bat = top10_batters.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Top Batters Report (CSV)",
            data=csv_bat,
            file_name=f"top_batters_chasing_{target_runs}_runs_last_over.csv",
            mime="text/csv"
        )

# ----------------- TAB 2: BOWLERS -----------------
with tab_bowl:
    st.markdown(f"### 🎯 Top Bowlers — Defending {target_runs} Runs in the Final Over")
    st.caption(f"Showing bowlers who delivered the qualifying 20th over defending {target_runs} runs (Min Opportunities: {min_opps}).")

    if df_bowlers.empty:
        st.info("No bowlers met the opportunity threshold under these filters.")
    else:
        top10_bowlers = df_bowlers.head(10).copy()

        # Display Table
        display_bowl = top10_bowlers[[
            "rank", "player_name", "opportunities", "conceded", "balls",
            "wickets", "economy", "dots", "defences", "pressure_rating", "sample_confidence"
        ]].rename(columns={
            "rank": "Rank",
            "player_name": "Bowler",
            "opportunities": "Opps",
            "conceded": "Runs Conceded",
            "balls": "Legal Balls",
            "wickets": "Wickets",
            "economy": "Economy",
            "dots": "Dot Balls",
            "defences": "Defences Won",
            "pressure_rating": "Pressure Rating",
            "sample_confidence": "Reliability"
        })

        st.dataframe(display_bowl, width="stretch", hide_index=True)

        # Plotly Comparison Chart
        st.markdown("#### 📈 Bowler Pressure Rating Comparison")
        fig_bowl = px.bar(
            top10_bowlers,
            x="pressure_rating",
            y="player_name",
            orientation="h",
            text="pressure_rating",
            color="defences",
            color_continuous_scale=["#151F30", "#3B82F6", "#16C784"],
            labels={"pressure_rating": "Pressure Rating (0-100)", "player_name": "Bowler", "defences": "Defences Won"},
            title=f"Top 10 Bowlers by Pressure Rating (Defending {target_runs} in 20th Over)"
        )
        layout_defs = get_plotly_layout_defaults()
        fig_bowl.update_layout(**layout_defs)
        fig_bowl.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bowl, width="stretch")

        # Interactive Bowler Selector from Top 10
        st.markdown("#### 🔎 Select a Bowler from Rankings for In-Depth Drill-Down:")
        col_b1, col_b2 = st.columns([3, 1])
        with col_b1:
            bowl_choice = st.selectbox(
                "Click to Inspect Bowler Performance Log:",
                top10_bowlers["player_name"].tolist(),
                key="bowl_sel_box"
            )
        with col_b2:
            if st.button("Open Bowler Drill-Down ➡️", key="btn_open_bowl"):
                chosen_row = top10_bowlers[top10_bowlers["player_name"] == bowl_choice].iloc[0]
                st.session_state.selected_pressure_player_id = int(chosen_row["bowler_id"])
                st.session_state.selected_pressure_player_type = "Bowler"
                st.rerun()

        # CSV Download
        csv_bowl = top10_bowlers.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Top Bowlers Report (CSV)",
            data=csv_bowl,
            file_name=f"top_bowlers_defending_{target_runs}_runs_last_over.csv",
            mime="text/csv"
        )

# ----------------- TAB 3: PLAYER DRILL-DOWN -----------------
with tab_drill:
    st.markdown("### 🔍 Individual Player Pressure Performance Analysis")

    # If user selected a player from direct search dropdown in sidebar
    active_player_id = st.session_state.selected_pressure_player_id
    active_player_type = st.session_state.selected_pressure_player_type

    if selected_player_search != "-- Select a Player to Inspect --":
        matched_p = pm[pm["player_name"] == selected_player_search]
        if not matched_p.empty:
            active_player_id = int(matched_p.iloc[0]["player_id"])
            # Detect whether batter or bowler in this scenario
            is_in_batters = not df_batters.empty and active_player_id in df_batters["player_id"].values
            is_in_bowlers = not df_bowlers.empty and active_player_id in df_bowlers["bowler_id"].values
            if is_in_batters and not is_in_bowlers:
                active_player_type = "Batter"
            elif is_in_bowlers and not is_in_batters:
                active_player_type = "Bowler"

    if active_player_id is None:
        st.info("👈 Select a player using the dropdown above, click a player from the rankings tables, or use the sidebar Direct Player Search.")
    else:
        # Option to switch role
        p_col1, p_col2 = st.columns([3, 1])
        with p_col2:
            active_player_type = st.radio("Analyze Role:", ["Batter", "Bowler"], index=0 if active_player_type == "Batter" else 1, horizontal=True)

        drill = get_player_pressure_drilldown(
            player_id=active_player_id,
            player_type=active_player_type,
            qualifying_matches=q_matches,
            bbb=bbb,
            players_master=pm,
            over_id=20
        )

        if not drill.get("found"):
            st.warning(f"No qualifying final-over records found for this player as a **{active_player_type}** in matches needing exactly **{target_runs} runs**.")
        else:
            st.markdown(f"## 🌟 {drill['player_name']} — Last-Over Performance Profile")

            # Extract player ranking row if available
            ranking_row = None
            if active_player_type == "Batter" and not df_batters.empty and active_player_id in df_batters["player_id"].values:
                ranking_row = df_batters[df_batters["player_id"] == active_player_id].iloc[0]
            elif active_player_type == "Bowler" and not df_bowlers.empty and active_player_id in df_bowlers["bowler_id"].values:
                ranking_row = df_bowlers[df_bowlers["bowler_id"] == active_player_id].iloc[0]

            # Player KPI Cards
            c_k1, c_k2, c_k3, c_k4 = st.columns(4)
            with c_k1:
                render_kpi("Qualifying Matches", f"{drill['match_count']}", f"Overs defended/chased", tone="primary")
            with c_k2:
                if ranking_row is not None:
                    score = ranking_row["pressure_rating"]
                    rank = ranking_row["rank"]
                    render_kpi("Pressure Rating", f"{score}/100", f"Overall Rank #{rank}", tone="success")
                else:
                    render_kpi("Pressure Rating", "Evaluated", "Based on match record", tone="primary")
            with c_k3:
                if active_player_type == "Batter":
                    chase_w = ranking_row["chase_wins"] if ranking_row is not None else sum(1 for m in drill["matches"] if "Success" in m["outcome"])
                    render_kpi("Successful Chases", f"{chase_w}/{drill['match_count']}", f"Target: {target_runs} runs", tone="primary")
                else:
                    def_w = ranking_row["defences"] if ranking_row is not None else sum(1 for m in drill["matches"] if "Success" in m["outcome"])
                    render_kpi("Successful Defences", f"{def_w}/{drill['match_count']}", f"Target: {target_runs} runs", tone="primary")
            with c_k4:
                if active_player_type == "Batter" and ranking_row is not None:
                    render_kpi("Total Runs & SR", f"{ranking_row['runs']} runs", f"Strike Rate: {ranking_row['strike_rate']}", tone="secondary")
                elif active_player_type == "Bowler" and ranking_row is not None:
                    render_kpi("Economy & Wickets", f"{ranking_row['economy']} rpo", f"{ranking_row['wickets']} wickets taken", tone="secondary")
                else:
                    render_kpi("Reliability", "Historical Record", "Ball-by-ball verified", tone="secondary")

            # Match-by-Match Breakdown
            st.markdown("### 📋 Match-by-Match Delivery Timelines & Outcomes")
            for idx, m in enumerate(drill["matches"], 1):
                outcome_color = "#16C784" if "Success" in m["outcome"] else "#F87171"
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #101827; border: 1px solid #1E293B; border-left: 4px solid {outcome_color}; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9;">{m['fixture']} ({m['competition']})</span>
                            <span style="background-color: {outcome_color}22; color: {outcome_color}; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;">{m['outcome']}</span>
                        </div>
                        <p style="color: #94A3B8; margin: 6px 0 10px 0; font-size: 0.85rem;">Date: {m['match_date']} | Target: {m['target']} runs | Runs Needed at start of 20th Over: <strong>{m['needed_start_over']} runs</strong></p>
                        <p style="color: #E2E8F0; margin: 4px 0;"><strong>Over Performance:</strong> {m['performance']}</p>
                        <p style="color: #60A5FA; margin: 4px 0; font-family: monospace; font-size: 0.95rem;"><strong>Ball Sequence:</strong> {m['deliveries']}</p>
                        <p style="color: #64748B; margin: 4px 0 0 0; font-size: 0.8rem;"><em>Official Result: {m['result_text']}</em></p>
                    </div>
                    """, unsafe_allow_html=True)

            # Ranking Explanation Card
            st.markdown("### 💡 Why Did This Player Receive This Ranking?")
            if ranking_row is not None:
                if active_player_type == "Batter":
                    st.info(f"""
                    **Ranking Evaluation for {drill['player_name']}:**
                    - **Chase Win Impact:** The batting team successfully crossed the line in **{ranking_row['chase_wins']} of {ranking_row['opportunities']}** qualifying opportunities ({ranking_row['chase_win_pct']}%).
                    - **Scoring Intensity:** Scored **{ranking_row['runs']} runs** from **{ranking_row['balls']} balls** at an explosive strike rate of **{ranking_row['strike_rate']}**.
                    - **Boundary Hitting:** Hit **{ranking_row['fours']} fours** and **{ranking_row['sixes']} sixes**.
                    - **Sample Reliability:** Evaluated across {ranking_row['opportunities']} match(es) with a confidence factor of `{ranking_row['sample_confidence']}`.
                    """)
                else:
                    st.info(f"""
                    **Ranking Evaluation for {drill['player_name']}:**
                    - **Defence Success:** The bowling team successfully defended the target in **{ranking_row['defences']} of {ranking_row['opportunities']}** qualifying opportunities ({ranking_row['defence_pct']}%).
                    - **Run Containment:** Conceded **{ranking_row['conceded']} runs** across **{ranking_row['balls']} legal balls**, maintaining a clutch economy of **{ranking_row['economy']}** against an allowable run rate of {target_runs}.
                    - **Clutch Wickets:** Took **{ranking_row['wickets']} wickets** and delivered **{ranking_row['dots']} dot balls** ({ranking_row['dot_pct']}% dot ball frequency).
                    - **Sample Reliability:** Evaluated across {ranking_row['opportunities']} match(es) with a confidence factor of `{ranking_row['sample_confidence']}`.
                    """)
