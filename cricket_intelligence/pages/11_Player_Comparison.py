"""
PLAYER COMPARISON & RADAR BENCHMARKING
Multi-player comparative analytics across phases, bowling types, and boundary frequencies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_all_data
from src.batting_analysis import get_batsman_kpis, get_batsman_phase_stats
from src.bowling_analysis import get_bowler_kpis, get_bowler_phase_stats
from src.shot_analysis import get_pace_vs_spin_stats
from src.ui_components import apply_custom_theme, render_banner, render_sidebar_branding, render_section_header, get_plotly_layout_defaults

st.set_page_config(page_title="Player Comparison | Cricket Intelligence", page_icon="⚖️", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
bbb = data["bbb"]
players_master = data["players_master"]

render_banner(
    "PLAYER COMPARISON & BENCHMARKING",
    "Side-by-side radar analysis, phase comparison, and multi-skill benchmarking"
)

comp_mode = st.radio("Select Comparison Discipline", ["🏏 Batsmen Comparison", "🎯 Bowlers Comparison"], horizontal=True)

st.divider()

if comp_mode.startswith("🏏"):
    # Multi-select batsmen
    player_options = dict(zip(
        players_master["player_id"],
        players_master["player_name"] + " (ID: " + players_master["player_id"].astype(str) + ")"
    ))

    # Default selection: Suryakumar Yadav (1699) & Jos Buttler (813)
    default_pids = [1699, 813] if 1699 in player_options and 813 in player_options else list(player_options.keys())[:2]

    selected_pids = st.multiselect(
        "Select Two or More Batsmen to Compare",
        options=list(player_options.keys()),
        default=default_pids,
        format_func=lambda x: player_options[x]
    )

    if len(selected_pids) < 2:
        st.info("Please select at least two batsmen to compare.")
        st.stop()

    # Gather stats for each player
    comparison_metrics = []
    radar_data = []

    for pid in selected_pids:
        p_name = players_master[players_master["player_id"] == pid]["player_name"].iloc[0]
        kpis = get_batsman_kpis(bbb, pid)
        phases = get_batsman_phase_stats(bbb, pid)
        pvs = get_pace_vs_spin_stats(bbb, pid)["summary"]

        pp_sr = phases[phases["Phase"] == "Powerplay"]["Strike Rate"].iloc[0] if len(phases[phases["Phase"] == "Powerplay"]) > 0 else 0.0
        mid_sr = phases[phases["Phase"] == "Middle"]["Strike Rate"].iloc[0] if len(phases[phases["Phase"] == "Middle"]) > 0 else 0.0
        death_sr = phases[phases["Phase"] == "Death"]["Strike Rate"].iloc[0] if len(phases[phases["Phase"] == "Death"]) > 0 else 0.0

        pace_sr = pvs[pvs["Bowling Type"] == "Pace"]["Strike Rate"].iloc[0] if len(pvs[pvs["Bowling Type"] == "Pace"]) > 0 else 0.0
        spin_sr = pvs[pvs["Bowling Type"] == "Spin"]["Strike Rate"].iloc[0] if len(pvs[pvs["Bowling Type"] == "Spin"]) > 0 else 0.0

        comparison_metrics.append({
            "Player": p_name,
            "Player ID": pid,
            "Total Runs": kpis["total_runs"],
            "Balls Faced": kpis["balls_faced"],
            "Strike Rate": kpis["strike_rate"],
            "Average": kpis["batting_average"],
            "Boundary %": kpis["boundary_pct"],
            "4s": kpis["fours"],
            "6s": kpis["sixes"],
            "Pace SR": pace_sr,
            "Spin SR": spin_sr,
            "Powerplay SR": pp_sr,
            "Middle SR": mid_sr,
            "Death SR": death_sr
        })

        radar_data.append({
            "name": p_name,
            "values": [kpis["strike_rate"], kpis["batting_average"], kpis["boundary_pct"], pace_sr, spin_sr, death_sr]
        })

    comp_df = pd.DataFrame(comparison_metrics)

    render_section_header("Head-to-Head Batsman Matrix")
    st.dataframe(comp_df, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    render_section_header("Multi-Dimensional Batting Radar Comparison")
    categories = ["Strike Rate", "Average", "Boundary %", "Pace SR", "Spin SR", "Death SR"]

    fig_radar = go.Figure()
    colors = ["#16C784", "#3B82F6", "#F59E0B", "#EF4444", "#A855F7", "#EC4899"]

    for i, p in enumerate(radar_data):
        r_vals = p["values"] + [p["values"][0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=p["name"],
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, gridcolor="#263449"),
            angularaxis=dict(gridcolor="#263449")
        ),
        plot_bgcolor="#080D18",
        paper_bgcolor="#151F30",
        font=dict(color="#F5F7FA", size=13),
        showlegend=True,
        height=540
    )
    st.plotly_chart(fig_radar, width="stretch")

else:
    # Bowlers Comparison
    bowlers_master = players_master[players_master["legal_balls"] > 50]
    bowler_options = dict(zip(
        bowlers_master["player_id"],
        bowlers_master["player_name"] + " (ID: " + bowlers_master["player_id"].astype(str) + ")"
    ))

    # Default: Arshdeep Singh (6464) & Rashid Khan (4722)
    default_bids = [6464, 4722] if 6464 in bowler_options and 4722 in bowler_options else list(bowler_options.keys())[:2]

    selected_bids = st.multiselect(
        "Select Two or More Bowlers to Compare",
        options=list(bowler_options.keys()),
        default=default_bids,
        format_func=lambda x: bowler_options[x]
    )

    if len(selected_bids) < 2:
        st.info("Please select at least two bowlers to compare.")
        st.stop()

    b_comp_metrics = []
    b_radar_data = []

    for bid in selected_bids:
        b_name = players_master[players_master["player_id"] == bid]["player_name"].iloc[0]
        kpis = get_bowler_kpis(bbb, bid)
        phases = get_bowler_phase_stats(bbb, bid)

        pp_econ = phases[phases["Phase"] == "Powerplay"]["Economy"].iloc[0] if len(phases[phases["Phase"] == "Powerplay"]) > 0 else 0.0
        death_econ = phases[phases["Phase"] == "Death"]["Economy"].iloc[0] if len(phases[phases["Phase"] == "Death"]) > 0 else 0.0

        b_comp_metrics.append({
            "Bowler": b_name,
            "Bowler ID": bid,
            "Wickets": kpis["wickets"],
            "Economy": kpis["economy"],
            "Average": kpis["average"],
            "Strike Rate": kpis["strike_rate"],
            "Dot %": kpis["dot_pct"],
            "Overs": kpis["overs"],
            "Runs Conceded": kpis["runs_conceded"],
            "Powerplay Econ": pp_econ,
            "Death Econ": death_econ
        })

        b_radar_data.append({
            "name": b_name,
            "values": [kpis["wickets"], kpis["dot_pct"], pp_econ, death_econ, kpis["economy"]]
        })

    b_df = pd.DataFrame(b_comp_metrics)
    render_section_header("Head-to-Head Bowler Matrix")
    st.dataframe(b_df, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Multi-Dimensional Bowling Radar Comparison")
    b_categories = ["Wickets", "Dot %", "Powerplay Econ", "Death Econ", "Overall Econ"]

    fig_b_radar = go.Figure()
    colors = ["#16C784", "#3B82F6", "#F59E0B", "#EF4444", "#A855F7"]

    for i, p in enumerate(b_radar_data):
        r_vals = p["values"] + [p["values"][0]]
        fig_b_radar.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=b_categories + [b_categories[0]],
            fill="toself",
            name=p["name"],
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig_b_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, gridcolor="#263449"),
            angularaxis=dict(gridcolor="#263449")
        ),
        plot_bgcolor="#080D18",
        paper_bgcolor="#151F30",
        font=dict(color="#F5F7FA", size=13),
        showlegend=True,
        height=540
    )
    st.plotly_chart(fig_b_radar, width="stretch")
