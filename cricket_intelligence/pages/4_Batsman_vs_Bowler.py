"""
BATSMAN VS BOWLER INTELLIGENCE & HEAD-TO-HEAD MATCHUP EXPLORER
Comprehensive matchup matrix and delivery-by-delivery head-to-head duel breakdown.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import load_all_data, get_lookup_maps
from src.matchup_analysis import get_batsman_vs_all_bowlers, get_h2h_matchup_details
from src.ui_components import apply_custom_theme, render_banner, render_kpi, render_sidebar_branding, render_section_header

st.set_page_config(page_title="Batsman vs Bowler | Cricket Intelligence", page_icon="⚔️", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
bbb = data["bbb"]
players_master = data["players_master"]

render_banner(
    "BATSMAN VS BOWLER INTELLIGENCE",
    "Comprehensive head-to-head matchup matrix, dismissal frequency, boundary cadence, and delivery histories"
)

# View Mode Selection: "1. All Bowlers Faced by Batsman" vs "2. Dedicated 1-on-1 Head-to-Head Duel"
mode = st.radio(
    "Select Matchup Exploration Mode",
    ["Batsman vs All Bowlers Matrix (Section 4.E)", "Dedicated 1-on-1 Bowler vs Batsman Duel (Section 6)"],
    horizontal=True
)

st.divider()

if mode.startswith("Batsman vs All Bowlers"):
    # SECTION 4.E
    # Batsman Selector
    bat_choices = dict(zip(
        players_master["player_id"],
        players_master["player_name"] + " (ID: " + players_master["player_id"].astype(str) + ")"
    ))
    
    current_pid = st.session_state.get("selected_player_id", 1699)
    if current_pid not in bat_choices:
        current_pid = list(bat_choices.keys())[0]

    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        chosen_pid = st.selectbox(
            "Select Batsman to Analyze",
            options=list(bat_choices.keys()),
            index=list(bat_choices.keys()).index(current_pid),
            format_func=lambda x: bat_choices[x]
        )
        st.session_state["selected_player_id"] = chosen_pid

    batsman_name = players_master[players_master["player_id"] == chosen_pid]["player_name"].iloc[0]

    render_section_header(f"All Bowlers Faced by {batsman_name} (ID: {chosen_pid})")
    
    b_matrix = get_batsman_vs_all_bowlers(bbb, chosen_pid)
    
    if len(b_matrix) > 0:
        st.dataframe(b_matrix, use_container_width=True, height=360)

        # Drill-down into a specific bowler from this table
        render_section_header("Inspect Deliveries Faced Against Specific Bowler")
        bowler_choices = dict(zip(b_matrix["Bowler ID"], b_matrix["Bowler"] + " (ID: " + b_matrix["Bowler ID"].astype(str) + ")"))
        sel_bid = st.selectbox("Select Bowler to inspect deliveries:", options=list(bowler_choices.keys()), format_func=lambda x: bowler_choices[x])

        h2h_balls = bbb[(bbb["striker_id"] == chosen_pid) & (bbb["bowler_id"] == sel_bid)].copy()
        
        # Match & Over filter for this drilldown
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            m_opts = ["All"] + sorted(h2h_balls["match_id"].unique().tolist())
            chosen_m = st.selectbox("Filter Match ID", m_opts)
        with c_f2:
            ov_opts = ["All"] + sorted(h2h_balls["overid"].unique().tolist())
            chosen_ov = st.selectbox("Filter Over", ov_opts)

        if chosen_m != "All":
            h2h_balls = h2h_balls[h2h_balls["match_id"] == chosen_m]
        if chosen_ov != "All":
            h2h_balls = h2h_balls[h2h_balls["overid"] == chosen_ov]

        st.dataframe(
            h2h_balls[[
                "match_id", "innings", "overid", "over_ball_num", "batter_runs", "extras_type",
                "stroke", "delivery_style", "pace_or_spin", "is_four", "is_six", "is_wicket", "dismissal_type"
            ]].rename(columns={
                "match_id": "Match ID", "innings": "Inn", "overid": "Over", "over_ball_num": "Ball #",
                "batter_runs": "Runs", "extras_type": "Extras", "stroke": "Shot Played",
                "delivery_style": "Style", "pace_or_spin": "Type", "is_four": "4", "is_six": "6",
                "is_wicket": "Out", "dismissal_type": "Dismissal"
            }),
            use_container_width=True,
            height=280
        )
    else:
        st.info("No recorded deliveries faced for this batsman.")

else:
    # SECTION 6: DEDICATED 1-ON-1 HEAD-TO-HEAD DUEL
    render_section_header("Bowler vs Batsman Head-to-Head Matchup Explorer")
    
    col_b1, col_b2 = st.columns(2)
    
    # Batsmen options
    bat_dict = dict(zip(players_master["player_id"], players_master["player_name"] + " (ID: " + players_master["player_id"].astype(str) + ")"))
    # Bowlers options
    bowlers_master = players_master[players_master["legal_balls"] > 0]
    bowl_dict = dict(zip(bowlers_master["player_id"], bowlers_master["player_name"] + " (ID: " + bowlers_master["player_id"].astype(str) + ")"))

    cur_bat = st.session_state.get("selected_player_id", 1699)
    if cur_bat not in bat_dict:
        cur_bat = list(bat_dict.keys())[0]

    cur_bowl = st.session_state.get("selected_bowler_id", 6464)
    if cur_bowl not in bowl_dict:
        cur_bowl = list(bowl_dict.keys())[0]

    with col_b1:
        sel_bat = st.selectbox("Select Batsman", options=list(bat_dict.keys()), index=list(bat_dict.keys()).index(cur_bat), format_func=lambda x: bat_dict[x])
    with col_b2:
        sel_bowl = st.selectbox("Select Bowler", options=list(bowl_dict.keys()), index=list(bowl_dict.keys()).index(cur_bowl), format_func=lambda x: bowl_dict[x])

    bat_name = players_master[players_master["player_id"] == sel_bat]["player_name"].iloc[0]
    bowl_name = bowlers_master[bowlers_master["player_id"] == sel_bowl]["player_name"].iloc[0]

    h2h_data = get_h2h_matchup_details(bbb, sel_bat, sel_bowl)

    if h2h_data["balls"] > 0:
        render_section_header(f"Duel: {bat_name} vs {bowl_name}")
        
        # KPI Row
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            render_kpi("Balls Faced", f"{h2h_data['balls']}", "Legal balls", tone='muted')
        with k2:
            render_kpi("Runs Scored", f"{h2h_data['runs']}", f"Off the bat", tone='primary')
        with k3:
            render_kpi("Strike Rate", f"{h2h_data['strike_rate']}", "Runs / 100 balls", tone='secondary')
        with k4:
            render_kpi("Dismissals", f"{h2h_data['dismissals']}", f"Times out", tone='primary')
        with k5:
            render_kpi("Boundaries", f"{h2h_data['fours']}x4 | {h2h_data['sixes']}x6", f"{h2h_data['boundary_pct']}% runs", tone='secondary')
        with k6:
            render_kpi("Dot Balls", f"{h2h_data['dots']}", f"{h2h_data['dot_pct']}% dot rate", tone='muted')

        st.markdown("<br>", unsafe_allow_html=True)

        tab_m1, tab_m2, tab_m3 = st.tabs(["Over Progression", "Delivery Style Breakdown", "Match History & Ball-by-Ball"])
        with tab_m1:
            st.dataframe(h2h_data["over_progression"], use_container_width=True, height=220)
        with tab_m2:
            st.dataframe(h2h_data["delivery_styles"], use_container_width=True, height=220)
        with tab_m3:
            render_section_header("Match-by-Match Records")
            st.dataframe(h2h_data["match_history"], use_container_width=True, height=180)
            render_section_header("Complete Ball-by-Ball History")
            st.dataframe(
                h2h_data["deliveries"][[
                    "match_id", "innings", "overid", "over_ball_num", "batter_runs", "stroke",
                    "delivery_style", "is_four", "is_six", "is_wicket", "dismissal_type"
                ]],
                use_container_width=True,
                height=260
            )
    else:
        st.warning(f"No recorded deliveries found between {bat_name} and {bowl_name} in the analytical dataset.")

