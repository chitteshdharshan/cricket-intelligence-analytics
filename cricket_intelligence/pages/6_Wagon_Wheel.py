"""
CRICKET GROUND VISUALIZATION & WAGON WHEEL INTELLIGENCE
Interactive cricket ground geometry, 17 fielding zones, transparent data disclosures,
and simulated shot trajectory explorer.
"""

import sys
from pathlib import Path
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_all_data
from src.wagon_wheel import (
    create_cricket_ground_figure,
    FIELD_ZONES,
    calculate_zone_statistics,
    get_zone_delivery_drilldown
)
from src.shot_analysis import get_on_off_side_analysis
from src.ui_components import apply_custom_theme, render_banner, render_sidebar_branding, render_section_header

st.set_page_config(page_title="Wagon Wheel | Cricket Intelligence", page_icon="🏟️", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
bbb = data["bbb"]
players_master = data["players_master"]

render_banner(
    "INTERACTIVE CRICKET GROUND & WAGON WHEEL",
    "Fielding zone mapping, 30-yard circle, boundary rope, and shot direction intelligence"
)

# Mandatory Data Integrity Alert as required by prompt Section 4.D & 9
st.markdown("""
    <div class="ci-disclaimer">
        <strong>⚠️ DATA INTEGRITY & DIRECTION SPECIFICATION:</strong><br>
        The provided datasets contain verified <code>stroke</code> names (e.g. <em>Cover Drive, Square Cut</em>) 
        and <code>on_or_off</code> indicators (<em>Off-side, On-side</em>), but do <strong>not</strong> contain raw 2D spatial coordinate tracking 
        or explicit fielding zone tags. In accordance with strict cricket analytical standards, we do <strong>not</strong> fabricate artificial zone numbers. 
        Below we present verified On-Side vs Off-Side scoring, standard field geometry for all 17 zones, and an optional directional vector simulation toggle.
    </div>
""", unsafe_allow_html=True)

# Batsman Selector
bat_choices = dict(zip(
    players_master["player_id"],
    players_master["player_name"] + " (ID: " + players_master["player_id"].astype(str) + ")"
))

current_pid = st.session_state.get("selected_player_id", 1699)
if current_pid not in bat_choices:
    current_pid = list(bat_choices.keys())[0]

sel_col1, sel_col2, sel_col3 = st.columns([2, 1, 1])
with sel_col1:
    chosen_pid = st.selectbox(
        "Select Batsman",
        options=list(bat_choices.keys()),
        index=list(bat_choices.keys()).index(current_pid),
        format_func=lambda x: bat_choices[x]
    )
    st.session_state["selected_player_id"] = chosen_pid

batsman_info = players_master[players_master["player_id"] == chosen_pid].iloc[0]
batsman_name = batsman_info["player_name"]
batting_hand = batsman_info["batting_style"]

# Filter shots for this batsman
p_bbb = bbb[bbb["striker_id"] == chosen_pid]

with sel_col2:
    filter_match = st.selectbox("Filter Match", ["All Matches"] + sorted(p_bbb["match_id"].unique().tolist()))
with sel_col3:
    render_vectors = st.toggle("Enable Shot Vector Simulation", value=True, help="Renders directional vectors derived from stroke category and handedness")

shots_to_render = p_bbb.copy()
if filter_match != "All Matches":
    shots_to_render = shots_to_render[shots_to_render["match_id"] == filter_match]

# Ground & Zone Layout
ground_col, table_col = st.columns([1.5, 1.5])

with ground_col:
    fig_ground = create_cricket_ground_figure(
        df_shots=shots_to_render,
        batsman_name=batsman_name,
        handedness=batting_hand,
        render_simulated_vectors=render_vectors
    )
    st.plotly_chart(fig_ground, width="stretch")

with table_col:
    render_section_header("The 17 Cricket Fielding Zones")
    zones_df = pd.DataFrame(FIELD_ZONES)
    st.dataframe(
        zones_df.rename(columns={
            "name": "Zone Name",
            "side": "Field Side",
            "min_deg": "Start Angle (°)",
            "max_deg": "End Angle (°)",
            "dist": "Depth"
        }),
        width="stretch",
        height=240
    )

    render_section_header("Verified Positional Scoring Split")
    side_split = get_on_off_side_analysis(bbb, chosen_pid)
    st.dataframe(side_split, width="stretch", height=130)

st.divider()

# Section 4.D: Zone-by-Zone Performance Table
render_section_header("Fielding Zone Scoring Analytics")
st.markdown("Metrics for each of the 17 fielding zones based on mapped shot directions:")

zone_stats_df = calculate_zone_statistics(shots_to_render, batting_hand)
st.dataframe(zone_stats_df, width="stretch", height=280)

# Drilldown into a specific fielding zone
render_section_header("Inspect Deliveries for Selected Fielding Zone")
available_zones = [z["name"] for z in FIELD_ZONES]
sel_drill_zone = st.selectbox("Select Fielding Zone to inspect deliveries:", available_zones)

zone_deliveries = get_zone_delivery_drilldown(shots_to_render, sel_drill_zone, batting_hand)
if len(zone_deliveries) > 0:
    st.dataframe(zone_deliveries, width="stretch", height=260)
else:
    st.info(f"No deliveries recorded in the {sel_drill_zone} zone for this batsman.")

st.markdown("<br>", unsafe_allow_html=True)

# Delivery Inspector
render_section_header("All Deliveries Faced by Batsman")
st.dataframe(
    shots_to_render[[
        "match_id", "innings", "overid", "over_ball_num", "bowler_name", "batter_runs",
        "stroke", "delivery_style", "on_or_off", "is_four", "is_six", "is_wicket"
    ]].rename(columns={
        "match_id": "Match ID", "innings": "Inn", "overid": "Over", "over_ball_num": "Ball #",
        "bowler_name": "Bowler", "batter_runs": "Runs", "stroke": "Shot Played",
        "delivery_style": "Delivery Style", "on_or_off": "Side", "is_four": "4", "is_six": "6", "is_wicket": "Wicket"
    }),
    width="stretch",
    height=280
)
