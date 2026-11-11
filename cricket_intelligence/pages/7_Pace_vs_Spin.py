"""
PACE VS SPIN BATTING PERFORMANCE INTELLIGENCE
Dedicated comparative module analyzing batsman vulnerability, cadence, and boundary metrics
against Fast Bowlers versus Spinners.
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
from src.shot_analysis import get_pace_vs_spin_stats
from src.ui_components import apply_custom_theme, render_banner, render_kpi, render_sidebar_branding, render_section_header, get_plotly_layout_defaults

st.set_page_config(page_title="Pace vs Spin | Cricket Intelligence", page_icon="⚡", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
bbb = data["bbb"]
players_master = data["players_master"]

render_banner(
    "PACE VS SPIN INTELLIGENCE",
    "Comparative analysis of batsman effectiveness, strike rates, dot percentages, and dismissal frequencies"
)

# Batsman Selector
bat_choices = dict(zip(
    players_master["player_id"],
    players_master["player_name"] + " (ID: " + players_master["player_id"].astype(str) + ")"
))

current_pid = st.session_state.get("selected_player_id", 1699)
if current_pid not in bat_choices:
    current_pid = list(bat_choices.keys())[0]

chosen_pid = st.sidebar.selectbox(
    "Choose Batsman",
    options=list(bat_choices.keys()),
    index=list(bat_choices.keys()).index(current_pid),
    format_func=lambda x: bat_choices[x]
)
st.session_state["selected_player_id"] = chosen_pid
batsman_name = players_master[players_master["player_id"] == chosen_pid]["player_name"].iloc[0]

render_section_header(f"Pace vs Spin Performance — {batsman_name} (ID: {chosen_pid})")

stats = get_pace_vs_spin_stats(bbb, chosen_pid)
sum_df = stats["summary"]
style_df = stats["styles"]

if len(sum_df) > 0:
    # Transposed Side-by-Side Comparison Table as requested in Section 4.F
    render_section_header("Core Comparative Matrix")
    
    # Transpose summary for clean side-by-side view
    metrics_to_show = [
        ("Balls Faced", "Balls faced"),
        ("Runs Scored", "Runs scored"),
        ("Strike Rate", "Strike rate"),
        ("Average", "Batting average"),
        ("Fours", "Fours (4s)"),
        ("Sixes", "Sixes (6s)"),
        ("Dot Balls", "Dot balls"),
        ("Dot %", "Dot-ball percentage"),
        ("Dismissals", "Dismissals"),
        ("Boundary %", "Boundary percentage"),
        ("Balls / 4", "Balls per four"),
        ("Balls / 6", "Balls per six"),
        ("Wicket Freq (Balls/Out)", "Balls per dismissal")
    ]
    
    pace_row = sum_df[sum_df["Bowling Type"] == "Pace"].iloc[0] if len(sum_df[sum_df["Bowling Type"] == "Pace"]) > 0 else {}
    spin_row = sum_df[sum_df["Bowling Type"] == "Spin"].iloc[0] if len(sum_df[sum_df["Bowling Type"] == "Spin"]) > 0 else {}

    comp_table = []
    for col_key, label in metrics_to_show:
        comp_table.append({
            "Metric": label,
            "Pace": pace_row.get(col_key, "N/A"),
            "Spin": spin_row.get(col_key, "N/A")
        })
    
    c_table_df = pd.DataFrame(comp_table)
    
    t_col1, t_col2 = st.columns([1.2, 1.8])
    with t_col1:
        st.dataframe(c_table_df, width="stretch", height=480)
    with t_col2:
        # Visual Comparison Charts
        fig_sr = px.bar(
            sum_df, x="Bowling Type", y="Strike Rate",
            color="Bowling Type",
            title="Strike Rate: Pace vs Spin",
            color_discrete_map={"Pace": "#3B82F6", "Spin": "#16C784"}
        )
        fig_sr.update_layout(**get_plotly_layout_defaults())
        st.plotly_chart(fig_sr, width="stretch")

        fig_dot = px.bar(
            sum_df, x="Bowling Type", y="Dot %",
            color="Bowling Type",
            title="Dot Ball Percentage: Pace vs Spin",
            color_discrete_map={"Pace": "#3B82F6", "Spin": "#16C784"}
        )
        fig_dot.update_layout(**get_plotly_layout_defaults())
        st.plotly_chart(fig_dot, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # Granular Bowling Types
    render_section_header("Granular Performance Against Specific Bowling Styles")
    st.dataframe(style_df, width="stretch", height=320)
else:
    st.info("No recorded deliveries for this batsman.")

