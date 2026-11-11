"""
SHOT TYPE & DELIVERY STYLE INTELLIGENCE
Detailed analysis of batsman shot distribution, productivity, delivery styles, and on/off side splits.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import load_all_data
from src.shot_analysis import get_stroke_analysis, get_delivery_type_analysis, get_on_off_side_analysis
from src.ui_components import apply_custom_theme, render_banner, render_kpi, render_sidebar_branding, render_section_header, get_plotly_layout_defaults

st.set_page_config(page_title="Shot Analysis | Cricket Intelligence", page_icon="🏏", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
bbb = data["bbb"]
players_master = data["players_master"]

render_banner(
    "SHOT & DELIVERY INTELLIGENCE",
    "Productive shot analysis, stroke execution, delivery style responses, and field-side distribution"
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

render_section_header(f"Performance Breakdown for {batsman_name} (ID: {chosen_pid})")

# Tabs for Shot Analysis, Delivery Types, and On/Off Side
tab_shot, tab_del, tab_side = st.tabs(["🏏 Shot Type Analysis (Stroke)", "🌀 Delivery Style Analysis", "📐 On-Side vs Off-Side Split"])

with tab_shot:
    render_section_header("Productive Shot Ranking (stroke Column)")
    stroke_df = get_stroke_analysis(bbb, chosen_pid)
    if len(stroke_df) > 0:
        c1, c2 = st.columns([1.6, 1.4])
        with c1:
            st.dataframe(stroke_df, width="stretch", height=360)
        with c2:
            fig_stroke = px.bar(
                stroke_df.head(10),
                x="Runs Scored", y="Shot Played",
                orientation="h",
                color="Strike Rate",
                color_continuous_scale="Viridis",
                title="Top 10 Most Productive Shots by Runs"
            )
            fig_stroke.update_layout(
                **{**get_plotly_layout_defaults(), "yaxis": dict(autorange="reversed")}
            )
            st.plotly_chart(fig_stroke, width="stretch")
    else:
        st.info("No recorded strokes available for this player.")

with tab_del:
    render_section_header("Performance Against Delivery Styles")
    st.markdown("""
        > [!NOTE]
        > Delivery styles are categorized directly from ball-tracking data. Unknown or unclassified deliveries are transparently reported without fabrication.
    """)
    del_df = get_delivery_type_analysis(bbb, chosen_pid)
    if len(del_df) > 0:
        c1, c2 = st.columns([1.6, 1.4])
        with c1:
            st.dataframe(del_df, width="stretch", height=360)
        with c2:
            fig_del = px.scatter(
                del_df[del_df["Balls Faced"] >= 15],
                x="Strike Rate", y="Dot %",
                size="Balls Faced", color="Runs Scored",
                hover_name="Delivery Style",
                title="Strike Rate vs Dot % by Delivery Style (min 15 balls)",
                color_continuous_scale="Plasma"
            )
            fig_del.update_layout(**get_plotly_layout_defaults())
            st.plotly_chart(fig_del, width="stretch")
    else:
        st.info("No delivery styles recorded.")

with tab_side:
    render_section_header("On-Side vs Off-Side Scoring Cadence")
    side_df = get_on_off_side_analysis(bbb, chosen_pid)
    if len(side_df) > 0:
        c1, c2 = st.columns([1.4, 1.6])
        with c1:
            st.dataframe(side_df, width="stretch", height=180)
        with c2:
            fig_pie = px.pie(
                side_df, names="Field Side", values="Runs Scored",
                title="Run Distribution: On-Side vs Off-Side",
                hole=0.45,
                color="Field Side",
                color_discrete_map={"Off-side": "#38BDF8", "On-side": "#F59E0B", "Unknown": "#64748B"}
            )
            fig_pie.update_layout(**get_plotly_layout_defaults())
            st.plotly_chart(fig_pie, width="stretch")

