"""
CRICKET INTELLIGENCE • INDIVIDUAL BATSMAN PROFILE & BALL-BY-BALL DOSSIER
Dedicated player analysis page with over-by-over breakdown and delivery-level inspector.
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

from src.data_loader import load_all_data, get_lookup_maps
from src.batting_analysis import (
    get_batsman_kpis,
    get_batsman_innings_breakdown,
    get_batsman_over_by_over,
    get_over_deliveries,
    get_batsman_phase_stats,
    apply_filters
)
from src.shot_analysis import (
    get_pace_vs_spin_stats,
    get_stroke_analysis,
    get_delivery_type_analysis,
    get_shot_performance_summary,
    get_shot_metrics,
    create_shot_comparison_chart
)
from src.ui_components import (
    apply_custom_theme,
    render_banner,
    render_kpi,
    render_sidebar_branding,
    render_section_header,
    get_plotly_layout_defaults
)

st.set_page_config(page_title="Batsman Analysis | Cricket Intelligence", page_icon="🏏", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
bbb = data["bbb"]
matches = data["matches"]
comps = data["comps"]
teams = data["teams"]
players_master = data["players_master"]
lookups = get_lookup_maps(data)

# Persistent player selection
if "selected_player_id" not in st.session_state:
    # Default to Suryakumar Yadav (ID: 1699) or highest run scorer
    st.session_state["selected_player_id"] = 1699

# Batsman Selector in Sidebar & Header
player_options = dict(zip(
    players_master["player_id"],
    players_master["player_name"] + " (ID: " + players_master["player_id"].astype(str) + ")"
))

st.sidebar.markdown("### 🏏 Select Batsman")
current_id = st.session_state.get("selected_player_id", 1699)
if current_id not in player_options:
    current_id = list(player_options.keys())[0]

chosen_player_id = st.sidebar.selectbox(
    "Choose Batsman",
    options=list(player_options.keys()),
    index=list(player_options.keys()).index(current_id),
    format_func=lambda x: player_options[x]
)
st.session_state["selected_player_id"] = chosen_player_id
player_id = chosen_player_id

# Player Master Info
p_info = players_master[players_master["player_id"] == player_id].iloc[0]
player_name = p_info["player_name"]

# Dynamic Match & Competition Filters for this Player
player_bbb = bbb[bbb["striker_id"] == player_id]
player_comps = ["All"] + sorted(player_bbb["comp_name"].dropna().unique().tolist())
player_seasons = ["All"] + sorted(player_bbb["season"].dropna().unique().tolist())
player_matches = ["All"] + sorted(player_bbb["match_id"].unique().tolist())
player_innings = ["All"] + sorted([int(x) for x in player_bbb["innings"].dropna().unique().tolist()])

st.sidebar.markdown("### 🎯 Granular Filters")
f_comp = st.sidebar.selectbox("Filter Competition", player_comps)
f_season = st.sidebar.selectbox("Filter Season", player_seasons)
f_match = st.sidebar.selectbox("Filter Match ID", player_matches)
f_innings = st.sidebar.selectbox("Filter Innings", player_innings)

# Date filter
min_d = pd.to_datetime(player_bbb["match_date"]).min()
max_d = pd.to_datetime(player_bbb["match_date"]).max()
f_dates = st.sidebar.date_input("Match Date Range", value=(min_d, max_d) if pd.notna(min_d) else None)

filters = {}
if f_comp != "All":
    c_sub = comps[comps["comp_name"] == f_comp]
    if len(c_sub) > 0:
        filters["comp_id"] = c_sub["comp_id"].iloc[0]
if f_season != "All":
    filters["season"] = f_season
if f_match != "All":
    filters["match_id"] = f_match
if f_innings != "All":
    filters["innings"] = int(f_innings)
if isinstance(f_dates, tuple) and len(f_dates) == 2:
    filters["start_date"] = f_dates[0]
    filters["end_date"] = f_dates[1]

# Header Banner
render_banner(
    f"{player_name} • Batting Dossier",
    f"Player ID: {player_id} • Batting: {p_info['batting_style']} • Bowling: {p_info['bowling_style']}",
    badge=p_info['batting_style']
)

# SECTION A: Player Information Dossier
render_section_header("Section A: Player Profile & Information", badge="Scouting Profile")
info_col1, info_col2, info_col3, info_col4 = st.columns(4)
with info_col1:
    st.markdown(f"**Full Name:** `{player_name}`")
    st.markdown(f"**Player ID:** `{player_id}`")
with info_col2:
    st.markdown(f"**Primary Squads:** {p_info['teams']}")
    st.markdown(f"**Batting Style:** `{p_info['batting_style']}`")
with info_col3:
    # Typical batting order position
    orders = player_bbb["batting_order"].dropna()
    mode_order = int(orders.mode().iloc[0]) if len(orders) > 0 else "N/A"
    st.markdown(f"**Primary Position:** `#{mode_order}`")
    st.markdown(f"**Matches Played:** `{p_info['matches_played']}`")
with info_col4:
    st.markdown(f"**Innings Batted:** `{p_info['bat_innings']}`")
    st.markdown(f"**Tournaments:** {len(player_comps)-1} competitions")

st.markdown("<br>", unsafe_allow_html=True)

# SECTION B: Overall Batting Statistics (KPI Cards)
render_section_header("Section B: Key Batting Performance Indicators", badge="Career Statistics")
kpis = get_batsman_kpis(bbb, player_id, filters)

k_row1 = st.columns(6)
with k_row1[0]:
    render_kpi("Total Runs", f"{kpis['total_runs']:,}", f"{kpis['innings']} Innings")
with k_row1[1]:
    render_kpi("Balls Faced", f"{kpis['balls_faced']:,}", "Legal balls faced")
with k_row1[2]:
    render_kpi("Batting Average", f"{kpis['batting_average']}", f"{kpis['dismissals']} dismissals")
with k_row1[3]:
    render_kpi("Strike Rate", f"{kpis['strike_rate']}", "Runs per 100 balls")
with k_row1[4]:
    render_kpi("Highest Score", f"{kpis['highest_score']}", f"{kpis['not_outs']} not outs")
with k_row1[5]:
    render_kpi("Boundary %", f"{kpis['boundary_pct']}%", "Runs in boundaries")

k_row2 = st.columns(6)
with k_row2[0]:
    render_kpi("Fours (4s)", f"{kpis['fours']}", f"{round(kpis['balls_faced']/kpis['fours'], 1) if kpis['fours']>0 else 0} balls/4")
with k_row2[1]:
    render_kpi("Sixes (6s)", f"{kpis['sixes']}", f"{round(kpis['balls_faced']/kpis['sixes'], 1) if kpis['sixes']>0 else 0} balls/6")
with k_row2[2]:
    render_kpi("Dot Balls", f"{kpis['dot_balls']}", f"{round(kpis['dot_balls']/kpis['balls_faced']*100, 1) if kpis['balls_faced']>0 else 0}% dot rate")
with k_row2[3]:
    render_kpi("Centuries (100+)", f"{kpis['centuries']}", "100s scored")
with k_row2[4]:
    render_kpi("Half-Centuries (50+)", f"{kpis['fifties']}", "50-99 scores")
with k_row2[5]:
    render_kpi("30+ Scores", f"{kpis['thirty_plus']}", "Aggressive starts")

st.markdown("<br>", unsafe_allow_html=True)

# SECTION C: Over-by-Over Batting Performance & Drill-Down
render_section_header("Section C: Over-by-Over Batting Performance & Drill-Down", badge="Over Cadence")
st.markdown("Inspect the batsman's scoring rate in each over. Select any over to immediately inspect every individual delivery faced.")

obo_df = get_batsman_over_by_over(bbb, player_id, match_id=filters.get("match_id"), filters=filters)

if len(obo_df) > 0:
    obo_display = obo_df[[
        "overid", "bowlers", "balls_faced", "runs", "fours", "sixes", "strike_rate", "dismissals", "innings_faced"
    ]].rename(columns={
        "overid": "Over",
        "bowlers": "Bowler(s)",
        "balls_faced": "Balls Faced",
        "runs": "Runs Scored",
        "fours": "4s",
        "sixes": "6s",
        "strike_rate": "Strike Rate",
        "dismissals": "Dismissals",
        "innings_faced": "Innings"
    })

    st.dataframe(
        obo_display,
        column_config={
            "Over": st.column_config.NumberColumn("Over", format="%d"),
            "Bowler(s)": st.column_config.TextColumn("Bowler(s)", width="large", help="All unique bowlers faced by batsman in this over"),
            "Balls Faced": st.column_config.NumberColumn("Balls Faced", format="%d"),
            "Runs Scored": st.column_config.NumberColumn("Runs Scored", format="%d"),
            "4s": st.column_config.NumberColumn("4s", format="%d"),
            "6s": st.column_config.NumberColumn("6s", format="%d"),
            "Strike Rate": st.column_config.NumberColumn("Strike Rate", format="%.2f"),
            "Dismissals": st.column_config.NumberColumn("Dismissals", format="%d"),
            "Innings": st.column_config.NumberColumn("Innings", format="%d")
        },
        width="stretch",
        height=320
    )

    with st.expander("📋 View Complete Bowler Roster per Over (Expanded List)"):
        for _, r in obo_df.iterrows():
            bowler_list_str = str(r["bowlers"])
            num_b = bowler_list_str.count(",") + 1 if bowler_list_str else 0
            st.markdown(f"**Over {r['overid']}** ({r['balls_faced']} balls, {r['runs']} runs • {num_b} unique bowler{'s' if num_b != 1 else ''}):  \n{bowler_list_str}")

    # Delivery inspection for selected over
    st.markdown("#### 🔬 Inspect Deliveries for Specific Over")
    avail_overs = sorted(obo_df["overid"].unique().tolist())
    selected_over = st.selectbox("Select Over to inspect individual deliveries:", avail_overs)

    deliveries_df = get_over_deliveries(bbb, player_id, selected_over, match_id=filters.get("match_id"))
    
    st.dataframe(
        deliveries_df.rename(columns={
            "match_id": "Match ID",
            "innings": "Inn",
            "overid": "Over",
            "over_ball_num": "Ball #",
            "bowler_id": "Bowler ID",
            "bowler_name": "Bowler Name",
            "batter_runs": "Runs",
            "extras_type": "Extras",
            "stroke": "Shot Played",
            "delivery_style": "Delivery Style",
            "pace_or_spin": "Pace/Spin",
            "is_four": "4",
            "is_six": "6",
            "is_wicket": "Wicket",
            "dismissal_type": "Dismissal"
        }),
        width="stretch",
        height=260
    )
else:
    st.info("No deliveries found for the selected filter combination.")

st.divider()

# SECTION D & E: Batting Phase Analysis & Innings Log
render_section_header("Section D: Batting Phase Analysis", badge="Phase Splits")
phase_df = get_batsman_phase_stats(bbb, player_id, filters)
if len(phase_df) > 0:
    p_col1, p_col2 = st.columns([1.5, 1])
    with p_col1:
        st.dataframe(phase_df, width="stretch", height=180)
    with p_col2:
        fig_phase = px.bar(
            phase_df, x="Phase", y="Strike Rate",
            color="Phase",
            title="Strike Rate by Innings Phase",
            color_discrete_map={"Powerplay": "#16C784", "Middle": "#3B82F6", "Death": "#EF4444"}
        )
        defaults_p = get_plotly_layout_defaults()
        fig_phase.update_layout(**defaults_p)
        st.plotly_chart(fig_phase, width="stretch")

st.markdown("<br>", unsafe_allow_html=True)

# Innings breakdown
render_section_header("Section E: Innings-by-Innings Performance Log", badge="Match History")
inn_df = get_batsman_innings_breakdown(bbb, player_id, filters)
if len(inn_df) > 0:
    st.dataframe(
        inn_df.rename(columns={
            "match_id": "Match ID",
            "innings": "Inn",
            "date": "Date",
            "competition": "Competition",
            "runs": "Runs",
            "balls": "Balls",
            "fours": "4s",
            "sixes": "6s",
            "strike_rate": "Strike Rate",
            "status": "Dismissal / Status"
        }),
        width="stretch",
        height=320
    )

st.divider()

# SECTION F: Complete Ball-by-Ball Delivery Explorer & Shot Performance Analysis
render_section_header("Section F: Ball-by-Ball Delivery Explorer & Shot Performance", badge="Delivery Drill-Down")
st.markdown("Comprehensive shot-wise analytics, boundary & dismissal breakdown, and delivery inspection.")

# Base filtered deliveries for this player respecting all global filters
filtered_player_bbb = apply_filters(player_bbb, filters)

# Compute full shot summary
shot_summary_df = get_shot_performance_summary(bbb, player_id, filters)

# Available shot types
available_shots = ["All"] + (sorted(filtered_player_bbb["stroke"].dropna().unique().tolist()) if len(filtered_player_bbb) > 0 else [])

# Interactivity State Management
if "active_shot_filter" not in st.session_state or st.session_state["active_shot_filter"] not in available_shots:
    st.session_state["active_shot_filter"] = "All"

# Sync table row click if present
table_state = st.session_state.get("shot_summary_table", {})
if isinstance(table_state, dict) and "selection" in table_state:
    selected_rows = table_state["selection"].get("rows", [])
    if selected_rows and selected_rows[0] < len(shot_summary_df):
        table_shot = shot_summary_df.iloc[selected_rows[0]]["Shot Played"]
        if table_shot in available_shots and table_shot != st.session_state["active_shot_filter"]:
            st.session_state["active_shot_filter"] = table_shot

# Sync chart point click if present
chart_state = st.session_state.get("shot_comparison_chart", {})
if isinstance(chart_state, dict) and "selection" in chart_state:
    selected_points = chart_state["selection"].get("points", [])
    if selected_points and "x" in selected_points[0]:
        chart_shot = selected_points[0]["x"]
        if chart_shot in available_shots and chart_shot != st.session_state["active_shot_filter"]:
            st.session_state["active_shot_filter"] = chart_shot

active_shot = st.session_state["active_shot_filter"]

# 1. SHOT PERFORMANCE SUMMARY SECTION
st.markdown("#### 🎯 Shot Performance Summary")

sel_col1, sel_col2 = st.columns([3, 1])
with sel_col1:
    chosen_shot = st.selectbox(
        "Select Shot for In-Depth Performance & Delivery Inspection:",
        options=available_shots,
        index=available_shots.index(active_shot) if active_shot in available_shots else 0,
        key="shot_summary_selector"
    )
    if chosen_shot != active_shot:
        st.session_state["active_shot_filter"] = chosen_shot
        active_shot = chosen_shot

with sel_col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Show All Shots", width="stretch", disabled=(active_shot == "All")):
        st.session_state["active_shot_filter"] = "All"
        st.rerun()

# 2 & 3. Calculate and display shot performance metrics and KPI cards
shot_metrics = get_shot_metrics(bbb, player_id, active_shot, filters)

if active_shot != "All":
    st.info(f"Displaying performance analytics for: **{active_shot}** ({shot_metrics['deliveries']} deliveries, {shot_metrics['runs']} runs)")
else:
    st.info(f"Displaying aggregate performance analytics across: **All Shots** ({shot_metrics['deliveries']} deliveries, {shot_metrics['runs']} runs)")

# Primary KPI Cards (Requirement 3: Fours, Sixes, Outs, Total Runs, Total Deliveries, Strike Rate)
k_shot1, k_shot2, k_shot3, k_shot4, k_shot5, k_shot6 = st.columns(6)
with k_shot1:
    render_kpi("Fours (4s)", f"{shot_metrics['fours']}", f"{shot_metrics['fours']*4} runs from 4s")
with k_shot2:
    render_kpi("Sixes (6s)", f"{shot_metrics['sixes']}", f"{shot_metrics['sixes']*6} runs from 6s")
with k_shot3:
    render_kpi("Outs (Wickets)", f"{shot_metrics['dismissals']}", f"{round(shot_metrics['balls']/shot_metrics['dismissals'], 1) if shot_metrics['dismissals']>0 else 0} balls/out")
with k_shot4:
    render_kpi("Total Runs", f"{shot_metrics['runs']:,}", f"From {shot_metrics['balls']} legal balls")
with k_shot5:
    render_kpi("Total Deliveries", f"{shot_metrics['deliveries']:,}", f"{shot_metrics['balls']} legal balls faced")
with k_shot6:
    render_kpi("Strike Rate", f"{shot_metrics['strike_rate']}", f"{shot_metrics['boundary_pct']}% boundary runs")

# Granular metrics row (Requirement 2: Dot balls, Singles, Twos, Threes, Strike rate, Boundary percentage)
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
g_col1, g_col2, g_col3, g_col4, g_col5, g_col6 = st.columns(6)
with g_col1:
    st.metric("Dot Balls", f"{shot_metrics['dots']}", f"{round(shot_metrics['dots']/shot_metrics['balls']*100, 1) if shot_metrics['balls']>0 else 0}% dot rate")
with g_col2:
    st.metric("Singles (1s)", f"{shot_metrics['singles']}", f"{round(shot_metrics['singles']/shot_metrics['balls']*100, 1) if shot_metrics['balls']>0 else 0}% of balls")
with g_col3:
    st.metric("Twos (2s)", f"{shot_metrics['twos']}", f"{round(shot_metrics['twos']/shot_metrics['balls']*100, 1) if shot_metrics['balls']>0 else 0}% of balls")
with g_col4:
    st.metric("Threes (3s)", f"{shot_metrics['threes']}", f"{round(shot_metrics['threes']/shot_metrics['balls']*100, 1) if shot_metrics['balls']>0 else 0}% of balls")
with g_col5:
    st.metric("Boundary %", f"{shot_metrics['boundary_pct']}%", f"{shot_metrics['fours']}x4, {shot_metrics['sixes']}x6")
with g_col6:
    st.metric("Balls Faced", f"{shot_metrics['balls']}", f"{round(shot_metrics['runs']/shot_metrics['balls'], 2) if shot_metrics['balls']>0 else 0} runs/ball")

st.markdown("<br>", unsafe_allow_html=True)

# 4 & 5. Shot-wise Performance Summary Table & Comparison Bar Chart
st.markdown("#### 📊 Shot-wise Performance Summary & Visual Comparison")
st.caption("💡 Tip: Click any row in the table or bar in the chart to automatically focus the delivery explorer on that shot.")

tab_tbl, tab_chart = st.tabs(["📋 Shot-wise Performance Summary Table (All Shots)", "📊 Boundary & Dismissal Comparison Chart"])

with tab_tbl:
    if len(shot_summary_df) > 0:
        st.dataframe(
            shot_summary_df[[
                "Shot Played", "Balls Played", "Runs Scored", "Fours", "Sixes",
                "Dismissals", "Dot Balls", "Strike Rate", "Boundary %"
            ]],
            column_config={
                "Shot Played": st.column_config.TextColumn("Shot Name", width="medium"),
                "Balls Played": st.column_config.NumberColumn("Balls Played", format="%d"),
                "Runs Scored": st.column_config.NumberColumn("Runs Scored", format="%d"),
                "Fours": st.column_config.NumberColumn("Fours", format="%d"),
                "Sixes": st.column_config.NumberColumn("Sixes", format="%d"),
                "Dismissals": st.column_config.NumberColumn("Dismissals", format="%d"),
                "Dot Balls": st.column_config.NumberColumn("Dot Balls", format="%d"),
                "Strike Rate": st.column_config.NumberColumn("Strike Rate", format="%.2f"),
                "Boundary %": st.column_config.NumberColumn("Boundary %", format="%.2f%%")
            },
            selection_mode="single-row",
            on_select="rerun",
            key="shot_summary_table",
            width="stretch",
            height=320
        )
    else:
        st.info("No shot data available for the selected filters.")

with tab_chart:
    if len(shot_summary_df) > 0:
        fig_shots = create_shot_comparison_chart(shot_summary_df, top_n=12, selected_shot=active_shot)
        st.plotly_chart(
            fig_shots,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key="shot_comparison_chart"
        )
    else:
        st.info("No shot data available to chart.")

st.divider()

# 6 & 8. BALL-BY-BALL DELIVERY EXPLORER
st.markdown("#### 🔬 Ball-by-Ball Delivery Explorer")
st.markdown("Inspect every individual ball faced by this batsman matching the selected criteria.")

exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)
with exp_c1:
    filter_bowler = st.selectbox("Filter Bowler", ["All"] + sorted(filtered_player_bbb["bowler_name"].dropna().unique().tolist()))
with exp_c2:
    filter_pace_spin = st.selectbox("Filter Pace / Spin", ["All", "Pace", "Spin"])
with exp_c3:
    filter_shot = st.selectbox(
        "Filter Shot Played",
        available_shots,
        index=available_shots.index(active_shot) if active_shot in available_shots else 0,
        key="delivery_explorer_shot_filter"
    )
    if filter_shot != active_shot:
        st.session_state["active_shot_filter"] = filter_shot
        active_shot = filter_shot

with exp_c4:
    filter_outcome = st.selectbox("Filter Ball Outcome", ["All", "Dot Ball", "Singles (1)", "Boundaries (4/6)", "Wickets"])

exp_df = filtered_player_bbb.copy()
if filter_bowler != "All":
    exp_df = exp_df[exp_df["bowler_name"] == filter_bowler]
if filter_pace_spin != "All":
    exp_df = exp_df[exp_df["pace_or_spin"] == filter_pace_spin]
if active_shot != "All":
    exp_df = exp_df[exp_df["stroke"] == active_shot]
if filter_outcome == "Dot Ball":
    exp_df = exp_df[exp_df["is_dot_ball_batter"]]
elif filter_outcome == "Singles (1)":
    exp_df = exp_df[exp_df["batter_runs"] == 1]
elif filter_outcome == "Boundaries (4/6)":
    exp_df = exp_df[exp_df["is_four"] | exp_df["is_six"]]
elif filter_outcome == "Wickets":
    exp_df = exp_df[exp_df["is_wicket"] & ((exp_df["dismissed_player_id"] == player_id) | (exp_df["dismissed_player_id"].isna()))]

# Pagination to guarantee smooth rendering and prevent WebSocket message overflow
total_delivs = len(exp_df)
deliv_page_size = 100
total_deliv_pages = max(1, (total_delivs + deliv_page_size - 1) // deliv_page_size)

if total_deliv_pages > 1:
    col_pg1, col_pg2 = st.columns([1, 3])
    with col_pg1:
        cur_deliv_page = st.number_input(
            f"Page (1 to {total_deliv_pages})",
            min_value=1,
            max_value=total_deliv_pages,
            value=1,
            step=1,
            key="batsman_deliv_page_input"
        )
    with col_pg2:
        start_idx = (cur_deliv_page - 1) * deliv_page_size
        end_idx = min(cur_deliv_page * deliv_page_size, total_delivs)
        st.caption(f"Showing deliveries {start_idx + 1} – {end_idx} of {total_delivs:,} matching deliveries")
    deliv_display = exp_df.iloc[start_idx:end_idx]
else:
    deliv_display = exp_df
    st.caption(f"Displaying all {total_delivs:,} matching deliveries.")

st.dataframe(
    deliv_display[[
        "match_id", "innings", "overid", "over_ball_num", "bowler_name", "bowler_id",
        "batter_runs", "stroke", "delivery_style", "pace_or_spin", "on_or_off",
        "is_four", "is_six", "is_wicket", "dismissal_type"
    ]].rename(columns={
        "match_id": "Match ID", "innings": "Inn", "overid": "Over", "over_ball_num": "Ball #",
        "bowler_name": "Bowler", "bowler_id": "Bowler ID", "batter_runs": "Runs",
        "stroke": "Shot Played", "delivery_style": "Delivery Style", "pace_or_spin": "Type",
        "on_or_off": "Side", "is_four": "4", "is_six": "6", "is_wicket": "Out",
        "dismissal_type": "Dismissal"
    }),
    width="stretch",
    height=340
)

