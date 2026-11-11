"""
ADVANCED BATTING & BOWLING PERFORMANCE ANALYZER
Comprehensive performance intelligence platform featuring:
1. Advanced Batting Analysis (Dynamic SR, Balls Per Boundary, Boundary %)
2. Advanced Bowling Analysis (Economy containment, legal balls, wickets)
3. Pace vs Spin Batting Analysis (Splits, comparative dominance filters)
4. Over-wise Batting & Bowling Analysis (Overs 1-20 progression and phase dynamics)
5. Yorker & Delivery Style Analysis (Verified yorkers, delivery classification)
6. Handedness Analysis (Bowler vs LHB/RHB and Batter vs Bowling Arm)
7. Multi-Player Comparison Engine
"""

import sys
from pathlib import Path

# Robustly ensure package directory and repository root are top of sys.path
_current_file = Path(__file__).resolve()
_pages_dir = _current_file.parent
_pkg_root = _pages_dir.parent          # cricket_intelligence directory
_repo_root = _pkg_root.parent          # repository root

for _p in [str(_repo_root), str(_pkg_root)]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from src.data_loader import load_all_data
    from src.ui_components import (
        apply_custom_theme,
        render_banner,
        render_sidebar_branding,
        render_kpi,
        render_section_header,
        get_plotly_layout_defaults
    )
    from src.advanced_batting_engine import (
        calculate_advanced_batting_stats,
        get_pace_vs_spin_batter_breakdown,
        get_pace_spin_comparative_table,
        filter_bbb_deliveries,
        get_batter_overwise_boundary_intelligence,
        get_batter_over_peaks,
        get_batter_match_innings_list,
        get_multi_batter_overwise_comparison
    )
    from src.advanced_bowling_engine import calculate_advanced_bowling_stats
    from src.overwise_analysis_engine import (
        get_batter_overwise_analysis,
        get_bowler_overwise_analysis,
        get_phase_wise_bowler_rankings,
        get_over_specific_bowler_rankings
    )
    from src.bowling_line_length_engine import (
        calculate_yorker_performance_stats,
        get_bowler_delivery_style_distribution,
        get_data_transparency_disclosure
    )
    from src.handedness_analysis_engine import (
        get_bowler_vs_handedness_stats,
        get_all_bowlers_handedness_rankings,
        get_batter_vs_bowling_arm_stats
    )
except (ImportError, ModuleNotFoundError):
    from cricket_intelligence.src.data_loader import load_all_data
    from cricket_intelligence.src.ui_components import (
        apply_custom_theme,
        render_banner,
        render_sidebar_branding,
        render_kpi,
        render_section_header,
        get_plotly_layout_defaults
    )
    from cricket_intelligence.src.advanced_batting_engine import (
        calculate_advanced_batting_stats,
        get_pace_vs_spin_batter_breakdown,
        get_pace_spin_comparative_table,
        filter_bbb_deliveries,
        get_batter_overwise_boundary_intelligence,
        get_batter_over_peaks,
        get_batter_match_innings_list,
        get_multi_batter_overwise_comparison
    )
    from cricket_intelligence.src.advanced_bowling_engine import calculate_advanced_bowling_stats
    from cricket_intelligence.src.overwise_analysis_engine import (
        get_batter_overwise_analysis,
        get_bowler_overwise_analysis,
        get_phase_wise_bowler_rankings,
        get_over_specific_bowler_rankings
    )
    from cricket_intelligence.src.bowling_line_length_engine import (
        calculate_yorker_performance_stats,
        get_bowler_delivery_style_distribution,
        get_data_transparency_disclosure
    )
    from cricket_intelligence.src.handedness_analysis_engine import (
        get_bowler_vs_handedness_stats,
        get_all_bowlers_handedness_rankings,
        get_batter_vs_bowling_arm_stats
    )

# Page Configuration
st.set_page_config(
    page_title="Advanced Performance Analyzer | Cricket Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()
render_sidebar_branding()

# Header Banner
render_banner(
    title="🔬 Advanced Batting & Bowling Performance Analyzer",
    subtitle="Deep situational filters, delivery telemetry, pace vs spin splits, over-by-over progression, and handedness dynamics",
    badge="WicketIQ Analytics Suite"
)

# Load Datasets
with st.spinner("Loading analytical datasets..."):
    data = load_all_data()
    bbb = data["bbb"]
    matches = data["matches"]
    comps = data["comps"]
    pm = data["players_master"]
    teams_df = data["teams"]

# Global Sidebar Filters
st.sidebar.markdown("### 🎛️ Global Data Controls")

format_options = ["All Formats"] + sorted(list(comps["match_format"].dropna().unique()))
sel_format = st.sidebar.selectbox("Match Format", format_options, index=0)

comp_options = ["All Tournaments"] + sorted(list(comps["comp_name"].dropna().unique()))
sel_comp = st.sidebar.selectbox("Tournament / League", comp_options, index=0)

season_options = ["All Seasons"] + sorted(list(comps["season"].dropna().unique()), reverse=True)
sel_season = st.sidebar.selectbox("Season", season_options, index=0)

team_options = ["All Teams"] + sorted(list(teams_df["team_name"].dropna().unique()))
sel_team = st.sidebar.selectbox("Team Filter", team_options, index=0)

# Filter base deliveries
filtered_bbb = filter_bbb_deliveries(
    bbb=bbb,
    format_name=sel_format,
    team_name=sel_team,
    season=sel_season,
    comp_name=sel_comp
)

# Overview KPIs
st.markdown("### 📊 Database Coverage & Scope")
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi("Filtered Deliveries", f"{len(filtered_bbb):,}", "Active ball-by-ball records", tone="primary")
with k2:
    render_kpi("Matches Analyzed", f"{filtered_bbb['match_id'].nunique()}", "Distinct fixtures", tone="secondary")
with k3:
    render_kpi("Active Batters", f"{filtered_bbb['striker_id'].nunique()}", "Unique strikers", tone="success")
with k4:
    render_kpi("Active Bowlers", f"{filtered_bbb['bowler_id'].nunique()}", "Unique bowlers", tone="primary")

# Main Multi-Tab Interface
tabs = st.tabs([
    "🏏 Batting Analysis",
    "🎯 Bowling Analysis",
    "⚡ Pace vs Spin",
    "📈 Over-wise Dynamics",
    "🎯 Yorker & Delivery Styles",
    "🔄 Handedness Dynamics",
    "👥 Multi-Player Comparison"
])

# =========================================================================
# TAB 1: ADVANCED BATTING ANALYSIS
# =========================================================================
with tabs[0]:
    st.markdown("### 🏏 Advanced Batting Performance & Boundary Frequency")
    st.caption("Filter batters dynamically by strike rate, balls per boundary frequency, balls faced, and bowling type.")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        min_balls_bat = st.number_input("Minimum Balls Faced", min_value=10, max_value=2000, value=50, step=10, key="min_b_bat")
    with f_col2:
        bat_bowl_type = st.selectbox("Facing Bowling Type", ["All", "Pace", "Spin"], index=0, key="bat_bowl_type")
    with f_col3:
        sr_filter_mode = st.selectbox("Strike Rate Filter", ["All", "Above", "Below", "Between"], index=0, key="sr_filt_mode")
    with f_col4:
        bpb_filter_mode = st.selectbox("Balls Per Boundary Filter", ["All", "Below", "Above", "Between"], index=0, key="bpb_filt_mode")

    # Dynamic SR Values
    sr_v1, sr_v2 = 0.0, 300.0
    if sr_filter_mode in ["Above", "Below"]:
        sr_v1 = st.slider("Strike Rate Threshold", min_value=50.0, max_value=250.0, value=150.0 if sr_filter_mode == "Above" else 110.0, step=5.0)
    elif sr_filter_mode == "Between":
        sr_range = st.slider("Strike Rate Range", min_value=50.0, max_value=250.0, value=(120.0, 160.0), step=5.0)
        sr_v1, sr_v2 = sr_range[0], sr_range[1]

    # Dynamic BPB Values
    bpb_v1, bpb_v2 = 0.0, 30.0
    if bpb_filter_mode in ["Below", "Above"]:
        bpb_v1 = st.slider("Balls Per Boundary Threshold", min_value=2.0, max_value=15.0, value=5.0 if bpb_filter_mode == "Below" else 8.0, step=0.5)
    elif bpb_filter_mode == "Between":
        bpb_range = st.slider("Balls Per Boundary Range", min_value=2.0, max_value=15.0, value=(3.0, 6.0), step=0.5)
        bpb_v1, bpb_v2 = bpb_range[0], bpb_range[1]

    # Calculate Batting Data
    df_adv_bat = calculate_advanced_batting_stats(
        bbb=filtered_bbb,
        min_balls_faced=int(min_balls_bat),
        sr_mode=sr_filter_mode,
        sr_val1=sr_v1,
        sr_val2=sr_v2,
        bpb_mode=bpb_filter_mode,
        bpb_val1=bpb_v1,
        bpb_val2=bpb_v2,
        bowling_type=bat_bowl_type
    )

    if df_adv_bat.empty:
        st.warning("No batters matched the specified filter thresholds.")
    else:
        st.markdown(f"**Found {len(df_adv_bat)} qualifying batters**")

        disp_cols = [
            "striker_name", "matches", "runs", "balls", "strike_rate",
            "batting_average", "fours", "sixes", "balls_per_boundary",
            "balls_per_four", "balls_per_six", "boundary_pct", "dot_pct"
        ]
        disp_bat = df_adv_bat[disp_cols].rename(columns={
            "striker_name": "Batter",
            "matches": "Matches",
            "runs": "Runs",
            "balls": "Balls Faced",
            "strike_rate": "Strike Rate",
            "batting_average": "Average",
            "fours": "4s",
            "sixes": "6s",
            "balls_per_boundary": "Balls/Boundary",
            "balls_per_four": "Balls/4",
            "balls_per_six": "Balls/6",
            "boundary_pct": "Boundary %",
            "dot_pct": "Dot %"
        })

        st.dataframe(disp_bat, width="stretch", hide_index=True)

        # Plotly Scatter: Strike Rate vs Balls Per Boundary
        st.markdown("#### 📈 Strike Rate vs Balls Per Boundary Efficiency")
        top_plot_bat = df_adv_bat.head(25)
        fig_bat_scatter = px.scatter(
            top_plot_bat,
            x="balls_per_boundary",
            y="strike_rate",
            size="runs",
            color="boundary_pct",
            hover_name="striker_name",
            color_continuous_scale=["#151F30", "#3B82F6", "#16C784"],
            labels={
                "balls_per_boundary": "Balls Per Boundary (Lower = More Frequent)",
                "strike_rate": "Batting Strike Rate",
                "boundary_pct": "Boundary Runs %"
            },
            title="Batting Intensity Matrix (Top 25 Scorers)"
        )
        layout_defs = get_plotly_layout_defaults()
        fig_bat_scatter.update_layout(**layout_defs)
        st.plotly_chart(fig_bat_scatter, width="stretch")

        # CSV Download
        csv_b = disp_bat.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Batting Analysis Report (CSV)",
            data=csv_b,
            file_name="wicket_iq_advanced_batting_analysis.csv",
            mime="text/csv"
        )

        # =====================================================================
        # OVER-WISE BATTING PERFORMANCE ANALYSIS (INTEGRATED FEATURE)
        # =====================================================================
        st.markdown("---")
        render_section_header("🏏 OVER-WISE BATTING PERFORMANCE ANALYSIS", badge="Dynamic Progression Engine")
        st.caption(
            "Granular over-by-over (Overs 1–20) breakdown of runs, balls, boundaries, boundary runs, dot balls, "
            "and strike rates, dynamically integrated with the qualifying batters above."
        )

        qualifying_batters = sorted(df_adv_bat["striker_name"].dropna().unique())
        
        # Default to a prominent qualifier from the requested list if available
        default_idx = 0
        preferred_names = [
            "Nehal Wadhera", "Shashank Singh", "Washington Sundar",
            "Angkrish Raghuvanshi", "Sunil Narine", "Yashasvi Jaiswal",
            "Heinrich Klaasen", "Shivam Dube", "Phil Salt", "Shreyas Iyer"
        ]
        for p_name in preferred_names:
            if p_name in qualifying_batters:
                default_idx = qualifying_batters.index(p_name)
                break

        # Analysis Mode and Batter Selection Controls
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1.6])
        with ctrl_col1:
            analysis_mode = st.radio(
                "Analysis Mode:",
                ["Overall Analysis (Aggregate across all matches)", "Match-wise Analysis (Specific fixture)"],
                index=0,
                key="ow_analysis_mode_radio"
            )
        with ctrl_col2:
            sel_ow_batter = st.selectbox(
                "🏏 Select Primary Batter:",
                qualifying_batters,
                index=default_idx,
                key="sel_ow_batter_dropdown"
            )
        with ctrl_col3:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            filter_active = st.checkbox("Show active overs only (Balls Faced > 0)", value=False, key="chk_ow_active")

        batter_row = df_adv_bat[df_adv_bat["striker_name"] == sel_ow_batter].iloc[0]
        batter_id = int(batter_row["striker_id"])
        is_match_mode = "Match-wise" in analysis_mode

        selected_match_id = None
        selected_innings = None

        if is_match_mode:
            match_list = get_batter_match_innings_list(filtered_bbb, batter_id)
            if match_list.empty:
                st.warning(f"No specific match innings found for {sel_ow_batter} under the active global filters.")
            else:
                sel_match_label = st.selectbox(
                    "📅 Select Match Fixture & Innings:",
                    match_list["label"].tolist(),
                    index=0,
                    key="sel_ow_match_fixture"
                )
                m_row = match_list[match_list["label"] == sel_match_label].iloc[0]
                selected_match_id = int(m_row["match_id"])
                selected_innings = int(m_row["innings"])

                st.markdown(
                    f"""
                    <div style="background-color: #151F30; border: 1px solid #263449; border-radius: 6px; padding: 10px 14px; margin-bottom: 14px;">
                        <span style="color: #94A3B8; font-size: 0.8rem; text-transform: uppercase;">Match Context:</span>
                        <strong style="color: #F5F7FA; margin-left: 6px;">{m_row['home_team']} vs {m_row['away_team']}</strong>
                        <span style="color: #16C784; margin-left: 10px;">Innings {selected_innings}</span>
                        <span style="color: #94A3B8; margin-left: 10px;">Date: {str(m_row['match_date'])[:10]}</span>
                        <span style="color: #3B82F6; margin-left: 12px; font-weight: 600;">Match Final: {m_row['runs']} runs off {m_row['balls']} balls ({m_row['fours']}x4, {m_row['sixes']}x6)</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info(
                "ℹ️ **Overall Analysis Documentation:** Performance is aggregated by match over number (Overs 1–20) "
                "across all selected fixtures meeting active filters. Over 1 represents the combined first overs of the batter's innings, "
                "Over 2 represents the second overs, and so on. Strike rates are computed from total runs divided by total balls faced."
            )

        # Compute Over-wise Batting Dataset
        df_ow_table = get_batter_overwise_boundary_intelligence(
            filtered_bbb,
            batter_id,
            batter_name=sel_ow_batter,
            match_id=selected_match_id,
            innings=selected_innings,
            only_active_overs=False
        )

        if df_ow_table.empty:
            st.warning("No delivery data available for this batter under current selection.")
        else:
            # Over-wise Maximum Performance KPIs (5 requested KPI cards)
            peaks = get_batter_over_peaks(
                filtered_bbb,
                batter_id,
                match_id=selected_match_id,
                innings=selected_innings
            )

            st.markdown("#### ⚡ Over-wise Maximum Performance")
            
            if is_match_mode:
                r_val = peaks["single_max_runs"]
                r_ov = f"Over {peaks['single_max_runs_over']}"
                f_val = peaks["single_max_fours"]
                f_ov = f"Over {peaks['single_max_fours_over']}"
                s_val = peaks["single_max_sixes"]
                s_ov = f"Over {peaks['single_max_sixes_over']}"
                b_val = peaks["single_max_boundaries"]
                b_ov = f"Over {peaks['single_max_boundaries_over']}"
                bnd_r_val = peaks["single_max_boundary_runs"]
                bnd_r_ov = f"Over {peaks['single_max_boundary_runs_over']}"
            else:
                r_val = peaks["single_max_runs"]
                r_ov = f"Over {peaks['single_max_runs_over']} (Agg: Ov {peaks['agg_max_runs_over']})"
                f_val = peaks["single_max_fours"]
                f_ov = f"Over {peaks['single_max_fours_over']} (Agg: Ov {peaks['agg_max_fours_over']})"
                s_val = peaks["single_max_sixes"]
                s_ov = f"Over {peaks['single_max_sixes_over']} (Agg: Ov {peaks['agg_max_sixes_over']})"
                b_val = peaks["single_max_boundaries"]
                b_ov = f"Over {peaks['single_max_boundaries_over']} (Agg: Ov {peaks['agg_max_boundaries_over']})"
                bnd_r_val = peaks["single_max_boundary_runs"]
                bnd_r_ov = f"Over {peaks['single_max_boundary_runs_over']} (Agg: Ov {peaks['agg_max_boundary_runs_over']})"

            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                render_kpi("Highest Runs in an Over", f"{r_val}", r_ov, tone="primary")
            with k2:
                render_kpi("Highest Fours in an Over", f"{f_val}", f_ov, tone="secondary")
            with k3:
                render_kpi("Highest Sixes in an Over", f"{s_val}", s_ov, tone="primary")
            with k4:
                render_kpi("Highest Total Boundaries", f"{b_val}", b_ov, tone="secondary")
            with k5:
                render_kpi("Highest Boundary Runs", f"{bnd_r_val}", bnd_r_ov, tone="primary")

            # Over-wise Charts (5 Interactive Plotly Visualizations)
            st.markdown("#### 📈 Over-wise Analytical Charts")
            
            chart_data = df_ow_table[df_ow_table["Balls Faced"] > 0] if filter_active else df_ow_table
            layout_defs = get_plotly_layout_defaults()

            c_tab1, c_tab2, c_tab3, c_tab4, c_tab5 = st.tabs([
                "🏏 1. Runs by Over",
                "💥 2. Fours vs Sixes by Over",
                "⚡ 3. Strike Rate by Over",
                "🎯 4. Boundary Distribution by Over",
                "🔥 5. Boundary Runs by Over"
            ])

            with c_tab1:
                fig_c1 = px.bar(
                    chart_data,
                    x="Over",
                    y="Runs",
                    text="Runs",
                    title=f"{sel_ow_batter} — Runs Scored by Over",
                    color="Runs",
                    color_continuous_scale=["#151F30", "#3B82F6", "#16C784"]
                )
                fig_c1.update_layout(**layout_defs)
                fig_c1.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                st.plotly_chart(fig_c1, width="stretch")

            with c_tab2:
                fig_c2 = go.Figure()
                fig_c2.add_trace(go.Bar(
                    x=chart_data["Over"],
                    y=chart_data["Fours"],
                    name="Fours (4s)",
                    marker_color="#3B82F6",
                    text=chart_data["Fours"],
                    textposition="auto"
                ))
                fig_c2.add_trace(go.Bar(
                    x=chart_data["Over"],
                    y=chart_data["Sixes"],
                    name="Sixes (6s)",
                    marker_color="#16C784",
                    text=chart_data["Sixes"],
                    textposition="auto"
                ))
                fig_c2.update_layout(
                    barmode="group",
                    title=f"{sel_ow_batter} — Fours vs Sixes Pattern by Over",
                    xaxis_title="Over Number",
                    yaxis_title="Count",
                    **layout_defs
                )
                fig_c2.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                st.plotly_chart(fig_c2, width="stretch")

            with c_tab3:
                sr_data = df_ow_table[df_ow_table["Balls Faced"] > 0]
                fig_c3 = px.line(
                    sr_data,
                    x="Over",
                    y="Strike Rate",
                    markers=True,
                    text="Strike Rate",
                    title=f"{sel_ow_batter} — Batting Strike Rate Progression by Over",
                    color_discrete_sequence=["#16C784"]
                )
                fig_c3.update_traces(textposition="top center")
                fig_c3.update_layout(**layout_defs)
                fig_c3.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                st.plotly_chart(fig_c3, width="stretch")

            with c_tab4:
                fig_c4 = go.Figure()
                fig_c4.add_trace(go.Bar(
                    x=chart_data["Over"],
                    y=chart_data["Fours"],
                    name="Fours (4s)",
                    marker_color="#3B82F6",
                    text=chart_data["Fours"],
                    textposition="auto"
                ))
                fig_c4.add_trace(go.Bar(
                    x=chart_data["Over"],
                    y=chart_data["Sixes"],
                    name="Sixes (6s)",
                    marker_color="#16C784",
                    text=chart_data["Sixes"],
                    textposition="auto"
                ))
                fig_c4.add_trace(go.Bar(
                    x=chart_data["Over"],
                    y=chart_data["Total Boundaries"],
                    name="Total Boundaries",
                    marker_color="#F59E0B",
                    text=chart_data["Total Boundaries"],
                    textposition="auto"
                ))
                fig_c4.update_layout(
                    barmode="group",
                    title=f"{sel_ow_batter} — Boundary Distribution (4s, 6s, and Total Boundaries) by Over",
                    xaxis_title="Over Number",
                    yaxis_title="Boundary Count",
                    **layout_defs
                )
                fig_c4.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                st.plotly_chart(fig_c4, width="stretch")

            with c_tab5:
                chart_data_copy = chart_data.copy()
                chart_data_copy["Runs from 4s"] = chart_data_copy["Fours"] * 4
                chart_data_copy["Runs from 6s"] = chart_data_copy["Sixes"] * 6

                fig_c5 = go.Figure()
                fig_c5.add_trace(go.Bar(
                    x=chart_data_copy["Over"],
                    y=chart_data_copy["Runs from 4s"],
                    name="Runs from 4s",
                    marker_color="#3B82F6",
                    text=chart_data_copy["Runs from 4s"],
                    textposition="auto"
                ))
                fig_c5.add_trace(go.Bar(
                    x=chart_data_copy["Over"],
                    y=chart_data_copy["Runs from 6s"],
                    name="Runs from 6s",
                    marker_color="#16C784",
                    text=chart_data_copy["Runs from 6s"],
                    textposition="auto"
                ))
                fig_c5.update_layout(
                    barmode="stack",
                    title=f"{sel_ow_batter} — Boundary Runs Breakdown (4s vs 6s) by Over",
                    xaxis_title="Over Number",
                    yaxis_title="Runs from Boundaries",
                    **layout_defs
                )
                fig_c5.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                st.plotly_chart(fig_c5, width="stretch")

            # Over-wise Batting Statistics Table (Detailed 13-Column Table)
            st.markdown("#### 📋 Over-wise Batting Statistics Table")
            
            table_to_show = df_ow_table[df_ow_table["Balls Faced"] > 0] if filter_active else df_ow_table

            st.dataframe(
                table_to_show,
                width="stretch",
                hide_index=True,
                column_config={
                    "Batter": st.column_config.TextColumn("Batter"),
                    "Over": st.column_config.NumberColumn("Over", format="%d"),
                    "Matches": st.column_config.NumberColumn("Matches", format="%d"),
                    "Runs": st.column_config.NumberColumn("Runs", format="%d"),
                    "Balls Faced": st.column_config.NumberColumn("Balls Faced", format="%d"),
                    "Strike Rate": st.column_config.NumberColumn("Strike Rate", format="%.2f"),
                    "Average": st.column_config.TextColumn("Average"),
                    "Fours": st.column_config.NumberColumn("4s", format="%d"),
                    "Sixes": st.column_config.NumberColumn("6s", format="%d"),
                    "Total Boundaries": st.column_config.NumberColumn("Total Boundaries", format="%d"),
                    "Boundary Runs": st.column_config.NumberColumn("Boundary Runs", format="%d"),
                    "Dot Balls": st.column_config.NumberColumn("Dot Balls", format="%d"),
                    "Balls Per Boundary": st.column_config.NumberColumn("Balls Per Boundary", format="%.2f")
                }
            )

            # Reconciled Summary Metrics
            tot_runs = int(df_ow_table["Runs"].sum())
            tot_balls = int(df_ow_table["Balls Faced"].sum())
            tot_4s = int(df_ow_table["Fours"].sum())
            tot_6s = int(df_ow_table["Sixes"].sum())
            tot_bnd = int(df_ow_table["Total Boundaries"].sum())
            tot_bnd_r = int(df_ow_table["Boundary Runs"].sum())
            tot_dots = int(df_ow_table["Dot Balls"].sum())
            tot_sr = round(tot_runs / tot_balls * 100.0, 2) if tot_balls > 0 else 0.0

            m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
            m_c1.metric("Total Runs", f"{tot_runs}")
            m_c2.metric("Balls Faced", f"{tot_balls}")
            m_c3.metric("4s / 6s", f"{tot_4s} / {tot_6s}")
            m_c4.metric("Total Boundaries", f"{tot_bnd}")
            m_c5.metric("Boundary Runs", f"{tot_bnd_r}")
            m_c6.metric("Overall SR", f"{tot_sr}")

            # CSV Download for Over-wise Table
            csv_ow = table_to_show.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"📥 Download {sel_ow_batter} Over-wise Data (CSV)",
                data=csv_ow,
                file_name=f"{sel_ow_batter.lower().replace(' ', '_')}_overwise_statistics.csv",
                mime="text/csv",
                key="btn_dl_ow_csv"
            )

            # Multi-Player Over-wise Comparison
            st.markdown("---")
            st.markdown("#### 👥 Multi-Player Over-wise Comparison")
            st.caption(
                "Compare over-by-over dynamics across multiple qualifying batters simultaneously from active filtered results. "
                "Each batter is rendered as an independent series in charts and separate rows in the comparison table."
            )

            comp_ctrl1, comp_ctrl2 = st.columns([3, 1])
            with comp_ctrl2:
                st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                sel_all_qualifiers = st.checkbox("Select all qualifying batters", value=False, key="chk_sel_all_qual")

            default_comp_list = [sel_ow_batter]
            other_batters = [b for b in qualifying_batters if b != sel_ow_batter]
            if other_batters:
                default_comp_list.append(other_batters[0])
            if len(other_batters) > 1:
                default_comp_list.append(other_batters[1])

            with comp_ctrl1:
                sel_comp_players = st.multiselect(
                    "Select Qualifying Batters to Compare:",
                    options=qualifying_batters,
                    default=qualifying_batters if sel_all_qualifiers else default_comp_list[:4],
                    key="ms_comp_qual_batters"
                )

            if len(sel_comp_players) < 2:
                st.info("ℹ️ Select at least 2 qualifying batters to display comparative multi-series charts and tables.")
            else:
                comp_ids = {}
                for p_nm in sel_comp_players:
                    p_match = df_adv_bat[df_adv_bat["striker_name"] == p_nm]
                    if not p_match.empty:
                        comp_ids[p_nm] = int(p_match.iloc[0]["striker_id"])

                df_comparison = get_multi_batter_overwise_comparison(
                    filtered_bbb,
                    comp_ids,
                    only_active_overs=filter_active
                )

                if df_comparison.empty:
                    st.warning("No delivery data available for the selected comparison group.")
                else:
                    # Comparison Visualizations (6 Comparative Charts)
                    cmp_tab_runs, cmp_tab_4s, cmp_tab_6s, cmp_tab_sr, cmp_tab_bnd, cmp_tab_bnd_r = st.tabs([
                        "🏏 Runs by Over",
                        "4️⃣ Fours by Over",
                        "6️⃣ Sixes by Over",
                        "⚡ Strike Rate by Over",
                        "🎯 Total Boundaries by Over",
                        "🔥 Boundary Runs by Over"
                    ])

                    with cmp_tab_runs:
                        fig_cmp_r = px.line(
                            df_comparison,
                            x="Over",
                            y="Runs",
                            color="Batter",
                            markers=True,
                            title="Batter Comparison — Runs by Over (Overs 1 to 20)"
                        )
                        fig_cmp_r.update_layout(**layout_defs)
                        fig_cmp_r.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                        st.plotly_chart(fig_cmp_r, width="stretch")

                    with cmp_tab_4s:
                        fig_cmp_4 = px.line(
                            df_comparison,
                            x="Over",
                            y="Fours",
                            color="Batter",
                            markers=True,
                            title="Batter Comparison — Fours (4s) Hit by Over"
                        )
                        fig_cmp_4.update_layout(**layout_defs)
                        fig_cmp_4.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                        st.plotly_chart(fig_cmp_4, width="stretch")

                    with cmp_tab_6s:
                        fig_cmp_6 = px.line(
                            df_comparison,
                            x="Over",
                            y="Sixes",
                            color="Batter",
                            markers=True,
                            title="Batter Comparison — Sixes (6s) Hit by Over"
                        )
                        fig_cmp_6.update_layout(**layout_defs)
                        fig_cmp_6.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                        st.plotly_chart(fig_cmp_6, width="stretch")

                    with cmp_tab_sr:
                        fig_cmp_s = px.line(
                            df_comparison[df_comparison["Balls Faced"] > 0],
                            x="Over",
                            y="Strike Rate",
                            color="Batter",
                            markers=True,
                            title="Batter Comparison — Strike Rate Trajectory by Over"
                        )
                        fig_cmp_s.update_layout(**layout_defs)
                        fig_cmp_s.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                        st.plotly_chart(fig_cmp_s, width="stretch")

                    with cmp_tab_bnd:
                        fig_cmp_b = px.line(
                            df_comparison,
                            x="Over",
                            y="Total Boundaries",
                            color="Batter",
                            markers=True,
                            title="Batter Comparison — Total Boundaries Hit by Over"
                        )
                        fig_cmp_b.update_layout(**layout_defs)
                        fig_cmp_b.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                        st.plotly_chart(fig_cmp_b, width="stretch")

                    with cmp_tab_bnd_r:
                        fig_cmp_br = px.line(
                            df_comparison,
                            x="Over",
                            y="Boundary Runs",
                            color="Batter",
                            markers=True,
                            title="Batter Comparison — Boundary Runs Scored by Over"
                        )
                        fig_cmp_br.update_layout(**layout_defs)
                        fig_cmp_br.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                        st.plotly_chart(fig_cmp_br, width="stretch")

                    # Multi-Batter Comparison Table
                    st.markdown("##### 📋 Multi-Player Over-wise Comparison Table")
                    st.dataframe(
                        df_comparison,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Batter": st.column_config.TextColumn("Batter"),
                            "Over": st.column_config.NumberColumn("Over", format="%d"),
                            "Matches": st.column_config.NumberColumn("Matches", format="%d"),
                            "Runs": st.column_config.NumberColumn("Runs", format="%d"),
                            "Balls Faced": st.column_config.NumberColumn("Balls Faced", format="%d"),
                            "Strike Rate": st.column_config.NumberColumn("Strike Rate", format="%.2f"),
                            "Average": st.column_config.TextColumn("Average"),
                            "Fours": st.column_config.NumberColumn("4s", format="%d"),
                            "Sixes": st.column_config.NumberColumn("6s", format="%d"),
                            "Total Boundaries": st.column_config.NumberColumn("Total Boundaries", format="%d"),
                            "Boundary Runs": st.column_config.NumberColumn("Boundary Runs", format="%d"),
                            "Dot Balls": st.column_config.NumberColumn("Dot Balls", format="%d"),
                            "Balls Per Boundary": st.column_config.NumberColumn("Balls Per Boundary", format="%.2f")
                        }
                    )

                    csv_comp = df_comparison.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Multi-Batter Over-wise Comparison (CSV)",
                        data=csv_comp,
                        file_name="wicket_iq_multi_batter_overwise_comparison.csv",
                        mime="text/csv",
                        key="btn_dl_comp_ow_csv"
                    )


# =========================================================================
# TAB 2: ADVANCED BOWLING ANALYSIS
# =========================================================================
with tabs[1]:
    st.markdown("### 🎯 Advanced Bowling Performance & Economy Containment")
    st.caption("Filter bowlers dynamically by economy rate, bowling discipline, wickets, phase, and style.")

    bw_col1, bw_col2, bw_col3, bw_col4 = st.columns(4)
    with bw_col1:
        min_ov_bowl = st.number_input("Minimum Overs Bowled", min_value=2.0, max_value=500.0, value=10.0, step=2.0, key="min_ov_bowl")
    with bw_col2:
        bw_pace_spin = st.selectbox("Bowling Type", ["All", "Pace", "Spin"], index=0, key="bw_pace_spin")
    with bw_col3:
        bw_phase = st.selectbox("Innings Phase", ["All Overs", "Powerplay", "Middle", "Death"], index=0, key="bw_phase")
    with bw_col4:
        bw_econ_mode = st.selectbox("Economy Filter", ["All", "Below", "Above", "Between"], index=0, key="bw_econ_mode")

    # Dynamic Economy Values
    ec_v1, ec_v2 = 0.0, 15.0
    if bw_econ_mode in ["Below", "Above"]:
        ec_v1 = st.slider("Economy Rate Threshold", min_value=4.0, max_value=12.0, value=7.0 if bw_econ_mode == "Below" else 9.5, step=0.25)
    elif bw_econ_mode == "Between":
        ec_range = st.slider("Economy Range", min_value=4.0, max_value=12.0, value=(6.0, 8.0), step=0.25)
        ec_v1, ec_v2 = ec_range[0], ec_range[1]

    # Calculate Bowling Data
    df_adv_bowl = calculate_advanced_bowling_stats(
        bbb=filtered_bbb,
        min_overs_bowled=float(min_ov_bowl),
        econ_mode=bw_econ_mode,
        econ_val1=ec_v1,
        econ_val2=ec_v2,
        bowling_type=bw_pace_spin,
        phase=bw_phase
    )

    if df_adv_bowl.empty:
        st.warning("No bowlers matched the specified economy and opportunity thresholds.")
    else:
        st.markdown(f"**Found {len(df_adv_bowl)} qualifying bowlers**")

        disp_bw_cols = [
            "bowler_name", "matches", "overs", "legal_balls", "conceded_runs",
            "wickets", "economy", "dot_pct", "boundary_pct", "bowling_sr", "bowling_avg", "bowling_style"
        ]
        disp_bowl = df_adv_bowl[disp_bw_cols].rename(columns={
            "bowler_name": "Bowler",
            "matches": "Matches",
            "overs": "Overs",
            "legal_balls": "Legal Balls",
            "conceded_runs": "Runs Conceded",
            "wickets": "Wickets",
            "economy": "Economy",
            "dot_pct": "Dot %",
            "boundary_pct": "Boundary Conceded %",
            "bowling_sr": "Bowling SR",
            "bowling_avg": "Bowling Avg",
            "bowling_style": "Style"
        })

        st.dataframe(disp_bowl, width="stretch", hide_index=True)

        # Plotly Chart: Economy vs Wickets
        st.markdown("#### 📈 Bowling Containment vs Wicket-Taking Matrix")
        top_plot_bw = df_adv_bowl.head(25)
        fig_bowl_scatter = px.scatter(
            top_plot_bw,
            x="economy",
            y="wickets",
            size="overs",
            color="dot_pct",
            hover_name="bowler_name",
            color_continuous_scale=["#151F30", "#3B82F6", "#16C784"],
            labels={
                "economy": "Economy Rate (Lower = More Economical)",
                "wickets": "Wickets Taken",
                "dot_pct": "Dot Ball %"
            },
            title="Bowling Dominance Matrix (Top 25 Wicket Takers)"
        )
        layout_defs = get_plotly_layout_defaults()
        fig_bowl_scatter.update_layout(**layout_defs)
        st.plotly_chart(fig_bowl_scatter, width="stretch")

        # CSV Download
        csv_bw = disp_bowl.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Bowling Analysis Report (CSV)",
            data=csv_bw,
            file_name="wicket_iq_advanced_bowling_analysis.csv",
            mime="text/csv"
        )

# =========================================================================
# TAB 3: PACE VS SPIN BATTING ANALYSIS
# =========================================================================
with tabs[2]:
    st.markdown("### ⚡ Pace vs Spin Batting Dynamics")
    st.caption("Inspect batter performance disaggregated across Fast/Pace and Spin bowling with comparative dominance filters.")

    ps_sub1, ps_sub2 = st.tabs(["Individual Batter Breakdown", "League-Wide Pace vs Spin Leaderboard"])

    with ps_sub1:
        st.markdown("#### 🔍 Select Batter for Comprehensive Pace vs Spin Breakdown:")
        sample_bat_list = sorted(list(pm["player_name"].dropna().unique()))
        sel_pvs_batter = st.selectbox("Select Batter", sample_bat_list, index=0, key="pvs_bat_sel")

        b_row = pm[pm["player_name"] == sel_pvs_batter]
        if not b_row.empty:
            b_id = int(b_row.iloc[0]["player_id"])
            pvs_table = get_pace_vs_spin_batter_breakdown(filtered_bbb, b_id)

            if pvs_table.empty:
                st.info("No recorded deliveries for this batter under active filters.")
            else:
                st.dataframe(pvs_table, width="stretch", hide_index=True)

                # Visual comparison chart
                fig_pvs = px.bar(
                    pvs_table[pvs_table["Bowling Type"] != "Overall"],
                    x="Bowling Type",
                    y="Strike Rate",
                    text="Strike Rate",
                    color="Bowling Type",
                    color_discrete_map={"Fast / Pace": "#3B82F6", "Spin": "#16C784"},
                    title=f"{sel_pvs_batter} — Strike Rate Comparison vs Pace & Spin"
                )
                fig_pvs.update_layout(**get_plotly_layout_defaults())
                st.plotly_chart(fig_pvs, width="stretch")

    with ps_sub2:
        st.markdown("#### 🏆 Dominance Filters (Pace Specialists vs Spin Specialists)")
        c_filter = st.selectbox(
            "Filter Dominance Trait:",
            [
                "All Qualified Batters",
                "Higher SR vs Pace",
                "Higher SR vs Spin",
                "More Frequent Boundaries vs Pace",
                "More Frequent Boundaries vs Spin",
                "Lower Dismissal Frequency vs Pace",
                "Lower Dismissal Frequency vs Spin"
            ],
            index=0
        )
        min_balls_each = st.slider("Minimum Balls Faced Against Each Bowling Type", min_value=15, max_value=150, value=30, step=5)

        comp_pvs_df = get_pace_spin_comparative_table(
            bbb=filtered_bbb,
            min_balls_each=min_balls_each,
            filter_criteria=c_filter if c_filter != "All Qualified Batters" else None
        )

        if comp_pvs_df.empty:
            st.info("No batters matched the specified criteria.")
        else:
            st.markdown(f"**Found {len(comp_pvs_df)} qualifying batters**")
            pvs_disp = comp_pvs_df[[
                "striker_name", "pace_runs", "pace_balls", "pace_sr", "pace_bpb",
                "spin_runs", "spin_balls", "spin_sr", "spin_bpb"
            ]].rename(columns={
                "striker_name": "Batter",
                "pace_runs": "Pace Runs",
                "pace_balls": "Pace Balls",
                "pace_sr": "Pace SR",
                "pace_bpb": "Pace Balls/Bnd",
                "spin_runs": "Spin Runs",
                "spin_balls": "Spin Balls",
                "spin_sr": "Spin SR",
                "spin_bpb": "Spin Balls/Bnd"
            })
            st.dataframe(pvs_disp, width="stretch", hide_index=True)

# =========================================================================
# TAB 4: OVER-WISE DYNAMICS
# =========================================================================
with tabs[3]:
    st.markdown("### 📈 Over-by-Over Progression (Overs 1 through 20)")
    st.caption("Analyze how batting strike rates and bowling economy rates evolve across every over number.")

    ov_tab1, ov_tab2, ov_tab3 = st.tabs([
        "Batting Over-wise Analysis",
        "Bowling Over-wise Analysis",
        "Phase-wise & Single Over Leaderboards"
    ])

    with ov_tab1:
        st.markdown("#### 🏏 Over-by-Over Batting Breakdown")
        sel_ov_bat = st.selectbox("Select Batter", sorted(list(pm["player_name"].dropna().unique())), index=0, key="ov_bat_sel")
        b_match = pm[pm["player_name"] == sel_ov_bat]
        if not b_match.empty:
            b_pid = int(b_match.iloc[0]["player_id"])
            df_b_ov = get_batter_overwise_analysis(filtered_bbb, b_pid)

            if df_b_ov.empty:
                st.info("No delivery records for this batter under active filters.")
            else:
                st.dataframe(df_b_ov, width="stretch", hide_index=True)

                # Line chart of Strike Rate across overs
                fig_ov_bat = px.line(
                    df_b_ov[df_b_ov["Balls Faced"] > 0],
                    x="Over",
                    y="Strike Rate",
                    markers=True,
                    title=f"{sel_ov_bat} — Strike Rate Evolution by Over Number",
                    color_discrete_sequence=["#16C784"]
                )
                fig_ov_bat.update_layout(**get_plotly_layout_defaults())
                fig_ov_bat.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                st.plotly_chart(fig_ov_bat, width="stretch")

    with ov_tab2:
        st.markdown("#### 🎯 Over-by-Over Bowling Economy Breakdown")
        sel_ov_bowl = st.selectbox("Select Bowler", sorted(list(pm["player_name"].dropna().unique())), index=0, key="ov_bowl_sel")
        bw_match = pm[pm["player_name"] == sel_ov_bowl]
        if not bw_match.empty:
            bw_pid = int(bw_match.iloc[0]["player_id"])
            df_bw_ov = get_bowler_overwise_analysis(filtered_bbb, bw_pid)

            if df_bw_ov.empty:
                st.info("No delivery records for this bowler under active filters.")
            else:
                st.dataframe(df_bw_ov, width="stretch", hide_index=True)

                # Bar chart of Economy by Over
                fig_ov_bw = px.bar(
                    df_bw_ov[df_bw_ov["Legal Balls"] > 0],
                    x="Over",
                    y="Economy",
                    text="Economy",
                    color="Phase",
                    color_discrete_map={"Powerplay": "#3B82F6", "Middle Overs": "#F59E0B", "Death Overs": "#EF4444"},
                    title=f"{sel_ov_bowl} — Economy Rate Across Overs 1 to 20"
                )
                fig_ov_bw.update_layout(**get_plotly_layout_defaults())
                fig_ov_bw.update_xaxes(tickmode="linear", tick0=1, dtick=1)
                st.plotly_chart(fig_ov_bw, width="stretch")

    with ov_tab3:
        st.markdown("#### 🏆 Phase-wise & Single Over Leaderboards")
        c_mode = st.radio("Rank By:", ["Innings Phase (Powerplay / Middle / Death)", "Specific Single Over Number (e.g. Over 20)"], horizontal=True)

        if "Innings Phase" in c_mode:
            phase_choice = st.selectbox("Select Phase", ["Powerplay", "Middle Overs", "Death Overs"], index=2)
            min_balls_ph = st.slider("Minimum Legal Balls in Phase", min_value=12, max_value=240, value=36, step=6)
            ph_ranks = get_phase_wise_bowler_rankings(filtered_bbb, phase_choice, min_balls_ph)
            if ph_ranks.empty:
                st.info("No bowlers qualified under selected filters.")
            else:
                st.dataframe(ph_ranks.rename(columns={
                    "bowler_name": "Bowler",
                    "matches": "Matches",
                    "overs": "Overs",
                    "conceded": "Runs Conceded",
                    "wickets": "Wickets",
                    "economy": "Economy",
                    "dot_pct": "Dot %",
                    "bowling_sr": "Bowling SR"
                }), width="stretch", hide_index=True)
        else:
            single_ov = st.slider("Select Single Over Number", min_value=1, max_value=20, value=20, step=1)
            min_balls_ov = st.slider("Minimum Legal Balls in This Over", min_value=6, max_value=120, value=18, step=6)
            ov_ranks = get_over_specific_bowler_rankings(filtered_bbb, single_ov, min_balls_ov)
            if ov_ranks.empty:
                st.info("No bowlers qualified under selected filters.")
            else:
                st.dataframe(ov_ranks.rename(columns={
                    "bowler_name": "Bowler",
                    "matches": "Matches",
                    "overs": "Overs",
                    "conceded": "Runs Conceded",
                    "wickets": "Wickets",
                    "economy": "Economy",
                    "dot_pct": "Dot %"
                }), width="stretch", hide_index=True)

# =========================================================================
# TAB 5: YORKER & DELIVERY STYLE ANALYSIS
# =========================================================================
with tabs[4]:
    st.markdown("### 🎯 Yorker Accuracy & Delivery Style Classifications")
    st.caption("Verified execution analysis for yorkers, wide yorkers, bouncers, full tosses, and movement trajectory.")

    # Data Transparency Notice
    disc = get_data_transparency_disclosure()
    with st.expander("ℹ️ Data Integrity & Hawk-Eye Telemetry Status Disclosure", expanded=False):
        st.markdown(f"""
        - **Verified Telemetry in Dataset:** {disc['verified_features']}
        - **Hawk-Eye Coordinate Notice:** {disc['limitations']}
        """)

    yk_sub1, yk_sub2 = st.tabs(["Yorker Leaderboard", "Individual Bowler Delivery Arsenal"])

    with yk_sub1:
        st.markdown("#### 🏆 Top Yorker Specialists")
        yk_c1, yk_c2, yk_c3 = st.columns(3)
        with yk_c1:
            min_tot_b = st.number_input("Minimum Total Career Balls", min_value=30, max_value=2000, value=60, step=30)
        with yk_c2:
            min_yk_b = st.number_input("Minimum Yorkers Bowled", min_value=3, max_value=300, value=5, step=2)
        with yk_c3:
            max_yk_ec = st.number_input("Max Yorker Economy", min_value=3.0, max_value=15.0, value=10.0, step=0.5)

        df_yorkers = calculate_yorker_performance_stats(
            bbb=filtered_bbb,
            min_total_balls=int(min_tot_b),
            min_yorkers=int(min_yk_b),
            max_yorker_economy=float(max_yk_ec)
        )

        if df_yorkers.empty:
            st.info("No bowlers matched the yorker threshold criteria.")
        else:
            disp_yk = df_yorkers[[
                "bowler_name", "yorker_balls", "total_balls", "yorker_pct",
                "yorker_runs", "yorker_wickets", "yorker_economy", "yorker_dot_pct", "yorker_boundary_pct"
            ]].rename(columns={
                "bowler_name": "Bowler",
                "yorker_balls": "Yorkers Bowled",
                "total_balls": "Total Deliveries",
                "yorker_pct": "Yorker %",
                "yorker_runs": "Runs Conceded",
                "yorker_wickets": "Wickets",
                "yorker_economy": "Economy on Yorkers",
                "yorker_dot_pct": "Dot % on Yorkers",
                "yorker_boundary_pct": "Boundary % on Yorkers"
            })
            st.dataframe(disp_yk, width="stretch", hide_index=True)

            # Top yorker bar chart
            fig_yk = px.bar(
                df_yorkers.head(15),
                x="yorker_balls",
                y="bowler_name",
                orientation="h",
                text="yorker_balls",
                color="yorker_economy",
                color_continuous_scale=["#16C784", "#3B82F6", "#EF4444"],
                labels={"yorker_balls": "Total Verified Yorkers", "bowler_name": "Bowler", "yorker_economy": "Economy on Yorkers"},
                title="Top 15 Bowlers by Total Verified Yorkers Bowled"
            )
            fig_yk.update_layout(**get_plotly_layout_defaults())
            fig_yk.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_yk, width="stretch")

    with yk_sub2:
        st.markdown("#### 🔬 Bowler Delivery Style Breakdown")
        sel_arsenal_bw = st.selectbox("Select Bowler to Inspect Arsenal:", sorted(list(pm["player_name"].dropna().unique())), index=0, key="arsenal_bw_sel")
        bw_rec = pm[pm["player_name"] == sel_arsenal_bw]
        if not bw_rec.empty:
            bw_id = int(bw_rec.iloc[0]["player_id"])
            df_styles = get_bowler_delivery_style_distribution(filtered_bbb, bw_id)
            if df_styles.empty:
                st.info("No delivery classifications available for this bowler under active filters.")
            else:
                st.dataframe(df_styles, width="stretch", hide_index=True)

                # Pie/Donut Chart of Delivery Styles
                fig_donut = px.pie(
                    df_styles,
                    names="Delivery Classification",
                    values="Deliveries",
                    hole=0.45,
                    title=f"{sel_arsenal_bw} — Delivery Arsenal Distribution"
                )
                fig_donut.update_layout(**get_plotly_layout_defaults())
                st.plotly_chart(fig_donut, width="stretch")

# =========================================================================
# TAB 6: HANDEDNESS DYNAMICS
# =========================================================================
with tabs[5]:
    st.markdown("### 🔄 Handedness Performance Dynamics")
    st.caption("Measure bowler effectiveness against Left-Handed vs Right-Handed batters, and batter mastery against Left-Arm vs Right-Arm bowling.")

    hand_sub1, hand_sub2, hand_sub3 = st.tabs([
        "Bowler vs LHB / RHB Splits",
        "Leaderboard Against LHB / RHB",
        "Batter vs Bowling Arm Splits"
    ])

    with hand_sub1:
        st.markdown("#### 🎯 Bowler Effectiveness: Against LHB vs RHB")
        sel_hand_bw = st.selectbox("Select Bowler", sorted(list(pm["player_name"].dropna().unique())), index=0, key="hand_bw_sel")
        bw_row = pm[pm["player_name"] == sel_hand_bw]
        if not bw_row.empty:
            bw_id = int(bw_row.iloc[0]["player_id"])
            df_hand_split = get_bowler_vs_handedness_stats(filtered_bbb, bw_id)
            if df_hand_split.empty:
                st.info("No delivery records for this bowler.")
            else:
                st.dataframe(df_hand_split, width="stretch", hide_index=True)

    with hand_sub2:
        st.markdown("#### 🏆 Bowler Rankings Against Specific Batter Handedness")
        c_hand = st.radio("Target Batter Handedness:", ["Left-Handed Batters (LHB)", "Right-Handed Batters (RHB)"], horizontal=True)
        hand_target = "LHB" if "Left" in c_hand else "RHB"
        hand_sort_mode = st.selectbox("Rank By:", ["Best Economy", "Most Wickets"], index=0)
        min_balls_h = st.slider("Minimum Legal Balls Bowled Against Handedness", min_value=18, max_value=240, value=36, step=6)

        df_all_hand = get_all_bowlers_handedness_rankings(filtered_bbb, hand_target, min_balls_h, hand_sort_mode)
        if df_all_hand.empty:
            st.info("No bowlers qualified under selected criteria.")
        else:
            st.dataframe(df_all_hand.rename(columns={
                "bowler_name": "Bowler",
                "matches": "Matches",
                "overs": "Overs",
                "conceded": "Runs Conceded",
                "wickets": "Wickets",
                "economy": "Economy",
                "dot_pct": "Dot %",
                "boundary_pct": "Boundary %",
                "bowling_sr": "Bowling SR"
            }), width="stretch", hide_index=True)

    with hand_sub3:
        st.markdown("#### 🏏 Batter Performance vs Bowling Arm (Right-Arm vs Left-Arm)")
        sel_arm_bat = st.selectbox("Select Batter", sorted(list(pm["player_name"].dropna().unique())), index=0, key="arm_bat_sel")
        bat_row = pm[pm["player_name"] == sel_arm_bat]
        if not bat_row.empty:
            bat_id = int(bat_row.iloc[0]["player_id"])
            df_arm_split = get_batter_vs_bowling_arm_stats(filtered_bbb, bat_id)
            if df_arm_split.empty:
                st.info("No delivery records for this batter.")
            else:
                st.dataframe(df_arm_split, width="stretch", hide_index=True)

# =========================================================================
# TAB 7: MULTI-PLAYER COMPARISON ENGINE
# =========================================================================
with tabs[6]:
    st.markdown("### 👥 Multi-Player Head-to-Head Comparison")
    st.caption("Select 2 or more batters or bowlers to compare key performance metrics side by side.")

    comp_role = st.radio("Comparison Category:", ["Batters", "Bowlers"], horizontal=True)

    if comp_role == "Batters":
        bat_pool = sorted(list(pm["player_name"].dropna().unique()))
        selected_batters = st.multiselect(
            "Select Batters to Compare (2 or more):",
            bat_pool,
            default=bat_pool[:2] if len(bat_pool) >= 2 else []
        )

        if len(selected_batters) < 2:
            st.info("Please select at least 2 batters to generate comparative metrics.")
        else:
            comp_rows = []
            for b_name in selected_batters:
                m_row = pm[pm["player_name"] == b_name]
                if m_row.empty:
                    continue
                pid = int(m_row.iloc[0]["player_id"])
                b_df = filtered_bbb[filtered_bbb["striker_id"] == pid]
                runs = int(b_df["batter_runs"].sum())
                balls = int(b_df["is_legal_ball_batter"].sum())
                fours = int(b_df["is_four"].sum())
                sixes = int(b_df["is_six"].sum())
                dots = int(b_df["is_dot_ball_batter"].sum())
                outs = int(((b_df["dismissed_player_id"] == pid) & (b_df["is_wicket"] == 1)).sum())
                sr = round((runs / balls * 100.0) if balls > 0 else 0.0, 2)
                avg = round((runs / outs) if outs > 0 else float(runs), 2)
                boundaries = fours + sixes
                bpb = round((balls / boundaries) if boundaries > 0 else 0.0, 2)
                dot_pct = round((dots / balls * 100.0) if balls > 0 else 0.0, 1)

                # Pace vs Spin SR
                pace_sub = b_df[b_df["pace_or_spin"] == "Pace"]
                spin_sub = b_df[b_df["pace_or_spin"] == "Spin"]
                p_sr = round((pace_sub["batter_runs"].sum() / pace_sub["is_legal_ball_batter"].sum() * 100.0) if pace_sub["is_legal_ball_batter"].sum() > 0 else 0.0, 2)
                s_sr = round((spin_sub["batter_runs"].sum() / spin_sub["is_legal_ball_batter"].sum() * 100.0) if spin_sub["is_legal_ball_batter"].sum() > 0 else 0.0, 2)

                comp_rows.append({
                    "Batter": b_name,
                    "Runs": runs,
                    "Balls": balls,
                    "Strike Rate": sr,
                    "Average": avg,
                    "Balls/Boundary": bpb if boundaries > 0 else np.nan,
                    "Dot %": f"{dot_pct}%",
                    "Pace SR": p_sr,
                    "Spin SR": s_sr,
                    "Fours": fours,
                    "Sixes": sixes
                })

            df_comp_b = pd.DataFrame(comp_rows)
            st.dataframe(df_comp_b, width="stretch", hide_index=True)

            # Comparative Bar Chart
            fig_comp_b = px.bar(
                df_comp_b,
                x="Batter",
                y="Strike Rate",
                color="Batter",
                text="Strike Rate",
                title="Batting Strike Rate Comparison"
            )
            fig_comp_b.update_layout(**get_plotly_layout_defaults())
            st.plotly_chart(fig_comp_b, width="stretch")

    else:
        bowl_pool = sorted(list(pm["player_name"].dropna().unique()))
        selected_bowlers = st.multiselect(
            "Select Bowlers to Compare (2 or more):",
            bowl_pool,
            default=bowl_pool[:2] if len(bowl_pool) >= 2 else []
        )

        if len(selected_bowlers) < 2:
            st.info("Please select at least 2 bowlers to generate comparative metrics.")
        else:
            comp_bw_rows = []
            for bw_name in selected_bowlers:
                m_row = pm[pm["player_name"] == bw_name]
                if m_row.empty:
                    continue
                bid = int(m_row.iloc[0]["player_id"])
                bw_df = filtered_bbb[filtered_bbb["bowler_id"] == bid]
                legal_balls = int(bw_df["is_legal_ball_bowler"].sum())
                conceded = int(bw_df["bowler_conceded_runs"].sum())
                wickets = int(bw_df["is_bowler_wicket"].sum())
                dots = int(bw_df["is_dot_ball_bowler"].sum())
                econ = round((conceded / legal_balls * 6.0) if legal_balls > 0 else 0.0, 2)
                dot_pct = round((dots / legal_balls * 100.0) if legal_balls > 0 else 0.0, 1)

                # Yorker %
                yorkers = int(bw_df[bw_df["delivery_style"].isin(["Yorker", "Wide Yorker"])]["is_legal_ball_bowler"].sum())
                yorker_pct = round((yorkers / legal_balls * 100.0) if legal_balls > 0 else 0.0, 1)

                # Phase Economy
                death_sub = bw_df[bw_df["phase"] == "Death"]
                death_econ = round((death_sub["bowler_conceded_runs"].sum() / death_sub["is_legal_ball_bowler"].sum() * 6.0) if death_sub["is_legal_ball_bowler"].sum() > 0 else 0.0, 2)

                comp_bw_rows.append({
                    "Bowler": bw_name,
                    "Overs": round(legal_balls / 6.0, 1),
                    "Wickets": wickets,
                    "Overall Economy": econ,
                    "Death Economy": death_econ if death_sub["is_legal_ball_bowler"].sum() > 0 else np.nan,
                    "Dot %": f"{dot_pct}%",
                    "Yorker %": f"{yorker_pct}%",
                    "Yorkers Bowled": yorkers
                })

            df_comp_bw = pd.DataFrame(comp_bw_rows)
            st.dataframe(df_comp_bw, width="stretch", hide_index=True)

            # Comparative Bar Chart
            fig_comp_bw = px.bar(
                df_comp_bw,
                x="Bowler",
                y="Overall Economy",
                color="Bowler",
                text="Overall Economy",
                title="Bowling Economy Rate Comparison (Lower = Better)"
            )
            fig_comp_bw.update_layout(**get_plotly_layout_defaults())
            st.plotly_chart(fig_comp_bw, width="stretch")
