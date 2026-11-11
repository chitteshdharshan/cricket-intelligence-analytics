"""
AI CRICKET STRATEGY ASSISTANT & DECISION INTELLIGENCE
Comprehensive tactical decision-support system featuring:
1. Strategy Chatbot
2. Match Situation Analyzer
3. Next Over Bowler Recommendation
4. Best Batter Recommendation
5. Partnership Analyzer
6. Player Matchup Analyzer
7. AI Squad & Playing XI Builder
8. Target Chase and Defence Planner
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
from src.ui_components import (
    apply_custom_theme,
    render_banner,
    render_sidebar_branding,
    render_kpi,
    render_section_header,
    get_plotly_layout_defaults
)
from src.player_recommendation_engine import recommend_batters, recommend_bowlers
from src.partnership_analyzer import analyze_partnership
from src.bowler_recommendation_engine import recommend_next_over_bowler
from src.match_situation_analyzer import analyze_target_chase, analyze_target_defence
from src.team_squad_builder import build_playing_xi
from src.matchup_analysis import get_h2h_matchup_details
from src.ai_assistant import answer_strategy_query, get_ai_api_key

# Page Configuration
st.set_page_config(
    page_title="AI Strategy Assistant | Cricket Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()
render_sidebar_branding(st.sidebar)

# Load Analytical Datasets
data = load_all_data()
bbb = data["bbb"]
matches = data["matches"]
teams = data["teams"]
comps = data["comps"]
players_master = data["players_master"]
lookups = get_lookup_maps(data)

# Header Banner
_, api_provider = get_ai_api_key()
engine_status_badge = f"Engine: {api_provider.upper()} API" if api_provider != "offline" else "Engine: Offline Analytics Model"

render_banner(
    "AI CRICKET STRATEGY ASSISTANT",
    "Grounded decision intelligence for match situations, bowler selections, squad formulation, and target planning",
    badge=engine_status_badge
)

# 8 Navigation Tabs
tabs = st.tabs([
    "💬 Strategy Chatbot",
    "⚡ Match Situation Analyzer",
    "🎯 Next Over Bowler",
    "🏏 Best Batter",
    "🤝 Partnership Analyzer",
    "🥊 Player Matchups",
    "🛡️ Squad & Playing XI",
    "📊 Chase & Defence Planner"
])

# -------------------------------------------------------------
# TAB 1: STRATEGY CHATBOT
# -------------------------------------------------------------
with tabs[0]:
    render_section_header("AI Cricket Strategy Chatbot", badge="Dual-Mode Natural Language Engine")
    st.markdown("""
        Ask tactical cricket strategy questions in natural language. Recommendations are grounded in 
        **200,999 verified deliveries** and structured across a comprehensive 7-part strategic briefing.
    """)

    # Suggested Prompts
    st.markdown("**Suggested Strategic Prompts:**")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    quick_query = None
    with col_p1:
        if st.button("Who is the best death-over finisher?", width="stretch"):
            quick_query = "Who is the best death-over finisher?"
    with col_p2:
        if st.button("Who should bowl the 16th over?", width="stretch"):
            quick_query = "Who should bowl the 16th over?"
    with col_p3:
        if st.button("Build the best Playing XI for CSK", width="stretch"):
            quick_query = "Build the best Playing XI for CSK"
    with col_p4:
        if st.button("How should we chase 190?", width="stretch"):
            quick_query = "How should we chase 190?"

    # Chat Session State
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "**Welcome to the Cricket Strategy Intelligence Center.**\n\n"
                    "I provide data-backed strategic recommendations for player selection, in-game bowling changes, "
                    "tactical chases, and squad composition. How can I assist your team today?"
                )
            }
        ]

    # Clear Chat Button
    c_btn, _ = st.columns([1.5, 8.5])
    with c_btn:
        if st.button("🗑️ Clear Conversation", width="stretch"):
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "Conversation cleared. Ready for your next strategic query."
                }
            ]
            st.rerun()

    # Display Chat Feed
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Process Input (either from quick prompt or chat input)
    user_input = st.chat_input("Enter your tactical cricket question (e.g. 'Who is the best batsman against spin?')...")
    active_prompt = quick_query or user_input

    if active_prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing historical deliveries and generating strategic briefing..."):
                response_payload = answer_strategy_query(active_prompt, data)
                content = response_payload["response_markdown"]
                st.markdown(content)
                st.caption(f"Evidence Source: {response_payload['engine_mode']}")
                st.session_state["chat_messages"].append({"role": "assistant", "content": content})


# -------------------------------------------------------------
# TAB 2: MATCH SITUATION ANALYZER
# -------------------------------------------------------------
with tabs[1]:
    render_section_header("Match Situation Tactical Blueprint", badge="Phase & Scenario Intelligence")

    s_c1, s_c2, s_c3 = st.columns(3)
    with s_c1:
        sel_phase = st.selectbox("Innings Phase", ["Powerplay (Overs 1-6)", "Middle Overs (Overs 7-15)", "Death Overs (Overs 16-20)"])
    with s_c2:
        sel_innings = st.selectbox("Innings Type", ["1st Innings (Setting Target)", "2nd Innings (Chasing Target)"])
    with s_c3:
        sel_state = st.selectbox("Match Dynamic", [
            "Normal Progression",
            "Batting Collapse / Early Wickets (3+ down)",
            "Established Batters at Crease",
            "High Required Run Rate (>10.0 rpo)",
            "Low Target Defense"
        ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Determine Tactical Blueprint
    bp_col1, bp_col2 = st.columns([1.5, 1.5])
    with bp_col1:
        st.markdown("### 📋 Recommended Batting Cadence")
        if "Collapse" in sel_state:
            st.warning("⚠️ **Batting Collapse Protocol**: Defer aerial shots. Work 1s and 2s to build a 30-ball partnership before re-accelerating.")
            b_recs = recommend_batters(bbb, players_master, situation="rebuilding", min_balls=30, top_n=3)
        elif "Death" in sel_phase or "High Required" in sel_state:
            st.success("⚡ **High-Impact Acceleration Protocol**: Target boundaries every 3 balls. Maximize death overs hitting.")
            b_recs = recommend_batters(bbb, players_master, situation="death", min_balls=25, top_n=3)
        elif "Powerplay" in sel_phase:
            st.info("🎯 **Powerplay Exploitation**: Capitalize on 2 boundary fielders. Target aerial shots over 30-yard circle.")
            b_recs = recommend_batters(bbb, players_master, situation="powerplay", min_balls=30, top_n=3)
        else:
            st.info("🔄 **Middle Overs Control**: Strike rotation + boundary per 8 balls. Attack favorable bowling matchups.")
            b_recs = recommend_batters(bbb, players_master, situation="middle", min_balls=35, top_n=3)

        if len(b_recs) > 0:
            st.dataframe(
                b_recs[["player_name", "runs", "balls", "strike_rate", "average", "boundary_pct"]].rename(columns={
                    "player_name": "Recommended Batter", "runs": "Runs", "balls": "Balls", "strike_rate": "SR", "average": "Avg", "boundary_pct": "Boundary %"
                }),
                width="stretch"
            )

    with bp_col2:
        st.markdown("### 🛡️ Recommended Bowling Strategy")
        if "Death" in sel_phase:
            st.error("🎯 **Death Overs Execution**: Pinpoint yorkers outside off-stump, slow bouncers into the pitch.")
            bw_recs = recommend_bowlers(bbb, players_master, situation="death", min_balls=24, top_n=3)
        elif "Powerplay" in sel_phase:
            st.info("🏏 **Powerplay Control**: Test hard lengths on 4th stump; seek early breakthroughs.")
            bw_recs = recommend_bowlers(bbb, players_master, situation="powerplay", min_balls=30, top_n=3)
        elif "Low Target" in sel_state:
            st.warning("🔒 **Run-Choke Protocol**: Pack ring fielders, eliminate singles, force false attacking strokes.")
            bw_recs = recommend_bowlers(bbb, players_master, situation="defending_low_total", min_balls=30, top_n=3)
        else:
            st.success("🌀 **Middle Overs Strangle**: Deploy spin in tandem to squeeze run rate and induce mistimed lofted shots.")
            bw_recs = recommend_bowlers(bbb, players_master, situation="middle", min_balls=30, top_n=3)

        if len(bw_recs) > 0:
            st.dataframe(
                bw_recs[["player_name", "overs", "runs", "wickets", "economy", "dot_pct"]].rename(columns={
                    "player_name": "Recommended Bowler", "overs": "Overs", "runs": "Runs", "wickets": "Wkts", "economy": "Econ", "dot_pct": "Dot %"
                }),
                width="stretch"
            )


# -------------------------------------------------------------
# TAB 3: NEXT OVER BOWLER RECOMMENDATION
# -------------------------------------------------------------
with tabs[2]:
    render_section_header("AI Next Over Bowler Recommendation", badge="Striker Matchup & Quota Optimizer")

    # Inputs
    nb_c1, nb_c2, nb_c3, nb_c4 = st.columns(4)

    # Player Choices
    all_players_dict = dict(zip(
        players_master["player_id"],
        players_master["player_name"] + " (" + players_master["batting_style"].astype(str).str[:3] + ")"
    ))

    # Bowlers Choices
    bowlers_pool = players_master[players_master["legal_balls"] >= 36]
    bowler_dict = dict(zip(
        bowlers_pool["player_id"],
        bowlers_pool["player_name"] + " (" + bowlers_pool["bowling_style"].astype(str) + ")"
    ))

    with nb_c1:
        # Default Suryakumar Yadav (1699)
        def_s = 1699 if 1699 in all_players_dict else list(all_players_dict.keys())[0]
        cur_striker = st.selectbox("Current Striker", list(all_players_dict.keys()), index=list(all_players_dict.keys()).index(def_s), format_func=lambda x: all_players_dict[x])
    with nb_c2:
        # Default Jos Buttler (813)
        def_ns = 813 if 813 in all_players_dict else list(all_players_dict.keys())[1]
        cur_non_striker = st.selectbox("Non-Striker", list(all_players_dict.keys()), index=list(all_players_dict.keys()).index(def_ns), format_func=lambda x: all_players_dict[x])
    with nb_c3:
        target_over = st.slider("Next Over to Bowl", min_value=1, max_value=20, value=16)
    with nb_c4:
        # Select Bowling Team to filter pool
        team_choices = sorted(teams["team_name"].unique().tolist())
        bowling_team_name = st.selectbox("Defending Team", team_choices, index=team_choices.index("Chennai Super Kings") if "Chennai Super Kings" in team_choices else 0)

    # Get bowling squad for this team
    b_tid = teams[teams["team_name"] == bowling_team_name]["team_id"].iloc[0]
    team_pids = data["player_map"][data["player_map"]["team_id"] == b_tid]["player_id"].unique()
    team_bowler_pool = [pid for pid in team_pids if pid in bowler_dict]
    if not team_bowler_pool:
        team_bowler_pool = list(bowler_dict.keys())[:10]

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Recommend Optimal Bowler for Next Over", width="stretch"):
        with st.spinner("Analyzing head-to-head match histories, phase economy, and tactical scores..."):
            rec_result = recommend_next_over_bowler(
                bbb=bbb,
                players_master=players_master,
                striker_id=cur_striker,
                non_striker_id=cur_non_striker,
                over_number=target_over,
                available_bowler_ids=team_bowler_pool
            )

            primary = rec_result["primary_recommendation"]
            if primary:
                st.markdown(f"### 🏆 Primary Recommendation: **{primary['bowler_name']}**")
                
                k_c1, k_c2, k_c3, k_c4, k_c5 = st.columns(5)
                with k_c1:
                    render_kpi("Tactical Index", f"{primary['tactical_score']}/100", "Composite suitablility", tone="primary")
                with k_c2:
                    render_kpi("Phase Economy", f"{primary['phase_economy']}", f"In {rec_result['phase']} overs", tone="secondary")
                with k_c3:
                    render_kpi("Phase Wickets", f"{primary['phase_wickets']}", f"Across {primary['phase_balls']} balls", tone="primary")
                with k_c4:
                    render_kpi("Striker H2H", f"{primary['striker_h2h']}", f"vs {rec_result['striker'][:12]}", tone="muted")
                with k_c5:
                    render_kpi("Non-Striker H2H", f"{primary['non_striker_h2h']}", f"vs {rec_result['non_striker'][:12]}", tone="muted")

                st.markdown("<br>", unsafe_allow_html=True)

                col_rat, col_plan = st.columns(2)
                with col_rat:
                    st.markdown("#### 🎯 Rationale for Selection")
                    for r in rec_result["rationale"]:
                        st.markdown(f"- {r}")
                with col_plan:
                    st.markdown("#### 📐 Recommended Tactical Execution Plan")
                    for p in rec_result["tactical_plan"]:
                        st.markdown(f"- {p}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 👥 Top 3 Ranked Bowling Candidates")
                st.dataframe(
                    rec_result["full_candidates_df"][[
                        "bowler_name", "category", "phase_economy", "phase_wickets",
                        "striker_h2h", "tactical_score"
                    ]].rename(columns={
                        "bowler_name": "Bowler", "category": "Type", "phase_economy": f"{rec_result['phase']} Econ",
                        "phase_wickets": "Wkts", "striker_h2h": "Striker H2H Duel", "tactical_score": "Tactical Score"
                    }),
                    width="stretch"
                )


# -------------------------------------------------------------
# TAB 4: BEST BATTER RECOMMENDATION
# -------------------------------------------------------------
with tabs[3]:
    render_section_header("AI Best Batter Recommendation Engine", badge="Role & Matchup Specialist Rankings")

    bb_c1, bb_c2, bb_c3 = st.columns([1.5, 1.5, 1])
    with bb_c1:
        situation_choice = st.selectbox(
            "Select Tactical Batting Requirement",
            [
                ("powerplay", "Top Opener / Powerplay Specialist (Overs 1-6)"),
                ("middle", "Top Middle-Order Anchor (Overs 7-15)"),
                ("death", "Top Finisher / Death Overs Specialist (Overs 16-20)"),
                ("against_spin", "Top Batter Against Spin Attacks"),
                ("against_pace", "Top Batter Against Express Pace"),
                ("boundary_hitters", "Highest Boundary Percentage Hitters"),
                ("rebuilding", "Best Rebuilding Batter (High Average / Low Out Freq)"),
                ("acceleration", "Rapid Acceleration Batter (Maximum Strike Rate)"),
                ("chase", "Master Chaser (2nd Innings Target Execution)")
            ],
            format_func=lambda x: x[1]
        )[0]
    with bb_c2:
        bb_team_filter = st.selectbox("Filter by Franchise/Team", ["All Teams"] + sorted(teams["team_name"].unique().tolist()))
    with bb_c3:
        bb_min_balls = st.slider("Minimum Balls Faced", min_value=15, max_value=150, value=40, step=5)

    # Team ID translation
    chosen_team_id = None
    if bb_team_filter != "All Teams":
        chosen_team_id = teams[teams["team_name"] == bb_team_filter]["team_id"].iloc[0]

    bat_rankings = recommend_batters(
        bbb=bbb,
        players_master=players_master,
        situation=situation_choice,
        team_id=chosen_team_id,
        min_balls=bb_min_balls,
        top_n=10
    )

    if len(bat_rankings) > 0:
        top_1 = bat_rankings.iloc[0]
        st.markdown(f"### 🥇 Top Ranked: **{top_1['player_name']}** (Rank Score: `{top_1['rank_score']:.1f}`)")
        
        c_k1, c_k2, c_k3, c_k4, c_k5 = st.columns(5)
        with c_k1:
            render_kpi("Strike Rate", f"{top_1['strike_rate']}", "Runs / 100 balls", tone="primary")
        with c_k2:
            render_kpi("Batting Average", f"{top_1['average']}", "Runs per out", tone="secondary")
        with c_k3:
            render_kpi("Boundary %", f"{top_1['boundary_pct']}%", f"{top_1['fours']}x4 | {top_1['sixes']}x6", tone="primary")
        with c_k4:
            render_kpi("Dot Ball %", f"{top_1['dot_pct']}%", "Unproductive balls", tone="muted")
        with c_k5:
            render_kpi("Sample Size", f"{top_1['balls']} balls", top_1["sample_flag"], tone="muted")

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            bat_rankings[[
                "player_name", "batting_style", "runs", "balls", "outs",
                "strike_rate", "average", "boundary_pct", "dot_pct", "sample_flag"
            ]].rename(columns={
                "player_name": "Batter", "batting_style": "Style", "runs": "Runs", "balls": "Balls",
                "outs": "Outs", "strike_rate": "Strike Rate", "average": "Average",
                "boundary_pct": "Boundary %", "dot_pct": "Dot %", "sample_flag": "Reliability"
            }),
            width="stretch"
        )
    else:
        st.info("No batters met the specified criteria. Try lowering the minimum balls filter.")


# -------------------------------------------------------------
# TAB 5: PARTNERSHIP ANALYZER
# -------------------------------------------------------------
with tabs[4]:
    render_section_header("Active Batting Partnership Analyzer", badge="LHB/RHB Dynamics & Counter-Tactics")

    pt_c1, pt_c2 = st.columns(2)
    with pt_c1:
        p1_choice = st.selectbox("Batter 1 (Striker)", list(all_players_dict.keys()), index=list(all_players_dict.keys()).index(1699), format_func=lambda x: all_players_dict[x], key="part_p1")
    with pt_c2:
        p2_choice = st.selectbox("Batter 2 (Non-Striker)", list(all_players_dict.keys()), index=list(all_players_dict.keys()).index(813), format_func=lambda x: all_players_dict[x], key="part_p2")

    part_analysis = analyze_partnership(bbb, players_master, p1_choice, p2_choice)
    b1_data = part_analysis["batter1"]
    b2_data = part_analysis["batter2"]

    st.markdown(f"### 🤝 Partnership Dynamics: **{part_analysis['combo_type']}**")
    st.info(f"**Tactical Role Assignment:** {part_analysis['role_strategy']}")

    col_side_a, col_side_b = st.columns(2)
    with col_side_a:
        st.markdown(f"#### 🏏 {b1_data['name']} (`{b1_data['batting_style']}`)")
        render_kpi("Overall Strike Rate", f"{b1_data['strike_rate']}", f"Boundary: {b1_data['boundary_pct']}%", tone="primary")
        render_kpi("Performance vs Pace", f"SR {b1_data['pace_sr']} | Avg {b1_data['pace_avg']}", f"Across {b1_data['pace_balls']} balls", tone="secondary")
        render_kpi("Performance vs Spin", f"SR {b1_data['spin_sr']} | Avg {b1_data['spin_avg']}", f"Across {b1_data['spin_balls']} balls", tone="muted")

    with col_side_b:
        st.markdown(f"#### 🏏 {b2_data['name']} (`{b2_data['batting_style']}`)")
        render_kpi("Overall Strike Rate", f"{b2_data['strike_rate']}", f"Boundary: {b2_data['boundary_pct']}%", tone="primary")
        render_kpi("Performance vs Pace", f"SR {b2_data['pace_sr']} | Avg {b2_data['pace_avg']}", f"Across {b2_data['pace_balls']} balls", tone="secondary")
        render_kpi("Performance vs Spin", f"SR {b2_data['spin_sr']} | Avg {b2_data['spin_avg']}", f"Across {b2_data['spin_balls']} balls", tone="muted")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Opposition Bowling Plan to Break the Partnership")
    for tactic in part_analysis["bowling_tactics"]:
        st.markdown(f"- {tactic}")


# -------------------------------------------------------------
# TAB 6: PLAYER MATCHUP ANALYZER
# -------------------------------------------------------------
with tabs[5]:
    render_section_header("Direct Player Matchup Duel Analyzer", badge="Historical Head-to-Head Records")

    mu_c1, mu_c2 = st.columns(2)
    with mu_c1:
        sel_mu_bat = st.selectbox("Select Batter", list(all_players_dict.keys()), index=list(all_players_dict.keys()).index(1699), format_func=lambda x: all_players_dict[x], key="mu_bat")
    with mu_c2:
        sel_mu_bowl = st.selectbox("Select Bowler", list(bowler_dict.keys()), index=list(bowler_dict.keys()).index(6464), format_func=lambda x: bowler_dict[x], key="mu_bowl")

    bat_name = players_master[players_master["player_id"] == sel_mu_bat]["player_name"].iloc[0]
    bowl_name = players_master[players_master["player_id"] == sel_mu_bowl]["player_name"].iloc[0]

    h2h = get_h2h_matchup_details(bbb, sel_mu_bat, sel_mu_bowl)

    if h2h["balls"] > 0:
        st.markdown(f"### ⚔️ Historical Duel: **{bat_name}** vs **{bowl_name}**")
        
        m_k1, m_k2, m_k3, m_k4, m_k5 = st.columns(5)
        with m_k1:
            render_kpi("Balls Faced", f"{h2h['balls']}", "Legal deliveries", tone="muted")
        with m_k2:
            render_kpi("Runs Scored", f"{h2h['runs']}", "Off the bat", tone="primary")
        with m_k3:
            render_kpi("Strike Rate", f"{h2h['strike_rate']}", "Runs per 100 balls", tone="secondary")
        with m_k4:
            render_kpi("Dismissals", f"{h2h['dismissals']}", f"Average: {h2h['average']}", tone="primary")
        with m_k5:
            render_kpi("Boundaries", f"{h2h['fours']}x4 | {h2h['sixes']}x6", f"{h2h['boundary_pct']}% in boundaries", tone="secondary")

        st.markdown("<br>", unsafe_allow_html=True)
        tab_dt, tab_log = st.tabs(["Delivery Style Breakdown", "Historical Ball-by-Ball Deliveries"])
        with tab_dt:
            st.dataframe(h2h["delivery_styles"], width="stretch")
        with tab_log:
            st.dataframe(h2h["deliveries"][[
                "match_id", "innings", "overid", "over_ball_num", "batter_runs", "stroke",
                "delivery_style", "is_four", "is_six", "is_wicket", "dismissal_type"
            ]], width="stretch")
    else:
        st.warning(f"No recorded historical deliveries found between {bat_name} and {bowl_name} in the dataset.")


# -------------------------------------------------------------
# TAB 7: AI SQUAD & PLAYING XI BUILDER
# -------------------------------------------------------------
with tabs[6]:
    render_section_header("AI Squad & Playing XI Builder", badge="Roster & Pitch Optimization")

    sq_c1, sq_c2, sq_c3 = st.columns(3)
    with sq_c1:
        sq_team = st.selectbox("Franchise / Team", sorted(teams["team_name"].unique().tolist()), index=0, key="sq_team")
    with sq_c2:
        sq_pitch = st.selectbox("Pitch Condition", ["Balanced", "Spin-Friendly (Dust Bowl)", "Pace-Friendly (Green Top)", "High-Scoring Flat Track"], key="sq_pitch")
    with sq_c3:
        sq_opp = st.selectbox("Opposition Team", ["General Matchup"] + sorted(teams["team_name"].unique().tolist()), index=0, key="sq_opp")

    if st.button("⚡ Formulate Optimal Playing XI & 15-Man Squad", width="stretch"):
        with st.spinner("Evaluating player catalog, roles, batting order, and pitch conditions..."):
            xi_result = build_playing_xi(sq_team, data, pitch_condition=sq_pitch, opposition_team=sq_opp)
            
            if "error" in xi_result:
                st.error(xi_result["error"])
            else:
                st.markdown(f"### 🛡️ Recommended Playing XI: **{xi_result['team_name']}**")
                st.info(xi_result["balance_summary"])

                # Display 1-11 Playing XI Cards
                xi_df = pd.DataFrame(xi_result["playing_xi"])
                st.dataframe(
                    xi_df[[
                        "batting_position", "player_name", "role", "tags",
                        "runs", "strike_rate", "wickets", "economy", "justification"
                    ]].rename(columns={
                        "batting_position": "Pos", "player_name": "Player", "role": "Assigned Role",
                        "tags": "Badges", "runs": "Runs", "strike_rate": "SR", "wickets": "Wkts",
                        "economy": "Econ", "justification": "Evidence & Rationale"
                    }),
                    width="stretch"
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🪑 Reserve Bench (4 Tactical Replacements)")
                st.dataframe(pd.DataFrame(xi_result["reserves"]), width="stretch")


# -------------------------------------------------------------
# TAB 8: TARGET CHASE AND DEFENCE PLANNER
# -------------------------------------------------------------
with tabs[7]:
    render_section_header("Target Chase & Defence Tactical Planner", badge="Real-Time Match Math")

    planner_mode = st.radio("Select Match State", ["Chasing a Target (2nd Innings)", "Defending a Total (1st Innings Defended)"], horizontal=True)

    if planner_mode.startswith("Chasing"):
        cp_c1, cp_c2, cp_c3, cp_c4 = st.columns(4)
        with cp_c1:
            chase_target = st.number_input("Target to Chase", min_value=50, max_value=300, value=185)
        with cp_c2:
            chase_current = st.number_input("Current Score", min_value=0, max_value=300, value=92)
        with cp_c3:
            chase_overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=11.2, step=0.1)
        with cp_c4:
            chase_wkts = st.number_input("Wickets Down", min_value=0, max_value=10, value=3)

        chase_plan = analyze_target_chase(chase_target, chase_current, chase_overs, chase_wkts)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### 🎯 Strategic Posture: **{chase_plan['strategic_posture']}**")
        st.warning(f"**Tempo Blueprint:** {chase_plan['tempo_guidance']}")

        k_ch1, k_ch2, k_ch3, k_ch4 = st.columns(4)
        with k_ch1:
            render_kpi("Required Run Rate", f"{chase_plan['rrr']} rpo", f"{chase_plan['runs_needed']} runs needed", tone="primary")
        with k_ch2:
            render_kpi("Current Run Rate", f"{chase_plan['crr']} rpo", "Current scoring tempo", tone="secondary")
        with k_ch3:
            render_kpi("Balls Remaining", f"{chase_plan['balls_remaining']}", f"{chase_plan['wickets_remaining']} wickets in hand", tone="muted")
        with k_ch4:
            render_kpi("Pressure Index", f"{chase_plan['pressure_index']}/100", f"Risk: {chase_plan['risk_level']}", tone="primary")

    else:
        dp_c1, dp_c2, dp_c3, dp_c4 = st.columns(4)
        with dp_c1:
            def_target = st.number_input("Target Defending", min_value=50, max_value=300, value=165)
        with dp_c2:
            def_conceded = st.number_input("Runs Conceded So Far", min_value=0, max_value=300, value=75)
        with dp_c3:
            def_overs = st.number_input("Overs Bowled", min_value=0.0, max_value=20.0, value=9.4, step=0.1)
        with dp_c4:
            def_wkts = st.number_input("Wickets Taken", min_value=0, max_value=10, value=4)

        def_plan = analyze_target_defence(def_target, def_conceded, def_overs, def_wkts)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### 🛡️ Defensive Posture: **{def_plan['defensive_posture']}**")
        st.info(f"**Field Strategy:** {def_plan['field_plan']}")

        k_df1, k_df2, k_df3, k_df4 = st.columns(4)
        with k_df1:
            render_kpi("Max Allowed Economy", f"{def_plan['max_defensive_econ']} rpo", f"{def_plan['runs_budget_left']} runs budget left", tone="primary")
        with k_df2:
            render_kpi("Opponent CRR", f"{def_plan['opp_crr']} rpo", "Current opposition scoring", tone="secondary")
        with k_df3:
            render_kpi("Balls Remaining", f"{def_plan['balls_remaining']}", f"{def_plan['wickets_needed']} wickets needed to win", tone="muted")
        with k_df4:
            render_kpi("Defensive Cushion", f"{def_plan['runs_budget_left']} runs", "Target buffer", tone="primary")

