"""
MATCH EXPLORER & FULL SCORECARD RECONSTRUCTION
Reconstructs complete match dossier: Playing XIs, scorecards, fall of wickets,
partnerships, over worms, and ball-by-ball inspector.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import load_all_data, get_lookup_maps
from src.match_analysis import get_match_full_details
from src.ui_components import apply_custom_theme, render_banner, render_kpi, render_sidebar_branding, render_section_header, get_plotly_layout_defaults

st.set_page_config(page_title="Match Explorer | Cricket Intelligence", page_icon="🏟️", layout="wide")
apply_custom_theme()
render_sidebar_branding()

data = load_all_data()
bbb = data["bbb"]
matches = data["matches"]
p_eleven = data["playing_eleven"]
lookups = get_lookup_maps(data)

render_banner(
    "MATCH EXPLORER & SCORECARD DOSSIER",
    "Complete match intelligence: Playing XI, batting & bowling scorecards, partnerships, fall of wickets, and ball logs"
)

# Match Selection Filters
col_m1, col_m2 = st.columns([1.5, 2.5])

with col_m1:
    team_opts = ["All Teams"] + sorted(data["teams"]["team_name"].unique().tolist())
    sel_t = st.selectbox("Filter Matches by Team", team_opts)

m_list = matches.copy()
if sel_t != "All Teams":
    m_list = m_list[(m_list["home_team"] == sel_t) | (m_list["away_team"] == sel_t)]

# Sort matches descending by date
m_list = m_list.sort_values(by="match_date", ascending=False)
match_options = dict(zip(
    m_list["match_id"],
    "Match " + m_list["match_id"].astype(str) + ": " + m_list["home_team"] + " vs " + m_list["away_team"] + " (" + m_list["match_date"].astype(str).str[:10] + ")"
))

if len(match_options) == 0:
    st.warning("No matches found for this filter.")
    st.stop()

with col_m2:
    selected_mid = st.selectbox(
        "Select Match to Explore",
        options=list(match_options.keys()),
        format_func=lambda x: match_options[x]
    )

# Retrieve full match intelligence
match_dossier = get_match_full_details(
    selected_mid,
    matches,
    bbb,
    p_eleven,
    lookups["player_names"]
)

meta = match_dossier["metadata"]

# Match Overview Card
render_section_header(f"{meta['home_team']} vs {meta['away_team']}")
st.markdown(f"**Match ID:** `{meta['match_id']}` | **Date:** `{str(meta['match_date'])[:10]}` | **Venue:** `{meta['venue_country']}` | **Stage:** `{meta['stage']}`")
st.markdown(f"**Toss:** Team opted to `{meta['toss_opted']}` | **Result:** `{meta['result']}`")

st.divider()

# Innings Tabs
inns = match_dossier["innings"]
inn_keys = sorted(inns.keys())

tabs = st.tabs([f"Innings {i} ({lookups['team_names'].get(inns[i]['batting_team_id'], 'Team')} {inns[i]['total_runs']}/{inns[i]['total_wickets']})" for i in inn_keys] + ["👥 Playing XIs", "📈 Match Worm / Progression", "🔍 Ball-by-Ball Log"])

for idx, inn_num in enumerate(inn_keys):
    inn_obj = inns[inn_num]
    with tabs[idx]:
        st.markdown(f"#### Innings {inn_num} Scorecard — {lookups['team_names'].get(inn_obj['batting_team_id'], 'Team')}")
        st.markdown(f"**Total Score:** `{inn_obj['total_runs']}/{inn_obj['total_wickets']}` in `{inn_obj['overs']}` overs | **Extras:** `{inn_obj['extras']}`")

        st.markdown("##### Batting Scorecard")
        st.dataframe(inn_obj["bat_scorecard"], width="stretch")

        st.markdown("##### Bowling Scorecard")
        st.dataframe(inn_obj["bowl_scorecard"], width="stretch")

        col_fow, col_part = st.columns(2)
        with col_fow:
            st.markdown("##### Fall of Wickets")
            st.dataframe(inn_obj["fow"], width="stretch", height=220)
        with col_part:
            st.markdown("##### Reconstructed Partnerships")
            st.dataframe(inn_obj["partnerships"], width="stretch", height=220)

with tabs[len(inn_keys)]:
    st.markdown("#### Official Playing XIs")
    xi_c1, xi_c2 = st.columns(2)
    with xi_c1:
        st.markdown(f"##### {meta['home_team']}")
        st.dataframe(
            match_dossier["home_xi"][["batting_order", "player_name", "player_id", "is_captain", "is_wk"]].rename(columns={
                "batting_order": "Bat Order", "player_name": "Player", "player_id": "ID", "is_captain": "Captain", "is_wk": "WK"
            }),
            width="stretch"
        )
    with xi_c2:
        st.markdown(f"##### {meta['away_team']}")
        st.dataframe(
            match_dossier["away_xi"][["batting_order", "player_name", "player_id", "is_captain", "is_wk"]].rename(columns={
                "batting_order": "Bat Order", "player_name": "Player", "player_id": "ID", "is_captain": "Captain", "is_wk": "WK"
            }),
            width="stretch"
        )

with tabs[len(inn_keys) + 1]:
    st.markdown("#### Over-by-Over Cumulative Worm")
    worm_dfs = []
    for inn_num in inn_keys:
        p = inns[inn_num]["progression"].copy()
        p["Innings"] = f"Inn {inn_num} ({lookups['team_names'].get(inns[inn_num]['batting_team_id'], 'Team')})"
        worm_dfs.append(p)
    if worm_dfs:
        full_worm = pd.concat(worm_dfs)
        fig_worm = px.line(
            full_worm, x="overid", y="cum_runs", color="Innings",
            title="Score Progression Across 20 Overs",
            labels={"overid": "Over Number", "cum_runs": "Cumulative Runs"},
            markers=True
        )
        fig_worm.update_layout(**get_plotly_layout_defaults())
        st.plotly_chart(fig_worm, width="stretch")

with tabs[len(inn_keys) + 2]:
    st.markdown("#### Complete Delivery-by-Delivery Log")
    st.dataframe(
        match_dossier["all_deliveries"][[
            "innings", "overid", "over_ball_num", "striker_name", "bowler_name", "batter_runs",
            "extras_type", "stroke", "delivery_style", "pace_or_spin", "is_four", "is_six", "is_wicket", "dismissal_type"
        ]].rename(columns={
            "innings": "Inn", "overid": "Over", "over_ball_num": "Ball #", "striker_name": "Striker",
            "bowler_name": "Bowler", "batter_runs": "Runs", "extras_type": "Extras", "stroke": "Shot",
            "delivery_style": "Style", "pace_or_spin": "Type", "is_four": "4", "is_six": "6",
            "is_wicket": "Wkt", "dismissal_type": "Dismissal"
        }),
        width="stretch",
        height=400
    )

