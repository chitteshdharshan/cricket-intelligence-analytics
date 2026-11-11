"""
Pressure Performance Analyzer Engine
Calculates genuine historical last-over pressure performance for batters and bowlers
under specific required run scenarios (e.g. 15 runs required in the 20th over).
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def get_qualifying_pressure_matches(
    bbb: pd.DataFrame,
    matches: pd.DataFrame,
    comps: Optional[pd.DataFrame] = None,
    target_runs: int = 15,
    match_format: Optional[str] = None,
    team_name: Optional[str] = None,
    season: Optional[str] = None,
    min_over: int = 20
) -> pd.DataFrame:
    """
    Identifies genuine historical 2nd innings matches where the batting team
    required exactly `target_runs` at the beginning of over `min_over` (default 20).
    """
    # 1. Calculate 1st innings total to establish target
    inn1_runs = (
        bbb[bbb["innings"] == 1]
        .groupby("match_id")["total_delivery_runs"]
        .sum()
        .reset_index(name="inn1_total")
    )

    # 2. Base match metadata
    m_info = matches.copy()
    if comps is not None and "comp_id" in m_info.columns and "comp_id" in comps.columns:
        comp_cols = [c for c in ["comp_id", "comp_name", "match_format", "season"] if c in comps.columns]
        m_info = m_info.merge(comps[comp_cols], on="comp_id", how="left")

    m_info = m_info.merge(inn1_runs, on="match_id", how="left")
    # Target = revised_target if rain-affected, else inn1_total + 1
    m_info["target"] = m_info["revised_target"].combine_first(m_info["inn1_total"] + 1)
    # Filter out matches where target could not be reliably established
    m_info = m_info.dropna(subset=["target"])
    m_info["target"] = m_info["target"].astype(int)

    # 3. Calculate 2nd innings runs scored before final over
    inn2 = bbb[bbb["innings"] == 2]
    # Identify matches that actually reached the final over
    m_reached_final = set(inn2[inn2["overid"] == min_over]["match_id"].unique())

    inn2_before_final = (
        inn2[inn2["overid"] < min_over]
        .groupby("match_id")["total_delivery_runs"]
        .sum()
        .reset_index(name="runs_before_final_over")
    )

    m_calc = m_info[m_info["match_id"].isin(m_reached_final)].merge(inn2_before_final, on="match_id", how="inner")
    m_calc["runs_needed_start_over"] = m_calc["target"] - m_calc["runs_before_final_over"]

    # 4. Determine chasing team and defending team
    inn2_teams = inn2.groupby("match_id").agg(
        chasing_team_id=("batting_team_id", "first"),
        defending_team_id=("bowling_team_id", "first")
    ).reset_index()
    m_calc = m_calc.merge(inn2_teams, on="match_id", how="left")

    # 5. Outcome verification: Chase won vs Defence won (excludes ties & no-results)
    m_calc["chase_won"] = m_calc["win_team_id"] == m_calc["chasing_team_id"]
    m_calc["defence_won"] = m_calc["win_team_id"] == m_calc["defending_team_id"]

    # 6. Apply scenario filter
    q_matches = m_calc[m_calc["runs_needed_start_over"] == target_runs].copy()

    # 7. Apply optional user filters
    if match_format and match_format != "All Formats" and "match_format" in q_matches.columns:
        q_matches = q_matches[q_matches["match_format"] == match_format]
    if season and season != "All Seasons" and "season" in q_matches.columns:
        q_matches = q_matches[q_matches["season"] == season]
    if team_name and team_name != "All Teams":
        q_matches = q_matches[(q_matches["home_team"] == team_name) | (q_matches["away_team"] == team_name)]

    return q_matches


def calculate_batter_pressure_rankings(
    qualifying_matches: pd.DataFrame,
    bbb: pd.DataFrame,
    target_runs: int = 15,
    min_opportunities: int = 1,
    over_id: int = 20
) -> pd.DataFrame:
    """
    Ranks batters who faced at least 1 delivery in the qualifying final overs.
    Formula considers scoring impact, boundary contribution, successful chases,
    and sample size reliability.
    """
    if qualifying_matches.empty:
        return pd.DataFrame()

    q_mids = set(qualifying_matches["match_id"].unique())
    over_balls = bbb[(bbb["match_id"].isin(q_mids)) & (bbb["innings"] == 2) & (bbb["overid"] == over_id)]

    if over_balls.empty:
        return pd.DataFrame()

    records = []
    for pid, pgroup in over_balls.groupby("striker_id"):
        pname = pgroup["striker_name"].iloc[0]
        p_mids = pgroup["match_id"].unique()
        opps = len(p_mids)

        if opps < min_opportunities:
            continue

        runs = int(pgroup["batter_runs"].sum())
        balls = int(pgroup["is_legal_ball_batter"].sum())
        fours = int(pgroup["is_four"].sum())
        sixes = int(pgroup["is_six"].sum())
        dots = int(pgroup["is_dot_ball_batter"].sum())

        sr = (runs / balls * 100.0) if balls > 0 else 0.0
        boundary_runs = (fours * 4) + (sixes * 6)
        boundary_pct = (boundary_runs / runs * 100.0) if runs > 0 else 0.0
        dot_pct = (dots / balls * 100.0) if balls > 0 else 0.0

        p_match_subset = qualifying_matches[qualifying_matches["match_id"].isin(p_mids)]
        chase_wins = int(p_match_subset["chase_won"].sum())
        chase_win_pct = (chase_wins / opps * 100.0) if opps > 0 else 0.0

        # Transparent Scoring Formula:
        # 1. Successful chase impact (35% weight)
        # 2. Strike rate intensity scaled to 250 SR (30% weight)
        # 3. Boundary contribution percentage (15% weight)
        # 4. Runs per opportunity relative to target (20% weight)
        # 5. Reliability dampening for single-ball/small samples
        sample_factor = min(1.0, 0.70 + 0.15 * opps)
        ball_faced_factor = min(1.0, balls / 3.0)  # penalize 1-ball cameos

        raw_score = (
            (chase_win_pct * 0.35) +
            (min(sr / 250.0, 1.0) * 30.0) +
            (min(boundary_pct / 100.0, 1.0) * 15.0) +
            (min((runs / opps) / float(target_runs), 1.0) * 20.0)
        )
        pressure_rating = round(raw_score * sample_factor * (0.85 + 0.15 * ball_faced_factor), 1)

        records.append({
            "player_id": pid,
            "player_name": pname,
            "opportunities": opps,
            "runs": runs,
            "balls": balls,
            "strike_rate": round(sr, 2),
            "fours": fours,
            "sixes": sixes,
            "dots": dots,
            "dot_pct": round(dot_pct, 1),
            "chase_wins": chase_wins,
            "chase_win_pct": round(chase_win_pct, 1),
            "pressure_rating": pressure_rating,
            "sample_confidence": "Robust" if opps >= 3 else ("Moderate" if opps == 2 else "Limited (1 match)")
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values(
        by=["pressure_rating", "chase_wins", "runs", "strike_rate"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    df["rank"] = df.index + 1
    return df


def calculate_bowler_pressure_rankings(
    qualifying_matches: pd.DataFrame,
    bbb: pd.DataFrame,
    target_runs: int = 15,
    min_opportunities: int = 1,
    over_id: int = 20
) -> pd.DataFrame:
    """
    Ranks bowlers who delivered the qualifying final overs.
    Formula considers successful defences, runs conceded vs target, wickets,
    dot ball %, and sample size reliability.
    """
    if qualifying_matches.empty:
        return pd.DataFrame()

    q_mids = set(qualifying_matches["match_id"].unique())
    over_balls = bbb[(bbb["match_id"].isin(q_mids)) & (bbb["innings"] == 2) & (bbb["overid"] == over_id)]

    if over_balls.empty:
        return pd.DataFrame()

    records = []
    for bid, bgroup in over_balls.groupby("bowler_id"):
        bname = bgroup["bowler_name"].iloc[0]
        b_mids = bgroup["match_id"].unique()
        opps = len(b_mids)

        if opps < min_opportunities:
            continue

        conceded = int(bgroup["bowler_conceded_runs"].sum())
        balls = int(bgroup["is_legal_ball_bowler"].sum())
        wickets = int(bgroup["is_bowler_wicket"].sum())
        dots = int(bgroup["is_dot_ball_bowler"].sum())

        econ = (conceded / balls * 6.0) if balls > 0 else 0.0
        dot_pct = (dots / balls * 100.0) if balls > 0 else 0.0

        b_match_subset = qualifying_matches[qualifying_matches["match_id"].isin(b_mids)]
        defences = int(b_match_subset["defence_won"].sum())
        def_pct = (defences / opps * 100.0) if opps > 0 else 0.0

        # Transparent Scoring Formula:
        # 1. Successful defence percentage (40% weight)
        # 2. Run containment vs target economy (30% weight)
        # 3. Wicket taking impact (20% weight: 2 wkts/over = 20 pts)
        # 4. Dot ball percentage (10% weight)
        # 5. Reliability dampening
        sample_factor = min(1.0, 0.70 + 0.15 * opps)
        econ_containment = max(0.0, (target_runs - econ) / float(target_runs)) * 30.0
        wkt_impact = min((wickets / opps) / 2.0, 1.0) * 20.0

        raw_score = (
            (def_pct * 0.40) +
            econ_containment +
            wkt_impact +
            (dot_pct * 0.10)
        )
        pressure_rating = round(raw_score * sample_factor, 1)

        records.append({
            "bowler_id": bid,
            "player_name": bname,
            "opportunities": opps,
            "conceded": conceded,
            "balls": balls,
            "overs": round(balls / 6.0, 1),
            "wickets": wickets,
            "economy": round(econ, 2),
            "dots": dots,
            "dot_pct": round(dot_pct, 1),
            "defences": defences,
            "defence_pct": round(def_pct, 1),
            "pressure_rating": pressure_rating,
            "sample_confidence": "Robust" if opps >= 3 else ("Moderate" if opps == 2 else "Limited (1 match)")
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values(
        by=["pressure_rating", "defences", "economy", "wickets"],
        ascending=[False, False, True, False]
    ).reset_index(drop=True)

    df["rank"] = df.index + 1
    return df


def get_player_pressure_drilldown(
    player_id: int,
    player_type: str,  # 'Batter' or 'Bowler'
    qualifying_matches: pd.DataFrame,
    bbb: pd.DataFrame,
    players_master: Optional[pd.DataFrame] = None,
    over_id: int = 20
) -> Dict[str, Any]:
    """
    Extracts match-by-match and ball-by-ball performance log for a selected player
    in qualifying pressure over situations.
    """
    q_mids = set(qualifying_matches["match_id"].unique())
    id_col = "striker_id" if player_type.lower() == "batter" else "bowler_id"

    p_balls = bbb[
        (bbb["match_id"].isin(q_mids)) &
        (bbb["innings"] == 2) &
        (bbb["overid"] == over_id) &
        (bbb[id_col] == player_id)
    ].copy()

    if p_balls.empty:
        return {"found": False}

    pname = p_balls["striker_name"].iloc[0] if player_type.lower() == "batter" else p_balls["bowler_name"].iloc[0]

    # Match-by-match aggregation
    matches_list = []
    for mid, mgroup in p_balls.groupby("match_id"):
        m_row = qualifying_matches[qualifying_matches["match_id"] == mid].iloc[0]

        if player_type.lower() == "batter":
            runs = int(mgroup["batter_runs"].sum())
            balls = int(mgroup["is_legal_ball_batter"].sum())
            fours = int(mgroup["is_four"].sum())
            sixes = int(mgroup["is_six"].sum())
            dots = int(mgroup["is_dot_ball_batter"].sum())
            sr = round((runs / balls * 100.0) if balls > 0 else 0.0, 2)
            won = bool(m_row["chase_won"])
            stat_summary = f"{runs} runs ({balls} balls, {fours}x4, {sixes}x6) — SR: {sr}"
        else:
            conceded = int(mgroup["bowler_conceded_runs"].sum())
            balls = int(mgroup["is_legal_ball_bowler"].sum())
            wickets = int(mgroup["is_bowler_wicket"].sum())
            dots = int(mgroup["is_dot_ball_bowler"].sum())
            econ = round((conceded / balls * 6.0) if balls > 0 else 0.0, 2)
            won = bool(m_row["defence_won"])
            stat_summary = f"{wickets}/{conceded} ({balls} balls) — Econ: {econ}"

        # Delivery log string (e.g. 1, 4, W, 0, 6, 2)
        delivery_seq = []
        for _, b in mgroup.sort_values(by="over_ball_num").iterrows():
            if b.get("is_bowler_wicket", 0):
                delivery_seq.append(f"W ({b.get('dismissal_type', 'out')})")
            elif b.get("extras_type") == "wides":
                delivery_seq.append(f"{b['total_delivery_runs']}Wd")
            elif b.get("extras_type") == "noballs":
                delivery_seq.append(f"{b['total_delivery_runs']}Nb")
            elif b.get("is_six", 0):
                delivery_seq.append("6 💥")
            elif b.get("is_four", 0):
                delivery_seq.append("4 🎯")
            elif b.get("is_dot_ball_batter", 0):
                delivery_seq.append("0 ⚪")
            else:
                delivery_seq.append(str(b.get("total_delivery_runs", 0)))

        matches_list.append({
            "match_id": mid,
            "match_date": str(m_row.get("match_date", "Unknown"))[:10],
            "competition": m_row.get("comp_name", "T20 Match"),
            "fixture": f"{m_row.get('home_team', '')} vs {m_row.get('away_team', '')}",
            "target": int(m_row.get("target", 0)),
            "needed_start_over": int(m_row.get("runs_needed_start_over", 0)),
            "performance": stat_summary,
            "deliveries": " → ".join(delivery_seq),
            "outcome": "Success (Won)" if won else "Defeat (Lost)",
            "result_text": m_row.get("result", "")
        })

    return {
        "found": True,
        "player_id": player_id,
        "player_name": pname,
        "player_type": player_type,
        "match_count": len(matches_list),
        "matches": matches_list,
        "deliveries_raw": p_balls
    }

