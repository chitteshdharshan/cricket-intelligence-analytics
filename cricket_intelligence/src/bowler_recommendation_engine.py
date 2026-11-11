"""
AI Cricket Strategy: Next Over Bowler Recommendation Engine
Analyzes striker/non-striker matchups, innings phase, bowling quotas, and tactical game state
to recommend the optimal bowler for the next over.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def recommend_next_over_bowler(
    bbb: pd.DataFrame,
    players_master: pd.DataFrame,
    striker_id: int,
    non_striker_id: int,
    over_number: int,
    available_bowler_ids: List[int],
    overs_bowled_map: Optional[Dict[int, float]] = None,
    last_bowler_id: Optional[int] = None,
    max_overs_per_bowler: int = 4
) -> Dict[str, Any]:
    """
    Evaluates every available bowler against the active striker and non-striker
    for the specific next over (1 to 20).
    """
    if overs_bowled_map is None:
        overs_bowled_map = {}

    # Phase detection
    if over_number <= 6:
        phase = "Powerplay"
    elif over_number <= 15:
        phase = "Middle"
    else:
        phase = "Death"

    striker_info = players_master[players_master["player_id"] == striker_id]
    striker_name = striker_info["player_name"].iloc[0] if len(striker_info) > 0 else f"Player {striker_id}"

    non_striker_info = players_master[players_master["player_id"] == non_striker_id]
    non_striker_name = non_striker_info["player_name"].iloc[0] if len(non_striker_info) > 0 else f"Player {non_striker_id}"

    # Filter out ineligible bowlers (exceeded quota or bowled consecutive over)
    eligible_ids = []
    for bid in available_bowler_ids:
        if bid == last_bowler_id:
            continue
        bowled = overs_bowled_map.get(bid, 0.0)
        if bowled < max_overs_per_bowler:
            eligible_ids.append(bid)

    if not eligible_ids:
        eligible_ids = [b for b in available_bowler_ids if b != last_bowler_id]
    if not eligible_ids:
        eligible_ids = available_bowler_ids

    # Evaluate each eligible bowler
    candidate_records = []
    for bid in eligible_ids:
        b_info = players_master[players_master["player_id"] == bid]
        if len(b_info) == 0:
            continue
        row = b_info.iloc[0]
        b_name = row["player_name"]
        style = str(row["bowling_style"])
        category = "Spin" if any(s in style.lower() for s in ["spin", "break", "slow", "orthodox"]) else "Pace"

        # 1. Direct H2H against striker
        h2h_s = bbb[(bbb["striker_id"] == striker_id) & (bbb["bowler_id"] == bid)]
        s_balls = int(h2h_s["is_legal_ball_bowler"].sum())
        s_runs = int(h2h_s["bowler_conceded_runs"].sum())
        s_outs = int(h2h_s["is_bowler_wicket"].sum())
        s_econ = round(s_runs / s_balls * 6.0, 2) if s_balls > 0 else None

        # 2. Direct H2H against non-striker
        h2h_ns = bbb[(bbb["striker_id"] == non_striker_id) & (bbb["bowler_id"] == bid)]
        ns_balls = int(h2h_ns["is_legal_ball_bowler"].sum())
        ns_runs = int(h2h_ns["bowler_conceded_runs"].sum())
        ns_outs = int(h2h_ns["is_bowler_wicket"].sum())
        ns_econ = round(ns_runs / ns_balls * 6.0, 2) if ns_balls > 0 else None

        # 3. Overall phase performance (Powerplay / Middle / Death)
        p_bbb = bbb[(bbb["bowler_id"] == bid) & (bbb["phase"] == phase)]
        p_balls = int(p_bbb["is_legal_ball_bowler"].sum())
        p_runs = int(p_bbb["bowler_conceded_runs"].sum())
        p_wkts = int(p_bbb["is_bowler_wicket"].sum())
        p_dots = int(p_bbb["is_dot_ball_bowler"].sum())
        p_econ = round(p_runs / p_balls * 6.0, 2) if p_balls > 0 else float(row["bowl_economy"])
        p_dot_pct = round(p_dots / p_balls * 100, 1) if p_balls > 0 else 30.0

        # Calculate Tactical Suitability Score
        score = 50.0

        # Phase economy contribution
        if phase == "Death":
            score += max(-25.0, min(25.0, (10.0 - p_econ) * 3.5))
            score += min(15.0, p_wkts * 1.5)
        elif phase == "Powerplay":
            score += max(-20.0, min(20.0, (8.5 - p_econ) * 3.0))
            score += (p_dot_pct - 35.0) * 0.4
        else:
            score += max(-20.0, min(20.0, (8.0 - p_econ) * 3.0))
            score += min(10.0, p_wkts * 1.0)

        # Matchup bonus/penalty (weighted heavily toward current striker)
        if s_balls >= 12:
            if s_outs > 0:
                score += (s_outs * 8.0)
            if s_econ is not None:
                score += max(-15.0, min(15.0, (8.5 - s_econ) * 2.0))

        if ns_balls >= 12 and ns_outs > 0:
            score += (ns_outs * 4.0)

        # Quota usage
        overs_bowled = overs_bowled_map.get(bid, 0.0)
        overs_left = max_overs_per_bowler - overs_bowled

        candidate_records.append({
            "player_id": bid,
            "bowler_name": b_name,
            "bowling_style": style,
            "category": category,
            "overs_bowled": overs_bowled,
            "overs_remaining": overs_left,
            "phase_economy": p_econ,
            "phase_wickets": p_wkts,
            "phase_balls": p_balls,
            "striker_h2h": f"{s_runs}/{s_outs} in {s_balls}b" if s_balls > 0 else "No recorded duel",
            "striker_h2h_balls": s_balls,
            "striker_h2h_outs": s_outs,
            "striker_h2h_econ": s_econ,
            "non_striker_h2h": f"{ns_runs}/{ns_outs} in {ns_balls}b" if ns_balls > 0 else "No recorded duel",
            "tactical_score": round(score, 1)
        })

    cand_df = pd.DataFrame(candidate_records).sort_values(by="tactical_score", ascending=False)
    top_3 = cand_df.head(3).to_dict(orient="records")

    primary = top_3[0] if top_3 else None
    rationale = []
    tactical_plan = []

    if primary:
        rationale.append(f"**Primary Choice:** `{primary['bowler_name']}` ({primary['category']} - {primary['bowling_style']}) is rated highest with tactical index {primary['tactical_score']}.")
        rationale.append(f"In the **{phase} phase**, operates at an economy of `{primary['phase_economy']}` with `{primary['phase_wickets']}` wickets across `{primary['phase_balls']}` balls.")
        if primary["striker_h2h_balls"] >= 6:
            rationale.append(f"Direct duel record against {striker_name}: {primary['striker_h2h']} (Econ: {primary['striker_h2h_econ']}).")
        
        # Tactical instructions
        if phase == "Death":
            tactical_plan.append("Target wide yorkers outside off-stump to deny leverage; utilize slower-ball bouncers on ball 2 and ball 5.")
        elif phase == "Powerplay":
            tactical_plan.append("Attack off-stump channel on a full-to-good length; keep slip in place for first 3 balls.")
        else:
            tactical_plan.append("Bowl stump-to-stump on tight back-of-a-length, forcing batters into uncomfortable cross-batted shots to boundary riders.")

    return {
        "over_number": over_number,
        "phase": phase,
        "striker": striker_name,
        "non_striker": non_striker_name,
        "top_3_bowlers": top_3,
        "primary_recommendation": primary,
        "rationale": rationale,
        "tactical_plan": tactical_plan,
        "full_candidates_df": cand_df
    }

