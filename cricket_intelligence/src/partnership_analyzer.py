"""
AI Cricket Strategy: Partnership Analyzer
In-depth comparative intelligence for active batting pairs:
LHB/RHB combinations, pace vs spin splits, tactical roles, and opposition breakdown plans.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def _get_single_batter_profile(bbb: pd.DataFrame, pm: pd.DataFrame, player_id: int) -> Dict[str, Any]:
    """Generates tactical profile for an individual batsman."""
    p_info = pm[pm["player_id"] == player_id]
    if len(p_info) == 0:
        return {"player_id": player_id, "name": f"Player {player_id}", "style": "Unknown", "balls": 0}

    row = p_info.iloc[0]
    p_bbb = bbb[bbb["striker_id"] == player_id]
    
    # Split by pace vs spin
    pace_b = p_bbb[p_bbb["pace_or_spin"] == "Pace"]
    spin_b = p_bbb[p_bbb["pace_or_spin"] == "Spin"]

    pace_runs = int(pace_b["batter_runs"].sum())
    pace_balls = int(pace_b["is_legal_ball_batter"].sum())
    pace_outs = int((pace_b["dismissed_player_id"] == player_id).sum())
    pace_sr = round(pace_runs / pace_balls * 100, 2) if pace_balls > 0 else 0.0
    pace_avg = round(pace_runs / pace_outs, 2) if pace_outs > 0 else pace_runs

    spin_runs = int(spin_b["batter_runs"].sum())
    spin_balls = int(spin_b["is_legal_ball_batter"].sum())
    spin_outs = int((spin_b["dismissed_player_id"] == player_id).sum())
    spin_sr = round(spin_runs / spin_balls * 100, 2) if spin_balls > 0 else 0.0
    spin_avg = round(spin_runs / spin_outs, 2) if spin_outs > 0 else spin_runs

    # Phase splits
    pp_b = p_bbb[p_bbb["overid"] <= 6]
    death_b = p_bbb[p_bbb["overid"] >= 16]

    pp_sr = round(pp_b["batter_runs"].sum() / pp_b["is_legal_ball_batter"].sum() * 100, 2) if pp_b["is_legal_ball_batter"].sum() > 0 else 0.0
    death_sr = round(death_b["batter_runs"].sum() / death_b["is_legal_ball_batter"].sum() * 100, 2) if death_b["is_legal_ball_batter"].sum() > 0 else 0.0

    total_runs = int(row["runs"])
    total_balls = int(row["balls_faced"])
    fours = int(row["fours"])
    sixes = int(row["sixes"])
    boundary_pct = round(((fours * 4) + (sixes * 6)) / total_runs * 100, 1) if total_runs > 0 else 0.0

    return {
        "player_id": player_id,
        "name": row["player_name"],
        "batting_style": str(row["batting_style"]),
        "total_runs": total_runs,
        "balls_faced": total_balls,
        "strike_rate": float(row["bat_strike_rate"]),
        "boundary_pct": boundary_pct,
        "pace_sr": pace_sr,
        "pace_avg": pace_avg,
        "pace_balls": pace_balls,
        "spin_sr": spin_sr,
        "spin_avg": spin_avg,
        "spin_balls": spin_balls,
        "powerplay_sr": pp_sr,
        "death_sr": death_sr,
    }


def analyze_partnership(
    bbb: pd.DataFrame,
    players_master: pd.DataFrame,
    batter1_id: int,
    batter2_id: int
) -> Dict[str, Any]:
    """
    Performs comprehensive partnership analysis between two active batters.
    """
    b1 = _get_single_batter_profile(bbb, players_master, batter1_id)
    b2 = _get_single_batter_profile(bbb, players_master, batter2_id)

    # Handedness combination
    h1 = "LHB" if "left" in b1["batting_style"].lower() else "RHB"
    h2 = "LHB" if "left" in b2["batting_style"].lower() else "RHB"
    is_left_right = (h1 != h2)
    combo_str = f"{h1} + {h2} (Left-Right Tandem)" if is_left_right else f"{h1} + {h2} (Same-Handed Tandem)"

    # Determine roles: who should attack vs anchor
    if b1["strike_rate"] >= b2["strike_rate"] + 15.0:
        aggressor = b1["name"]
        anchor = b2["name"]
        role_strategy = f"{b1['name']} should lead the scoring acceleration (SR {b1['strike_rate']}), while {b2['name']} rotates strike and anchors."
    elif b2["strike_rate"] >= b1["strike_rate"] + 15.0:
        aggressor = b2["name"]
        anchor = b1["name"]
        role_strategy = f"{b2['name']} should lead the scoring acceleration (SR {b2['strike_rate']}), while {b1['name']} rotates strike and anchors."
    else:
        aggressor = "Dual Aggressors"
        anchor = "Dynamic Rotation"
        role_strategy = f"Both batters possess comparable striking abilities (SR {b1['strike_rate']} vs {b2['strike_rate']}). Target matchups dynamically based on bowling type."

    # Vulnerability analysis
    spin_vulnerable = []
    pace_vulnerable = []

    if b1["spin_sr"] < 120 and b1["spin_balls"] >= 30:
        spin_vulnerable.append(b1["name"])
    if b2["spin_sr"] < 120 and b2["spin_balls"] >= 30:
        spin_vulnerable.append(b2["name"])

    if b1["pace_sr"] < 120 and b1["pace_balls"] >= 30:
        pace_vulnerable.append(b1["name"])
    if b2["pace_sr"] < 120 and b2["pace_balls"] >= 30:
        pace_vulnerable.append(b2["name"])

    # Bowling tactics
    bowling_tactics = []
    if is_left_right:
        bowling_tactics.append("Disrupt field setting rhythm by deploying bowlers with consistent channel discipline; prioritize round-the-wicket angles to neutralize the LHB.")
    else:
        bowling_tactics.append(f"Standardized off-side/leg-side field placings remain stable across overs due to identical handedness ({h1}).")

    if spin_vulnerable:
        bowling_tactics.append(f"Deploy attacking spin against {', '.join(spin_vulnerable)}: keep mid-wicket back and pack the ring on the off-side.")
    if pace_vulnerable:
        bowling_tactics.append(f"Test {', '.join(pace_vulnerable)} with back-of-a-length pace bowling and hard lengths into the body.")

    # Historical joint performance if available
    joint_bbb = bbb[(bbb["striker_id"].isin([batter1_id, batter2_id])) & (bbb["non_striker_id"].isin([batter1_id, batter2_id]))]
    joint_balls = int(joint_bbb["is_legal_ball_batter"].sum())
    joint_runs = int(joint_bbb["batter_runs"].sum())
    joint_sr = round(joint_runs / joint_balls * 100, 2) if joint_balls > 0 else 0.0

    return {
        "batter1": b1,
        "batter2": b2,
        "combo_type": combo_str,
        "is_left_right": is_left_right,
        "aggressor": aggressor,
        "anchor": anchor,
        "role_strategy": role_strategy,
        "spin_vulnerable": spin_vulnerable,
        "pace_vulnerable": pace_vulnerable,
        "bowling_tactics": bowling_tactics,
        "joint_history": {
            "balls": joint_balls,
            "runs": joint_runs,
            "strike_rate": joint_sr
        }
    }

