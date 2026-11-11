"""
Batsman vs Bowler Matchup Intelligence Engine for Cricket Intelligence.
Computes head-to-head statistics, dismissal logs, and ball-by-ball histories.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def get_batsman_vs_all_bowlers(df: pd.DataFrame, batsman_id: int, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Computes performance of a batsman against every bowler they have faced:
    Bowler, Bowler ID, Balls, Runs (bat), Total runs, 4s, 6s, Dots, Singles, Doubles, Triples,
    Dismissals, Strike Rate, Boundary %, 4 Frequency, 6 Frequency, Dismissal Frequency.
    """
    sub = df[df["striker_id"] == batsman_id]
    if len(sub) == 0:
        return pd.DataFrame()

    summary = []
    for bowler_id, g in sub.groupby("bowler_id"):
        bowler_name = g["bowler_name"].iloc[0]
        legal_balls = int(g["is_legal_ball_batter"].sum())
        bat_runs = int(g["batter_runs"].sum())
        total_runs = int(g["total_delivery_runs"].sum())
        fours = int(g["is_four"].sum())
        sixes = int(g["is_six"].sum())
        dots = int(g["is_dot_ball_batter"].sum())
        singles = int(((g["batter_runs"] == 1) & g["is_legal_ball_batter"]).sum())
        doubles = int(((g["batter_runs"] == 2) & g["is_legal_ball_batter"]).sum())
        triples = int(((g["batter_runs"] == 3) & g["is_legal_ball_batter"]).sum())
        dismissals = int((g["is_wicket"] & ((g["dismissed_player_id"] == batsman_id) | (g["dismissed_player_id"].isna()))).sum())

        sr = round(bat_runs / legal_balls * 100, 2) if legal_balls > 0 else 0.0
        boundary_runs = (fours * 4) + (sixes * 6)
        boundary_pct = round(boundary_runs / bat_runs * 100, 2) if bat_runs > 0 else 0.0
        
        four_freq = round(legal_balls / fours, 1) if fours > 0 else 0.0
        six_freq = round(legal_balls / sixes, 1) if sixes > 0 else 0.0
        dismissal_freq = round(legal_balls / dismissals, 1) if dismissals > 0 else 0.0

        summary.append({
            "Bowler": bowler_name,
            "Bowler ID": bowler_id,
            "Balls": legal_balls,
            "Runs (Bat)": bat_runs,
            "Total Runs": total_runs,
            "4s": fours,
            "6s": sixes,
            "Dots": dots,
            "1s": singles,
            "2s": doubles,
            "3s": triples,
            "Dismissals": dismissals,
            "Strike Rate": sr,
            "Boundary %": boundary_pct,
            "Balls / 4": four_freq,
            "Balls / 6": six_freq,
            "Balls / Out": dismissal_freq
        })

    res = pd.DataFrame(summary)
    return res.sort_values(by="Balls", ascending=False)

def get_h2h_matchup_details(df: pd.DataFrame, batsman_id: int, bowler_id: int) -> Dict[str, Any]:
    """
    Computes comprehensive head-to-head statistics between a specific batsman and bowler.
    """
    h2h = df[(df["striker_id"] == batsman_id) & (df["bowler_id"] == bowler_id)]
    if len(h2h) == 0:
        return {"balls": 0, "runs": 0, "sr": 0.0, "deliveries": pd.DataFrame()}

    legal_balls = int(h2h["is_legal_ball_batter"].sum())
    bat_runs = int(h2h["batter_runs"].sum())
    fours = int(h2h["is_four"].sum())
    sixes = int(h2h["is_six"].sum())
    dots = int(h2h["is_dot_ball_batter"].sum())
    dismissals = int((h2h["is_wicket"] & ((h2h["dismissed_player_id"] == batsman_id) | (h2h["dismissed_player_id"].isna()))).sum())

    sr = round(bat_runs / legal_balls * 100, 2) if legal_balls > 0 else 0.0
    dot_pct = round(dots / legal_balls * 100, 2) if legal_balls > 0 else 0.0
    boundary_pct = round(((fours * 4 + sixes * 6) / bat_runs * 100), 2) if bat_runs > 0 else 0.0
    avg = round(bat_runs / dismissals, 2) if dismissals > 0 else bat_runs

    # Over by over progression in this matchup
    over_summary = h2h.groupby("overid").agg(
        balls=("is_legal_ball_batter", "sum"),
        runs=("batter_runs", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dismissals=("is_wicket", lambda x: (x & ((h2h.loc[x.index, "dismissed_player_id"] == batsman_id) | (h2h.loc[x.index, "dismissed_player_id"].isna()))).sum())
    ).reset_index()
    over_summary["sr"] = np.where(over_summary["balls"] > 0, (over_summary["runs"] / over_summary["balls"] * 100).round(2), 0.0)

    # Delivery style breakdown
    style_summary = h2h.groupby("delivery_style").agg(
        balls=("is_legal_ball_batter", "sum"),
        runs=("batter_runs", "sum"),
        dismissals=("is_wicket", lambda x: (x & ((h2h.loc[x.index, "dismissed_player_id"] == batsman_id) | (h2h.loc[x.index, "dismissed_player_id"].isna()))).sum())
    ).reset_index()
    style_summary["sr"] = np.where(style_summary["balls"] > 0, (style_summary["runs"] / style_summary["balls"] * 100).round(2), 0.0)

    # Match by match breakdown
    match_summary = h2h.groupby("match_id").agg(
        date=("match_date", "first"),
        comp=("comp_name", "first"),
        balls=("is_legal_ball_batter", "sum"),
        runs=("batter_runs", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dismissals=("is_wicket", lambda x: (x & ((h2h.loc[x.index, "dismissed_player_id"] == batsman_id) | (h2h.loc[x.index, "dismissed_player_id"].isna()))).sum())
    ).reset_index()
    match_summary["sr"] = np.where(match_summary["balls"] > 0, (match_summary["runs"] / match_summary["balls"] * 100).round(2), 0.0)

    return {
        "balls": legal_balls,
        "runs": bat_runs,
        "strike_rate": sr,
        "average": avg,
        "fours": fours,
        "sixes": sixes,
        "dots": dots,
        "dot_pct": dot_pct,
        "dismissals": dismissals,
        "boundary_pct": boundary_pct,
        "over_progression": over_summary,
        "delivery_styles": style_summary,
        "match_history": match_summary,
        "deliveries": h2h.sort_values(by=["match_date", "overid", "over_ball_num"], ascending=[False, True, True])
    }

