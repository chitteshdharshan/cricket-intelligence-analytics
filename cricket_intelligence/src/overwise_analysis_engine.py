"""
Over-wise Batting & Bowling Analysis Engine
Provides granular delivery breakdowns over-by-over (Overs 1 through 20),
identifying phase dynamics, death-over specialists, and acceleration patterns.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def get_batter_overwise_analysis(
    bbb: pd.DataFrame,
    batter_id: int,
    max_overs: int = 20
) -> pd.DataFrame:
    """
    Computes over-by-over performance for a specific batter from Over 1 through 20.
    """
    df = bbb[(bbb["striker_id"] == batter_id) & (bbb["overid"] <= max_overs)]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for ov in range(1, max_overs + 1):
        sub = df[df["overid"] == ov]
        runs = int(sub["batter_runs"].sum())
        balls = int(sub["is_legal_ball_batter"].sum())
        fours = int(sub["is_four"].sum())
        sixes = int(sub["is_six"].sum())
        dots = int(sub["is_dot_ball_batter"].sum())
        outs = int(((sub["dismissed_player_id"] == batter_id) & (sub["is_wicket"] == 1)).sum())

        sr = round((runs / balls * 100.0) if balls > 0 else 0.0, 2)
        boundaries = fours + sixes
        bpb = round((balls / boundaries) if boundaries > 0 else 0.0, 2)
        dot_pct = round((dots / balls * 100.0) if balls > 0 else 0.0, 1)

        # Innings phase
        phase = "Powerplay" if ov <= 6 else ("Middle Overs" if ov <= 15 else "Death Overs")

        rows.append({
            "Over": ov,
            "Phase": phase,
            "Runs": runs,
            "Balls Faced": balls,
            "Strike Rate": sr,
            "Fours": fours,
            "Sixes": sixes,
            "Dots": dots,
            "Dot Ball %": dot_pct,
            "Dismissals": outs,
            "Balls Per Boundary": bpb if boundaries > 0 else np.nan
        })

    return pd.DataFrame(rows)


def get_bowler_overwise_analysis(
    bbb: pd.DataFrame,
    bowler_id: int,
    max_overs: int = 20
) -> pd.DataFrame:
    """
    Computes over-by-over bowling economy and wicket impact for a bowler from Over 1 through 20.
    """
    df = bbb[(bbb["bowler_id"] == bowler_id) & (bbb["overid"] <= max_overs)]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for ov in range(1, max_overs + 1):
        sub = df[df["overid"] == ov]
        legal_balls = int(sub["is_legal_ball_bowler"].sum())
        conceded = int(sub["bowler_conceded_runs"].sum())
        wickets = int(sub["is_bowler_wicket"].sum())
        dots = int(sub["is_dot_ball_bowler"].sum())
        fours = int(sub["is_four"].sum())
        sixes = int(sub["is_six"].sum())

        econ = round((conceded / legal_balls * 6.0) if legal_balls > 0 else 0.0, 2)
        dot_pct = round((dots / legal_balls * 100.0) if legal_balls > 0 else 0.0, 1)
        phase = "Powerplay" if ov <= 6 else ("Middle Overs" if ov <= 15 else "Death Overs")

        rows.append({
            "Over": ov,
            "Phase": phase,
            "Legal Balls": legal_balls,
            "Overs Bowled": round(legal_balls / 6.0, 1),
            "Runs Conceded": conceded,
            "Wickets": wickets,
            "Economy": econ,
            "Dots": dots,
            "Dot Ball %": dot_pct,
            "Boundaries Conceded": fours + sixes
        })

    return pd.DataFrame(rows)


def get_phase_wise_bowler_rankings(
    bbb: pd.DataFrame,
    phase: str = "Death Overs",  # 'Powerplay', 'Middle Overs', 'Death Overs'
    min_legal_balls: int = 36
) -> pd.DataFrame:
    """
    Ranks bowlers strictly within a selected phase (Powerplay 1-6, Middle 7-15, Death 16-20).
    """
    phase_map = {
        "Powerplay": "Powerplay",
        "Middle Overs": "Middle",
        "Death Overs": "Death"
    }
    p_val = phase_map.get(phase, phase)
    sub = bbb[bbb["phase"] == p_val]

    if sub.empty:
        return pd.DataFrame()

    agg = sub.groupby(["bowler_id", "bowler_name"]).agg(
        matches=("match_id", "nunique"),
        legal_balls=("is_legal_ball_bowler", "sum"),
        conceded=("bowler_conceded_runs", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dots=("is_dot_ball_bowler", "sum")
    ).reset_index()

    agg = agg[agg["legal_balls"] >= min_legal_balls].copy()
    if agg.empty:
        return pd.DataFrame()

    agg["overs"] = (agg["legal_balls"] / 6.0).round(1)
    agg["economy"] = (agg["conceded"] / agg["legal_balls"] * 6.0).round(2)
    agg["dot_pct"] = (agg["dots"] / agg["legal_balls"] * 100.0).round(1)
    agg["bowling_sr"] = np.where(
        agg["wickets"] > 0,
        (agg["legal_balls"] / agg["wickets"]).round(1),
        np.nan
    )

    return agg.sort_values(by=["economy", "wickets"], ascending=[True, False]).reset_index(drop=True)


def get_over_specific_bowler_rankings(
    bbb: pd.DataFrame,
    over_number: int = 20,
    min_legal_balls: int = 18
) -> pd.DataFrame:
    """
    Identifies the best bowlers for a specific single over number (e.g. Over 20 or Over 19).
    """
    sub = bbb[bbb["overid"] == over_number]
    if sub.empty:
        return pd.DataFrame()

    agg = sub.groupby(["bowler_id", "bowler_name"]).agg(
        matches=("match_id", "nunique"),
        legal_balls=("is_legal_ball_bowler", "sum"),
        conceded=("bowler_conceded_runs", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dots=("is_dot_ball_bowler", "sum")
    ).reset_index()

    agg = agg[agg["legal_balls"] >= min_legal_balls].copy()
    if agg.empty:
        return pd.DataFrame()

    agg["overs"] = (agg["legal_balls"] / 6.0).round(1)
    agg["economy"] = (agg["conceded"] / agg["legal_balls"] * 6.0).round(2)
    agg["dot_pct"] = (agg["dots"] / agg["legal_balls"] * 100.0).round(1)

    return agg.sort_values(by=["economy", "wickets"], ascending=[True, False]).reset_index(drop=True)
