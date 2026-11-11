"""
Advanced Bowling Performance Analysis Engine
Provides granular economy rate filtering, conceded runs calculations,
wicket impact, and bowling discipline metrics.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def calculate_advanced_bowling_stats(
    bbb: pd.DataFrame,
    min_overs_bowled: float = 10.0,
    econ_mode: str = "All",  # 'All', 'Below', 'Above', 'Between'
    econ_val1: float = 0.0,
    econ_val2: float = 12.0,
    bowling_type: str = "All",  # 'All', 'Pace', 'Spin'
    phase: str = "All Overs",  # 'All Overs', 'Powerplay', 'Middle', 'Death'
    bowling_style: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculates advanced bowling metrics across all bowlers including economy rate,
    dot ball %, boundary conceded %, bowling strike rate, and average.
    """
    df = bbb
    if bowling_type in ["Pace", "Spin"]:
        df = df[df["pace_or_spin"] == bowling_type]
    if phase != "All Overs" and "phase" in df.columns:
        df = df[df["phase"] == phase]
    if bowling_style and bowling_style != "All Styles" and "bowling_type" in df.columns:
        df = df[df["bowling_type"] == bowling_style]

    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["bowler_id", "bowler_name"]).agg(
        matches=("match_id", "nunique"),
        innings=("match_id", lambda s: len(s.unique())),
        legal_balls=("is_legal_ball_bowler", "sum"),
        conceded_runs=("bowler_conceded_runs", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dots=("is_dot_ball_bowler", "sum"),
        fours_conceded=("is_four", "sum"),
        sixes_conceded=("is_six", "sum"),
        bowling_style=("bowling_type", "first")
    ).reset_index()

    agg["overs"] = (agg["legal_balls"] / 6.0).round(1)
    agg = agg[agg["overs"] >= min_overs_bowled].copy()

    if agg.empty:
        return pd.DataFrame()

    # Derived Metrics
    agg["economy"] = np.where(
        agg["legal_balls"] > 0,
        (agg["conceded_runs"] / agg["legal_balls"] * 6.0).round(2),
        0.0
    )
    agg["dot_pct"] = np.where(
        agg["legal_balls"] > 0,
        (agg["dots"] / agg["legal_balls"] * 100.0).round(1),
        0.0
    )
    boundaries = agg["fours_conceded"] + agg["sixes_conceded"]
    agg["boundary_pct"] = np.where(
        agg["legal_balls"] > 0,
        (boundaries / agg["legal_balls"] * 100.0).round(1),
        0.0
    )
    agg["bowling_sr"] = np.where(
        agg["wickets"] > 0,
        (agg["legal_balls"] / agg["wickets"]).round(1),
        np.nan
    )
    agg["bowling_avg"] = np.where(
        agg["wickets"] > 0,
        (agg["conceded_runs"] / agg["wickets"]).round(2),
        np.nan
    )

    # Apply Economy Filter
    if econ_mode == "Below":
        agg = agg[agg["economy"] <= econ_val1]
    elif econ_mode == "Above":
        agg = agg[agg["economy"] >= econ_val1]
    elif econ_mode == "Between":
        low, high = min(econ_val1, econ_val2), max(econ_val1, econ_val2)
        agg = agg[(agg["economy"] >= low) & (agg["economy"] <= high)]

    return agg.sort_values(by=["wickets", "economy"], ascending=[False, True]).reset_index(drop=True)

