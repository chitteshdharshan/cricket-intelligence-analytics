"""
Bowling Performance & Intelligence Module for Cricket Intelligence.
Computes canonical bowling metrics, phases, handedness split, and delivery effectiveness.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def apply_bowler_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Applies common filters for bowler analysis."""
    if not filters:
        return df
    filtered = df
    if "comp_id" in filters and filters["comp_id"] is not None and filters["comp_id"] != "All":
        filtered = filtered[filtered["comp_id"] == int(filters["comp_id"])]
    if "season" in filters and filters["season"] is not None and filters["season"] != "All":
        filtered = filtered[filtered["season"] == str(filters["season"])]
    if "team_id" in filters and filters["team_id"] is not None and filters["team_id"] != "All":
        filtered = filtered[filtered["bowling_team_id"] == int(filters["team_id"])]
    if "match_id" in filters and filters["match_id"] is not None and filters["match_id"] != "All":
        filtered = filtered[filtered["match_id"] == int(filters["match_id"])]
    if "start_date" in filters and filters["start_date"] is not None:
        filtered = filtered[pd.to_datetime(filtered["match_date"]) >= pd.to_datetime(filters["start_date"])]
    if "end_date" in filters and filters["end_date"] is not None:
        filtered = filtered[pd.to_datetime(filtered["match_date"]) <= pd.to_datetime(filters["end_date"])]
    return filtered

def get_bowler_kpis(df: pd.DataFrame, bowler_id: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Computes all standard bowling KPIs for a bowler:
    Wickets, Legal balls, Overs, Runs conceded, Economy, Bowling average, Strike rate, Dot %, Boundaries conceded.
    """
    b_df = apply_bowler_filters(df[df["bowler_id"] == bowler_id], filters)
    if len(b_df) == 0:
        return {
            "wickets": 0, "legal_balls": 0, "overs": "0.0", "runs_conceded": 0,
            "economy": 0.0, "average": 0.0, "strike_rate": 0.0, "dot_balls": 0,
            "dot_pct": 0.0, "fours_conceded": 0, "sixes_conceded": 0,
            "matches": 0, "innings": 0, "best_figures": "0/0 (0.0 ov)"
        }

    legal_balls = int(b_df["is_legal_ball_bowler"].sum())
    runs_conceded = int(b_df["bowler_conceded_runs"].sum())
    wickets = int(b_df["is_bowler_wicket"].sum())
    dot_balls = int(b_df["is_dot_ball_bowler"].sum())
    fours = int(b_df["is_four"].sum())
    sixes = int(b_df["is_six"].sum())

    overs_str = f"{legal_balls // 6}.{legal_balls % 6}"
    economy = round((runs_conceded / legal_balls * 6.0), 2) if legal_balls > 0 else 0.0
    average = round(runs_conceded / wickets, 2) if wickets > 0 else 0.0
    strike_rate = round(legal_balls / wickets, 2) if wickets > 0 else 0.0
    dot_pct = round((dot_balls / legal_balls * 100), 2) if legal_balls > 0 else 0.0

    matches_count = b_df["match_id"].nunique()
    innings_count = b_df.groupby(["match_id", "innings"]).ngroups

    # Best bowling figures in an innings
    inn_figures = b_df.groupby(["match_id", "innings"]).agg(
        w=("is_bowler_wicket", "sum"),
        r=("bowler_conceded_runs", "sum"),
        b=("is_legal_ball_bowler", "sum")
    ).reset_index().sort_values(by=["w", "r"], ascending=[False, True])

    if len(inn_figures) > 0:
        top_f = inn_figures.iloc[0]
        ov_f = f"{top_f['b'] // 6}.{top_f['b'] % 6}"
        best_figures = f"{top_f['w']}/{top_f['r']} ({ov_f} ov)"
    else:
        best_figures = "0/0"

    return {
        "wickets": wickets,
        "legal_balls": legal_balls,
        "overs": overs_str,
        "runs_conceded": runs_conceded,
        "economy": economy,
        "average": average,
        "strike_rate": strike_rate,
        "dot_balls": dot_balls,
        "dot_pct": dot_pct,
        "fours_conceded": fours,
        "sixes_conceded": sixes,
        "matches": matches_count,
        "innings": innings_count,
        "best_figures": best_figures
    }

def get_bowler_phase_stats(df: pd.DataFrame, bowler_id: int, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Computes bowler performance across Powerplay, Middle, and Death phases."""
    b_df = apply_bowler_filters(df[df["bowler_id"] == bowler_id], filters)
    if len(b_df) == 0:
        return pd.DataFrame()

    phases = ["Powerplay", "Middle", "Death"]
    summary = []
    for ph in phases:
        sub = b_df[b_df["phase"] == ph]
        balls = int(sub["is_legal_ball_bowler"].sum())
        runs = int(sub["bowler_conceded_runs"].sum())
        wkts = int(sub["is_bowler_wicket"].sum())
        dots = int(sub["is_dot_ball_bowler"].sum())
        fours = int(sub["is_four"].sum())
        sixes = int(sub["is_six"].sum())

        overs = f"{balls // 6}.{balls % 6}"
        econ = round(runs / balls * 6.0, 2) if balls > 0 else 0.0
        avg = round(runs / wkts, 2) if wkts > 0 else 0.0
        sr = round(balls / wkts, 2) if wkts > 0 else 0.0
        dot_pct = round(dots / balls * 100, 2) if balls > 0 else 0.0

        summary.append({
            "Phase": ph,
            "Overs": overs,
            "Legal Balls": balls,
            "Runs Conceded": runs,
            "Wickets": wkts,
            "Economy": econ,
            "Average": avg,
            "Strike Rate": sr,
            "Dot %": dot_pct,
            "Fours Conceded": fours,
            "Sixes Conceded": sixes
        })
    return pd.DataFrame(summary)

def get_bowler_handedness_split(df: pd.DataFrame, bowler_id: int, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Computes bowler performance vs Right Hand Batsmen (RHB) and Left Hand Batsmen (LHB)."""
    b_df = apply_bowler_filters(df[df["bowler_id"] == bowler_id], filters)
    if len(b_df) == 0:
        return pd.DataFrame()

    res = []
    for hand in ["RHB", "LHB"]:
        sub = b_df[b_df["batting_type"] == hand]
        balls = int(sub["is_legal_ball_bowler"].sum())
        runs = int(sub["bowler_conceded_runs"].sum())
        wkts = int(sub["is_bowler_wicket"].sum())
        dots = int(sub["is_dot_ball_bowler"].sum())
        econ = round(runs / balls * 6.0, 2) if balls > 0 else 0.0
        avg = round(runs / wkts, 2) if wkts > 0 else 0.0
        sr = round(balls / wkts, 2) if wkts > 0 else 0.0
        dot_pct = round(dots / balls * 100, 2) if balls > 0 else 0.0

        res.append({
            "Batsman Hand": "Right-Hand Bat (RHB)" if hand == "RHB" else "Left-Hand Bat (LHB)",
            "Legal Balls": balls,
            "Overs": f"{balls // 6}.{balls % 6}",
            "Runs Conceded": runs,
            "Wickets": wkts,
            "Economy": econ,
            "Average": avg,
            "Strike Rate": sr,
            "Dot %": dot_pct
        })
    return pd.DataFrame(res)

def get_bowler_dismissal_types(df: pd.DataFrame, bowler_id: int) -> pd.DataFrame:
    """Returns breakdown of bowler's wickets by dismissal type."""
    b_df = df[(df["bowler_id"] == bowler_id) & (df["is_bowler_wicket"] == True)]
    if len(b_df) == 0:
        return pd.DataFrame()

    counts = b_df["dismissal_type"].value_counts().reset_index()
    counts.columns = ["Dismissal Type", "Wickets"]
    counts["Percentage"] = (counts["Wickets"] / counts["Wickets"].sum() * 100).round(1)
    return counts

def get_bowler_delivery_styles(df: pd.DataFrame, bowler_id: int) -> pd.DataFrame:
    """Computes stats per delivery style (Inswing, Yorker, Bouncer, Seamed In, etc.)."""
    b_df = df[df["bowler_id"] == bowler_id]
    if len(b_df) == 0:
        return pd.DataFrame()

    grouped = b_df.groupby("delivery_style").agg(
        balls=("is_legal_ball_bowler", "sum"),
        runs=("bowler_conceded_runs", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dots=("is_dot_ball_bowler", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum")
    ).reset_index()

    grouped["Economy"] = np.where(grouped["balls"] > 0, (grouped["runs"] / grouped["balls"] * 6.0).round(2), 0.0)
    grouped["Strike Rate"] = np.where(grouped["wickets"] > 0, (grouped["balls"] / grouped["wickets"]).round(2), 0.0)
    grouped["Dot %"] = np.where(grouped["balls"] > 0, (grouped["dots"] / grouped["balls"] * 100).round(2), 0.0)
    grouped = grouped.rename(columns={"delivery_style": "Delivery Style", "balls": "Balls", "runs": "Runs", "wickets": "Wickets", "dots": "Dots"})
    return grouped.sort_values(by="Balls", ascending=False)

