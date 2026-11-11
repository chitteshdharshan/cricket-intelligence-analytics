"""
Handedness Performance Analysis Engine
Evaluates bowler performance against Left-Handed Batters (LHB) vs Right-Handed Batters (RHB),
and batter performance against Right-Arm vs Left-Arm bowling.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def get_bowler_vs_handedness_stats(
    bbb: pd.DataFrame,
    bowler_id: int
) -> pd.DataFrame:
    """
    Computes performance table for a bowler: Against LHB | Against RHB | Overall.
    """
    df = bbb[bbb["bowler_id"] == bowler_id].copy()
    if df.empty:
        return pd.DataFrame()

    categories = [
        ("Against LHB", df[df["batting_type"] == "LHB"]),
        ("Against RHB", df[df["batting_type"] == "RHB"]),
        ("Overall", df)
    ]

    metrics = [
        "Runs Conceded",
        "Legal Balls",
        "Overs",
        "Wickets",
        "Economy Rate",
        "Dot Ball %",
        "Boundary %",
        "Bowling Strike Rate"
    ]

    result_dict: Dict[str, List[Any]] = {"Statistic": metrics}

    for col_label, sub in categories:
        legal_balls = int(sub["is_legal_ball_bowler"].sum())
        conceded = int(sub["bowler_conceded_runs"].sum())
        wickets = int(sub["is_bowler_wicket"].sum())
        dots = int(sub["is_dot_ball_bowler"].sum())
        fours = int(sub["is_four"].sum())
        sixes = int(sub["is_six"].sum())

        overs = round(legal_balls / 6.0, 1)
        econ = round((conceded / legal_balls * 6.0) if legal_balls > 0 else 0.0, 2)
        dot_pct = round((dots / legal_balls * 100.0) if legal_balls > 0 else 0.0, 1)
        boundary_pct = round(((fours + sixes) / legal_balls * 100.0) if legal_balls > 0 else 0.0, 1)
        bowling_sr = round((legal_balls / wickets) if wickets > 0 else 0.0, 1)

        result_dict[col_label] = [
            conceded,
            legal_balls,
            overs,
            wickets,
            econ,
            f"{dot_pct}%",
            f"{boundary_pct}%",
            bowling_sr if wickets > 0 else "N/A"
        ]

    return pd.DataFrame(result_dict)


def get_all_bowlers_handedness_rankings(
    bbb: pd.DataFrame,
    target_hand: str = "LHB",  # 'LHB' or 'RHB'
    min_legal_balls: int = 36,
    filter_mode: str = "Best Economy"  # 'Best Economy', 'Most Wickets', 'Lower Economy vs Hand'
) -> pd.DataFrame:
    """
    Ranks all bowlers against a specific batter handedness (LHB or RHB).
    """
    df = bbb[bbb["batting_type"] == target_hand]
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["bowler_id", "bowler_name"]).agg(
        matches=("match_id", "nunique"),
        legal_balls=("is_legal_ball_bowler", "sum"),
        conceded=("bowler_conceded_runs", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dots=("is_dot_ball_bowler", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum")
    ).reset_index()

    agg = agg[agg["legal_balls"] >= min_legal_balls].copy()
    if agg.empty:
        return pd.DataFrame()

    agg["overs"] = (agg["legal_balls"] / 6.0).round(1)
    agg["economy"] = (agg["conceded"] / agg["legal_balls"] * 6.0).round(2)
    agg["dot_pct"] = (agg["dots"] / agg["legal_balls"] * 100.0).round(1)
    boundaries = agg["fours"] + agg["sixes"]
    agg["boundary_pct"] = (boundaries / agg["legal_balls"] * 100.0).round(1)
    agg["bowling_sr"] = np.where(
        agg["wickets"] > 0,
        (agg["legal_balls"] / agg["wickets"]).round(1),
        np.nan
    )

    if filter_mode == "Most Wickets":
        return agg.sort_values(by=["wickets", "economy"], ascending=[False, True]).reset_index(drop=True)
    return agg.sort_values(by=["economy", "wickets"], ascending=[True, False]).reset_index(drop=True)


def get_batter_vs_bowling_arm_stats(
    bbb: pd.DataFrame,
    batter_id: int
) -> pd.DataFrame:
    """
    Computes performance of a batter against Right-Arm vs Left-Arm bowling,
    including pace and spin splits.
    """
    df = bbb[bbb["striker_id"] == batter_id].copy()
    if df.empty:
        return pd.DataFrame()

    # Classify bowling arm
    df["arm"] = np.where(df["bowling_type"].str.contains("Left", case=False, na=False), "Left-Arm", "Right-Arm")

    categories = [
        ("Right-Arm Pace", df[(df["arm"] == "Right-Arm") & (df["pace_or_spin"] == "Pace")]),
        ("Left-Arm Pace", df[(df["arm"] == "Left-Arm") & (df["pace_or_spin"] == "Pace")]),
        ("Right-Arm Spin", df[(df["arm"] == "Right-Arm") & (df["pace_or_spin"] == "Spin")]),
        ("Left-Arm Spin", df[(df["arm"] == "Left-Arm") & (df["pace_or_spin"] == "Spin")]),
        ("All Right-Arm", df[df["arm"] == "Right-Arm"]),
        ("All Left-Arm", df[df["arm"] == "Left-Arm"]),
        ("Overall", df)
    ]

    rows = []
    for cat_name, sub in categories:
        runs = int(sub["batter_runs"].sum())
        balls = int(sub["is_legal_ball_batter"].sum())
        fours = int(sub["is_four"].sum())
        sixes = int(sub["is_six"].sum())
        dots = int(sub["is_dot_ball_batter"].sum())
        outs = int(((sub["dismissed_player_id"] == batter_id) & (sub["is_wicket"] == 1)).sum())

        sr = round((runs / balls * 100.0) if balls > 0 else 0.0, 2)
        avg = round((runs / outs) if outs > 0 else float(runs), 2)
        boundaries = fours + sixes
        bpb = round((balls / boundaries) if boundaries > 0 else 0.0, 2)
        dot_pct = round((dots / balls * 100.0) if balls > 0 else 0.0, 1)

        rows.append({
            "Bowling Category": cat_name,
            "Runs": runs,
            "Balls Faced": balls,
            "Strike Rate": sr,
            "Batting Average": avg,
            "Fours": fours,
            "Sixes": sixes,
            "Balls Per Boundary": bpb if boundaries > 0 else np.nan,
            "Dot Ball %": dot_pct,
            "Dismissals": outs
        })

    return pd.DataFrame(rows)
