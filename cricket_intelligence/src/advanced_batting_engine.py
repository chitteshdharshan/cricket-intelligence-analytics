"""
Advanced Batting Performance Analysis Engine
Provides granular strike rate filtering, balls per boundary analysis,
and pace vs spin performance breakdowns.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def filter_bbb_deliveries(
    bbb: pd.DataFrame,
    format_name: Optional[str] = None,
    team_name: Optional[str] = None,
    season: Optional[str] = None,
    comp_name: Optional[str] = None,
    phase: Optional[str] = None,
    bowling_pace_or_spin: Optional[str] = None
) -> pd.DataFrame:
    """Filters ball-by-ball dataframe by common match and situational filters."""
    df = bbb
    if format_name and format_name != "All Formats" and "match_format" in df.columns:
        df = df[df["match_format"] == format_name]
    if season and season != "All Seasons" and "season" in df.columns:
        df = df[df["season"] == season]
    if comp_name and comp_name != "All Tournaments" and "comp_name" in df.columns:
        df = df[df["comp_name"] == comp_name]
    if phase and phase != "All Overs" and "phase" in df.columns:
        df = df[df["phase"] == phase]
    if bowling_pace_or_spin and bowling_pace_or_spin != "All" and "pace_or_spin" in df.columns:
        df = df[df["pace_or_spin"] == bowling_pace_or_spin]
    if team_name and team_name != "All Teams":
        df = df[(df["home_team"] == team_name) | (df["away_team"] == team_name)]
    return df


def calculate_advanced_batting_stats(
    bbb: pd.DataFrame,
    min_balls_faced: int = 30,
    sr_mode: str = "All",  # 'All', 'Above', 'Below', 'Between'
    sr_val1: float = 0.0,
    sr_val2: float = 300.0,
    bpb_mode: str = "All",  # 'All', 'Below', 'Above', 'Between'
    bpb_val1: float = 0.0,
    bpb_val2: float = 30.0,
    bowling_type: str = "All"  # 'All', 'Pace', 'Spin'
) -> pd.DataFrame:
    """
    Calculates advanced batting metrics across all batters including strike rate,
    balls per four, balls per six, balls per boundary, dot ball %, and boundary %.
    """
    df = bbb
    if bowling_type in ["Pace", "Spin"]:
        df = df[df["pace_or_spin"] == bowling_type]

    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["striker_id", "striker_name"]).agg(
        matches=("match_id", "nunique"),
        innings=("match_id", lambda s: len(s.unique())),
        runs=("batter_runs", "sum"),
        balls=("is_legal_ball_batter", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dots=("is_dot_ball_batter", "sum"),
        dismissals=("is_wicket", lambda w: ((df.loc[w.index, "dismissed_player_id"] == df.loc[w.index, "striker_id"]) & (w == 1)).sum())
    ).reset_index()

    agg = agg[agg["balls"] >= min_balls_faced].copy()
    if agg.empty:
        return pd.DataFrame()

    # Derived Metrics
    agg["strike_rate"] = (agg["runs"] / agg["balls"] * 100.0).round(2)
    agg["batting_average"] = np.where(
        agg["dismissals"] > 0,
        (agg["runs"] / agg["dismissals"]).round(2),
        agg["runs"].astype(float)
    )
    agg["boundaries_total"] = agg["fours"] + agg["sixes"]
    agg["boundary_runs"] = (agg["fours"] * 4) + (agg["sixes"] * 6)
    agg["boundary_pct"] = np.where(
        agg["runs"] > 0,
        (agg["boundary_runs"] / agg["runs"] * 100.0).round(1),
        0.0
    )
    agg["dot_pct"] = (agg["dots"] / agg["balls"] * 100.0).round(1)

    # Balls per boundary calculations
    agg["balls_per_four"] = np.where(
        agg["fours"] > 0,
        (agg["balls"] / agg["fours"]).round(2),
        np.nan
    )
    agg["balls_per_six"] = np.where(
        agg["sixes"] > 0,
        (agg["balls"] / agg["sixes"]).round(2),
        np.nan
    )
    agg["balls_per_boundary"] = np.where(
        agg["boundaries_total"] > 0,
        (agg["balls"] / agg["boundaries_total"]).round(2),
        np.nan
    )

    # Apply Strike Rate Filter
    if sr_mode == "Above":
        agg = agg[agg["strike_rate"] >= sr_val1]
    elif sr_mode == "Below":
        agg = agg[agg["strike_rate"] <= sr_val1]
    elif sr_mode == "Between":
        low, high = min(sr_val1, sr_val2), max(sr_val1, sr_val2)
        agg = agg[(agg["strike_rate"] >= low) & (agg["strike_rate"] <= high)]

    # Apply Balls Per Boundary Filter
    if bpb_mode == "Below":
        agg = agg[agg["balls_per_boundary"].fillna(999) <= bpb_val1]
    elif bpb_mode == "Above":
        agg = agg[agg["balls_per_boundary"].fillna(0) >= bpb_val1]
    elif bpb_mode == "Between":
        low, high = min(bpb_val1, bpb_val2), max(bpb_val1, bpb_val2)
        agg = agg[(agg["balls_per_boundary"] >= low) & (agg["balls_per_boundary"] <= high)]

    return agg.sort_values(by=["runs", "strike_rate"], ascending=[False, False]).reset_index(drop=True)


def get_pace_vs_spin_batter_breakdown(
    bbb: pd.DataFrame,
    batter_id: int
) -> pd.DataFrame:
    """
    Generates a comparative breakdown table for a single batter:
    Fast/Pace vs Spin vs Overall.
    """
    b_df = bbb[bbb["striker_id"] == batter_id]
    if b_df.empty:
        return pd.DataFrame()

    categories = [
        ("Fast / Pace", b_df[b_df["pace_or_spin"] == "Pace"]),
        ("Spin", b_df[b_df["pace_or_spin"] == "Spin"]),
        ("Overall", b_df)
    ]

    rows = []
    for cat_name, sub_df in categories:
        runs = int(sub_df["batter_runs"].sum())
        balls = int(sub_df["is_legal_ball_batter"].sum())
        fours = int(sub_df["is_four"].sum())
        sixes = int(sub_df["is_six"].sum())
        dots = int(sub_df["is_dot_ball_batter"].sum())
        dismissals = int(((sub_df["dismissed_player_id"] == batter_id) & (sub_df["is_wicket"] == 1)).sum())

        sr = round((runs / balls * 100.0) if balls > 0 else 0.0, 2)
        avg = round((runs / dismissals) if dismissals > 0 else float(runs), 2)
        boundaries = fours + sixes
        bpb = round((balls / boundaries) if boundaries > 0 else 0.0, 2)
        dot_pct = round((dots / balls * 100.0) if balls > 0 else 0.0, 1)

        rows.append({
            "Bowling Type": cat_name,
            "Runs": runs,
            "Balls Faced": balls,
            "Strike Rate": sr,
            "Batting Average": avg,
            "Fours": fours,
            "Sixes": sixes,
            "Balls Per Boundary": bpb if boundaries > 0 else "N/A",
            "Dot Ball %": dot_pct,
            "Dismissals": dismissals
        })

    return pd.DataFrame(rows)


def get_pace_spin_comparative_table(
    bbb: pd.DataFrame,
    min_balls_each: int = 30,
    filter_criteria: Optional[str] = None
) -> pd.DataFrame:
    """
    Compares all batters against Pace vs Spin with filter options:
    - Higher strike rate against Pace
    - Higher strike rate against Spin
    - Better boundary frequency against Pace
    - Better boundary frequency against Spin
    - Lower dismissal frequency against Pace / Spin
    """
    pace_df = bbb[bbb["pace_or_spin"] == "Pace"].groupby(["striker_id", "striker_name"]).agg(
        pace_runs=("batter_runs", "sum"),
        pace_balls=("is_legal_ball_batter", "sum"),
        pace_4s=("is_four", "sum"),
        pace_6s=("is_six", "sum"),
        pace_outs=("is_wicket", lambda w: ((bbb.loc[w.index, "dismissed_player_id"] == bbb.loc[w.index, "striker_id"]) & (w == 1)).sum())
    ).reset_index()

    spin_df = bbb[bbb["pace_or_spin"] == "Spin"].groupby(["striker_id", "striker_name"]).agg(
        spin_runs=("batter_runs", "sum"),
        spin_balls=("is_legal_ball_batter", "sum"),
        spin_4s=("is_four", "sum"),
        spin_6s=("is_six", "sum"),
        spin_outs=("is_wicket", lambda w: ((bbb.loc[w.index, "dismissed_player_id"] == bbb.loc[w.index, "striker_id"]) & (w == 1)).sum())
    ).reset_index()

    merged = pace_df.merge(spin_df, on=["striker_id", "striker_name"], how="inner")
    merged = merged[(merged["pace_balls"] >= min_balls_each) & (merged["spin_balls"] >= min_balls_each)].copy()

    if merged.empty:
        return pd.DataFrame()

    merged["pace_sr"] = (merged["pace_runs"] / merged["pace_balls"] * 100.0).round(2)
    merged["spin_sr"] = (merged["spin_runs"] / merged["spin_balls"] * 100.0).round(2)
    merged["pace_bpb"] = np.where(
        (merged["pace_4s"] + merged["pace_6s"]) > 0,
        (merged["pace_balls"] / (merged["pace_4s"] + merged["pace_6s"])).round(2),
        999.0
    )
    merged["spin_bpb"] = np.where(
        (merged["spin_4s"] + merged["spin_6s"]) > 0,
        (merged["spin_balls"] / (merged["spin_4s"] + merged["spin_6s"])).round(2),
        999.0
    )
    merged["pace_bpd"] = np.where(
        merged["pace_outs"] > 0,
        (merged["pace_balls"] / merged["pace_outs"]).round(1),
        merged["pace_balls"].astype(float)
    )
    merged["spin_bpd"] = np.where(
        merged["spin_outs"] > 0,
        (merged["spin_balls"] / merged["spin_outs"]).round(1),
        merged["spin_balls"].astype(float)
    )

    if filter_criteria == "Higher SR vs Pace":
        merged = merged[merged["pace_sr"] > merged["spin_sr"]]
    elif filter_criteria == "Higher SR vs Spin":
        merged = merged[merged["spin_sr"] > merged["pace_sr"]]
    elif filter_criteria == "More Frequent Boundaries vs Pace":
        merged = merged[merged["pace_bpb"] < merged["spin_bpb"]]
    elif filter_criteria == "More Frequent Boundaries vs Spin":
        merged = merged[merged["spin_bpb"] < merged["pace_bpb"]]
    elif filter_criteria == "Lower Dismissal Frequency vs Pace":
        merged = merged[merged["pace_bpd"] > merged["spin_bpd"]]
    elif filter_criteria == "Lower Dismissal Frequency vs Spin":
        merged = merged[merged["spin_bpd"] > merged["pace_bpd"]]

    return merged.sort_values(by=["pace_runs", "spin_runs"], ascending=[False, False]).reset_index(drop=True)
