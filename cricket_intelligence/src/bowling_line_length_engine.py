"""
Bowling Delivery Style & Yorker Performance Engine
Evaluates verified yorkers, bouncers, full tosses, and trajectory classifications,
and provides transparent disclosures regarding Hawk-Eye line and length telemetry.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


YORKER_STYLES = ["Yorker", "Wide Yorker"]
BOUNCER_STYLES = ["Bouncer", "Slow Bouncer"]
FULL_TOSS_STYLES = ["Full Toss", "Beamer"]


def calculate_yorker_performance_stats(
    bbb: pd.DataFrame,
    min_total_balls: int = 60,
    min_yorkers: int = 5,
    max_yorker_economy: Optional[float] = None,
    min_yorker_pct: Optional[float] = None
) -> pd.DataFrame:
    """
    Evaluates bowler proficiency with verified yorkers (including wide yorkers).
    Calculates execution accuracy, runs conceded, wickets, economy, and dot ball %.
    """
    total_bowler = bbb.groupby(["bowler_id", "bowler_name"]).agg(
        total_balls=("is_legal_ball_bowler", "sum")
    ).reset_index()

    total_bowler = total_bowler[total_bowler["total_balls"] >= min_total_balls]
    if total_bowler.empty:
        return pd.DataFrame()

    eligible_bids = set(total_bowler["bowler_id"])
    yorker_df = bbb[(bbb["bowler_id"].isin(eligible_bids)) & (bbb["delivery_style"].isin(YORKER_STYLES))]

    if yorker_df.empty:
        return pd.DataFrame()

    agg = yorker_df.groupby(["bowler_id", "bowler_name"]).agg(
        yorker_balls=("is_legal_ball_bowler", "sum"),
        yorker_runs=("bowler_conceded_runs", "sum"),
        yorker_wickets=("is_bowler_wicket", "sum"),
        yorker_dots=("is_dot_ball_bowler", "sum"),
        yorker_fours=("is_four", "sum"),
        yorker_sixes=("is_six", "sum")
    ).reset_index()

    agg = agg[agg["yorker_balls"] >= min_yorkers].copy()
    if agg.empty:
        return pd.DataFrame()

    merged = agg.merge(total_bowler, on=["bowler_id", "bowler_name"])
    merged["yorker_pct"] = (merged["yorker_balls"] / merged["total_balls"] * 100.0).round(1)
    merged["yorker_economy"] = (merged["yorker_runs"] / merged["yorker_balls"] * 6.0).round(2)
    merged["yorker_dot_pct"] = (merged["yorker_dots"] / merged["yorker_balls"] * 100.0).round(1)
    boundaries = merged["yorker_fours"] + merged["yorker_sixes"]
    merged["yorker_boundary_pct"] = (boundaries / merged["yorker_balls"] * 100.0).round(1)

    if max_yorker_economy is not None:
        merged = merged[merged["yorker_economy"] <= max_yorker_economy]
    if min_yorker_pct is not None:
        merged = merged[merged["yorker_pct"] >= min_yorker_pct]

    return merged.sort_values(by=["yorker_balls", "yorker_wickets", "yorker_economy"], ascending=[False, False, True]).reset_index(drop=True)


def get_bowler_delivery_style_distribution(
    bbb: pd.DataFrame,
    bowler_id: int
) -> pd.DataFrame:
    """
    Breaks down every verified delivery style used by a specific bowler:
    Deliveries, % of Total, Runs Conceded, Wickets, Economy, Dot %, Boundary %.
    """
    df = bbb[bbb["bowler_id"] == bowler_id].copy()
    if df.empty:
        return pd.DataFrame()

    total_legal = df["is_legal_ball_bowler"].sum()
    if total_legal == 0:
        return pd.DataFrame()

    rows = []
    for style, sgroup in df.groupby("delivery_style"):
        if style == "Unknown / Unclassified" or pd.isna(style):
            continue
        balls = int(sgroup["is_legal_ball_bowler"].sum())
        if balls == 0:
            continue
        pct = round(balls / total_legal * 100.0, 1)
        runs = int(sgroup["bowler_conceded_runs"].sum())
        wickets = int(sgroup["is_bowler_wicket"].sum())
        dots = int(sgroup["is_dot_ball_bowler"].sum())
        fours = int(sgroup["is_four"].sum())
        sixes = int(sgroup["is_six"].sum())
        econ = round(runs / balls * 6.0, 2)
        dot_pct = round(dots / balls * 100.0, 1)
        boundary_pct = round((fours + sixes) / balls * 100.0, 1)

        rows.append({
            "Delivery Classification": style,
            "Deliveries": balls,
            "% of Bowler Total": pct,
            "Runs Conceded": runs,
            "Wickets": wickets,
            "Economy": econ,
            "Dot Ball %": dot_pct,
            "Boundary %": boundary_pct
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(by="Deliveries", ascending=False).reset_index(drop=True)


def get_data_transparency_disclosure() -> Dict[str, str]:
    """
    Returns verified status of delivery tracking telemetry in the current dataset.
    """
    return {
        "verified_features": (
            "Verified delivery style classifications are fully tracked and analyzed: "
            "Yorkers, Wide Yorkers, Full Tosses, Beamers, Bouncers, Slow Bouncers, "
            "Inswing, Outswing, Seam, Cutters, Googly, and Orthodox spin."
        ),
        "limitations": (
            "Standard 6-zone Hawk-Eye pitch-length telemetry (Good length, Back of length, Short, Full "
            "measured in meters from the stumps) requires ball-tracking optical camera data, which is "
            "not recorded in standard ball-by-ball scorecards. In strict compliance with WicketIQ data "
            "integrity principles, missing camera tracking coordinates are honestly documented rather than guessed."
        )
    }
