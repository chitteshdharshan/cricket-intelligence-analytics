"""
AI Cricket Strategy: Player Recommendation Engine
Calculates situation-specific rankings for batters and bowlers using verified historical deliveries.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def recommend_batters(
    bbb: pd.DataFrame,
    players_master: pd.DataFrame,
    situation: str = "powerplay",
    team_id: Optional[int] = None,
    min_balls: int = 30,
    top_n: int = 5,
    player_subset: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Ranks eligible batters for specific match situations.
    Situations supported:
      - 'powerplay' (overs 1-6)
      - 'middle' (overs 7-15)
      - 'death' / 'finisher' (overs 16-20)
      - 'against_spin' (bowling type == Spin)
      - 'against_pace' (bowling type == Pace)
      - 'boundary_hitters' (highest boundary percentage)
      - 'rebuilding' (high average, low dismissal frequency)
      - 'acceleration' (high strike rate, high boundary frequency)
      - 'chase' (2nd innings performance)
    """
    sub_bbb = bbb.copy()

    # Filter by situation
    situation = situation.lower()
    if situation in ["powerplay", "opener"]:
        sub_bbb = sub_bbb[sub_bbb["overid"] <= 6]
    elif situation in ["middle", "middle_order"]:
        sub_bbb = sub_bbb[(sub_bbb["overid"] >= 7) & (sub_bbb["overid"] <= 15)]
    elif situation in ["death", "finisher"]:
        sub_bbb = sub_bbb[sub_bbb["overid"] >= 16]
    elif situation == "against_spin":
        sub_bbb = sub_bbb[sub_bbb["pace_or_spin"] == "Spin"]
    elif situation == "against_pace":
        sub_bbb = sub_bbb[sub_bbb["pace_or_spin"] == "Pace"]
    elif situation == "chase":
        sub_bbb = sub_bbb[sub_bbb["innings"] == 2]

    # Filter player pool if restricted to team or subset
    if player_subset is not None:
        sub_bbb = sub_bbb[sub_bbb["striker_id"].isin(player_subset)]
    elif team_id is not None:
        sub_bbb = sub_bbb[sub_bbb["batting_team_id"] == team_id]

    if len(sub_bbb) == 0:
        return pd.DataFrame()

    # Aggregate core performance
    agg = sub_bbb.groupby(["striker_id", "striker_name"]).agg(
        runs=("batter_runs", "sum"),
        balls=("is_legal_ball_batter", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dots=("is_dot_ball_batter", "sum"),
        matches=("match_id", "nunique"),
        innings=("innings", "count")  # delivery count proxy
    ).reset_index()

    # Calculate outs where this specific striker was dismissed
    outs_df = sub_bbb[sub_bbb["dismissed_player_id"] == sub_bbb["striker_id"]].groupby("striker_id").size().to_dict()
    agg["outs"] = agg["striker_id"].map(outs_df).fillna(0).astype(int)

    # Calculate cricket metrics
    agg["strike_rate"] = np.where(agg["balls"] > 0, (agg["runs"] / agg["balls"] * 100).round(2), 0.0)
    agg["average"] = np.where(agg["outs"] > 0, (agg["runs"] / agg["outs"]).round(2), agg["runs"].astype(float))
    agg["boundary_pct"] = np.where(
        agg["runs"] > 0,
        (((agg["fours"] * 4) + (agg["sixes"] * 6)) / agg["runs"] * 100).round(2),
        0.0
    )
    agg["dot_pct"] = np.where(agg["balls"] > 0, (agg["dots"] / agg["balls"] * 100).round(1), 0.0)
    agg["balls_per_boundary"] = np.where(
        (agg["fours"] + agg["sixes"]) > 0,
        (agg["balls"] / (agg["fours"] + agg["sixes"])).round(1),
        999.0
    )
    agg["balls_per_dismissal"] = np.where(
        agg["outs"] > 0,
        (agg["balls"] / agg["outs"]).round(1),
        agg["balls"].astype(float)
    )

    # Attach style from players_master
    style_map = players_master.set_index("player_id")["batting_style"].to_dict()
    agg["batting_style"] = agg["striker_id"].map(style_map).fillna("Right Hand Bat")

    # Filter by minimum sample size
    effective_min_balls = min_balls
    filtered = agg[agg["balls"] >= effective_min_balls].copy()
    if len(filtered) < top_n and min_balls > 15:
        # Fallback to lower threshold if dataset is sparse
        effective_min_balls = max(15, min_balls // 2)
        filtered = agg[agg["balls"] >= effective_min_balls].copy()

    if len(filtered) == 0:
        filtered = agg[agg["balls"] >= 5].copy()

    # Ranking logic per situation
    if situation in ["death", "finisher", "acceleration"]:
        # Weighted score: high SR + boundary % with dot penalty
        filtered["rank_score"] = filtered["strike_rate"] * 0.7 + filtered["boundary_pct"] * 0.3
    elif situation in ["powerplay", "opener"]:
        filtered["rank_score"] = filtered["strike_rate"] * 0.5 + (100 - filtered["dot_pct"]) * 0.3 + filtered["boundary_pct"] * 0.2
    elif situation in ["against_spin", "middle", "middle_order"]:
        filtered["rank_score"] = filtered["average"] * 0.5 + filtered["strike_rate"] * 0.5
    elif situation == "rebuilding":
        filtered["rank_score"] = filtered["average"] * 0.6 + (100 - filtered["dot_pct"]) * 0.4
    elif situation == "boundary_hitters":
        filtered["rank_score"] = filtered["boundary_pct"] * 0.6 + filtered["strike_rate"] * 0.4
    else:
        filtered["rank_score"] = filtered["strike_rate"] * 0.5 + filtered["average"] * 0.5

    ranked = filtered.sort_values(by="rank_score", ascending=False).head(top_n).copy()
    ranked["sample_flag"] = np.where(ranked["balls"] < 40, "Low Sample (<40b)", "Verified Sample")

    return ranked.rename(columns={
        "striker_id": "player_id",
        "striker_name": "player_name"
    })


def recommend_bowlers(
    bbb: pd.DataFrame,
    players_master: pd.DataFrame,
    situation: str = "death",
    team_id: Optional[int] = None,
    min_balls: int = 24,
    top_n: int = 5,
    bowler_subset: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Ranks eligible bowlers for specific match situations.
    Situations supported:
      - 'powerplay' (overs 1-6)
      - 'middle' (overs 7-15)
      - 'death' (overs 16-20)
      - 'defending_low_total' (overall economy & dot-ball dominance)
      - 'wicket_taking' (lowest bowling strike rate)
      - 'spin_attack' (spinners only)
      - 'pace_attack' (pacers only)
    """
    sub_bbb = bbb.copy()

    # Filter by situation
    situation = situation.lower()
    if situation == "powerplay":
        sub_bbb = sub_bbb[sub_bbb["overid"] <= 6]
    elif situation == "middle":
        sub_bbb = sub_bbb[(sub_bbb["overid"] >= 7) & (sub_bbb["overid"] <= 15)]
    elif situation == "death":
        sub_bbb = sub_bbb[sub_bbb["overid"] >= 16]
    elif situation == "spin_attack":
        sub_bbb = sub_bbb[sub_bbb["pace_or_spin"] == "Spin"]
    elif situation == "pace_attack":
        sub_bbb = sub_bbb[sub_bbb["pace_or_spin"] == "Pace"]

    # Filter player pool if restricted to team or subset
    if bowler_subset is not None:
        sub_bbb = sub_bbb[sub_bbb["bowler_id"].isin(bowler_subset)]
    elif team_id is not None:
        sub_bbb = sub_bbb[sub_bbb["bowling_team_id"] == team_id]

    if len(sub_bbb) == 0:
        return pd.DataFrame()

    agg = sub_bbb.groupby(["bowler_id", "bowler_name"]).agg(
        runs=("bowler_conceded_runs", "sum"),
        balls=("is_legal_ball_bowler", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dots=("is_dot_ball_bowler", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        matches=("match_id", "nunique")
    ).reset_index()

    agg["economy"] = np.where(agg["balls"] > 0, (agg["runs"] / agg["balls"] * 6.0).round(2), 0.0)
    agg["dot_pct"] = np.where(agg["balls"] > 0, (agg["dots"] / agg["balls"] * 100).round(1), 0.0)
    agg["bowling_sr"] = np.where(agg["wickets"] > 0, (agg["balls"] / agg["wickets"]).round(1), 999.0)
    agg["overs"] = (agg["balls"] // 6).astype(str) + "." + (agg["balls"] % 6).astype(str)

    # Attach style from players_master
    style_map = players_master.set_index("player_id")["bowling_style"].to_dict()
    agg["bowling_style"] = agg["bowler_id"].map(style_map).fillna("Unknown")

    # Filter sample size
    filtered = agg[agg["balls"] >= min_balls].copy()
    if len(filtered) < top_n and min_balls > 18:
        filtered = agg[agg["balls"] >= max(12, min_balls // 2)].copy()

    if len(filtered) == 0:
        filtered = agg[agg["balls"] >= 6].copy()

    # Ranking logic
    if situation in ["death", "defending_low_total"]:
        # Primary: low economy, secondary: high dot %, tertiary: wickets
        filtered["rank_score"] = (20.0 - filtered["economy"]) * 0.6 + (filtered["dot_pct"] * 0.2) + (filtered["wickets"] * 0.2)
    elif situation == "wicket_taking":
        filtered["rank_score"] = (100.0 - np.clip(filtered["bowling_sr"], 0, 100)) * 0.6 + filtered["wickets"] * 0.4
    elif situation == "powerplay":
        filtered["rank_score"] = (15.0 - filtered["economy"]) * 0.5 + (filtered["dot_pct"] * 0.3) + (filtered["wickets"] * 0.2)
    else:
        filtered["rank_score"] = (18.0 - filtered["economy"]) * 0.5 + filtered["wickets"] * 0.3 + (filtered["dot_pct"] * 0.2)

    ranked = filtered.sort_values(by="rank_score", ascending=False).head(top_n).copy()
    ranked["sample_flag"] = np.where(ranked["balls"] < 36, "Low Sample (<36b)", "Verified Sample")

    return ranked.rename(columns={
        "bowler_id": "player_id",
        "bowler_name": "player_name"
    })

