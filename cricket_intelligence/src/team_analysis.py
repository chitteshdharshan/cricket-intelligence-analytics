"""
Team Performance Intelligence Module for Cricket Intelligence.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def get_team_kpis(matches_df: pd.DataFrame, bbb_df: pd.DataFrame, team_id: int) -> Dict[str, Any]:
    """
    Computes team-level metrics:
    Total matches, Wins, Losses, Win %, Average score, Highest score, Lowest score,
    Batting strike rate, Bowling economy, Wickets taken, Powerplay & Death over metrics.
    """
    # Matches played by team
    t_matches = matches_df[(matches_df["home_team_id"] == team_id) | (matches_df["away_team_id"] == team_id)]
    total_matches = len(t_matches)
    if total_matches == 0:
        return {}

    wins = int((t_matches["win_team_id"] == team_id).sum())
    # Exclude No Result / ties without winner from clean losses
    results_with_winner = t_matches[t_matches["win_team_id"].notna()]
    losses = int((results_with_winner["win_team_id"] != team_id).sum())
    win_pct = round(wins / total_matches * 100, 1) if total_matches > 0 else 0.0

    # Team batting deliveries
    bat_df = bbb_df[bbb_df["batting_team_id"] == team_id]
    inn_totals = bat_df.groupby(["match_id", "innings"])["total_delivery_runs"].sum()
    
    avg_score = round(inn_totals.mean(), 1) if len(inn_totals) > 0 else 0.0
    highest_score = int(inn_totals.max()) if len(inn_totals) > 0 else 0
    lowest_score = int(inn_totals.min()) if len(inn_totals) > 0 else 0

    total_bat_runs = int(bat_df["batter_runs"].sum())
    total_bat_balls = int(bat_df["is_legal_ball_batter"].sum())
    bat_sr = round(total_bat_runs / total_bat_balls * 100, 2) if total_bat_balls > 0 else 0.0

    # Team bowling deliveries
    bowl_df = bbb_df[bbb_df["bowling_team_id"] == team_id]
    total_bowl_runs = int(bowl_df["bowler_conceded_runs"].sum())
    total_bowl_balls = int(bowl_df["is_legal_ball_bowler"].sum())
    bowl_econ = round(total_bowl_runs / total_bowl_balls * 6.0, 2) if total_bowl_balls > 0 else 0.0
    wickets_taken = int(bowl_df["is_bowler_wicket"].sum())

    # Phase analysis
    pp_bat = bat_df[bat_df["phase"] == "Powerplay"]
    pp_run_rate = round(pp_bat["total_delivery_runs"].sum() / (pp_bat["is_legal_ball_batter"].sum() / 6.0), 2) if pp_bat["is_legal_ball_batter"].sum() > 0 else 0.0

    death_bat = bat_df[bat_df["phase"] == "Death"]
    death_run_rate = round(death_bat["total_delivery_runs"].sum() / (death_bat["is_legal_ball_batter"].sum() / 6.0), 2) if death_bat["is_legal_ball_batter"].sum() > 0 else 0.0

    return {
        "matches": total_matches,
        "wins": wins,
        "losses": losses,
        "win_pct": win_pct,
        "avg_score": avg_score,
        "highest_score": highest_score,
        "lowest_score": lowest_score,
        "bat_sr": bat_sr,
        "bowl_econ": bowl_econ,
        "wickets_taken": wickets_taken,
        "pp_run_rate": pp_run_rate,
        "death_run_rate": death_run_rate
    }

def get_team_player_contributions(bbb_df: pd.DataFrame, team_id: int) -> Dict[str, pd.DataFrame]:
    """Returns top batsmen and top bowlers for a team."""
    bat_sub = bbb_df[bbb_df["batting_team_id"] == team_id]
    bowl_sub = bbb_df[bbb_df["bowling_team_id"] == team_id]

    top_batsmen = bat_sub.groupby(["striker_id", "striker_name"]).agg(
        runs=("batter_runs", "sum"),
        balls=("is_legal_ball_batter", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        innings=("match_id", "nunique")
    ).reset_index()
    top_batsmen["SR"] = np.where(top_batsmen["balls"] > 0, (top_batsmen["runs"] / top_batsmen["balls"] * 100).round(2), 0.0)
    top_batsmen = top_batsmen.sort_values(by="runs", ascending=False).rename(columns={
        "striker_id": "Player ID", "striker_name": "Player", "runs": "Runs", "balls": "Balls", "fours": "4s", "sixes": "6s", "innings": "Innings"
    })

    top_bowlers = bowl_sub.groupby(["bowler_id", "bowler_name"]).agg(
        wickets=("is_bowler_wicket", "sum"),
        runs_conceded=("bowler_conceded_runs", "sum"),
        legal_balls=("is_legal_ball_bowler", "sum"),
        innings=("match_id", "nunique")
    ).reset_index()
    top_bowlers["Economy"] = np.where(top_bowlers["legal_balls"] > 0, (top_bowlers["runs_conceded"] / top_bowlers["legal_balls"] * 6.0).round(2), 0.0)
    top_bowlers["Overs"] = (top_bowlers["legal_balls"] // 6).astype(str) + "." + (top_bowlers["legal_balls"] % 6).astype(str)
    top_bowlers = top_bowlers.sort_values(by="wickets", ascending=False).rename(columns={
        "bowler_id": "Bowler ID", "bowler_name": "Bowler", "wickets": "Wickets", "runs_conceded": "Runs Conceded"
    })

    return {"batsmen": top_batsmen, "bowlers": top_bowlers}

