"""
Batting Performance & Ball-by-Ball Intelligence Module.
Implements exact cricket calculations, over-by-over analysis, and delivery explorer.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

def apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Applies common filters (comp_id, season, team_id, match_id, date range) to ball-by-ball dataframe."""
    if not filters:
        return df
    filtered = df
    if "comp_id" in filters and filters["comp_id"] is not None and filters["comp_id"] != "All":
        filtered = filtered[filtered["comp_id"] == int(filters["comp_id"])]
    if "season" in filters and filters["season"] is not None and filters["season"] != "All":
        filtered = filtered[filtered["season"] == str(filters["season"])]
    if "team_id" in filters and filters["team_id"] is not None and filters["team_id"] != "All":
        filtered = filtered[filtered["batting_team_id"] == int(filters["team_id"])]
    if "match_id" in filters and filters["match_id"] is not None and filters["match_id"] != "All":
        filtered = filtered[filtered["match_id"] == int(filters["match_id"])]
    if "innings" in filters and filters["innings"] is not None and filters["innings"] != "All":
        filtered = filtered[filtered["innings"] == int(filters["innings"])]
    if "start_date" in filters and filters["start_date"] is not None:
        filtered = filtered[pd.to_datetime(filtered["match_date"]) >= pd.to_datetime(filters["start_date"])]
    if "end_date" in filters and filters["end_date"] is not None:
        filtered = filtered[pd.to_datetime(filtered["match_date"]) <= pd.to_datetime(filters["end_date"])]
    return filtered

def get_batsman_kpis(df: pd.DataFrame, player_id: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Computes all standard batting KPIs for a batsman.
    Returns: Runs, Balls faced, Average, Strike rate, Highest score, 30s, 50s, 100s, 4s, 6s, Dots, Boundary %.
    """
    p_df = apply_filters(df[df["striker_id"] == player_id], filters)
    if len(p_df) == 0:
        return {
            "total_runs": 0, "balls_faced": 0, "batting_average": 0.0, "strike_rate": 0.0,
            "highest_score": "0", "thirty_plus": 0, "fifties": 0, "centuries": 0,
            "fours": 0, "sixes": 0, "dot_balls": 0, "boundary_pct": 0.0, "innings": 0,
            "matches": 0, "dismissals": 0, "not_outs": 0
        }

    total_runs = int(p_df["batter_runs"].sum())
    balls_faced = int(p_df["is_legal_ball_batter"].sum())
    fours = int(p_df["is_four"].sum())
    sixes = int(p_df["is_six"].sum())
    dot_balls = int(p_df["is_dot_ball_batter"].sum())

    strike_rate = round((total_runs / balls_faced * 100), 2) if balls_faced > 0 else 0.0
    boundary_runs = (fours * 4) + (sixes * 6)
    boundary_pct = round((boundary_runs / total_runs * 100), 2) if total_runs > 0 else 0.0

    # Innings breakdown
    # Check dismissals: striker dismissed in innings
    # A batsman was dismissed if a wicket fell where dismissed_player_id == player_id or striker got out
    inn_group = p_df.groupby(["match_id", "innings"]).agg(
        runs=("batter_runs", "sum"),
        balls=("is_legal_ball_batter", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum")
    ).reset_index()

    # Determine dismissals per innings from full df
    m_ids = inn_group["match_id"].unique()
    dismissal_events = df[
        (df["match_id"].isin(m_ids)) &
        (df["is_wicket"] == True) &
        ((df["dismissed_player_id"] == player_id) | ((df["striker_id"] == player_id) & (df["dismissed_player_id"].isna())))
    ]
    dismissed_innings = set(zip(dismissal_events["match_id"], dismissal_events["innings"]))

    inn_group["is_out"] = inn_group.apply(lambda r: (r["match_id"], r["innings"]) in dismissed_innings, axis=1)

    innings_count = len(inn_group)
    matches_count = inn_group["match_id"].nunique()
    total_dismissals = int(inn_group["is_out"].sum())
    not_outs = innings_count - total_dismissals

    if total_dismissals > 0:
        batting_average = round(total_runs / total_dismissals, 2)
    else:
        batting_average = float(total_runs) if total_runs > 0 else 0.0

    # Highest score
    if len(inn_group) > 0:
        top_inn = inn_group.sort_values(by=["runs", "is_out"], ascending=[False, True]).iloc[0]
        highest_score = f"{top_inn['runs']}{'*' if not top_inn['is_out'] else ''}"
        thirty_plus = int((inn_group["runs"] >= 30).sum())
        fifties = int(((inn_group["runs"] >= 50) & (inn_group["runs"] < 100)).sum())
        centuries = int((inn_group["runs"] >= 100).sum())
    else:
        highest_score = "0"
        thirty_plus = 0
        fifties = 0
        centuries = 0

    return {
        "total_runs": total_runs,
        "balls_faced": balls_faced,
        "batting_average": batting_average,
        "strike_rate": strike_rate,
        "highest_score": highest_score,
        "thirty_plus": thirty_plus,
        "fifties": fifties,
        "centuries": centuries,
        "fours": fours,
        "sixes": sixes,
        "dot_balls": dot_balls,
        "boundary_pct": boundary_pct,
        "innings": innings_count,
        "matches": matches_count,
        "dismissals": total_dismissals,
        "not_outs": not_outs
    }

def get_batsman_innings_breakdown(df: pd.DataFrame, player_id: int, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Returns an innings-by-innings list for the batsman."""
    p_df = apply_filters(df[df["striker_id"] == player_id], filters)
    if len(p_df) == 0:
        return pd.DataFrame()

    innings_list = []
    for (m_id, inn), g in p_df.groupby(["match_id", "innings"]):
        runs = g["batter_runs"].sum()
        balls = g["is_legal_ball_batter"].sum()
        fours = g["is_four"].sum()
        sixes = g["is_six"].sum()
        sr = round((runs / balls * 100), 2) if balls > 0 else 0.0
        
        # Check dismissal
        w_row = g[g["is_wicket"] & ((g["dismissed_player_id"] == player_id) | (g["dismissed_player_id"].isna()))]
        if len(w_row) > 0:
            dismissal = w_row.iloc[0]["dismissal_type"]
            bowler = w_row.iloc[0]["bowler_name"]
            status = f"c/b {bowler} ({dismissal})" if dismissal != "RunOut" else f"Run Out"
            out = True
        else:
            status = "Not Out"
            out = False

        m_date = str(g["match_date"].iloc[0])[:10] if "match_date" in g.columns else ""
        comp = g["comp_name"].iloc[0] if "comp_name" in g.columns else ""
        
        # Opponent team
        bowling_team_id = g["bowling_team_id"].iloc[0]
        
        innings_list.append({
            "match_id": m_id,
            "innings": inn,
            "date": m_date,
            "competition": comp,
            "runs": runs,
            "balls": balls,
            "fours": fours,
            "sixes": sixes,
            "strike_rate": sr,
            "status": status,
            "is_out": out
        })

    res = pd.DataFrame(innings_list)
    return res.sort_values(by=["date", "match_id"], ascending=[False, False])

def get_batsman_over_by_over(df: pd.DataFrame, player_id: int, match_id: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Computes over-by-over table for the batsman:
    Over, Bowler(s) (actual unique bowler names), Balls faced, Runs, 4s, 6s, Strike Rate, Dismissals, Innings.
    """
    p_df = apply_filters(df[df["striker_id"] == player_id], filters)
    if match_id is not None and match_id != "All":
        p_df = p_df[p_df["match_id"] == int(match_id)]
    
    if len(p_df) == 0:
        return pd.DataFrame()

    if match_id is not None and match_id != "All":
        # Group by over in that specific match
        res = p_df.groupby("overid").agg(
            bowlers=("bowler_name", lambda x: ", ".join(sorted(x.dropna().unique()))),
            balls_faced=("is_legal_ball_batter", "sum"),
            runs=("batter_runs", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            dismissals=("is_wicket", lambda x: (x & (p_df.loc[x.index, "dismissed_player_id"] == player_id)).sum())
        ).reset_index()
        res["innings_faced"] = 1
    else:
        # Career aggregate by over number (1 to 20)
        res = p_df.groupby("overid").agg(
            bowlers=("bowler_name", lambda x: ", ".join(sorted(x.dropna().unique()))),
            balls_faced=("is_legal_ball_batter", "sum"),
            runs=("batter_runs", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            dismissals=("is_wicket", lambda x: (x & (p_df.loc[x.index, "dismissed_player_id"] == player_id)).sum()),
            innings_faced=("match_id", "nunique")
        ).reset_index()

    res["strike_rate"] = np.where(res["balls_faced"] > 0, (res["runs"] / res["balls_faced"] * 100).round(2), 0.0)
    res = res.sort_values(by="overid")
    return res

def get_over_deliveries(df: pd.DataFrame, player_id: int, over_id: int, match_id: Optional[int] = None) -> pd.DataFrame:
    """
    Returns every individual delivery in that over:
    Match ID, Innings, Over, Ball, Bowler name, Bowler ID, Runs scored, Extras, Shot played,
    Delivery type, Pace or spin, Boundary, Six, Wicket.
    """
    sub = df[(df["striker_id"] == player_id) & (df["overid"] == over_id)]
    if match_id is not None and match_id != "All":
        sub = sub[sub["match_id"] == int(match_id)]
        
    cols = [
        "match_id", "innings", "overid", "over_ball_num", "bowler_id", "bowler_name",
        "batter_runs", "extras_type", "stroke", "delivery_style", "pace_or_spin",
        "is_four", "is_six", "is_wicket", "dismissal_type"
    ]
    available_cols = [c for c in cols if c in sub.columns]
    res = sub[available_cols].copy().sort_values(by=["match_id", "innings", "over_ball_num"])
    return res

def get_batsman_phase_stats(df: pd.DataFrame, player_id: int, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Computes Powerplay (1-6), Middle (7-15), Death (16-20) phase analysis for a batsman.
    """
    p_df = apply_filters(df[df["striker_id"] == player_id], filters)
    if len(p_df) == 0:
        return pd.DataFrame()

    phases = ["Powerplay", "Middle", "Death"]
    summary = []
    for ph in phases:
        sub = p_df[p_df["phase"] == ph]
        balls = int(sub["is_legal_ball_batter"].sum())
        runs = int(sub["batter_runs"].sum())
        fours = int(sub["is_four"].sum())
        sixes = int(sub["is_six"].sum())
        dots = int(sub["is_dot_ball_batter"].sum())
        dismissals = int((sub["is_wicket"] & ((sub["dismissed_player_id"] == player_id) | (sub["dismissed_player_id"].isna()))).sum())
        sr = round((runs / balls * 100), 2) if balls > 0 else 0.0
        boundary_pct = round(((fours * 4 + sixes * 6) / runs * 100), 2) if runs > 0 else 0.0
        dot_pct = round((dots / balls * 100), 2) if balls > 0 else 0.0
        avg = round(runs / dismissals, 2) if dismissals > 0 else (runs if runs > 0 else 0.0)

        summary.append({
            "Phase": ph,
            "Overs": "1-6" if ph == "Powerplay" else ("7-15" if ph == "Middle" else "16-20"),
            "Balls Faced": balls,
            "Runs": runs,
            "Strike Rate": sr,
            "Average": avg,
            "Fours": fours,
            "Sixes": sixes,
            "Dot Balls": dots,
            "Dot %": dot_pct,
            "Dismissals": dismissals,
            "Boundary %": boundary_pct
        })
    return pd.DataFrame(summary)

