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


# ===========================================================================
# OVER-WISE BOWLING PERFORMANCE ENGINE
# ===========================================================================

def get_bowler_overwise_runs_economy(
    bbb: pd.DataFrame,
    bowler_id: int,
    bowler_name: Optional[str] = None,
    match_id: Optional[int] = None,
    innings: Optional[int] = None,
    only_active_overs: bool = False
) -> pd.DataFrame:
    """
    Returns a detailed over-wise bowling statistics table for a specific bowler.

    Economy = (Runs Conceded / Legal Balls) × 6
    - bowler_conceded_runs: runs attributed to the bowler (batter runs + wides +
      no-ball extras charged to bowler; byes & leg-byes excluded automatically).
    - is_legal_ball_bowler: True for legal deliveries (non-wide).
    - is_bowler_wicket: True for wickets credited to the bowler.
    - is_dot_ball_bowler: True when a legal ball is bowled and no runs are scored.

    For incomplete overs, economy is calculated on actual legal balls bowled.
    When no legal balls exist in an over, Economy is returned as None (displayed N/A).

    Columns returned:
        Bowler, Over, Runs Conceded, Legal Balls, Wickets, Economy,
        Dot Balls, Fours Conceded, Sixes Conceded
    """
    df = bbb[bbb["bowler_id"] == bowler_id].copy()
    if match_id is not None:
        df = df[df["match_id"] == match_id]
    if innings is not None:
        df = df[df["innings"] == innings]

    if df.empty:
        return pd.DataFrame()

    b_name = bowler_name or (df["bowler_name"].iloc[0] if "bowler_name" in df.columns else f"Bowler {bowler_id}")

    # Aggregate per over (1–20 for T20)
    agg = df.groupby("overid").agg(
        matches=("match_id", "nunique"),
        runs_conceded=("bowler_conceded_runs", "sum"),
        legal_balls=("is_legal_ball_bowler", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dot_balls=("is_dot_ball_bowler", "sum"),
        fours_conceded=("is_four", "sum"),
        sixes_conceded=("is_six", "sum"),
    ).reset_index()

    # Ensure all 20 overs represented
    all_overs = pd.DataFrame({"overid": range(1, 21)})
    agg = all_overs.merge(agg, on="overid", how="left").fillna(0)
    for col in ["matches", "runs_conceded", "legal_balls", "wickets", "dot_balls", "fours_conceded", "sixes_conceded"]:
        agg[col] = agg[col].astype(int)

    # Economy: aggregate (total runs / total legal balls × 6)
    agg["economy"] = agg.apply(
        lambda r: round(r["runs_conceded"] / r["legal_balls"] * 6.0, 2)
        if r["legal_balls"] > 0 else None,
        axis=1
    )

    if only_active_overs:
        agg = agg[agg["legal_balls"] > 0]

    agg.insert(0, "Bowler", b_name)
    agg.rename(columns={
        "overid": "Over",
        "matches": "Matches",
        "runs_conceded": "Runs Conceded",
        "legal_balls": "Legal Balls",
        "wickets": "Wickets",
        "dot_balls": "Dot Balls",
        "fours_conceded": "Fours Conceded",
        "sixes_conceded": "Sixes Conceded",
        "economy": "Economy"
    }, inplace=True)

    return agg[["Bowler", "Over", "Matches", "Runs Conceded", "Legal Balls", "Wickets",
                "Economy", "Dot Balls", "Fours Conceded", "Sixes Conceded"]].reset_index(drop=True)


def get_bowler_overwise_matchwise(
    bbb: pd.DataFrame,
    bowler_id: int,
    match_id: int,
    innings: Optional[int] = None
) -> pd.DataFrame:
    """
    Returns delivery-level over-by-over records for a specific bowler in a specific match.
    Each row represents one over bowled in that match/innings.
    """
    df = bbb[(bbb["bowler_id"] == bowler_id) & (bbb["match_id"] == match_id)].copy()
    if innings is not None:
        df = df[df["innings"] == innings]

    if df.empty:
        return pd.DataFrame()

    b_name = df["bowler_name"].iloc[0] if "bowler_name" in df.columns else f"Bowler {bowler_id}"

    agg = df.groupby(["match_id", "innings", "overid"]).agg(
        runs_conceded=("bowler_conceded_runs", "sum"),
        legal_balls=("is_legal_ball_bowler", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dot_balls=("is_dot_ball_bowler", "sum"),
        fours_conceded=("is_four", "sum"),
        sixes_conceded=("is_six", "sum"),
    ).reset_index()

    agg["economy"] = agg.apply(
        lambda r: round(r["runs_conceded"] / r["legal_balls"] * 6.0, 2)
        if r["legal_balls"] > 0 else None,
        axis=1
    )

    agg.insert(0, "Bowler", b_name)
    agg.rename(columns={
        "match_id": "Match ID",
        "innings": "Innings",
        "overid": "Over",
        "runs_conceded": "Runs Conceded",
        "legal_balls": "Legal Balls",
        "wickets": "Wickets",
        "dot_balls": "Dot Balls",
        "fours_conceded": "Fours Conceded",
        "sixes_conceded": "Sixes Conceded",
        "economy": "Economy"
    }, inplace=True)

    return agg[["Bowler", "Match ID", "Innings", "Over", "Runs Conceded", "Legal Balls",
                "Wickets", "Economy", "Dot Balls", "Fours Conceded", "Sixes Conceded"]].reset_index(drop=True)


def get_bowler_over_peaks(bbb: pd.DataFrame, bowler_id: int) -> Dict[str, Any]:
    """
    Computes per-bowler aggregate over-wise performance peaks.

    Returns a dict with:
      agg_max_runs, agg_max_runs_over,
      agg_min_runs, agg_min_runs_over,
      agg_best_economy, agg_best_economy_over,
      agg_worst_economy, agg_worst_economy_over,
      agg_max_wickets, agg_max_wickets_over,
      agg_max_dots, agg_max_dots_over,
      total_runs, total_legal_balls, overall_economy
    """
    df = bbb[bbb["bowler_id"] == bowler_id].copy()
    if df.empty:
        return {
            "agg_max_runs": 0, "agg_max_runs_over": 1,
            "agg_min_runs": 0, "agg_min_runs_over": 1,
            "agg_best_economy": None, "agg_best_economy_over": 1,
            "agg_worst_economy": None, "agg_worst_economy_over": 1,
            "agg_max_wickets": 0, "agg_max_wickets_over": 1,
            "agg_max_dots": 0, "agg_max_dots_over": 1,
            "total_runs": 0, "total_legal_balls": 0, "overall_economy": None
        }

    agg = df.groupby("overid").agg(
        runs_conceded=("bowler_conceded_runs", "sum"),
        legal_balls=("is_legal_ball_bowler", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dot_balls=("is_dot_ball_bowler", "sum"),
    ).reset_index()

    agg["economy"] = agg.apply(
        lambda r: r["runs_conceded"] / r["legal_balls"] * 6.0 if r["legal_balls"] > 0 else None,
        axis=1
    )

    active = agg[agg["legal_balls"] > 0]

    def _peak(col, ascending=False):
        if active.empty:
            return 0, 1
        idx = active[col].idxmax() if not ascending else active[col].idxmin()
        return int(active.loc[idx, col]) if col != "economy" else round(float(active.loc[idx, col]), 2), int(active.loc[idx, "overid"])

    max_runs, max_runs_ov = _peak("runs_conceded", ascending=False)
    min_runs, min_runs_ov = _peak("runs_conceded", ascending=True)

    # Economy peaks on active overs
    best_econ, best_econ_ov, worst_econ, worst_econ_ov = None, 1, None, 1
    if not active.empty:
        econ_valid = active.dropna(subset=["economy"])
        if not econ_valid.empty:
            best_idx = econ_valid["economy"].idxmin()
            worst_idx = econ_valid["economy"].idxmax()
            best_econ = round(float(econ_valid.loc[best_idx, "economy"]), 2)
            best_econ_ov = int(econ_valid.loc[best_idx, "overid"])
            worst_econ = round(float(econ_valid.loc[worst_idx, "economy"]), 2)
            worst_econ_ov = int(econ_valid.loc[worst_idx, "overid"])

    max_wkts, max_wkts_ov = _peak("wickets", ascending=False)
    max_dots, max_dots_ov = _peak("dot_balls", ascending=False)

    total_runs = int(agg["runs_conceded"].sum())
    total_lb = int(agg["legal_balls"].sum())
    overall_econ = round(total_runs / total_lb * 6.0, 2) if total_lb > 0 else None

    return {
        "agg_max_runs": max_runs, "agg_max_runs_over": max_runs_ov,
        "agg_min_runs": min_runs, "agg_min_runs_over": min_runs_ov,
        "agg_best_economy": best_econ, "agg_best_economy_over": best_econ_ov,
        "agg_worst_economy": worst_econ, "agg_worst_economy_over": worst_econ_ov,
        "agg_max_wickets": max_wkts, "agg_max_wickets_over": max_wkts_ov,
        "agg_max_dots": max_dots, "agg_max_dots_over": max_dots_ov,
        "total_runs": total_runs, "total_legal_balls": total_lb, "overall_economy": overall_econ
    }


def get_bowler_match_innings_list(bbb: pd.DataFrame, bowler_id: int) -> pd.DataFrame:
    """
    Returns a list of distinct match/innings combinations the bowler participated in.
    Includes match_id, innings, and basic identifiers from the delivery data.
    """
    df = bbb[bbb["bowler_id"] == bowler_id].copy()
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["match_id", "innings"]).agg(
        match_date=("match_date", "first"),
        comp_name=("comp_name", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
    ).reset_index()

    return agg.sort_values("match_date", ascending=False).reset_index(drop=True)


def get_multi_bowler_overwise_comparison(
    bbb: pd.DataFrame,
    bowler_ids_map: Dict[str, int],
    only_active_overs: bool = False
) -> pd.DataFrame:
    """
    Computes overwise performance across multiple bowlers for comparative visualization.
    bowler_ids_map: {bowler_name: bowler_id, ...}
    """
    all_dfs = []
    for b_name, b_id in bowler_ids_map.items():
        sub_df = get_bowler_overwise_runs_economy(bbb, b_id, bowler_name=b_name, only_active_overs=only_active_overs)
        if not sub_df.empty:
            all_dfs.append(sub_df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

