"""
Match Analysis & Detailed Scorecard Reconstruction Module.
Reconstructs batting scorecards, bowling scorecards, fall of wickets, partnerships,
and over-by-over progression.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

def get_match_full_details(
    match_id: int,
    matches_df: pd.DataFrame,
    bbb_df: pd.DataFrame,
    p_eleven_df: pd.DataFrame,
    player_names: Dict[int, str]
) -> Dict[str, Any]:
    """
    Returns full match intelligence dossier:
    - Metadata (teams, toss, venue, result)
    - Playing XIs
    - Innings-by-innings scorecards (Batting, Bowling, Extras)
    - Fall of wickets
    - Partnerships
    - Over-by-over progression
    - Ball-by-ball log
    """
    m_row = matches_df[matches_df["match_id"] == match_id]
    if len(m_row) == 0:
        return {}
    m_info = m_row.iloc[0].to_dict()

    # Playing XI
    m_pe = p_eleven_df[p_eleven_df["match_id"] == match_id].copy()
    m_pe["player_name"] = m_pe["player_id"].map(player_names).fillna("Unknown")
    
    home_team_id = m_info["home_team_id"]
    away_team_id = m_info["away_team_id"]
    
    home_xi = m_pe[m_pe["team_id"] == home_team_id].sort_values(by="batting_order")
    away_xi = m_pe[m_pe["team_id"] == away_team_id].sort_values(by="batting_order")

    # Match deliveries
    m_bbb = bbb_df[bbb_df["match_id"] == match_id].copy().sort_values(by=["innings", "overid", "over_ball_num"])
    
    innings_data = {}
    for inn in sorted(m_bbb["innings"].unique()):
        inn_df = m_bbb[m_bbb["innings"] == inn]
        batting_team_id = inn_df["batting_team_id"].iloc[0]
        bowling_team_id = inn_df["bowling_team_id"].iloc[0]

        # Batting scorecard
        batters = []
        for striker_id, g in inn_df.groupby("striker_id", sort=False):
            striker_name = g["striker_name"].iloc[0]
            runs = int(g["batter_runs"].sum())
            balls = int(g["is_legal_ball_batter"].sum())
            fours = int(g["is_four"].sum())
            sixes = int(g["is_six"].sum())
            sr = round(runs / balls * 100, 2) if balls > 0 else 0.0

            # Dismissal
            w_row = g[g["is_wicket"] & ((g["dismissed_player_id"] == striker_id) | (g["dismissed_player_id"].isna()))]
            if len(w_row) > 0:
                dtype = w_row.iloc[0]["dismissal_type"]
                bowler = w_row.iloc[0]["bowler_name"]
                if dtype == "Bowled":
                    status = f"b {bowler}"
                elif dtype == "Caught":
                    status = f"c Sub b {bowler}"
                elif dtype == "LBW":
                    status = f"lbw b {bowler}"
                elif dtype == "Stump":
                    status = f"st Keeper b {bowler}"
                elif dtype == "RunOut":
                    status = "run out"
                else:
                    status = f"{dtype} b {bowler}"
            else:
                status = "not out"

            batters.append({
                "Player ID": striker_id,
                "Batter": striker_name,
                "Dismissal": status,
                "Runs": runs,
                "Balls": balls,
                "4s": fours,
                "6s": sixes,
                "SR": sr
            })

        bat_scorecard = pd.DataFrame(batters)

        # Bowling scorecard
        bowlers = []
        for bowler_id, g in inn_df.groupby("bowler_id", sort=False):
            bowler_name = g["bowler_name"].iloc[0]
            legal_balls = int(g["is_legal_ball_bowler"].sum())
            runs = int(g["bowler_conceded_runs"].sum())
            wkts = int(g["is_bowler_wicket"].sum())
            dots = int(g["is_dot_ball_bowler"].sum())
            overs_str = f"{legal_balls // 6}.{legal_balls % 6}"
            econ = round(runs / legal_balls * 6.0, 2) if legal_balls > 0 else 0.0

            bowlers.append({
                "Bowler ID": bowler_id,
                "Bowler": bowler_name,
                "Overs": overs_str,
                "Maidens": 0,
                "Runs": runs,
                "Wickets": wkts,
                "Economy": econ,
                "Dots": dots
            })

        bowl_scorecard = pd.DataFrame(bowlers)

        # Extras breakdown
        wides = int((inn_df["extras_type"] == "wide").sum())
        noballs = int((inn_df["extras_type"] == "Overstep").sum())
        legbyes = int((inn_df["extras_type"] == "legbyes").sum())
        byes = int((inn_df["extras_type"] == "byes").sum())
        extras_2 = int(inn_df["extras_2_runs"].sum())
        total_extras = wides + noballs + legbyes + byes + extras_2
        total_runs = int(inn_df["total_delivery_runs"].sum())
        total_wickets = int(inn_df["is_wicket"].sum())
        legal_balls_inn = int(inn_df["is_legal_ball_bowler"].sum())
        overs_inn = f"{legal_balls_inn // 6}.{legal_balls_inn % 6}"

        # Fall of Wickets
        fow = []
        cum_runs = 0
        wkt_count = 0
        for _, row in inn_df.iterrows():
            cum_runs += int(row["total_delivery_runs"])
            if row["is_wicket"]:
                wkt_count += 1
                fow.append({
                    "Wicket #": wkt_count,
                    "Score": f"{cum_runs}/{wkt_count}",
                    "Over": f"{row['overid']}.{row['over_ball_num']}",
                    "Dismissed Player": row["striker_name"],
                    "Dismissal": row["dismissal_type"],
                    "Bowler": row["bowler_name"]
                })
        fow_df = pd.DataFrame(fow)

        # Reconstruct Partnerships
        partnerships = []
        cur_p_runs = 0
        cur_p_balls = 0
        p_striker = None
        p_non_striker = None
        for _, row in inn_df.iterrows():
            cur_p_runs += int(row["total_delivery_runs"])
            if row["is_legal_ball_batter"]:
                cur_p_balls += 1
            p_striker = row["striker_name"]
            p_non_striker = row["non_striker_name"]
            if row["is_wicket"]:
                partnerships.append({
                    "Partnership #": len(partnerships) + 1,
                    "Batters": f"{p_striker} & {p_non_striker}",
                    "Runs": cur_p_runs,
                    "Balls": cur_p_balls
                })
                cur_p_runs = 0
                cur_p_balls = 0
        if cur_p_runs > 0 or cur_p_balls > 0:
            partnerships.append({
                "Partnership #": len(partnerships) + 1,
                "Batters": f"{p_striker} & {p_non_striker}*",
                "Runs": cur_p_runs,
                "Balls": cur_p_balls
            })
        part_df = pd.DataFrame(partnerships)

        # Over-by-over progression
        prog = inn_df.groupby("overid").agg(
            over_runs=("total_delivery_runs", "sum"),
            wickets=("is_wicket", "sum")
        ).reset_index()
        prog["cum_runs"] = prog["over_runs"].cumsum()

        innings_data[inn] = {
            "batting_team_id": batting_team_id,
            "bowling_team_id": bowling_team_id,
            "total_runs": total_runs,
            "total_wickets": total_wickets,
            "overs": overs_inn,
            "extras": f"{total_extras} (w {wides}, nb {noballs}, lb {legbyes}, b {byes})",
            "bat_scorecard": bat_scorecard,
            "bowl_scorecard": bowl_scorecard,
            "fow": fow_df,
            "partnerships": part_df,
            "progression": prog,
            "deliveries": inn_df
        }

    return {
        "metadata": m_info,
        "home_xi": home_xi,
        "away_xi": away_xi,
        "innings": innings_data,
        "all_deliveries": m_bbb
    }

