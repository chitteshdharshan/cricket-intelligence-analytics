"""
Data Cleaning and Preprocessing Pipeline for Cricket Intelligence.
Processes raw CSVs into highly optimized, pre-indexed parquet files and metadata tables.
"""

import os
import pandas as pd
import numpy as np

def clean_and_process_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> dict:
    """
    Cleans raw cricket datasets, performs relational integrity checks,
    computes canonical cricket metrics and exports optimized Parquet files.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load raw datasets
    comps = pd.read_csv(os.path.join(raw_dir, "competitions.csv"))
    teams = pd.read_csv(os.path.join(raw_dir, "teams.csv"))
    matches = pd.read_csv(os.path.join(raw_dir, "matches.csv"))
    playing_eleven = pd.read_csv(os.path.join(raw_dir, "playing_eleven.csv"))
    player_map = pd.read_csv(os.path.join(raw_dir, "competition_team_player_mapping.csv"))
    bbb = pd.read_csv(os.path.join(raw_dir, "ball_by_ball.csv"))

    # 2. Competitions cleaning
    comps["season"] = comps["season"].astype(str)
    comps["comp_name"] = comps["comp_name"].str.strip()

    # 3. Teams cleaning
    teams["team_name"] = teams["team_name"].str.strip()
    team_dict = teams.groupby("team_id")["team_name"].first().to_dict()

    # 4. Matches cleaning
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    matches["match_year"] = matches["match_date"].dt.year
    matches["stage"] = matches["stage"].fillna("League").str.strip()
    matches["venue_country"] = matches["venue_country"].fillna("Unknown").str.strip()
    matches["result"] = matches["result"].fillna("No Result").str.strip()
    
    # Match lookup map for fast join
    match_lookup = matches[["match_id", "comp_id", "match_date", "match_year", "stage", "venue_country", "home_team", "away_team", "result"]].copy()
    match_lookup = match_lookup.merge(comps[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left")

    # 5. Ball by ball enrichment & cricket calculations
    # Join match details
    bbb = bbb.merge(match_lookup, on="match_id", how="left")
    
    # Fill text categories
    bbb["stroke"] = bbb["stroke"].fillna("Unknown / Unclassified").str.strip()
    bbb["delivery_style"] = bbb["delivery_style"].fillna("Unknown / Unclassified").str.strip()
    bbb["pace_or_spin"] = bbb["pace_or_spin"].fillna("Unknown").str.strip()
    bbb["bowling_type"] = bbb["bowling_type"].fillna("Unknown").str.strip()
    bbb["batting_type"] = bbb["batting_type"].fillna("Unknown").str.strip()
    bbb["on_or_off"] = bbb["on_or_off"].fillna("Unknown").str.strip()
    bbb["over_or_around"] = bbb["over_or_around"].fillna("Unknown").str.strip()
    bbb["extras_type"] = bbb["extras_type"].fillna("none").str.strip()
    bbb["dismissal_type"] = bbb["dismissal_type"].fillna("not_out").str.strip()
    bbb["extras_2_runs"] = bbb["extras_2_runs"].fillna(0).astype(int)

    # Batting metrics
    # Wides do not count as balls faced by the batter. Overstep and other extras do.
    bbb["is_legal_ball_batter"] = (bbb["extras_type"] != "wide")
    bbb["batter_runs"] = bbb["runs"].astype(int)
    bbb["is_dot_ball_batter"] = bbb["is_legal_ball_batter"] & (bbb["batter_runs"] == 0)

    # Bowling metrics
    # Legal balls for bowler: neither wide nor overstep (no-ball)
    bbb["is_legal_ball_bowler"] = ~bbb["extras_type"].isin(["wide", "Overstep"])
    
    # Extras conceded by bowler: wide adds 1 + extras_2_runs, no-ball adds 1 + extras_2_runs
    # Byes, leg-byes, penalties are team extras and NOT charged to bowler
    bowler_extras = np.where(
        bbb["extras_type"] == "wide", 1 + bbb["extras_2_runs"],
        np.where(bbb["extras_type"] == "Overstep", 1 + bbb["extras_2_runs"], 0)
    )
    bbb["bowler_conceded_runs"] = bbb["batter_runs"] + bowler_extras
    
    # Dot ball for bowler: legal delivery where 0 runs are conceded (no batter runs, no bowler extras)
    bbb["is_dot_ball_bowler"] = bbb["is_legal_ball_bowler"] & (bbb["bowler_conceded_runs"] == 0)

    # Total delivery runs (including byes and legbyes)
    delivery_extras = np.where(
        bbb["extras_type"].isin(["wide", "Overstep", "legbyes", "byes", "other"]),
        1 + bbb["extras_2_runs"],
        np.where(bbb["extras_type"] == "Penalty", 5, 0)
    )
    bbb["total_delivery_runs"] = bbb["batter_runs"] + delivery_extras

    # Bowler credited wickets: exclude RunOut, Retire Out, obstructing the field
    non_bowler_dismissals = ["RunOut", "Retire Out", "Obstracting the field", "Obstructing the field", "not_out"]
    bbb["is_bowler_wicket"] = bbb["is_wicket"] & (~bbb["dismissal_type"].isin(non_bowler_dismissals))

    # Innings phases
    # T20 phases: Powerplay (1-6), Middle (7-15), Death (16-20)
    def assign_phase(over):
        if over <= 6:
            return "Powerplay"
        elif over <= 15:
            return "Middle"
        else:
            return "Death"
    bbb["phase"] = bbb["overid"].apply(assign_phase)

    # 6. Build Master Player Directory
    # Combine information from mapping, playing_eleven, and bbb
    player_names = player_map.groupby("player_id")["player_name"].first().to_dict()
    player_teams = player_map.groupby("player_id")["team_name"].unique().apply(lambda x: ", ".join(sorted(x))).to_dict()
    player_comps = player_map.groupby("player_id")["comp_name"].unique().apply(lambda x: ", ".join(sorted(x))).to_dict()

    # Batting and Bowling style mode
    bat_style = bbb[bbb["batting_type"] != "Unknown"].groupby("striker_id")["batting_type"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
    ).to_dict()
    bowl_style = bbb[bbb["bowling_type"] != "Unknown"].groupby("bowler_id")["bowling_type"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
    ).to_dict()

    # Matches played count
    played_counts = playing_eleven[playing_eleven["is_played"] == True].groupby("player_id")["match_id"].nunique().to_dict()
    
    # Career summary from ball by ball
    bat_summary = bbb.groupby("striker_id").agg(
        runs=("batter_runs", "sum"),
        balls_faced=("is_legal_ball_batter", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        bat_innings=("match_id", "nunique")
    ).reset_index()

    bowl_summary = bbb.groupby("bowler_id").agg(
        wickets=("is_bowler_wicket", "sum"),
        runs_conceded=("bowler_conceded_runs", "sum"),
        legal_balls=("is_legal_ball_bowler", "sum"),
        bowl_innings=("match_id", "nunique")
    ).reset_index()

    all_player_ids = sorted(list(set(player_map["player_id"]).union(set(bbb["striker_id"])).union(set(bbb["bowler_id"]))))
    
    players_df = pd.DataFrame({"player_id": all_player_ids})
    players_df["player_name"] = players_df["player_id"].map(player_names).fillna("Unknown")
    players_df["teams"] = players_df["player_id"].map(player_teams).fillna("Unknown")
    players_df["competitions"] = players_df["player_id"].map(player_comps).fillna("Unknown")
    players_df["batting_style"] = players_df["player_id"].map(bat_style).fillna("Unknown")
    players_df["bowling_style"] = players_df["player_id"].map(bowl_style).fillna("Unknown")
    players_df["matches_played"] = players_df["player_id"].map(played_counts).fillna(0).astype(int)
    
    players_df = players_df.merge(bat_summary, left_on="player_id", right_on="striker_id", how="left").drop(columns=["striker_id"])
    players_df = players_df.merge(bowl_summary, left_on="player_id", right_on="bowler_id", how="left").drop(columns=["bowler_id"])
    
    players_df["runs"] = players_df["runs"].fillna(0).astype(int)
    players_df["balls_faced"] = players_df["balls_faced"].fillna(0).astype(int)
    players_df["fours"] = players_df["fours"].fillna(0).astype(int)
    players_df["sixes"] = players_df["sixes"].fillna(0).astype(int)
    players_df["bat_innings"] = players_df["bat_innings"].fillna(0).astype(int)
    players_df["wickets"] = players_df["wickets"].fillna(0).astype(int)
    players_df["runs_conceded"] = players_df["runs_conceded"].fillna(0).astype(int)
    players_df["legal_balls"] = players_df["legal_balls"].fillna(0).astype(int)
    players_df["bowl_innings"] = players_df["bowl_innings"].fillna(0).astype(int)
    
    # Calculate career strike rate and economy
    players_df["bat_strike_rate"] = np.where(
        players_df["balls_faced"] > 0,
        (players_df["runs"] / players_df["balls_faced"]) * 100,
        0.0
    ).round(2)
    
    players_df["bowl_economy"] = np.where(
        players_df["legal_balls"] > 0,
        (players_df["runs_conceded"] / players_df["legal_balls"]) * 6.0,
        0.0
    ).round(2)

    # 7. Save processed Parquet files
    bbb.to_parquet(os.path.join(processed_dir, "ball_by_ball.parquet"), index=False)
    matches.to_parquet(os.path.join(processed_dir, "matches.parquet"), index=False)
    comps.to_parquet(os.path.join(processed_dir, "competitions.parquet"), index=False)
    teams.to_parquet(os.path.join(processed_dir, "teams.parquet"), index=False)
    playing_eleven.to_parquet(os.path.join(processed_dir, "playing_eleven.parquet"), index=False)
    player_map.to_parquet(os.path.join(processed_dir, "competition_team_player_mapping.parquet"), index=False)
    players_df.to_parquet(os.path.join(processed_dir, "players_master.parquet"), index=False)

    print("Data cleaning and Parquet caching complete.")
    print(f"Total deliveries processed: {len(bbb):,}")
    print(f"Total players cataloged: {len(players_df):,}")
    print(f"Total matches cataloged: {len(matches):,}")
    return {
        "bbb": bbb,
        "matches": matches,
        "comps": comps,
        "teams": teams,
        "playing_eleven": playing_eleven,
        "player_map": player_map,
        "players_master": players_df
    }

if __name__ == "__main__":
    clean_and_process_data()

