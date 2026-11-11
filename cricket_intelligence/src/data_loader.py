"""
Optimized Data Loader with Caching for Cricket Intelligence.
"""

import os
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

@st.cache_data(show_spinner=False)
def load_all_data(base_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads all pre-processed Parquet tables with memory caching.
    Handles relative path resolution dynamically.
    """
    if base_path is None:
        # Check if running from cricket_intelligence or parent directory
        if os.path.exists("data/processed/ball_by_ball.parquet"):
            data_dir = "data/processed"
        elif os.path.exists("cricket_intelligence/data/processed/ball_by_ball.parquet"):
            data_dir = "cricket_intelligence/data/processed"
        else:
            # Run cleaning pipeline if processed files don't exist
            from src.data_cleaning import clean_and_process_data
            clean_and_process_data()
            data_dir = "data/processed" if os.path.exists("data/processed") else "cricket_intelligence/data/processed"
    else:
        data_dir = base_path

    bbb = pd.read_parquet(os.path.join(data_dir, "ball_by_ball.parquet"))
    matches = pd.read_parquet(os.path.join(data_dir, "matches.parquet"))
    comps = pd.read_parquet(os.path.join(data_dir, "competitions.parquet"))
    teams = pd.read_parquet(os.path.join(data_dir, "teams.parquet"))
    playing_eleven = pd.read_parquet(os.path.join(data_dir, "playing_eleven.parquet"))
    player_map = pd.read_parquet(os.path.join(data_dir, "competition_team_player_mapping.parquet"))
    players_master = pd.read_parquet(os.path.join(data_dir, "players_master.parquet"))

    return {
        "bbb": bbb,
        "matches": matches,
        "comps": comps,
        "teams": teams,
        "playing_eleven": playing_eleven,
        "player_map": player_map,
        "players_master": players_master
    }

def get_lookup_maps(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Builds fast in-memory key-value maps for IDs to display names.
    """
    player_map = data["players_master"].set_index("player_id")["player_name"].to_dict()
    team_map = data["teams"].groupby("team_id")["team_name"].first().to_dict()
    comp_map = data["comps"].set_index("comp_id")["comp_name"].to_dict()
    
    # Matches lookup
    matches = data["matches"]
    match_labels = {}
    for _, row in matches.iterrows():
        date_str = str(row["match_date"])[:10]
        match_labels[row["match_id"]] = f"Match {row['match_id']}: {row['home_team']} vs {row['away_team']} ({date_str})"

    return {
        "player_names": player_map,
        "team_names": team_map,
        "comp_names": comp_map,
        "match_labels": match_labels
    }

