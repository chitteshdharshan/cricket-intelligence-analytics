"""
Database and Query Interface for Cricket Intelligence.
Uses DuckDB in-memory OLAP query engine for fast multi-table aggregations.
"""

import duckdb
import pandas as pd
from typing import Dict, Any, Optional

class CricketDatabase:
    """
    DuckDB OLAP Engine initialized with the analytical tables.
    """
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.con = duckdb.connect(database=":memory:")
        for name, df in data_dict.items():
            self.con.register(name, df)

    def query(self, sql: str) -> pd.DataFrame:
        """Executes a SQL query against the in-memory cricket tables."""
        return self.con.execute(sql).df()

    def get_player_career_record(self, player_id: int) -> Dict[str, Any]:
        """Returns unified batting and bowling career stats for a player."""
        bat_query = f"""
            SELECT 
                COUNT(DISTINCT match_id) as matches,
                COUNT(DISTINCT match_id || '_' || innings) as innings,
                SUM(batter_runs) as runs,
                SUM(CASE WHEN is_legal_ball_batter THEN 1 ELSE 0 END) as balls,
                SUM(CASE WHEN is_four THEN 1 ELSE 0 END) as fours,
                SUM(CASE WHEN is_six THEN 1 ELSE 0 END) as sixes,
                SUM(CASE WHEN is_dot_ball_batter THEN 1 ELSE 0 END) as dots,
                SUM(CASE WHEN is_wicket AND dismissed_player_id = {player_id} THEN 1 ELSE 0 END) as dismissals
            FROM bbb
            WHERE striker_id = {player_id}
        """
        bowl_query = f"""
            SELECT 
                COUNT(DISTINCT match_id) as bowl_matches,
                COUNT(DISTINCT match_id || '_' || innings) as bowl_innings,
                SUM(CASE WHEN is_legal_ball_bowler THEN 1 ELSE 0 END) as legal_balls,
                SUM(bowler_conceded_runs) as runs_conceded,
                SUM(CASE WHEN is_bowler_wicket THEN 1 ELSE 0 END) as wickets,
                SUM(CASE WHEN is_dot_ball_bowler THEN 1 ELSE 0 END) as dot_balls
            FROM bbb
            WHERE bowler_id = {player_id}
        """
        bat_res = self.query(bat_query).iloc[0].to_dict()
        bowl_res = self.query(bowl_query).iloc[0].to_dict()
        return {"batting": bat_res, "bowling": bowl_res}

