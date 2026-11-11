"""
AI Cricket Strategy: Team Squad Builder & Playing XI Recommendation Engine
Builds verified 15-player squads and tactical Playing XIs for any team using historical records,
role profiling, pitch adjustments, and bowling balance requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def get_team_roster(team_identifier: Any, data: Dict[str, pd.DataFrame]) -> tuple[Optional[int], str, pd.DataFrame]:
    """Resolves team ID and returns all cataloged players for that franchise/team."""
    teams_df = data["teams"]
    pm = data["players_master"]
    pmap = data["player_map"]
    pe = data["playing_eleven"]

    if isinstance(team_identifier, int):
        match = teams_df[teams_df["team_id"] == team_identifier]
    else:
        # Match by name
        q = str(team_identifier).strip().lower()
        match = teams_df[teams_df["team_name"].str.lower().str.contains(q, regex=False)]
        if len(match) == 0:
            # Fallback exact word search
            match = teams_df[teams_df["team_name"].str.lower() == q]

    if len(match) == 0:
        # Default to first team if unmatched
        t_id = 1
        t_name = "Chennai Super Kings"
    else:
        t_id = int(match["team_id"].iloc[0])
        t_name = str(match["team_name"].iloc[0])

    # Get players belonging to this team
    pids = pmap[pmap["team_id"] == t_id]["player_id"].unique()
    roster = pm[pm["player_id"].isin(pids)].copy()

    # Enrich roles from playing_eleven history
    wk_pids = set(pe[(pe["player_id"].isin(pids)) & (pe["is_wk"] == True)]["player_id"].unique())
    cap_pids = set(pe[(pe["player_id"].isin(pids)) & (pe["is_captain"] == True)]["player_id"].unique())

    roster["is_wk"] = roster["player_id"].isin(wk_pids)
    roster["is_captain"] = roster["player_id"].isin(cap_pids)

    # Classify bowling styles
    bstyle = roster["bowling_style"].astype(str).str.lower()
    roster["is_pacer"] = bstyle.str.contains("fast|pace|medium") & (roster["legal_balls"] >= 18)
    roster["is_spinner"] = bstyle.str.contains("spin|break|slow|orthodox") & (roster["legal_balls"] >= 18)
    roster["is_bowler"] = (roster["legal_balls"] >= 36) & (roster["wickets"] >= 3)
    roster["is_allrounder"] = (roster["runs"] >= 80) & (roster["wickets"] >= 4) & (roster["legal_balls"] >= 36)

    # Batting ratings
    roster["bat_score"] = (roster["runs"] * 0.4) + (roster["bat_strike_rate"] * 1.5)
    # Bowling ratings
    roster["bowl_score"] = (roster["wickets"] * 15.0) + ((15.0 - np.clip(roster["bowl_economy"], 5, 15)) * 10.0)

    return t_id, t_name, roster


def build_playing_xi(
    team_identifier: Any,
    data: Dict[str, pd.DataFrame],
    pitch_condition: str = "Balanced",
    opposition_team: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs an optimal, tactically balanced Playing XI with verified batting order,
    designated roles, wicketkeeper, bowling options, captain, and 4 reserve bench players.
    """
    t_id, t_name, roster = get_team_roster(team_identifier, data)
    bbb = data["bbb"]

    if len(roster) < 11:
        # Not enough players in team roster
        return {"error": f"Insufficient players ({len(roster)}) in {t_name} roster to formulate a Playing XI."}

    pitch = pitch_condition.lower()
    selected_pids = []
    lineup = []

    # 1. Select Wicketkeeper (Crucial mandatory position)
    wks = roster[roster["is_wk"]].sort_values(by="bat_score", ascending=False)
    if len(wks) > 0:
        wk_pick = wks.iloc[0]
    else:
        # Fallback to top batter
        wk_pick = roster.sort_values(by="runs", ascending=False).iloc[0]
    selected_pids.append(wk_pick["player_id"])

    # 2. Select Captain
    cap_cands = roster[roster["is_captain"]].sort_values(by=["matches_played", "runs"], ascending=False)
    if len(cap_cands) > 0:
        captain_pid = cap_cands.iloc[0]["player_id"]
    else:
        captain_pid = roster.sort_values(by="matches_played", ascending=False).iloc[0]["player_id"]

    # 3. Select Primary Bowlers based on Pitch Condition
    if "spin" in pitch:
        num_spinners = 3
        num_pacers = 2
    elif "pace" in pitch:
        num_spinners = 1
        num_pacers = 4
    else:
        num_spinners = 2
        num_pacers = 3

    available_pacers = roster[roster["is_pacer"] & ~roster["player_id"].isin(selected_pids)].sort_values(by="bowl_score", ascending=False)
    picked_pacers = available_pacers.head(num_pacers)
    for _, p in picked_pacers.iterrows():
        selected_pids.append(p["player_id"])

    available_spinners = roster[roster["is_spinner"] & ~roster["player_id"].isin(selected_pids)].sort_values(by="bowl_score", ascending=False)
    picked_spinners = available_spinners.head(num_spinners)
    for _, s in picked_spinners.iterrows():
        selected_pids.append(s["player_id"])

    # 4. Select All-rounders (Balance creators)
    available_ars = roster[roster["is_allrounder"] & ~roster["player_id"].isin(selected_pids)].sort_values(by="bat_score", ascending=False)
    picked_ars = available_ars.head(2)
    for _, ar in picked_ars.iterrows():
        selected_pids.append(ar["player_id"])

    # 5. Fill remaining spots with top specialty batters
    spots_remaining = 11 - len(selected_pids)
    available_batters = roster[~roster["player_id"].isin(selected_pids)].sort_values(by="bat_score", ascending=False)
    picked_batters = available_batters.head(spots_remaining)
    for _, b in picked_batters.iterrows():
        selected_pids.append(b["player_id"])

    # 6. Re-order into optimal 1-11 Batting Order
    xi_players = roster[roster["player_id"].isin(selected_pids)].copy()

    # Determine batting order:
    # 1-2: Openers (High SR & PP ability)
    # 3-4: Top Order Anchor / Stroke Makers
    # 5-6: Middle Order / All-rounders
    # 7: Finisher
    # 8-11: Specialty Bowlers
    batters_pool = xi_players[~xi_players["is_bowler"] | xi_players["is_allrounder"] | (xi_players["player_id"] == wk_pick["player_id"])].sort_values(by="bat_strike_rate", ascending=False)
    bowlers_pool = xi_players[xi_players["is_bowler"] & ~xi_players["is_allrounder"] & (xi_players["player_id"] != wk_pick["player_id"])].sort_values(by="bowl_score", ascending=False)

    ordered_players = list(batters_pool["player_id"])
    for bid in bowlers_pool["player_id"]:
        if bid not in ordered_players:
            ordered_players.append(bid)

    # Pad or trim to exactly 11
    final_11_pids = ordered_players[:11]

    # Build detailed player cards
    bowling_options_count = 0
    vice_captain_pid = None

    for idx, pid in enumerate(final_11_pids, start=1):
        p_row = roster[roster["player_id"] == pid].iloc[0]
        p_name = p_row["player_name"]

        # Tag role
        is_wk = (pid == wk_pick["player_id"])
        is_c = (pid == captain_pid)
        is_ar = bool(p_row["is_allrounder"])
        is_pac = bool(p_row["is_pacer"])
        is_spin = bool(p_row["is_spinner"])

        if is_pac or is_spin:
            bowling_options_count += 1

        if idx <= 2:
            role_desc = "Opening Batsman"
        elif idx == 3:
            role_desc = "Top Order Anchor (No. 3)"
        elif idx <= 5:
            role_desc = "Middle Order Stroke Maker"
        elif idx <= 7:
            role_desc = "Finisher / Dynamic All-Rounder" if is_ar else "Lower Middle Order Finisher"
        else:
            role_desc = f"Primary Strike Bowler ({'Pace' if is_pac else 'Spin'})"

        tags = []
        if is_c:
            tags.append("Captain (C)")
        elif vice_captain_pid is None and idx <= 6:
            vice_captain_pid = pid
            tags.append("Vice-Captain (VC)")

        if is_wk:
            tags.append("Wicketkeeper (WK)")
        if is_ar:
            tags.append("All-Rounder")
        elif is_pac:
            tags.append("Pacer")
        elif is_spin:
            tags.append("Spinner")

        justification = (
            f"Career: {p_row['runs']} runs (SR {p_row['bat_strike_rate']}), "
            f"{p_row['wickets']} wickets (Econ {p_row['bowl_economy']}) in {p_row['matches_played']} matches."
        )

        lineup.append({
            "batting_position": idx,
            "player_id": pid,
            "player_name": p_name,
            "role": role_desc,
            "tags": ", ".join(tags) if tags else "Specialist",
            "batting_style": str(p_row["batting_style"]),
            "bowling_style": str(p_row["bowling_style"]),
            "runs": int(p_row["runs"]),
            "strike_rate": float(p_row["bat_strike_rate"]),
            "wickets": int(p_row["wickets"]),
            "economy": float(p_row["bowl_economy"]),
            "justification": justification
        })

    # 7. Select 4 Reserve Bench Players (Reserve Batter, Reserve Spinner, Reserve Pacer, Reserve WK/All-Rounder)
    bench_pids = [pid for pid in roster["player_id"] if pid not in final_11_pids]
    bench_pool = roster[roster["player_id"].isin(bench_pids)].copy()

    reserves = []
    # Reserve batter
    r_bat = bench_pool[~bench_pool["is_bowler"]].sort_values(by="bat_score", ascending=False)
    if len(r_bat) > 0:
        res = r_bat.iloc[0]
        reserves.append({"role": "Reserve Batter", "player_name": res["player_name"], "stats": f"{res['runs']} runs, SR {res['bat_strike_rate']}"})
        bench_pool = bench_pool[bench_pool["player_id"] != res["player_id"]]

    # Reserve pacer
    r_pac = bench_pool[bench_pool["is_pacer"]].sort_values(by="bowl_score", ascending=False)
    if len(r_pac) > 0:
        res = r_pac.iloc[0]
        reserves.append({"role": "Reserve Fast Bowler", "player_name": res["player_name"], "stats": f"{res['wickets']} wkts, Econ {res['bowl_economy']}"})
        bench_pool = bench_pool[bench_pool["player_id"] != res["player_id"]]

    # Reserve spinner
    r_sp = bench_pool[bench_pool["is_spinner"]].sort_values(by="bowl_score", ascending=False)
    if len(r_sp) > 0:
        res = r_sp.iloc[0]
        reserves.append({"role": "Reserve Spinner", "player_name": res["player_name"], "stats": f"{res['wickets']} wkts, Econ {res['bowl_economy']}"})
        bench_pool = bench_pool[bench_pool["player_id"] != res["player_id"]]

    # Reserve utility / keeper
    r_util = bench_pool.sort_values(by=["bat_score", "bowl_score"], ascending=False)
    if len(r_util) > 0:
        res = r_util.iloc[0]
        reserves.append({"role": "Reserve Utility / Bench", "player_name": res["player_name"], "stats": f"{res['runs']} runs, {res['wickets']} wkts"})

    # Complete 15-player squad
    squad_15 = list(lineup)
    for r in reserves:
        squad_15.append({
            "batting_position": "Bench",
            "player_id": 0,
            "player_name": r["player_name"],
            "role": r["role"],
            "tags": "Reserve",
            "batting_style": "-",
            "bowling_style": "-",
            "runs": 0,
            "strike_rate": 0.0,
            "wickets": 0,
            "economy": 0.0,
            "justification": r["stats"]
        })

    balance_summary = (
        f"**Tactical Balance Check:** {bowling_options_count} bowling options selected. "
        f"Pitch tuning: `{pitch_condition.title()}`. "
        f"Designated Wicketkeeper: `{wk_pick['player_name']}`. "
        f"Designated Captain: `{roster[roster['player_id'] == captain_pid]['player_name'].iloc[0]}`."
    )

    return {
        "team_id": t_id,
        "team_name": t_name,
        "pitch_condition": pitch_condition,
        "opposition_team": opposition_team or "General Matchup",
        "playing_xi": lineup,
        "reserves": reserves,
        "squad_15": squad_15,
        "bowling_options_count": bowling_options_count,
        "balance_summary": balance_summary
    }
