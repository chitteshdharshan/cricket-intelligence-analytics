"""
AI Cricket Strategy: Match Situation Analyzer & Target Chase/Defence Planner
Mathematical model evaluating target chases, defensive run-budgets, collapse recovery,
and phase-by-phase tactical execution blueprints.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def analyze_target_chase(
    target: int,
    current_runs: int,
    overs_completed: float,
    wickets_down: int,
    total_overs: int = 20
) -> Dict[str, Any]:
    """
    Computes real-time target chasing parameters and strategic acceleration roadmaps.
    """
    full_overs = int(overs_completed)
    extra_balls = int(round((overs_completed - full_overs) * 10))
    balls_bowled = full_overs * 6 + extra_balls
    total_balls = total_overs * 6

    balls_remaining = max(1, total_balls - balls_bowled)
    runs_needed = max(0, target - current_runs)
    wickets_remaining = max(0, 10 - wickets_down)

    rrr = round(runs_needed / (balls_remaining / 6.0), 2)
    crr = round(current_runs / (max(1, balls_bowled) / 6.0), 2)

    # Pressure Index (0 to 100)
    # Scaled by RRR and wickets lost
    pressure_index = min(100.0, round((rrr * 5.0) + (wickets_down * 5.5), 1))

    # Strategic Posture Categorization
    if wickets_down >= 4 and balls_bowled <= 48:
        strategic_posture = "Collapse Recovery & Partnership Rebuilding"
        risk_level = "High Caution"
        tempo = "Preserve wickets, rotate strike with singles/twos, defer high-risk aerial shots until over 14."
    elif rrr >= 12.0:
        strategic_posture = "Ultra-Aggressive Boundary Blitz"
        risk_level = "Maximum Risk"
        tempo = "Require at least 2 boundaries per over; deploy power-hitters with high death strike rates immediately."
    elif rrr >= 9.5:
        strategic_posture = "Calculated Acceleration"
        risk_level = "Moderate Aggression"
        tempo = "Target 1 boundary per over + hard running; attack the opposition's 5th bowler."
    elif rrr >= 7.0:
        strategic_posture = "Controlled Accumulation"
        risk_level = "Balanced"
        tempo = "Low-risk strike rotation; keep required run rate manageable and avoid double-wicket overs."
    else:
        strategic_posture = "Clinical Cruise Control"
        risk_level = "Minimal Risk"
        tempo = "Low risk singles; punish loose deliveries and seal victory with wickets in hand."

    # Phase-by-Phase Roadmap
    current_over_num = full_overs + 1
    phase_roadmap = []

    if current_over_num <= 6:
        phase_roadmap.append({
            "phase": "Powerplay (Overs 1-6)",
            "focus": "Exploit 2-fielder restriction; target 8.5-9.0 RPO without losing more than 1 wicket."
        })
    if current_over_num <= 15:
        phase_roadmap.append({
            "phase": "Middle Overs (Overs 7-15)",
            "focus": "Minimize dot-ball percentage (<30%); run hard between wickets and attack matching matchups."
        })
    phase_roadmap.append({
        "phase": "Death Overs (Overs 16-20)",
        "focus": "Designate designated finishers; capitalize on short boundaries and pace variations."
    })

    return {
        "target": target,
        "current_runs": current_runs,
        "runs_needed": runs_needed,
        "balls_remaining": balls_remaining,
        "wickets_remaining": wickets_remaining,
        "crr": crr,
        "rrr": rrr,
        "pressure_index": pressure_index,
        "strategic_posture": strategic_posture,
        "risk_level": risk_level,
        "tempo_guidance": tempo,
        "phase_roadmap": phase_roadmap
    }


def analyze_target_defence(
    target_to_defend: int,
    current_runs_conceded: int,
    overs_completed: float,
    wickets_taken: int,
    total_overs: int = 20
) -> Dict[str, Any]:
    """
    Computes defensive run-budget, required bowling economy, and bowling allocation strategies.
    """
    full_overs = int(overs_completed)
    extra_balls = int(round((overs_completed - full_overs) * 10))
    balls_bowled = full_overs * 6 + extra_balls
    total_balls = total_overs * 6

    balls_remaining = max(1, total_balls - balls_bowled)
    run_budget_remaining = max(0, target_to_defend - current_runs_conceded)
    wickets_needed = max(0, 10 - wickets_taken)

    # Maximum economy allowed for opposition to lose
    max_defensive_econ = round(run_budget_remaining / (balls_remaining / 6.0), 2)
    opp_crr = round(current_runs_conceded / (max(1, balls_bowled) / 6.0), 2)

    if wickets_taken >= 6:
        defensive_posture = "All-Out Wicket Choke (Scenting Victory)"
        field_plan = "Keep catching fielders in the ring; bowl aggressively at the stumps to clean up the tail."
    elif max_defensive_econ <= 6.5:
        defensive_posture = "Extreme Run-Denial Defense"
        field_plan = "Set boundary sweepers at deep mid-wicket, long-on and deep point; bowl disciplined 5th-stump lines."
    elif max_defensive_econ <= 9.0:
        defensive_posture = "Balanced Strategic Defense"
        field_plan = "Mix wicket-taking deliveries with defensive boundary protection; prioritize dot balls."
    else:
        defensive_posture = "Under Siege (Defensive Crisis)"
        field_plan = "Create pressure through tight 3-ball spells; attack with primary strike bowler to buy a wicket."

    return {
        "target_to_defend": target_to_defend,
        "runs_conceded": current_runs_conceded,
        "runs_budget_left": run_budget_remaining,
        "balls_remaining": balls_remaining,
        "wickets_needed": wickets_needed,
        "opp_crr": opp_crr,
        "max_defensive_econ": max_defensive_econ,
        "defensive_posture": defensive_posture,
        "field_plan": field_plan
    }

