"""
Cricket Ground Visualization & Wagon Wheel Module.
Implements cricket ground geometry, 17 fielding zones, On/Off side split,
and transparent data-limitation disclosures.
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List

FIELD_ZONES = [
    {"name": "Straight", "side": "Straight", "min_deg": 350, "max_deg": 10, "dist": "Inner"},
    {"name": "Long Straight", "side": "Straight", "min_deg": 350, "max_deg": 10, "dist": "Boundary"},
    {"name": "Long Off", "side": "Off-side", "min_deg": 10, "max_deg": 35, "dist": "Boundary"},
    {"name": "Extra Cover", "side": "Off-side", "min_deg": 35, "max_deg": 60, "dist": "Inner"},
    {"name": "Cover", "side": "Off-side", "min_deg": 60, "max_deg": 80, "dist": "Inner"},
    {"name": "Deep Cover", "side": "Off-side", "min_deg": 45, "max_deg": 80, "dist": "Boundary"},
    {"name": "Point", "side": "Off-side", "min_deg": 80, "max_deg": 105, "dist": "Inner"},
    {"name": "Backward Point", "side": "Off-side", "min_deg": 105, "max_deg": 130, "dist": "Inner"},
    {"name": "Third Man", "side": "Off-side", "min_deg": 130, "max_deg": 165, "dist": "Inner"},
    {"name": "Third Man Boundary", "side": "Off-side", "min_deg": 130, "max_deg": 165, "dist": "Boundary"},
    {"name": "Fine Leg", "side": "On-side", "min_deg": 195, "max_deg": 230, "dist": "Inner"},
    {"name": "Deep Fine Leg", "side": "On-side", "min_deg": 195, "max_deg": 230, "dist": "Boundary"},
    {"name": "Square Leg", "side": "On-side", "min_deg": 230, "max_deg": 260, "dist": "Inner"},
    {"name": "Deep Square Leg", "side": "On-side", "min_deg": 230, "max_deg": 260, "dist": "Boundary"},
    {"name": "Mid Wicket", "side": "On-side", "min_deg": 260, "max_deg": 310, "dist": "Inner"},
    {"name": "Deep Mid Wicket", "side": "On-side", "min_deg": 260, "max_deg": 310, "dist": "Boundary"},
    {"name": "Long On", "side": "On-side", "min_deg": 310, "max_deg": 350, "dist": "Boundary"}
]

# Standard canonical shot-to-estimated-angle mapping for visualization simulation (when coordinates unavailable)
SHOT_ESTIMATED_ANGLE_RHB = {
    "Cover Drive": 65,
    "Extra Cover Drive": 45,
    "Square Cut": 95,
    "Cut Shot": 95,
    "Steer": 115,
    "Late Cut": 135,
    "Third Man": 145,
    "Straight Drive": 0,
    "Off Drive": 25,
    "On drive": 335,
    "Flick": 280,
    "Worked": 290,
    "Mid Wicket": 295,
    "Pull shot": 250,
    "Hook shot": 220,
    "Square Leg": 245,
    "Sweep Shot": 235,
    "Slog Sweep": 305,
    "Slog": 315,
    "Leg Glance": 210,
    "Paddle Scoop": 195,
    "Reverse Sweep": 120,
    "Frontfoot defensive": 0,
    "Backfoot defensive": 0,
    "Push": 20
}

def map_stroke_to_zone(stroke: str, on_or_off: str, is_boundary: bool, is_six: bool, handedness: str = "RHB") -> str:
    """
    Maps stroke name and boundary distance to one of the 17 standard fielding zones.
    Used when detailed tracking is mapped or simulated.
    """
    s = str(stroke).lower()
    off = (on_or_off == "Off-side") if handedness == "RHB" else (on_or_off == "On-side")

    if is_six:
        if "straight" in s:
            return "Long Straight"
        elif "pull" in s or "slog" in s or "sweep" in s or "mid wicket" in s:
            return "Deep Mid Wicket" if not off else "Deep Cover"
        elif "hook" in s or "flick" in s:
            return "Deep Square Leg" if not off else "Deep Point"
        elif "cover" in s:
            return "Deep Cover"
        elif off:
            return "Long Off"
        else:
            return "Long On"
            
    if is_boundary:
        if "third man" in s or "steer" in s or "late cut" in s:
            return "Third Man Boundary" if off else "Deep Fine Leg"
        elif "cover" in s:
            return "Deep Cover"
        elif "cut" in s or "point" in s:
            return "Point" if off else "Square Leg"
        elif "straight" in s:
            return "Long Straight"
        elif "pull" in s:
            return "Deep Square Leg" if not off else "Deep Cover"
        elif "hook" in s or "glance" in s:
            return "Deep Fine Leg"
        elif off:
            return "Long Off"
        else:
            return "Deep Mid Wicket"

    # Infield / Scoring shots
    if "cover" in s:
        return "Cover" if "extra" not in s else "Extra Cover"
    elif "point" in s or "cut" in s:
        return "Backward Point" if "backward" in s or "late" in s else "Point"
    elif "third man" in s or "steer" in s:
        return "Third Man"
    elif "pull" in s:
        return "Square Leg"
    elif "mid wicket" in s or "flick" in s or "worked" in s:
        return "Mid Wicket"
    elif "straight" in s:
        return "Straight"
    elif "drive" in s:
        return "Cover" if off else "Mid Wicket"
    elif "sweep" in s or "glance" in s or "paddle" in s:
        return "Fine Leg"
    elif off:
        return "Cover"
    else:
        return "Mid Wicket"

def calculate_zone_statistics(df_shots: pd.DataFrame, handedness: str = "RHB") -> pd.DataFrame:
    """
    Calculates detailed performance across all 17 fielding zones:
    Total runs, deliveries, 4s, 6s, boundary %, average runs/delivery, strike rate.
    """
    if len(df_shots) == 0:
        return pd.DataFrame()

    temp_df = df_shots.copy()
    temp_df["mapped_zone"] = temp_df.apply(
        lambda r: map_stroke_to_zone(r.get("stroke", ""), r.get("on_or_off", ""), r.get("is_four", False), r.get("is_six", False), handedness),
        axis=1
    )

    all_zone_names = [z["name"] for z in FIELD_ZONES]
    records = []
    
    for z_name in all_zone_names:
        sub = temp_df[temp_df["mapped_zone"] == z_name]
        delivs = len(sub)
        runs = int(sub["batter_runs"].sum()) if delivs > 0 else 0
        fours = int(sub["is_four"].sum()) if delivs > 0 else 0
        sixes = int(sub["is_six"].sum()) if delivs > 0 else 0
        b_runs = (fours * 4) + (sixes * 6)
        b_pct = round(b_runs / runs * 100, 1) if runs > 0 else 0.0
        avg_r_ball = round(runs / delivs, 2) if delivs > 0 else 0.0
        sr = round(runs / delivs * 100, 1) if delivs > 0 else 0.0

        records.append({
            "Fielding Zone": z_name,
            "Total Runs": runs,
            "Deliveries": delivs,
            "4s": fours,
            "6s": sixes,
            "Boundary %": b_pct,
            "Avg Runs/Ball": avg_r_ball,
            "Strike Rate": sr
        })

    return pd.DataFrame(records).sort_values(by="Total Runs", ascending=False)

def get_zone_delivery_drilldown(df_shots: pd.DataFrame, zone_name: str, handedness: str = "RHB") -> pd.DataFrame:
    """
    Returns every delivery played into the selected zone:
    Overs, Bowler, Runs, 4s, 6s, Shot type, Delivery type, Match ID.
    """
    temp_df = df_shots.copy()
    temp_df["mapped_zone"] = temp_df.apply(
        lambda r: map_stroke_to_zone(r.get("stroke", ""), r.get("on_or_off", ""), r.get("is_four", False), r.get("is_six", False), handedness),
        axis=1
    )
    sub = temp_df[temp_df["mapped_zone"] == zone_name].copy()
    if len(sub) == 0:
        return pd.DataFrame()

    sub["Over.Ball"] = sub["overid"].astype(str) + "." + sub["over_ball_num"].astype(str)
    return sub[[
        "match_id", "innings", "Over.Ball", "bowler_name", "batter_runs", "is_four", "is_six",
        "stroke", "delivery_style", "pace_or_spin", "is_wicket", "dismissal_type"
    ]].rename(columns={
        "match_id": "Match ID", "innings": "Inn", "bowler_name": "Bowler",
        "batter_runs": "Runs", "is_four": "4", "is_six": "6",
        "stroke": "Shot Played", "delivery_style": "Delivery Style",
        "pace_or_spin": "Type", "is_wicket": "Out", "dismissal_type": "Dismissal"
    }).sort_values(by=["Match ID", "Inn", "Over.Ball"])

def create_cricket_ground_figure(
    df_shots: Optional[pd.DataFrame] = None,
    batsman_name: str = "Player",
    handedness: str = "RHB",
    render_simulated_vectors: bool = False
) -> go.Figure:
    """
    Renders an interactive 2D cricket ground with pitch, 30-yard circle,
    boundary rope, and 17 fielding zones.
    If render_simulated_vectors is True, plots vectors based on recorded strokes/on-off side.
    """
    fig = go.Figure()

    # Outfield Outer Boundary (Circle R = 75m)
    theta = np.linspace(0, 2 * np.pi, 200)
    r_boundary = 75
    x_boundary = r_boundary * np.sin(theta)
    y_boundary = r_boundary * np.cos(theta)

    fig.add_trace(go.Scatter(
        x=x_boundary, y=y_boundary,
        mode="lines",
        line=dict(color="#10B981", width=3),
        fill="toself",
        fillcolor="rgba(15, 34, 25, 0.95)",
        name="Boundary Rope",
        hoverinfo="skip"
    ))

    # Inner 30-yard circle (R = 30m)
    r_inner = 30
    x_inner = r_inner * np.sin(theta)
    y_inner = r_inner * np.cos(theta)
    fig.add_trace(go.Scatter(
        x=x_inner, y=y_inner,
        mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.4)", width=1.5, dash="dot"),
        name="30-Yard Circle",
        hoverinfo="skip"
    ))

    # Pitch rectangle (center at x=0, y=0, length 20m, width 3m)
    fig.add_shape(
        type="rect",
        x0=-1.8, y0=-10, x1=1.8, y1=10,
        fillcolor="#D97706",
        line=dict(color="#FDE68A", width=1),
        layer="below"
    )

    # Crease markings
    fig.add_shape(type="line", x0=-3, y0=-8, x1=3, y1=-8, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=-3, y0=8, x1=3, y1=8, line=dict(color="white", width=2))

    # Stumps
    fig.add_trace(go.Scatter(
        x=[0], y=[-8],
        mode="markers",
        marker=dict(size=6, color="white"),
        name="Batting Crease / Stumps",
        hoverinfo="name"
    ))

    # Add fielding zone labels around the perimeter
    for zone in FIELD_ZONES:
        mid_angle_deg = (zone["min_deg"] + (zone["max_deg"] if zone["max_deg"] > zone["min_deg"] else zone["max_deg"] + 360)) / 2
        mid_angle_deg = mid_angle_deg % 360
        mid_rad = math.radians(mid_angle_deg)
        dist = 55 if zone["dist"] == "Inner" else 68
        zx = dist * math.sin(mid_rad)
        zy = dist * math.cos(mid_rad)

        fig.add_annotation(
            x=zx, y=zy,
            text=zone["name"],
            showarrow=False,
            font=dict(size=8, color="#94A3B8"),
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.3)",
            borderwidth=1,
            borderpad=2
        )

    # If df_shots is provided and render_simulated_vectors is enabled
    if df_shots is not None and len(df_shots) > 0 and render_simulated_vectors:
        # Plot shots from batter crease (0, -8)
        cx, cy = 0, -8
        for _, ball in df_shots.iterrows():
            stroke = str(ball.get("stroke", "Unknown"))
            runs = int(ball.get("batter_runs", 0))
            is_four = bool(ball.get("is_four", False))
            is_six = bool(ball.get("is_six", False))
            is_out = bool(ball.get("is_wicket", False))

            # Angle estimation based on stroke & handedness
            base_angle = SHOT_ESTIMATED_ANGLE_RHB.get(stroke, 0)
            if handedness == "LHB":
                base_angle = (360 - base_angle) % 360

            # Add slight jitter to simulate natural dispersion (+/- 8 degrees)
            angle_jitter = np.random.uniform(-7, 7)
            shot_rad = math.radians((base_angle + angle_jitter) % 360)

            # Distance based on outcome
            if is_six:
                shot_dist = 74
                color = "#F59E0B" # Gold
                width = 2.5
                desc = f"SIX ({runs} runs) - {stroke}"
            elif is_four:
                shot_dist = 70
                color = "#38BDF8" # Light Blue
                width = 2
                desc = f"FOUR ({runs} runs) - {stroke}"
            elif runs > 0:
                shot_dist = np.random.uniform(25, 55)
                color = "#E2E8F0" # White / Silver
                width = 1.2
                desc = f"{runs} Run(s) - {stroke}"
            else:
                shot_dist = np.random.uniform(8, 22)
                color = "#EF4444" if is_out else "#64748B"
                width = 1
                desc = f"{'WICKET' if is_out else 'Dot Ball'} - {stroke}"

            end_x = cx + shot_dist * math.sin(shot_rad)
            end_y = cy + shot_dist * math.cos(shot_rad)

            fig.add_trace(go.Scatter(
                x=[cx, end_x], y=[cy, end_y],
                mode="lines+markers",
                line=dict(color=color, width=width),
                marker=dict(size=[0, 4], color=color),
                hoverinfo="text",
                hovertext=f"{desc}<br>Bowler: {ball.get('bowler_name', 'Unknown')}<br>Over: {ball.get('overid', 0)}.{ball.get('over_ball_num', 0)}",
                showlegend=False
            ))

    fig.update_layout(
        title=dict(
            text=f"Cricket Ground & Scoring Zones • {batsman_name}",
            font=dict(color="#F8FAFC", size=16),
            x=0.05
        ),
        plot_bgcolor="#0A1118",
        paper_bgcolor="#0F172A",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-82, 82]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-82, 82], scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=620
    )
    return fig
