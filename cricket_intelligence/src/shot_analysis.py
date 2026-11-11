"""
Shot Type, Delivery Style, Pace vs Spin & Positional Analysis Engine.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional

def get_pace_vs_spin_stats(df: pd.DataFrame, player_id: int, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Computes comprehensive comparison between Pace and Spin bowling for a batsman.
    Returns overall Pace vs Spin KPI table and granular bowling style breakdown.
    """
    p_df = df[df["striker_id"] == player_id]
    if len(p_df) == 0:
        return {"summary": pd.DataFrame(), "styles": pd.DataFrame()}

    summary = []
    for p_or_s in ["Pace", "Spin"]:
        sub = p_df[p_df["pace_or_spin"] == p_or_s]
        balls = int(sub["is_legal_ball_batter"].sum())
        runs = int(sub["batter_runs"].sum())
        fours = int(sub["is_four"].sum())
        sixes = int(sub["is_six"].sum())
        dots = int(sub["is_dot_ball_batter"].sum())
        dismissals = int((sub["is_wicket"] & ((sub["dismissed_player_id"] == player_id) | (sub["dismissed_player_id"].isna()))).sum())

        sr = round(runs / balls * 100, 2) if balls > 0 else 0.0
        avg = round(runs / dismissals, 2) if dismissals > 0 else runs
        boundary_runs = (fours * 4) + (sixes * 6)
        boundary_pct = round(boundary_runs / runs * 100, 2) if runs > 0 else 0.0
        dot_pct = round(dots / balls * 100, 2) if balls > 0 else 0.0
        wkt_freq = round(balls / dismissals, 1) if dismissals > 0 else 0.0
        four_freq = round(balls / fours, 1) if fours > 0 else 0.0
        six_freq = round(balls / sixes, 1) if sixes > 0 else 0.0

        summary.append({
            "Bowling Type": p_or_s,
            "Balls Faced": balls,
            "Runs Scored": runs,
            "Strike Rate": sr,
            "Average": avg,
            "Fours": fours,
            "Sixes": sixes,
            "Dot Balls": dots,
            "Dot %": dot_pct,
            "Dismissals": dismissals,
            "Boundary %": boundary_pct,
            "Balls / 4": four_freq,
            "Balls / 6": six_freq,
            "Wicket Freq (Balls/Out)": wkt_freq
        })

    sum_df = pd.DataFrame(summary)

    # Detailed bowling style breakdown
    style_group = p_df.groupby(["pace_or_spin", "bowling_type"]).agg(
        balls=("is_legal_ball_batter", "sum"),
        runs=("batter_runs", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dots=("is_dot_ball_batter", "sum"),
        dismissals=("is_wicket", lambda x: (x & ((p_df.loc[x.index, "dismissed_player_id"] == player_id) | (p_df.loc[x.index, "dismissed_player_id"].isna()))).sum())
    ).reset_index()

    style_group["Strike Rate"] = np.where(style_group["balls"] > 0, (style_group["runs"] / style_group["balls"] * 100).round(2), 0.0)
    style_group["Average"] = np.where(style_group["dismissals"] > 0, (style_group["runs"] / style_group["dismissals"]).round(2), style_group["runs"])
    style_group["Dot %"] = np.where(style_group["balls"] > 0, (style_group["dots"] / style_group["balls"] * 100).round(2), 0.0)

    style_df = style_group.sort_values(by="balls", ascending=False).rename(columns={
        "pace_or_spin": "Category",
        "bowling_type": "Bowling Style",
        "balls": "Balls",
        "runs": "Runs",
        "fours": "4s",
        "sixes": "6s",
        "dots": "Dots",
        "dismissals": "Outs"
    })

    return {"summary": sum_df, "styles": style_df}

def get_delivery_type_analysis(df: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """
    Computes batsman performance against individual delivery types
    (Yorker, Bouncer, Full Toss, Inswing, Outswing, Off Break, Leg Break, Googly, etc.).
    Preserves and reports unclassified / unknown deliveries.
    """
    p_df = df[df["striker_id"] == player_id]
    if len(p_df) == 0:
        return pd.DataFrame()

    res = p_df.groupby("delivery_style").agg(
        balls=("is_legal_ball_batter", "sum"),
        runs=("batter_runs", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dots=("is_dot_ball_batter", "sum"),
        dismissals=("is_wicket", lambda x: (x & ((p_df.loc[x.index, "dismissed_player_id"] == player_id) | (p_df.loc[x.index, "dismissed_player_id"].isna()))).sum())
    ).reset_index()

    res["Strike Rate"] = np.where(res["balls"] > 0, (res["runs"] / res["balls"] * 100).round(2), 0.0)
    res["Dot %"] = np.where(res["balls"] > 0, (res["dots"] / res["balls"] * 100).round(2), 0.0)
    res["Boundary %"] = np.where(res["runs"] > 0, (((res["fours"] * 4 + res["sixes"] * 6) / res["runs"]) * 100).round(2), 0.0)

    res = res.rename(columns={
        "delivery_style": "Delivery Style",
        "balls": "Balls Faced",
        "runs": "Runs Scored",
        "fours": "4s",
        "sixes": "6s",
        "dots": "Dots",
        "dismissals": "Dismissals"
    })
    return res.sort_values(by="Balls Faced", ascending=False)

def get_stroke_analysis(df: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """
    Computes performance by stroke type (Cover Drive, Pull, Flick, Square Cut, Straight Drive, etc.).
    """
    p_df = df[df["striker_id"] == player_id]
    if len(p_df) == 0:
        return pd.DataFrame()

    res = p_df.groupby("stroke").agg(
        balls=("is_legal_ball_batter", "sum"),
        runs=("batter_runs", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dots=("is_dot_ball_batter", "sum"),
        dismissals=("is_wicket", lambda x: (x & ((p_df.loc[x.index, "dismissed_player_id"] == player_id) | (p_df.loc[x.index, "dismissed_player_id"].isna()))).sum())
    ).reset_index()

    res["Runs / Ball"] = np.where(res["balls"] > 0, (res["runs"] / res["balls"]).round(2), 0.0)
    res["Strike Rate"] = np.where(res["balls"] > 0, (res["runs"] / res["balls"] * 100).round(2), 0.0)
    res["Boundary %"] = np.where(res["runs"] > 0, (((res["fours"] * 4 + res["sixes"] * 6) / res["runs"]) * 100).round(2), 0.0)

    res = res.rename(columns={
        "stroke": "Shot Played",
        "balls": "Times Played",
        "runs": "Runs Scored",
        "fours": "4s",
        "sixes": "6s",
        "dots": "Dots",
        "dismissals": "Dismissals"
    })
    return res.sort_values(by="Runs Scored", ascending=False)


def get_on_off_side_analysis(df: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """
    Computes On-side vs Off-side scoring split from the on_or_off column.
    """
    p_df = df[df["striker_id"] == player_id]
    if len(p_df) == 0:
        return pd.DataFrame()

    res = p_df.groupby("on_or_off").agg(
        deliveries=("is_legal_ball_batter", "sum"),
        runs=("batter_runs", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dots=("is_dot_ball_batter", "sum")
    ).reset_index()

    res["Strike Rate"] = np.where(res["deliveries"] > 0, (res["runs"] / res["deliveries"] * 100).round(2), 0.0)
    res["Runs Share %"] = np.where(res["runs"].sum() > 0, (res["runs"] / res["runs"].sum() * 100).round(1), 0.0)
    res = res.rename(columns={
        "on_or_off": "Field Side",
        "deliveries": "Deliveries Faced",
        "runs": "Runs Scored",
        "fours": "4s",
        "sixes": "6s",
        "dots": "Dots"
    })
    return res.sort_values(by="Runs Scored", ascending=False)


def get_shot_performance_summary(
    df: pd.DataFrame,
    player_id: int,
    filters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Computes comprehensive shot-wise batting performance summary across all shot types.
    Calculates exact deliveries, legal balls faced, runs, fours, sixes, dismissals of
    the selected batsman, dot balls, singles, twos, threes, strike rate, and boundary %.
    Strictly respects player, competition, season, match, innings, and date filters.
    """
    from src.batting_analysis import apply_filters

    p_df = df[df["striker_id"] == player_id].copy()
    if filters:
        p_df = apply_filters(p_df, filters)

    if len(p_df) == 0:
        return pd.DataFrame(columns=[
            "Shot Played", "Balls Played", "Total Deliveries", "Runs Scored",
            "Fours", "Sixes", "Dismissals", "Dot Balls", "Singles", "Twos", "Threes",
            "Strike Rate", "Boundary %"
        ])

    p_df["stroke"] = p_df["stroke"].fillna("Unknown / Unclassified").astype(str).str.strip()
    p_df["stroke"] = p_df["stroke"].replace("", "Unknown / Unclassified")

    rows = []
    for shot, sub in p_df.groupby("stroke"):
        deliveries = len(sub)
        balls = int(sub["is_legal_ball_batter"].sum())
        runs = int(sub["batter_runs"].sum())
        fours = int(sub["is_four"].sum())
        sixes = int(sub["is_six"].sum())
        dots = int(sub["is_dot_ball_batter"].sum())
        singles = int((sub["batter_runs"] == 1).sum())
        twos = int((sub["batter_runs"] == 2).sum())
        threes = int((sub["batter_runs"] == 3).sum())

        # Count outs of selected batsman using dismissed_player_id and verified dismissal fields
        # Excludes wickets of partner (e.g. non-striker run out)
        outs = int((sub["is_wicket"] & ((sub["dismissed_player_id"] == player_id) | (sub["dismissed_player_id"].isna()))).sum())

        sr = round(runs / balls * 100, 2) if balls > 0 else 0.0
        boundary_runs = (fours * 4) + (sixes * 6)
        boundary_pct = round(boundary_runs / runs * 100, 2) if runs > 0 else 0.0

        rows.append({
            "Shot Played": shot,
            "Balls Played": balls,
            "Total Deliveries": deliveries,
            "Runs Scored": runs,
            "Fours": fours,
            "Sixes": sixes,
            "Dismissals": outs,
            "Dot Balls": dots,
            "Singles": singles,
            "Twos": twos,
            "Threes": threes,
            "Strike Rate": sr,
            "Boundary %": boundary_pct
        })

    res = pd.DataFrame(rows)
    if len(res) > 0:
        res = res.sort_values(by=["Runs Scored", "Balls Played"], ascending=[False, False]).reset_index(drop=True)
    return res


def get_shot_metrics(
    df: pd.DataFrame,
    player_id: int,
    shot_name: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Computes exact detailed metrics for a particular shot (or all shots combined if shot_name is None or 'All').
    Returns total deliveries, runs, fours, sixes, dismissals, dot balls, singles, twos, threes, strike rate, boundary %.
    """
    from src.batting_analysis import apply_filters

    p_df = df[df["striker_id"] == player_id].copy()
    if filters:
        p_df = apply_filters(p_df, filters)

    p_df["stroke"] = p_df["stroke"].fillna("Unknown / Unclassified").astype(str).str.strip()
    p_df["stroke"] = p_df["stroke"].replace("", "Unknown / Unclassified")

    if shot_name and shot_name != "All":
        p_df = p_df[p_df["stroke"] == shot_name]

    deliveries = len(p_df)
    balls = int(p_df["is_legal_ball_batter"].sum()) if deliveries > 0 else 0
    runs = int(p_df["batter_runs"].sum()) if deliveries > 0 else 0
    fours = int(p_df["is_four"].sum()) if deliveries > 0 else 0
    sixes = int(p_df["is_six"].sum()) if deliveries > 0 else 0
    dots = int(p_df["is_dot_ball_batter"].sum()) if deliveries > 0 else 0
    singles = int((p_df["batter_runs"] == 1).sum()) if deliveries > 0 else 0
    twos = int((p_df["batter_runs"] == 2).sum()) if deliveries > 0 else 0
    threes = int((p_df["batter_runs"] == 3).sum()) if deliveries > 0 else 0

    outs = int((p_df["is_wicket"] & ((p_df["dismissed_player_id"] == player_id) | (p_df["dismissed_player_id"].isna()))).sum()) if deliveries > 0 else 0

    sr = round(runs / balls * 100, 2) if balls > 0 else 0.0
    boundary_runs = (fours * 4) + (sixes * 6)
    boundary_pct = round(boundary_runs / runs * 100, 2) if runs > 0 else 0.0

    return {
        "shot_name": shot_name if shot_name else "All Shots",
        "deliveries": deliveries,
        "balls": balls,
        "runs": runs,
        "fours": fours,
        "sixes": sixes,
        "dismissals": outs,
        "dots": dots,
        "singles": singles,
        "twos": twos,
        "threes": threes,
        "strike_rate": sr,
        "boundary_pct": boundary_pct
    }


def create_shot_comparison_chart(
    summary_df: pd.DataFrame,
    top_n: int = 12,
    selected_shot: Optional[str] = None
) -> go.Figure:
    """
    Interactive grouped bar chart comparing Fours, Sixes, and Dismissals across shot types.
    """
    if len(summary_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No Shot Data Available",
            plot_bgcolor="#0F172A",
            paper_bgcolor="#0B0F17",
            font=dict(color="#F8FAFC")
        )
        return fig

    plot_df = summary_df.head(top_n).copy()
    if selected_shot and selected_shot != "All" and selected_shot not in plot_df["Shot Played"].values:
        sel_row = summary_df[summary_df["Shot Played"] == selected_shot]
        if len(sel_row) > 0:
            plot_df = pd.concat([plot_df, sel_row]).drop_duplicates(subset=["Shot Played"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["Shot Played"],
        y=plot_df["Fours"],
        name="Fours (4s)",
        marker_color="#38BDF8"
    ))
    fig.add_trace(go.Bar(
        x=plot_df["Shot Played"],
        y=plot_df["Sixes"],
        name="Sixes (6s)",
        marker_color="#A855F7"
    ))
    fig.add_trace(go.Bar(
        x=plot_df["Shot Played"],
        y=plot_df["Dismissals"],
        name="Dismissals (Outs)",
        marker_color="#EF4444"
    ))

    fig.update_layout(
        barmode="group",
        title=f"Fours, Sixes & Dismissals by Shot Type (Top {len(plot_df)} Scoring Shots)",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0B0F17",
        font=dict(color="#F8FAFC", size=12),
        xaxis=dict(title="Shot Played", gridcolor="#1E293B", tickangle=-30),
        yaxis=dict(title="Total Events", gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=20, t=60, b=80)
    )
    return fig

