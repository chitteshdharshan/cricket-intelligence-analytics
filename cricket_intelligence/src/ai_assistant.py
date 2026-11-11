"""
AI Cricket Strategy Assistant: Core Dispatcher & Dual-Mode Narrative Synthesizer
Parses natural language strategy queries, queries the statistical recommendation engines,
and generates structured 7-part strategic briefings via Gemini/OpenAI API or local deterministic fallback.
"""

import os
import re
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd

from src.player_recommendation_engine import recommend_batters, recommend_bowlers
from src.partnership_analyzer import analyze_partnership
from src.bowler_recommendation_engine import recommend_next_over_bowler
from src.match_situation_analyzer import analyze_target_chase, analyze_target_defence
from src.team_squad_builder import build_playing_xi


def _call_gemini_api(prompt: str, api_key: str) -> Optional[str]:
    """Invokes Google Gemini Flash API via lightweight REST endpoint (zero extra dependencies)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024}
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            return res_json["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Gemini API request failed: {e}")
        return None


def _call_openai_api(prompt: str, api_key: str) -> Optional[str]:
    """Invokes OpenAI API if configured."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API request failed: {e}")
        return None


def get_ai_api_key() -> tuple[Optional[str], str]:
    """Checks for configured Gemini or OpenAI keys across secrets and environment."""
    # Check Streamlit secrets first
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"], "gemini"
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"], "openai"
    except Exception:
        pass

    # Check environment variables
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY"), "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("OPENAI_API_KEY"), "openai"

    return None, "offline"


def answer_strategy_query(query: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Main entry point for strategy chatbot queries.
    Parses intent, pulls verified historical evidence, and formats the 7-part response.
    """
    q = query.strip().lower()
    bbb = data["bbb"]
    pm = data["players_master"]

    # 1. Intent Detection
    is_death = any(w in q for w in ["death", "finisher", "final over", "last over", "19th", "20th", "18th", "17th", "16th"])
    is_powerplay = any(w in q for w in ["powerplay", "opener", "opening", "first 6", "1 to 6"])
    is_spin = any(w in q for w in ["spin", "spinner", "turning", "slow bowler"])
    is_pace = any(w in q for w in ["pace", "fast bowler", "pacer", "seam", "swing"])
    is_bowler = any(w in q for w in ["bowl", "bowler", "bowling", "defend", "economy", "wicket", "overs"])
    is_squad = any(w in q for w in ["playing xi", "squad", "team", "lineup", "xi for", "build"])
    is_chase = any(w in q for w in ["chase", "chasing", "target", "required"])
    is_defend = any(w in q for w in ["defend", "defending", "low total"])
    is_partnership = any(w in q for w in ["partnership", "break this", "pair", "together"])

    # Extract team if present
    team_name_match = None
    for t in data["teams"]["team_name"].unique():
        if t.lower() in q:
            team_name_match = t
            break

    # Extract target number if mentioned (e.g. "defend 160", "chase 190")
    nums = re.findall(r"\b\d{2,3}\b", q)
    target_num = int(nums[0]) if nums else 165

    # 2. Execute Analytics Engine based on intent
    evidence_type = "general"
    evidence_data: Any = None

    if is_squad or team_name_match:
        evidence_type = "playing_xi"
        t_id = team_name_match or "Chennai Super Kings"
        evidence_data = build_playing_xi(t_id, data, pitch_condition="Spin-Friendly" if is_spin else "Balanced")

    elif is_chase:
        evidence_type = "target_chase"
        evidence_data = analyze_target_chase(target=target_num, current_runs=target_num // 2, overs_completed=10.0, wickets_down=3)

    elif is_defend:
        evidence_type = "target_defend"
        evidence_data = analyze_target_defence(target_to_defend=target_num, current_runs_conceded=target_num // 2, overs_completed=10.0, wickets_taken=3)

    elif is_bowler:
        evidence_type = "bowlers"
        sit = "death" if is_death else ("powerplay" if is_powerplay else ("spin_attack" if is_spin else "wicket_taking"))
        evidence_data = recommend_bowlers(bbb, pm, situation=sit, min_balls=24, top_n=5)

    else:
        # Batters
        evidence_type = "batters"
        sit = "death" if is_death else ("powerplay" if is_powerplay else ("against_spin" if is_spin else ("against_pace" if is_pace else "acceleration")))
        evidence_data = recommend_batters(bbb, pm, situation=sit, min_balls=30, top_n=5)

    # 3. Format into the 7-Part Structured Response
    api_key, api_provider = get_ai_api_key()
    llm_narrative = None

    if api_key:
        prompt = f"""
You are an elite Cricket Strategy Intelligence Director. Answer the user's question using ONLY the provided verified analytical data.
Question: "{query}"

Analytical Evidence:
{json.dumps(evidence_data.to_dict(orient="records") if isinstance(evidence_data, pd.DataFrame) else evidence_data, default=str)}

Format your response strictly following these 7 parts in Markdown:
1. **Match Situation**: Concise operational context.
2. **Recommended Players / Tactics**: Primary picks with exact roles.
3. **Relevant Player Statistics**: Key data points from the evidence.
4. **Reasons for Selection**: Tactical justification based on phase/style.
5. **Alternative Options**: Secondary contingency options.
6. **Potential Weaknesses / Risks**: Sample size flags, matchups, or pressure factors.
7. **Final Strategic Recommendation**: Clear executive decision.
"""
        if api_provider == "gemini":
            llm_narrative = _call_gemini_api(prompt, api_key)
        elif api_provider == "openai":
            llm_narrative = _call_openai_api(prompt, api_key)

    # Fallback deterministic generator if LLM not available or failed
    if not llm_narrative:
        llm_narrative = _generate_deterministic_response(query, evidence_type, evidence_data)

    return {
        "query": query,
        "evidence_type": evidence_type,
        "evidence_data": evidence_data,
        "response_markdown": llm_narrative,
        "engine_mode": f"Live LLM ({api_provider.title()})" if (api_key and llm_narrative and api_provider != "offline") else "Deterministic Analytics Engine (Offline Fallback)"
    }


def _generate_deterministic_response(query: str, evidence_type: str, data_payload: Any) -> str:
    """Generates the official 7-part structured cricket strategy briefing deterministically."""
    lines = []

    if evidence_type == "batters" and isinstance(data_payload, pd.DataFrame) and len(data_payload) > 0:
        top_player = data_payload.iloc[0]
        alt_players = data_payload.iloc[1:3]

        lines.append(f"### 🎯 Strategic Intelligence Briefing: Batting Selection")
        lines.append(f"**1. Match Situation:** Analyzing historical performance response for: *\"{query}\"* across 200,999 recorded deliveries.\n")
        lines.append(f"**2. Recommended Primary Batter:** **{top_player['player_name']}** (ID: `{top_player['player_id']}`)\n")
        
        lines.append("**3. Relevant Player Statistics (Verified Historical Deliveries):**")
        lines.append(f"| Player | Runs | Balls | SR | Batting Avg | Boundary % | Dot % | Sample Flag |")
        lines.append(f"|---|---|---|---|---|---|---|---|")
        for _, r in data_payload.head(5).iterrows():
            lines.append(f"| **{r['player_name']}** | {r['runs']} | {r['balls']} | **{r['strike_rate']}** | {r['average']} | {r['boundary_pct']}% | {r['dot_pct']}% | `{r['sample_flag']}` |")
        lines.append("")

        lines.append(f"**4. Reasons for Selection:**")
        lines.append(f"- **Primary Striking Power:** `{top_player['player_name']}` leads the category with a strike rate of **{top_player['strike_rate']}** across {top_player['balls']} balls.")
        lines.append(f"- **Boundary Cadence:** Scoring {top_player['boundary_pct']}% of runs in boundaries with an average of {top_player['average']}.")
        lines.append(f"- **Strike Rotation:** Dot-ball percentage restricted to {top_player['dot_pct']}%.")
        lines.append("")

        lines.append(f"**5. Alternative Player Options:**")
        for _, alt in alt_players.iterrows():
            lines.append(f"- **{alt['player_name']}**: {alt['runs']} runs in {alt['balls']} balls (SR: {alt['strike_rate']}, Avg: {alt['average']}).")
        lines.append("")

        lines.append(f"**6. Potential Weaknesses or Risks:**")
        lines.append(f"- Watch for sample size: `{top_player['sample_flag']}`. If opposition targets with contrary bowling styles, maintain anchor support at the non-striker's end.")
        lines.append("")

        lines.append(f"**7. Final Strategic Recommendation:**")
        lines.append(f"Deploy **{top_player['player_name']}** as the primary batter for this phase. Instruct to attack designated boundary channels while keeping risk measured.")

    elif evidence_type == "bowlers" and isinstance(data_payload, pd.DataFrame) and len(data_payload) > 0:
        top_bowler = data_payload.iloc[0]
        alt_bowlers = data_payload.iloc[1:3]

        lines.append(f"### 🎯 Strategic Intelligence Briefing: Bowling Recommendation")
        lines.append(f"**1. Match Situation:** Analyzing bowler tactical metrics for: *\"{query}\"* across verified competition records.\n")
        lines.append(f"**2. Recommended Primary Bowler:** **{top_bowler['player_name']}** (ID: `{top_bowler['player_id']}`)\n")

        lines.append("**3. Relevant Player Statistics (Verified Historical Spells):**")
        lines.append(f"| Bowler | Overs | Runs | Wickets | Economy | Dot % | Bowling SR | Sample Flag |")
        lines.append(f"|---|---|---|---|---|---|---|---|")
        for _, r in data_payload.head(5).iterrows():
            lines.append(f"| **{r['player_name']}** | {r['overs']} | {r['runs']} | **{r['wickets']}** | **{r['economy']}** | {r['dot_pct']}% | {r['bowling_sr']} | `{r['sample_flag']}` |")
        lines.append("")

        lines.append(f"**4. Reasons for Selection:**")
        lines.append(f"- **Economy Control:** `{top_bowler['player_name']}` operates at a stingy economy of **{top_bowler['economy']}** rpo across {top_bowler['balls']} legal deliveries.")
        lines.append(f"- **Pressure Builder:** Maintains a {top_bowler['dot_pct']}% dot-ball percentage with {top_bowler['wickets']} dismissals.")
        lines.append("")

        lines.append(f"**5. Alternative Bowling Options:**")
        for _, alt in alt_bowlers.iterrows():
            lines.append(f"- **{alt['player_name']}**: {alt['wickets']} wkts in {alt['overs']} overs (Econ: {alt['economy']}, Dot%: {alt['dot_pct']}%).")
        lines.append("")

        lines.append(f"**6. Potential Weaknesses or Risks:**")
        lines.append(f"- Boundary mitigation: Ensure sweeper covers are stationed on the dominant boundary side if batting pair attempts cross-batted hits.")
        lines.append("")

        lines.append(f"**7. Final Strategic Recommendation:**")
        lines.append(f"Hand the ball to **{top_bowler['player_name']}**. Focus on tight 4th-stump lines and yorker length to contain run rate.")

    elif evidence_type == "playing_xi" and isinstance(data_payload, dict):
        xi = data_payload.get("playing_xi", [])
        lines.append(f"### 🛡️ Strategic Intelligence Briefing: Optimal Playing XI for {data_payload.get('team_name', 'Team')}")
        lines.append(f"**1. Match Situation:** Formulation of balanced Playing XI under `{data_payload.get('pitch_condition', 'Balanced')}` conditions against `{data_payload.get('opposition_team', 'General Matchup')}`.\n")
        lines.append(f"**2. Recommended Lineup Overview:** Full 11-player squad with {data_payload.get('bowling_options_count', 5)} active bowling options.\n")

        lines.append("**3. Recommended Playing XI Lineup:**")
        lines.append(f"| No | Player | Tactical Role | Tags | Career Stats |")
        lines.append(f"|---|---|---|---|---|")
        for p in xi:
            lines.append(f"| **{p['batting_position']}** | **{p['player_name']}** | {p['role']} | `{p['tags']}` | {p['runs']} runs (SR {p['strike_rate']}), {p['wickets']} wkts (Econ {p['economy']}) |")
        lines.append("")

        lines.append(f"**4. Reasons for Selection:**")
        lines.append(f"- **Top-Order Firepower:** Positions 1-3 provide rapid powerplay capitalization.")
        lines.append(f"- **Middle-Order Spine:** All-rounders at 5-7 ensure deep batting with 5+ legitimate bowling options.")
        lines.append(f"- **Pitch Tuning:** Bowling attack customized to `{data_payload.get('pitch_condition', 'Balanced')}` specifications.")
        lines.append("")

        lines.append(f"**5. Alternative Bench Options (Reserves):**")
        for r in data_payload.get("reserves", []):
            lines.append(f"- **{r['role']}**: `{r['player_name']}` ({r['stats']})")
        lines.append("")

        lines.append(f"**6. Potential Weaknesses or Risks:**")
        lines.append(f"- Guard against top-order collapse; ensure all-rounders in positions 5-7 are prepared to stabilize in the middle overs.")
        lines.append("")

        lines.append(f"**7. Final Strategic Recommendation:**")
        lines.append(f"Field this exact Playing XI. {data_payload.get('balance_summary', '')}")

    elif evidence_type == "target_chase" and isinstance(data_payload, dict):
        lines.append(f"### 📊 Strategic Intelligence Briefing: Target Chase Strategy")
        lines.append(f"**1. Match Situation:** Chasing **{data_payload['target']}** runs. Current score: **{data_payload['current_runs']}** with **{data_payload['wickets_remaining']}** wickets in hand.")
        lines.append(f"**2. Current Formula:** **{data_payload['runs_needed']} runs needed off {data_payload['balls_remaining']} balls** (CRR: `{data_payload['crr']}` | Required Run Rate: `{data_payload['rrr']}` rpo).\n")
        
        lines.append("**3. Core Situation Matrix:**")
        lines.append(f"- **Strategic Posture:** `{data_payload['strategic_posture']}`")
        lines.append(f"- **Risk Level:** `{data_payload['risk_level']}`")
        lines.append(f"- **Pressure Index:** `{data_payload['pressure_index']}/100`")
        lines.append("")

        lines.append(f"**4. Reasons for Strategic Approach:**")
        lines.append(f"- {data_payload['tempo_guidance']}")
        lines.append("")

        lines.append(f"**5. Alternative Contingency Plan:**")
        lines.append("- If an early wicket falls in this over, drop tempo by 1.5 RPO for 6 balls before counter-attacking.")
        lines.append("")

        lines.append(f"**6. Potential Risks:**")
        lines.append(f"- Do not leave more than 35 runs for the final 2 overs. Ensure primary finisher arrives at the crease before ball 100.")
        lines.append("")

        lines.append(f"**7. Final Strategic Recommendation:**")
        lines.append(f"Target the 5th bowling option in the middle overs to bring the Required Run Rate below 9.0 before the death overs.")

    else:
        lines.append(f"### 🎯 Strategic Intelligence Response")
        lines.append(f"**1. Match Situation:** Analyzing cricket strategy request: *\"{query}\"*.\n")
        lines.append(f"**2. Tactical Finding:** Query processed against the 200,999 delivery analytical database.\n")
        lines.append(f"**3. Data Verification:** Metrics calculated using standard ICC legal ball rules and verified match IDs.\n")
        lines.append(f"**4. Strategic Guidance:** For targeted situation optimization, specify a player role (e.g. *\"best death overs finisher\"*, *\"who should bowl the 16th over\"*, or *\"Playing XI for CSK\"*).")

    return "\n".join(lines)
