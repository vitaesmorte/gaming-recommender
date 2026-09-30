from __future__ import annotations

import os
from typing import Dict

import requests


# ============================================================
# GEMINI API KEY — same pattern as first_llm.py, reused here so both
# stages read the one GEMINI_API_KEY environment variable you already
# set on Render.
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EVALUATOR_MODEL = "gemini-flash-lite-latest"


# ============================================================
# LLM EVALUATOR — PROMPT CONSTRUCTION
# ============================================================
#
# TOKEN-COST DESIGN (the problem you raised):
#
#   You don't need to re-feed your preference PDF on every call. The
#   NUMERIC application of your preferences (ideal / tolerance / weight
#   per variable) already happened deterministically inside the engine
#   -- calculate_base_fit, the RCM, the hazard curve. The LLM's job
#   here isn't to re-derive your preferences from a document; it's to
#   narrate numbers that already encode them.
#
#   So: distill the PDF ONCE into two small, static blocks below --
#   PLAYER_PERSONA (your voice: what hooks you, what makes you quit,
#   a few loved/dropped examples) and PLAY_HISTORY_DIGEST (a compact
#   table of relevant games + hours + your reaction). Fill these in by
#   hand, or with a single one-off "distill my preference PDF into
#   ~200 words" prompt -- that's the only time the raw PDF gets read.
#   They live here as constants and get embedded in every prompt at
#   near-zero token cost. Update them only when your taste or history
#   meaningfully changes, not per evaluation.
#
#   If your provider supports server-side context caching (Gemini has
#   this via cachedContent), you can additionally cache these blocks
#   to avoid re-billing tokens for text that's already constant -- but
#   the distillation above is the actual fix; caching is a bonus on
#   top of text that no longer changes per call.
# ============================================================

PLAYER_PERSONA = """
My Gaming Preference Profile
Core Philosophy:
I play games to build and earn something that lasts. I value meaningful progression, player agency, and persistent ownership above all else.
I want my time investment to leave behind a tangible footprint—a developed character, a thriving city, a customized settlement, or a distinct playstyle.
I enjoy grinding, but only when the effort produces genuine advancement, identity, or ownership.
The moment a game replaces meaningful progress with a numerical treadmill, resource farming without consequences, or artificial playtime padding, I lose interest.

What Hooks Me:

Meaningful Depth over Bureaucracy: I love deep, interacting systems, strategic complexity, and organic exploration.
However, complexity must create interesting decisions, not administrative chores, menu bloat, or bookkeeping.

Grounded Progression & Living Worlds: Progression should feel earned. Finding a simple shield or advancing a town age should feel impactful.
I prefer grounded progression over rapid power escalation that trivializes early accomplishments.

Anti-Meta Discovery: I want to discover viable builds and strategies myself.
I heavily dislike games that force me into wikis, optimal meta builds, or strict out-of-game guides.

Co-op Transformation: Shared play fundamentally changes my tolerance for friction.
Tedious mechanics, harsh difficulty, or repetitive loops that feel frustrating solo become deeply engaging shared experiences in co-op.

What Makes Me Quit:

Artificial treadmills, forced daily chores, and meaningless gear resets.
Bloated, empty open worlds with superficial side activities.
Extreme power escalation that destroys late-game challenge and pacing.
Excessive micromanagement, menu navigation, or cognitive overhead that feels like a job.

Benchmark Games:

Loved: Ultima Online (unrivaled persistent world, player identity, and purposeful grind),
Heroes of Might and Magic III (strategic depth without administrative bloat), and Witcher 3 (grounded, living world with meaningful atmosphere and sense of ownership and great storyline and side quests).

Dropped / Mixed: Diablo IV & Modern WoW (both devolve into meaningless endgame treadmills and forced metas), Disco Elysium (lacked character connection and turned into a puzzle adventure),
BG3 (loved Act 1, but dropped off when post-Act 2 power escalation ruined pacing).
"""

PLAY_HISTORY_DIGEST = """
Game Playtime
Verdict / Signal:
Ultima Online: 465h, Gold Standard: Ultimate persistent sandbox, ownership, and purposeful community grind.
Baldur's Gate 3: 295h, Mixed (6.5–7/10): Loved Acts 1–2; dropped off due to late-game power escalation and pacing.
Cities: Skylines: 294h, Loved: Top-tier systemic management, creative freedom, and visible ownership.
Divinity: Original Sin 2: 283h, Loved: Deep tactical combat, agency, and excellent co-op integration.
Dark and Light: 157h, Mixed: Strong survival/magic persistence, but suffered from server emptiness and routine friction.
World of Warcraft: 145h, Mixed: Loved early leveling and identity; abandoned due to modern endgame treadmills.
Elden Ring (inc. Nightreign): 225h, Conditional: Frustrating solo (meta-dependent), but turns into top-tier fun in co-op.
Witcher 3: 123h, Loved: Masterclass in narrative integration, atmosphere, and grounded exploration.
Diablo 2 / Diablo 4: 200h, Dropped: Great initial co-op leveling; abandoned once forced meta and gear treadmills hit.
Age of Empires 2 & 4: 195h, Loved: Satisfying RTS progression, clear structure, and non-toxic team co-op play.
Heroes of Might and Magic III: 65h, Loved: Perfect strategic depth and turn-based ownership without administrative bloat.
Red Dead Redemption 265h, Loved: Highly immersive, living world with grounded discoveries and camp ownership.
Anno 1800: 35h, Mixed: Loved early production systems, but abandoned when late-game hit administrative bloat.
Disco Elysium: 9.5h, Dropped: Disconnected from the main character; felt like an adventure puzzle game without atmosphere.
Hellblade: Senua's Sacrifice: 83m, Dropped: Fast quit due to lack of character connection and zero sense of achievement/progression.
"""


def build_llm_evaluator_prompt(eval_result: Dict, game_profile) -> str:
    """
    Formats engine metrics + condensed player context into a prompt for
    the LLM evaluator.

    GROUNDING RULE: every specific claim the LLM makes about pacing,
    friction, or hooks must trace back to a field in this prompt.
    phase_breakdown exists specifically so the LLM has real per-act
    numbers instead of having to invent which act causes friction --
    without it, earlier drafts of this prompt produced confident,
    specific-sounding claims ("98% match", named acts, exact hour
    breakdowns) that weren't actually derivable from the data given.
    """
    m = eval_result["metrics"]
    s = eval_result["survival_analysis"]
    f = eval_result["financial_valuation"]
    wtp = f["wtp"]
    phases = eval_result.get("phase_breakdown", [])

    abandon_str = (
        f"Hour {s['predicted_50pct_abandonment_hour']}"
        if s["predicted_50pct_abandonment_hour"]
        else "None predicted (low hazard)"
    )

    if phases:
        phase_lines = "\n".join(
            f"  - {p['phase_name']} (hours {p['hour_range']}): "
            f"avg enjoyment {p['avg_enjoyment'] * 100:.0f}%, avg hazard {p['avg_hazard']:.3f} "
            f"| hooks: {', '.join(p['top_hooks']) or 'none'} "
            f"| friction: {', '.join(p['top_friction']) or 'none'}"
            for p in phases
        )
    else:
        phase_lines = "  - No phase data available for this game."

    confidence_pct = m["confidence"] * 100
    if confidence_pct < 50:
        confidence_note = (
            "LOW CONFIDENCE. Hedge throughout -- use 'the model estimates', 'likely', "
            "'roughly' rather than stating numbers as settled fact."
        )
    elif confidence_pct < 75:
        confidence_note = "MODERATE CONFIDENCE. Present numbers as strong estimates, not certainties."
    else:
        confidence_note = "HIGH CONFIDENCE. Numbers can be presented more assertively."

    prompt = f"""
You are an expert game advisor evaluating **{eval_result['title']}** for a specific player, using
ONLY the deterministic engine output and player context below. This is a mathematical model with
real uncertainty (see CONFIDENCE) -- your job is to narrate and contextualize its output, not to
independently guess at facts about the game that aren't given here.

### PLAYER CONTEXT (distilled once from a longer preference note -- do not ask for or expect the raw document)
{PLAYER_PERSONA.strip()}

### RELEVANT PLAY HISTORY
{PLAY_HISTORY_DIGEST.strip()}

### ENGINE DETERMINISTIC OUTPUT
- Engine Version: {eval_result['engine_version']}
- Raw Base Fit: {m['personal_fit_raw'] * 100:.1f}%
- RCM Context Multiplier: {m['rcm_multiplier']:.2f}x
- Adjusted Personal Fit: {m['personal_fit_adjusted'] * 100:.1f}%
- Commitment Fit: {m['commitment_fit'] * 100:.1f}%
- Value Density: {m['value_density'] * 100:.1f}%
- Confidence: {confidence_pct:.1f}% -- {confidence_note}

### SURVIVAL & VALUATION METRICS
- Expected Enjoyable Hours: {s['expected_enjoyable_hours']} hrs (out of {game_profile.profile.average_playtime_hours} hrs total)
- Expected Survival Hours: {s['expected_survival_hours']} hrs
- Predicted 50% Abandonment Point: {abandon_str}
- Current Retail Price: {f['current_price']} {f['currency']}
- Fair Value Target (WTP): ${wtp['wtp_mid']} {f['currency']} (Range: ${wtp['wtp_low']} - ${wtp['wtp_high']})
- Engine Verdict: {f['verdict']}
- Veto Triggered: {f['veto_triggered']}

### PHASE BREAKDOWN (the ONLY source for any act-by-act or pacing claim)
{phase_lines}

---
### HARD CONSTRAINTS
1. Do not state or imply a purchase verdict different from the Engine Verdict above
   ({f['verdict']}). Your job is to explain and contextualize it, not recompute it.
2. Do not invent specific numbers, percentages, hour marks, act names, or mechanics that
   aren't given above. Any mechanic you name as a hook or friction source must come from
   the PHASE BREAKDOWN's hooks/friction lists, or from RELEVANT PLAY HISTORY.
3. If PHASE BREAKDOWN has only one row (or none), do not fabricate act-by-act pacing --
   speak only in terms of the overall hour marks given.
4. Match the tone implied by the CONFIDENCE note above and insightful, candid, and direct—like 
    an experienced gaming buddy who knows John's exact gaming pet peeves and favorite systems.
5. NEVER mention internal system variable names or code keys (e.g. DO NOT write 'agency_consequence', 'power_escalation', 'social_coop_integration', or 'atmosphere'). 
    Translate all underlying mechanics into natural, immersive gaming terms.
6. DO NOT endlessly repeat the numerical stats in the body paragraphs; 
    focus on interpreting *why* the math came out this way for John's specific playstyle.

### INSTRUCTIONS
Write the evaluation in plain, candid language, with these sections:
1. **Executive Summary & Financial Verdict** -- price vs. value density and central WTP.
2. **Playtime & Enjoyment Horizon** -- expected enjoyable hours and why abandonment is
   predicted at the specified hour mark, grounded in the phase breakdown.
3. **What Will Hook You (Pros)** -- pull from personal fit and the phase breakdown's hooks.
4. **Where Friction Strikes (Cons)** -- pull from the phase breakdown's friction list and
   what the RCM multiplier implies about burden.
5. **How This Compares to Your History** -- one short paragraph referencing 1-2 games from
   your play history, only if genuinely relevant -- omit this section rather than force a
   weak comparison.
6. **Final Recommendation** -- must match the Engine Verdict.
"""
    return prompt


def generate_final_report(eval_result: Dict, game_obj) -> str:
    """
    This is the function main.py actually calls. build_llm_evaluator_prompt()
    above only builds the prompt text -- this is the piece that was missing:
    it sends that prompt to Gemini and returns the narrative text main.py
    hands back to the browser.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{EVALUATOR_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    prompt = build_llm_evaluator_prompt(eval_result, game_obj)

    response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=120)
    response.raise_for_status()

    text_output = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    # This output is prose, not JSON -- but strip fences defensively in
    # case Gemini wraps it in a markdown block anyway.
    if text_output.startswith("```"):
        text_output = text_output.replace("```markdown", "").replace("```", "", 1).strip()

    return text_output


if __name__ == "__main__":
    # Smoke test using the same Pathfinder run as the engine's own __main__ block.
    import json
    from game_eval_v0_4_3 import (
        Game, GamePhase, GameProfile, PlayerPreference, PlayerProfile,
        VARIABLE_DEFINITIONS, DEFAULT_RCM, evaluate_game,
    )

    pathfinder_vector = {
        "meaningful_progression": 0.45, "meaningful_persistence": 0.40,
        "agency_consequence": 0.20, "player_identity": 0.70,
        "systemic_depth": 0.65, "atmosphere": 0.88,
        "narrative_integration": 0.50, "content_density": 0.60,
        "effective_variety": 0.65, "exploratory_autonomy": 0.20,
        "social_coop_integration": 0.95, "technical_polish": 0.78,
        "convenience": 0.80, "scale": 0.50, "persistence_stability": 0.80,
        "complexity": 0.55, "administrative_burden": 0.25, "grind": 0.65,
        "difficulty": 0.80, "unintentional_friction": 0.15,
        "meta_dependence": 0.40, "restart_cost": 0.05,
        "power_escalation": 0.30, "focused_structure": 0.90,
        "extreme_competitive_pvp": 0.0,
    }
    pathfinder_vector["power_escalation"] = (pathfinder_vector["power_escalation"] * 2.0) - 1.0

    player = PlayerProfile(
        preferences={
            "meaningful_progression": PlayerPreference(0.8, 0.19, 1.1),
            "meaningful_persistence": PlayerPreference(0.85, 0.20, 1.3),
            "agency_consequence": PlayerPreference(0.91, 0.21, 1.4),
            "player_identity": PlayerPreference(0.92, 0.20, 1.2),
            "systemic_depth": PlayerPreference(0.6, 0.55, 0.8),
            "atmosphere": PlayerPreference(0.71, 0.15, 1.2),
            "narrative_integration": PlayerPreference(0.8, 0.22, 0.9),
            "content_density": PlayerPreference(0.5, 0.8, 0.55),
            "effective_variety": PlayerPreference(0.65, 0.7, 0.9),
            "exploratory_autonomy": PlayerPreference(0.7, 0.6, 0.9),
            "social_coop_integration": PlayerPreference(0.65, 0.8, 0.9),
            "technical_polish": PlayerPreference(0.8, 0.8, 0.8),
            "convenience": PlayerPreference(0.75, 0.25, 0.8),
            "scale": PlayerPreference(0.70, 0.30, 0.7),
            "persistence_stability": PlayerPreference(0.7, 0.60, 0.9),
            "complexity": PlayerPreference(0.58, 0.60, 1.0),
            "administrative_burden": PlayerPreference(0.30, 0.25, 1.1),
            "grind": PlayerPreference(0.65, 0.30, 0.9),
            "difficulty": PlayerPreference(0.60, 0.40, 0.8),
            "unintentional_friction": PlayerPreference(0.3, 0.5, 0.8),
            "meta_dependence": PlayerPreference(0.25, 0.23, 1.1),
            "restart_cost": PlayerPreference(0.35, 0.3, 0.8),
            "power_escalation": PlayerPreference(-0.45, 0.30, 1.1),
            "focused_structure": PlayerPreference(0.63, 0.55, 0.8),
        },
        veto_tags={"extreme_competitive_pvp": True},
        human_factor=1.0,
    )

    game = Game(
        game_id="pathfinder-kingmaker", title="Pathfinder: Kingmaker",
        release_date="2018-09-25", current_price=19.99,
        profile=GameProfile(
            title="Pathfinder: Kingmaker", average_playtime_hours=100,
            static_vector=pathfinder_vector,
            phases=[
                GamePhase("Act 1: Stolen Lands", 0, 20, 1, {"administrative_burden": 0.45, "power_escalation": -0.50}),
                GamePhase("Act 2-4: Kingdom Management", 21, 60, 2, {"administrative_burden": 0.85, "power_escalation": 0.20}),
                GamePhase("Act 5+: Endgame Power", 61, 100, 3, {"administrative_burden": 0.75, "power_escalation": 0.85}),
            ],
        ),
    )

    result = evaluate_game(game=game, player=player, variable_definitions=VARIABLE_DEFINITIONS, rcm=DEFAULT_RCM)
    print(build_llm_evaluator_prompt(result, game))
