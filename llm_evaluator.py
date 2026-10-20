from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import requests

from gaming_his import (
    CALIBRATION_NOTE,
    format_history_digest,
    select_relevant_history,
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EVALUATOR_MODEL = "gemini-3.5-flash-lite"

PLAYER_PERSONA = """
My Gaming Preference Profile
Core Philosophy:
I play games to build and earn something that lasts. I value meaningful progression, player agency, and persistent ownership above all else.
I want my time investment to leave behind a tangible footprint—a developed character, a thriving city, a customized settlement, or a distinct playstyle.
I enjoy grinding, but only when the effort produces genuine advancement, identity, or ownership.
The moment a game replaces meaningful progress with a numerical treadmill, resource farming without consequences, or artificial playtime padding, I lose interest.

What Hooks Me:
Meaningful Depth over Bureaucracy: Deep, interacting systems and organic exploration without menu bloat or administrative chores.
Grounded Progression & Living Worlds: Earned progression without rapid power escalation that trivializes early accomplishments.
Anti-Meta Discovery: Viable builds discovered organically without forced meta guides.
Co-op Transformation: Shared play eliminates friction and turns tedious loops into engaging co-op experiences.

What Makes Me Quit:
Artificial treadmills, forced daily chores, and meaningless gear resets.
Bloated, empty open worlds with superficial side activities.
Extreme power escalation that destroys late-game challenge and pacing.
Excessive micromanagement or cognitive overhead that feels like a job.
"""


def build_llm_evaluator_prompt(
    eval_result: Dict,
    game_profile,
    game_tags: Optional[List[str]] = None
) -> str:
    m = eval_result["metrics"]
    s = eval_result["survival_analysis"]
    f = eval_result["financial_valuation"]
    wtp = f["wtp"]
    phases = eval_result.get("phase_breakdown", [])
    history_ctx = eval_result.get("history_context", {})
    best_overlap = history_ctx.get("best_overlap", 0)
    rec_score = m["recommendation_score"]
    verdict_tone = f.get("verdict_tone") or (
        "strong_positive"           if rec_score >= 0.80 else
        "positive"                  if rec_score >= 0.60 else
        "positive_with_reservations" if rec_score >= 0.40 else
        "marginal"                  if rec_score >= 0.15 else
        "skip_recommended"
    )

    # How far above fair value is the current price? This is what
    # distinguishes "wait for a small sale" from "skip entirely" -- both
    # can be DEEP SALE ONLY, but the LLM needs to know which one this is.
    price_over_wtp = f['current_price'] / max(wtp['wtp_central'], 0.01)
    if price_over_wtp <= 1.5:
        price_gap_note = (
            f"The current price is {price_over_wtp:.1f}x the central fair-value "
            "estimate -- a SMALL gap. Framing: 'worth buying soon, wait for a "
            "modest discount.'"
        )
    elif price_over_wtp <= 2.5:
        price_gap_note = (
            f"The current price is {price_over_wtp:.1f}x the central fair-value "
            "estimate -- a MODERATE gap. Framing: 'a good game, but wait for a "
            "real sale below $X' where X is the WTP ceiling."
        )
    else:
        price_gap_note = (
            f"The current price is {price_over_wtp:.1f}x the central fair-value "
            "estimate -- a LARGE gap. Framing: 'not worth the money at any "
            "realistic price.'"
        )

    tags = list(game_tags) if game_tags else list(getattr(game_profile, "tags", []) or [])
    relevant_matches = select_relevant_history(
        tags,
        top_n=3,
        min_overlap=1,
        current_title=eval_result.get("title"),
    )
    dynamic_history_digest = format_history_digest(relevant_matches)

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

    if not relevant_matches:
        novelty_note = (
            "\nNOVELTY RISK: No games in the player's history share this game's genre, "
            "mechanics, or design character. This title sits OUTSIDE the player's demonstrated "
            "comfort zone. You MUST state this explicitly, advise caution, and — when the "
            "engine verdict is 'DEEP SALE ONLY' — describe the practical recommendation as a "
            "SKIP for this player, not merely 'cheap but maybe'."
        )
    else:
        novelty_note = ""

    avg_hours = game_profile.profile.average_playtime_hours or 1
    enjoyable_ratio = s["expected_enjoyable_hours"] / avg_hours

    # --- NEW FINISH SIGNAL LOGIC ---
    finish_signal = (
        avg_hours <= 20
        and (s["expected_survival_hours"] / avg_hours) >= 0.70
    )
    finish_note = (
        "STRONG -- the model expects you to reach the end of this short experience. "
        "Cite this in the Playtime section as evidence the game holds attention to the end; "
        "do NOT frame short completion as 'running out of content'."
        if finish_signal else
        "not applicable"
    )

    score_disconnect_note = ""
    if m["personal_fit_adjusted"] >= 0.60 and enjoyable_ratio < 0.35:
        score_disconnect_note = (
            "\nNOTE -- SCORE DISCONNECT: Topical Fit is decent "
            f"({m['personal_fit_adjusted'] * 100:.0f}%) but Expected Enjoyable Hours is only "
            f"{enjoyable_ratio * 100:.0f}% of the total playtime. You MUST explicitly explain "
            "this gap in the Executive Summary -- say plainly that topical fit measures "
            "surface-level alignment, while the survival math measures how long that alignment "
            "actually survives sustained play, and that the second number is what drove the "
            "recommendation score and the verdict. Do not present these two numbers as if they agree."
        )

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
ONLY the deterministic engine output and player context below.

### PLAYER CONTEXT
{PLAYER_PERSONA.strip()}

### RELEVANT PLAY HISTORY
CRITICAL CALIBRATION RULE: {CALIBRATION_NOTE}

{dynamic_history_digest}

### ENGINE DETERMINISTIC OUTPUT
- Engine Version: {eval_result['engine_version']}
- Recommendation Score (survival + novelty aware, 0-100) -- THIS is the number the verdict
  is built from: {m['recommendation_score'] * 100:.1f}%
- Topical Fit (static similarity on paper, 0-100) -- NOT the same thing as the score above:
  {m['personal_fit_adjusted'] * 100:.1f}%{score_disconnect_note}
- Raw Static Topical Fit (pre-RCM, diagnostic only): {m['personal_fit_raw'] * 100:.1f}%
- RCM Context Multiplier: {m['rcm_multiplier']:.2f}x
- Commitment Fit: {m['commitment_fit'] * 100:.1f}%
- Value Density: {m['value_density'] * 100:.1f}%
- History Tag Overlap (best single-game match): {best_overlap} tag(s){novelty_note}
- Finish Signal: {finish_note}
- Confidence: {confidence_pct:.1f}% -- {confidence_note}

### SURVIVAL & VALUATION METRICS
- Expected Enjoyable Hours: {s['expected_enjoyable_hours']} hrs (out of {game_profile.profile.average_playtime_hours} hrs total)
- Expected Survival Hours: {s['expected_survival_hours']} hrs
- Predicted 50% Abandonment Point: {abandon_str}
- Current Retail Price: {f['current_price']} {f['currency']}
- Willingness-to-Pay Range: {wtp['wtp_low']} - {wtp['wtp_high']} {f['currency']} (Central: {wtp['wtp_central']} {f['currency']})
- Price vs. WTP: {price_gap_note}
- Engine Verdict: {f['verdict']}
- Verdict Tone: {verdict_tone}
- Veto Triggered: {f['veto_triggered']}

### PHASE BREAKDOWN (the ONLY source for any act-by-act or pacing claim)
{phase_lines}

---
### HARD CONSTRAINTS
1. Do not state or imply a purchase verdict different from the Engine Verdict above ({f['verdict']}).
2. Do not invent specific numbers, percentages, hour marks, act names, or mechanics not given above.
3. If PHASE BREAKDOWN has only one row (or none), do not fabricate act-by-act pacing.
4. Match tone implied by CONFIDENCE note -- insightful, candid, direct gaming advisor.
5. NEVER mention internal system variable names or code keys (e.g. DO NOT write 'agency_consequence', 'power_escalation').
6. DO NOT endlessly repeat numerical stats; focus on interpreting *why* the math came out this way.
7. Section 5 ("How This Compares to Your History") is MANDATORY. Never omit it, never write
   "omitted due to no matches". If the history digest says no close matches exist, write a
   short paragraph stating plainly that this game sits outside the player's demonstrated taste
   profile, and name 1-2 of the closest history games by genre or mechanic even if the overlap
   is loose. A "this is unlike anything you've played" comparison is still a comparison.

### VERDICT TONE — HOW TO CLOSE (ENFORCED)
The Engine Verdict is the FINANCIAL decision (does today's price match WTP?).
The Verdict Tone is the EMOTIONAL REGISTER of your closing paragraph. It is
equally binding and OVERRIDES any implied tone in the Engine Verdict string.

STYLE RULES:
- Write like a knowledgeable friend, not a magazine critic.
- Average sentence length 12-18 words. Never exceed 25.
- One idea per sentence. No compound-complex chains.
- Active voice: "the game rewards exploration", not "exploration is rewarded".
- Ban these words entirely: delve, tapestry, robust, myriad, navigate (as metaphor),
  journey (as metaphor), landscape (as metaphor), realm, resonate, visceral, seamless,
  holistic, leverage, ecosystem, curate, elevate, dynamic (as filler).
- No hedging stacks: "it could potentially be argued that" → "it probably is".
- Cite any given number at most ONCE. If you mention the 60% topical fit in the
  executive summary, never repeat it later.
- If you have nothing new to say, stop writing. Do not pad.

TONE BANDS:
- strong_positive (score >= 80): strong endorsement. Do NOT hedge.
- positive (60-80): clear recommend. Do NOT frame as "wait and see."
- positive_with_reservations (40-60): SOFT RECOMMEND. You MUST name at least
  one specific thing the game does well and explain WHY that thing is worth
  experiencing, even if the price is currently above fair value. The closing
  should read like "worth it when the price comes down" -- NOT like "a clinical
  financial mismatch." FORBIDDEN phrasings: "skip", "pass on it", "avoid",
  "bypass", "not worth your time", "strictly pass", "completely bypass".
- marginal (15-40): low-risk curiosity. Say "wait for a deep sale" and name
  the specific reason the fit is weak. Do NOT lean into strengths.
- skip_recommended (< 15): a clear "pass" is allowed and appropriate.

CRITICAL: The Verdict Tone is about the GAME, and the Engine Verdict is about
the PRICE. When the tone is positive_with_reservations or stronger, your
closing MUST explain why the game itself is worth trying. You may recommend
waiting for a discount. You may NOT make the game sound like a bad purchase
at any price. Read the "Price vs. WTP" gap note above -- if it says the gap is
SMALL or MODERATE, the closing must reflect that the game is a reasonable
acquisition at the right price, not a hard skip.

### INSTRUCTIONS
Write the evaluation in plain, candid language using ONLY these section headings.
Do NOT prefix any heading with a number, bullet, or letter — write the heading text alone.

**Executive Summary & Financial Verdict**
Open by contrasting the two scores explicitly: "On paper this is a X% topical match, but
the practical recommendation score is Y% because…". These are different numbers — never
present them as if they agree. Then cover price vs. value density and central WTP.

**Playtime & Enjoyment Horizon**
Expected enjoyable hours and why abandonment is predicted at the specified hour mark,
grounded in the phase breakdown.

**What Will Hook You (PROS)**
Pull from topical fit and phase breakdown's hooks.

**Where Friction Strikes (CONS)**
Pull from phase breakdown's friction list and RCM burden. Highlight specific mechanics,
pacing issues, or scaling problems that threaten an early drop-off.

**How This Compares to Your History**
MANDATORY. One short paragraph. If direct matches exist, cite 1-3 of them. If no direct matches exist, say so explicitly
and cite the closest genre neighbor from history (e.g. "closest analog in your library is
X, but even that overlaps only loosely on genre"). Never write "omitted" or "section skipped.

**Final Recommendation**
A FULL PARAGRAPH of at least 4 sentences, minimum 250 characters. Match the Engine
Verdict ({f['verdict']}). Explain the practical next step (buy now / wait for a specific
discount target / skip). A bare word like "BUY" or "WAIT" is a failed response.
"""
    return prompt


# ============================================================
# FINAL RECOMMENDATION REPAIR
# ============================================================

_BARE_VERDICT_TOKENS = {
    "buy", "sale", "wait", "skip", "buy / sale", "buy/sale",
    "deep sale only", "buy — full price", "buy - full price",
    "buy full price",
}


def _extract_final_recommendation_section(text: str) -> str:
    matches = list(re.finditer(r"final\s+recommendation", text, re.IGNORECASE))
    if not matches:
        return ""
    last = matches[-1]
    return text[last.end():].strip()


def _needs_recommendation_repair(text: str, min_chars: int = 220) -> bool:
    section = _extract_final_recommendation_section(text)
    if not section:
        return True
    stripped = section.strip().strip("*_`.#").strip()
    if len(stripped) < min_chars:
        return True
    if stripped.lower().rstrip(".!: ") in _BARE_VERDICT_TOKENS:
        return True
    return False


def _replace_final_recommendation(text: str, new_paragraph: str) -> str:
    """Replace everything from the LAST 'Final Recommendation' heading onward."""
    matches = list(re.finditer(r"final\s+recommendation", text, re.IGNORECASE))
    if not matches:
        return text.rstrip() + "\n\n### Final Recommendation\n\n" + new_paragraph
    last = matches[-1]
    prefix = text[:last.start()].rstrip()
    return prefix + "\n\n### Final Recommendation\n\n" + new_paragraph


def _generate_recommendation_paragraph(eval_result: Dict) -> str:
    f = eval_result["financial_valuation"]
    wtp = f["wtp"]

    prompt = f"""You are writing ONE concluding paragraph of a candid, personal game review
for **{eval_result['title']}**.

Facts you must respect:
- Engine verdict: {f['verdict']}
- Current retail price: {f['current_price']} {f['currency']}
- Fair-value (WTP) range: {wtp['wtp_low']} - {wtp['wtp_high']} {f['currency']} (central {wtp['wtp_central']} {f['currency']})

Write a SINGLE paragraph of 4-6 sentences, plain prose, no bullets, no headings.
It must:
1. State the practical next step in plain language (e.g. "buy now", "wait for a sale
   around $X", "skip unless..."), matching the engine verdict.
2. If the verdict is not a full-price buy, name a specific discount price target.
3. Briefly name the game's strongest appeal and its single biggest risk of drop-off.
4. Sound like a knowledgeable friend, not a formal report.
5. NOT begin with the words "BUY", "WAIT", "SKIP", or "SALE" as a standalone label.

Output ONLY the paragraph — no heading, no preamble, no markdown.
"""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{EVALUATOR_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=120)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate_final_report(eval_result: Dict, game_obj, game_tags: Optional[List[str]] = None) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{EVALUATOR_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    prompt = build_llm_evaluator_prompt(eval_result, game_obj, game_tags=game_tags)

    response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=120)
    response.raise_for_status()
    text_output = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    if text_output.startswith("```"):
        text_output = text_output.replace("```markdown", "").replace("```", "", 1).strip()

    if _needs_recommendation_repair(text_output):
        try:
            replacement = _generate_recommendation_paragraph(eval_result)
            text_output = _replace_final_recommendation(text_output, replacement)
        except Exception:
            pass

    return text_output
