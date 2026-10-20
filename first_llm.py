import os
import json
import re
import requests

from gaming_his import TAG_VOCABULARY

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

VARIABLES = {
    "meaningful_progression": "How strongly the game provides meaningful, consequential advancement over time — including mechanical/statistical progression, character power growth, AND narrative character development, personal growth, or story-arc advancement. A game whose protagonist grows through a story counts as strongly here.",
    "meaningful_persistence": "How strongly the player's actions and achievements create lasting, meaningful progress or state.",
    "agency_consequence": "How strongly the player's decisions and actions create meaningful consequences — including branching narrative choices, moral decisions, and character-defining moments, not only mechanical world-state changes.",
    "player_identity": "How strongly the game allows the player to develop a meaningful personal identity — through mechanical builds and playstyle, AND through narrative role, character personality, and story-driven self-definition.",
    "systemic_depth": "How much meaningful depth exists in the game's interacting systems and decision-making.",
    "atmosphere": "How strongly the game builds immersion, mood, tone, world cohesion, and thematic resonance through audiovisual and world design.",
    "narrative_integration": "How meaningfully the story, worldbuilding, and lore are integrated with gameplay mechanics and player action.",
    "content_density": "How richly the game fills its world and runtime with unique activities, meaningful encounters, and high-value gameplay.",
    "effective_variety": "How meaningfully distinct and functional the available playstyles, tactical options, challenges, content, or builds are from one another.",
    "exploratory_autonomy": "How much meaningful freedom the player has to explore, discover, experiment, and choose their own path.",
    "social_coop_integration": "How meaningfully multiplayer, cooperation, social interaction, or shared goals are integrated into the game's experience.",
    "technical_polish": "How well the game executes technically, including stability, controls, performance, interface, audiovisual implementation, and overall polish.",
    "convenience": "How effectively the game respects player time through quality-of-life features, accessible interfaces, low downtime, and streamlined systems.",
    "scale": "How grand the overall scope, physical world size, systemic breadth, or operational capability is within the game.",
    "persistence_stability": "How reliably and predictably the game maintains world state, player progress, structural continuity, and long-term save safety.",
    "complexity": "How much complexity the player must understand and manage in order to play effectively.",
    "administrative_burden": "How much unnecessary management, bookkeeping, menu work, micromanagement, or cognitive overhead the game creates.",
    "grind": "How much repeated effort, resource accumulation, farming, or routine activity is required for progression.",
    "difficulty": "How demanding the game is in terms of mechanical skill, strategic thinking, failure, learning, or decision-making.",
    "unintentional_friction": "How much unnecessary friction interferes with the player's intended experience without providing meaningful challenge or decision-making.",
    "meta_dependence": "How heavily the player must rely on external guides, wikis, community information, or out-of-game tools to play effectively.",
    "restart_cost": "How severely the game penalizes failure or death in terms of lost time, progress, resources, or reset momentum.",
    "power_escalation": "How strongly the player's power, capabilities, resources, or influence increase relative to the beginning of the game.",
    "focused_structure": "How strongly the game provides a coherent, directed structure rather than leaving the player to create their own direction.",
    "extreme_competitive_pvp": "How heavily the game CENTERS ITS DESIGN on direct player-versus-player dominance with high skill floors and aggressive rank/skill-based rivalry. Score HIGH (0.7+) only for games whose core loop IS competitive PvP (ranked shooters, MOBAs, competitive fighting games, hardcore PvP MMOs). Score LOW (0.2 or below) for games that HAVE a PvP mode but whose primary experience is cooperative, solo, campaign, or strategy play. Optional PvP is not 'extreme competitive PvP'. A turn-based strategy game with skirmish PvP modes scores 0.15 or lower"
}

# Tag vocabulary now lives in gaming_his.py (TAG_VOCABULARY) -- that's the
# single source of truth both this file and select_relevant_history() draw
# from, so the research LLM can never be asked to use a tag the history
# matcher doesn't also know about, or vice versa.
CONTROLLED_TAGS = TAG_VOCABULARY

def _parse_llm_json(raw_text: str) -> dict:
    """Robustly parse Gemini's JSON output."""
    text = raw_text.strip()

    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text, count=1)
        text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        repaired = re.sub(r",(\s*[}\]])", r"\1", text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            start = max(0, e.pos - 200)
            end = min(len(text), e.pos + 200)
            raise ValueError(
                f"Failed to parse LLM JSON at char {e.pos}: {e.msg}\n"
                f"--- CONTEXT AROUND FAILURE ---\n...{text[start:end]}...\n"
                f"--- FULL OUTPUT (first 2000 chars) ---\n{raw_text[:2000]}"
            )

def extract_game_data(game_name: str, price: float) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-3.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
    )
    variable_text = "\n".join(
        f"{i + 1}. {name}: {definition}"
        for i, (name, definition) in enumerate(VARIABLES.items())
    )
    tag_vocab = ", ".join(CONTROLLED_TAGS)

    prompt = f"""
You are the research and game-analysis layer of a personal Game Recommendation Engine.
GAME: {game_name}

Evaluate the following 25 variables and retrieve basic metadata for this specific game.

VARIABLES:
{variable_text}

OUTPUT:
Return ONLY a valid JSON object with:
- game (string)
- release_year (integer, e.g. 2023 -- use the year of the original release)
- tagline (string, 4-8 words, DESCRIPTIVE -- genre + setting + core hook.
  Example: "Narrative-driven fantasy RPG with deep faction politics."
  Example: "Colony sim with generation-spanning survival management."
  Do NOT write a marketing blurb. Write a one-glance description.)
- average_playtime_hours (number)
- current_price (float)
- currency (string, e.g. "USD")
- tags (array of 3-8 lowercase, hyphenated keywords describing this game's genre, mechanics,
  and design character. STRONGLY prefer reusing items from this controlled vocabulary so
  tags remain consistent across games. Only invent a new tag if nothing here fits):
  {tag_vocab}
- variables (object with all 25 variables as keys, each containing estimate (0.0-1.0),
  range, confidence (0.0-1.0), and evidence list)

No markdown, no backticks, no text before or after.

  RULES, IN ORDER:
  1. ANCHOR FIRST: The FIRST tag MUST be the primary genre anchor, chosen from this list
     if any applies: crpg, arpg, rpg, action-rpg, action-adventure, open-world, sandbox,
     souls-like, mmo, survival, city-builder, management, simulation, grand-strategy, rts,
     turn-based-strategy, tactical-stealth, tactical-combat, horde-shooter, looter-shooter,
     fps, fighting, puzzle, puzzle-adventure, puzzle-horror, narrative-adventure,
     narrative-comedy, stealth-action.

     A party-based D&D-style RPG → "crpg" FIRST, then flavor tags.
     A Diablo-like → "arpg" FIRST.
     A zoo/tycoon sim → "management" or "simulation" FIRST.
     A WWII grand campaign → "grand-strategy" FIRST.

  2. NEVER invent a genre synonym when a vocabulary anchor exists. Do NOT write
     "isometric-rpg" or "fantasy-rpg" or "kingdom-management" — write "crpg" or "rpg".

  3. Then add 2-7 flavor/mechanic/craft tags from this vocabulary:
     {tag_vocab}

  4. Only invent a brand-new tag if NOTHING in the vocabulary describes a defining trait
     of the game. New tags are a last resort, not a shortcut.
  CRITICAL: Always include the primary GENRE tag from the vocabulary when it applies --
  do not skip it in favor of flavor tags. A party-based D&D-style RPG is a "crpg" first
  and foremost; an isometric turn-based fantasy RPG is a "crpg"; a Diablo-like is an
  "arpg". Genre tags are the anchor for cross-game matching.):
  {tag_vocab}
- variables (object with all 25 variables as keys, each containing estimate (0.0-1.0),
  range, confidence (0.0-1.0), and evidence list)

No markdown, no backticks, no text before or after.
"""

    response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=120)
    response.raise_for_status()

    candidate = response.json()["candidates"][0]
    finish_reason = candidate.get("finishReason", "STOP")
    text_output = candidate["content"]["parts"][0]["text"].strip()

    if finish_reason == "MAX_TOKENS":
        raise ValueError(
            "Gemini response was truncated by MAX_TOKENS. The extraction prompt "
            "is too large or the model was too verbose."
        )

    raw_data = _parse_llm_json(text_output)
    raw_variables = raw_data.get("variables", {})

    game_vector = {}
    confidence_scores = []

    for var_name, var_data in raw_variables.items():
        estimate = var_data.get("estimate", 0.0)
        confidence = var_data.get("confidence", 0.85)
        confidence_scores.append(confidence)

        if var_name == "power_escalation":
            estimate = (estimate * 2.0) - 1.0

        game_vector[var_name] = estimate

    evidence_quality = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.85
    average_playtime_hours = float(raw_data.get("average_playtime_hours", 40))

    raw_tags = raw_data.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []
    tags = []
    for t in raw_tags:
        s = str(t).strip().lower().replace(" ", "-")
        if s and s not in tags:
            tags.append(s)
    tags = tags[:10]

    return {
        "game_name": game_name,
        "price": price,
        "average_playtime_hours": average_playtime_hours,
        "game_vector": game_vector,
        "evidence_quality": evidence_quality,
        "tags": tags,
        "release_year": int(raw_data.get("release_year") or 0),
        "tagline": str(raw_data.get("tagline") or "").strip(),
    }
