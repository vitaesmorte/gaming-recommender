import os
import json
import requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

VARIABLES = {
    "meaningful_progression": "How strongly the game provides meaningful, consequential advancement over time.",
    "meaningful_persistence": "How strongly the player's actions and achievements create lasting, meaningful progress or state.",
    "agency_consequence": "How strongly the player's decisions and actions create meaningful consequences in the game.",
    "player_identity": "How strongly the game allows the player to develop a meaningful personal identity, role, build, creation, or way of playing.",
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
    "extreme_competitive_pvp": "How heavily the game prioritizes direct player-versus-player dominance, high skill floors, and aggressive rank or skill-based rivalry over casual or cooperative play."
}

def extract_game_data(game_name: str, price: float) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={GEMINI_API_KEY}"
    variable_text = "\n".join(f"{i + 1}. {name}: {definition}" for i, (name, definition) in enumerate(VARIABLES.items()))

    prompt = f"""
You are the research and game-analysis layer of a personal Game Recommendation Engine.
GAME: {game_name}

Evaluate the following 25 variables and retrieve basic metadata for this specific game.

VARIABLES:
{variable_text}

OUTPUT:
Return ONLY a valid JSON object with:
- game (string)
- average_playtime_hours (number)
- current_price (float)
- currency (string, e.g. "USD")
- variables (object with all 25 variables as keys, each containing estimate (0.0-1.0), range, confidence (0.0-1.0), and evidence list)

No markdown, no backticks, no text before or after.
"""

    response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=120)
    response.raise_for_status()

    text_output = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    if text_output.startswith("```"):
        text_output = text_output.replace("```json", "").replace("```", "").strip()

    raw_data = json.loads(text_output)
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

    return {
        "game_name": game_name,
        "price": price,
        "average_playtime_hours": average_playtime_hours,
        "game_vector": game_vector,
        "evidence_quality": evidence_quality
    }
