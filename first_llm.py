import os
import json
import requests
from fastapi import FastAPI

# Import Stage 2 (Math Engine) and Stage 3 (Final Evaluator LLM)
import game_eval_claude_v0_4_3
import llm_evaluator

app = FastAPI()


# ============================================================
# GEMINI API KEY (STEP 5 FIX: Read from environment variable)
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# GAME ASSISTANT — 25 VARIABLE GAME VECTOR RESEARCH
# ============================================================

VARIABLES = {

    "meaningful_progression":
        "How strongly the game provides meaningful, consequential advancement over time.",

    "meaningful_persistence":
        "How strongly the player's actions and achievements create lasting, meaningful progress or state.",

    "agency_consequence":
        "How strongly the player's decisions and actions create meaningful consequences in the game.",

    "player_identity":
        "How strongly the game allows the player to develop a meaningful personal identity, role, build, creation, or way of playing.",

    "systemic_depth":
        "How much meaningful depth exists in the game's interacting systems and decision-making.",

    "atmosphere":
        "How strongly the game builds immersion, mood, tone, world cohesion, and thematic resonance through audiovisual and world design.",

    "narrative_integration":
        "How meaningfully the story, worldbuilding, and lore are integrated with gameplay mechanics and player action.",

    "content_density":
        "How richly the game fills its world and runtime with unique activities, meaningful encounters, and high-value gameplay.",

    "effective_variety":
        "How meaningfully distinct and functional the available playstyles, tactical options, challenges, content, or builds are from one another.",

    "exploratory_autonomy":
        "How much meaningful freedom the player has to explore, discover, experiment, and choose their own path.",

    "social_coop_integration":
        "How meaningfully multiplayer, cooperation, social interaction, or shared goals are integrated into the game's experience.",

    "technical_polish":
        "How well the game executes technically, including stability, controls, performance, interface, audiovisual implementation, and overall polish.",

    "convenience":
        "How effectively the game respects player time through quality-of-life features, accessible interfaces, low downtime, and streamlined systems.",

    "scale":
        "How grand the overall scope, physical world size, systemic breadth, or operational capability is within the game.",

    "persistence_stability":
        "How reliably and predictably the game maintains world state, player progress, structural continuity, and long-term save safety.",

    "complexity":
        "How much complexity the player must understand and manage in order to play effectively.",

    "administrative_burden":
        "How much unnecessary management, bookkeeping, menu work, micromanagement, or cognitive overhead the game creates.",

    "grind":
        "How much repeated effort, resource accumulation, farming, or routine activity is required for progression.",

    "difficulty":
        "How demanding the game is in terms of mechanical skill, strategic thinking, failure, learning, or decision-making.",

    "unintentional_friction":
        "How much unnecessary friction interferes with the player's intended experience without providing meaningful challenge or decision-making.",

    "meta_dependence":
        "How heavily the player must rely on external guides, wikis, community information, or out-of-game tools to play effectively.",

    "restart_cost":
        "How severely the game penalizes failure or death in terms of lost time, progress, resources, or reset momentum.",

    "power_escalation":
        "How strongly the player's power, capabilities, resources, or influence increase relative to the beginning of the game.",

    "focused_structure":
        "How strongly the game provides a coherent, directed structure rather than leaving the player to create their own direction.",

    # STEP 1 FIX: Added 25th variable (veto tag)
    "extreme_competitive_pvp":
        "How heavily the game prioritizes direct player-versus-player dominance, high skill floors, and aggressive rank or skill-based rivalry over casual or cooperative play."

}


# ============================================================
# EVALUATE GAME
# ============================================================

@app.get("/evaluate")
def evaluate_game(game: str = "Ultima Online"):

    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY environment variable is not set."}

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-flash-lite-latest:generateContent"
        f"?key={GEMINI_API_KEY}"
    )


    # --------------------------------------------------------
    # Build variable list for Gemini
    # --------------------------------------------------------

    variable_text = "\n".join(
        f"{i + 1}. {name}: {definition}"
        for i, (name, definition)
        in enumerate(VARIABLES.items())
    )


    # --------------------------------------------------------
    # Research prompt (STEP 1 & 3 FIX: Updated to 25 variables + metadata request)
    # --------------------------------------------------------

    prompt = f"""
You are the research and game-analysis layer of a personal
Game Recommendation Engine.

GAME:
{game}


YOUR TASK
---------

Evaluate the following 25 variables and retrieve basic metadata for this specific game.


IMPORTANT — ADAPT TO THE GAME
-----------------------------

These variables must be interpreted according to the actual
genre, structure and mechanics of THIS game.

Do NOT assume that every game is an RPG.

For example:

In an RPG, meaningful progression may involve character
development, abilities, equipment and builds.

In a city builder, meaningful progression may involve city
development, infrastructure, economic growth, technology,
population and increasingly complex systems.

In a strategy game, meaningful progression may involve
territory, economy, technology, military development or
strategic options.

In a multiplayer game, progression may involve character
development, mastery, teamwork, social progression or
persistent development.

Therefore:

1. First understand how this particular game works.

2. Determine how each variable manifests in THIS game.

3. Evaluate only mechanisms that are actually relevant.

4. Do not force RPG concepts onto non-RPG games.

5. Do not assume that the same evidence categories apply to
   every game.

The purpose is to create a GAME VECTOR, not a review.


SOURCES
-------

Use these source types when researching:

- Official documentation
- Developer information
- Game databases / wikis
- Specialist reviews
- Steam reviews
- Reddit / community discussions

Use factual evidence and recurring player experiences where
appropriate.


IMPORTANT DISTINCTIONS
----------------------

Be careful to distinguish:

- quantity vs meaningfulness
- nominal variety vs effective variety
- complexity vs depth
- convenience vs meaningful progression
- freedom vs meaningful agency
- scale vs content density
- difficulty vs unnecessary friction

Do not assume that a larger number automatically means a
better result.

Do not evaluate whether the player personally likes the
variable.

The estimate describes the GAME.


VARIABLES
---------

{variable_text}


OUTPUT
------

Return an object with:

- game
- average_playtime_hours (estimated main story/average completion time in hours as a number)
- current_price (current estimated retail price as a float number, e.g. 59.99)
- currency (currency code, e.g. "USD")
- variables

The variables object must contain exactly the 25 variable names
listed above.

Every variable must contain:

- estimate
- range
- confidence
- evidence

Definitions:

estimate:
A number from 0.0 to 1.0 representing the estimated strength
of the variable in this game.

range:
An uncertainty range [low, high] around the estimate.

confidence:
A number from 0.0 to 1.0 representing how confident you are
in the estimate based on the available evidence.

evidence:
A short list of factual observations supporting the estimate.

Keep evidence concise.

Do not provide a personal recommendation.

Do not provide a purchase recommendation.

Do not compare the game with the player's preferences.

Return ONLY valid JSON.

Do not use markdown.

Do not use code fences.

Do not include commentary before or after the JSON.
"""


    # --------------------------------------------------------
    # Gemini request
    # --------------------------------------------------------

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }


    response = requests.post(
        url,
        json=data,
        timeout=120
    )


    # Raise an error if Gemini returns an HTTP error
    response.raise_for_status()

    response_data = response.json()


    # --------------------------------------------------------
    # Extract Gemini response
    # --------------------------------------------------------

    text_output = (
        response_data["candidates"][0]
        ["content"]["parts"][0]["text"]
    )

    text_output = text_output.strip()


    if text_output.startswith("```"):
        text_output = text_output.replace("```json", "", 1)
        text_output = text_output.replace("```", "", 1)
        text_output = text_output.strip()


    # --------------------------------------------------------
    # Convert Gemini JSON text into actual Python JSON
    # --------------------------------------------------------

    try:
        stage1_result = json.loads(text_output)
    except json.JSONDecodeError:
        return {
            "game": game,
            "raw_result": text_output,
            "error": "Gemini returned invalid JSON"
        }


    # --------------------------------------------------------
    # PROCESS DATA FOR ENGINE (STEPS 2, 3, & 4 FIXES)
    # --------------------------------------------------------

    raw_variables = stage1_result.get("variables", {})
    game_vector = {}
    confidence_scores = []

    for var_name, var_data in raw_variables.items():
        estimate = var_data.get("estimate", 0.0)
        confidence = var_data.get("confidence", 0.85)

        confidence_scores.append(confidence)

        # STEP 2 FIX: Convert power_escalation from 0..1 to -1..1 spectrum
        if var_name == "power_escalation":
            estimate = (estimate * 2.0) - 1.0

        game_vector[var_name] = estimate

    # STEP 4 FIX: Calculate average confidence across all variables for evidence_quality
    evidence_quality = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores else 0.85
    )

    # STEP 3 FIX: Extract metadata provided by Gemini
    average_playtime_hours = stage1_result.get("average_playtime_hours", 0)
    current_price = stage1_result.get("current_price", 0.0)
    currency = stage1_result.get("currency", "USD")


    # --------------------------------------------------------
    # STAGE 2: CALL GAME EVALUATION ENGINE (STEP 6 FIX)
    # --------------------------------------------------------

    engine_output = game_eval_claude_v0_4_3.evaluate_game(
        game_vector=game_vector,
        average_playtime_hours=average_playtime_hours,
        current_price=current_price,
        currency=currency,
        evidence_quality=evidence_quality
    )


    # --------------------------------------------------------
    # STAGE 3: CALL SECOND LLM EVALUATOR (STEP 6 FIX)
    # --------------------------------------------------------

    final_narrative = llm_evaluator.evaluate(
        game_name=game,
        engine_output=engine_output
    )


    # --------------------------------------------------------
    # Return final result to website UI
    # --------------------------------------------------------

    return final_narrative
