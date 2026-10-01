from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import exp
from typing import Dict, List, Optional, Tuple


# ============================================================
# GAME ASSISTANT — DETERMINISTIC EVALUATION ENGINE
# Version: 0.4.3 (RCM Architecture + Gaussian Scaling + 25 Variables)
#
# Changelog vs 0.4.2:
#   - Added phase-level grounding data for the LLM narration layer:
#     summarize_phase_friction() and summarize_phases() condense the
#     hour-by-hour survival curve into one row per GamePhase, with
#     average enjoyment/hazard and the top "hook" and "friction"
#     variables active in that phase (ranked by player weight x
#     alignment). evaluate_game() now returns this as
#     "phase_breakdown". This exists so an LLM evaluator has real
#     per-act numbers to cite instead of inventing act names, hour
#     marks, or friction causes -- a hallucination risk observed in
#     early LLM-narration drafts.
# ============================================================


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class ScaleType(str, Enum):
    QUALITY = "quality"
    INTENSITY_BURDEN = "intensity_burden"
    SPECTRUM = "spectrum"
    BOOLEAN_TAG = "boolean_tag"


@dataclass
class GameVariableDefinition:
    variable_id: str
    domain: str
    scale_type: ScaleType
    min_value: float
    max_value: float
    is_vetoable: bool = False
    description: str = ""

    def validate(self, value: float) -> bool:
        return self.min_value <= value <= self.max_value


# ============================================================
# 1. COMPLETE VARIABLE DEFINITIONS
# ============================================================

VARIABLE_DEFINITIONS: Dict[str, GameVariableDefinition] = {
    # Quality Variables (0.0 to 1.0)
    "meaningful_progression": GameVariableDefinition("meaningful_progression", "systemic_depth", ScaleType.QUALITY, 0.0, 1.0),
    "meaningful_persistence": GameVariableDefinition("meaningful_persistence", "systemic_depth", ScaleType.QUALITY, 0.0, 1.0),
    "agency_consequence": GameVariableDefinition("agency_consequence", "dynamic_agency", ScaleType.QUALITY, 0.0, 1.0),
    "player_identity": GameVariableDefinition("player_identity", "dynamic_agency", ScaleType.QUALITY, 0.0, 1.0),
    "systemic_depth": GameVariableDefinition("systemic_depth", "systemic_depth", ScaleType.QUALITY, 0.0, 1.0),
    "atmosphere": GameVariableDefinition("atmosphere", "narrative_vibe", ScaleType.QUALITY, 0.0, 1.0),
    "narrative_integration": GameVariableDefinition("narrative_integration", "narrative_vibe", ScaleType.QUALITY, 0.0, 1.0),
    "content_density": GameVariableDefinition("content_density", "world_content", ScaleType.QUALITY, 0.0, 1.0),
    "effective_variety": GameVariableDefinition("effective_variety", "world_content", ScaleType.QUALITY, 0.0, 1.0),
    "exploratory_autonomy": GameVariableDefinition("exploratory_autonomy", "dynamic_agency", ScaleType.QUALITY, 0.0, 1.0),
    "social_coop_integration": GameVariableDefinition("social_coop_integration", "world_content", ScaleType.QUALITY, 0.0, 1.0),
    "technical_polish": GameVariableDefinition("technical_polish", "execution_value", ScaleType.QUALITY, 0.0, 1.0),
    "convenience": GameVariableDefinition("convenience", "execution_value", ScaleType.QUALITY, 0.0, 1.0),
    "scale": GameVariableDefinition("scale", "world_content", ScaleType.QUALITY, 0.0, 1.0),
    "persistence_stability": GameVariableDefinition("persistence_stability", "systemic_depth", ScaleType.QUALITY, 0.0, 1.0),

    # Intensity / Burden Variables (0.0 to 1.0)
    "complexity": GameVariableDefinition("complexity", "systemic_depth", ScaleType.INTENSITY_BURDEN, 0.0, 1.0),
    "administrative_burden": GameVariableDefinition("administrative_burden", "systemic_depth", ScaleType.INTENSITY_BURDEN, 0.0, 1.0),
    "grind": GameVariableDefinition("grind", "systemic_depth", ScaleType.INTENSITY_BURDEN, 0.0, 1.0),
    "difficulty": GameVariableDefinition("difficulty", "execution_value", ScaleType.INTENSITY_BURDEN, 0.0, 1.0),
    "unintentional_friction": GameVariableDefinition("unintentional_friction", "execution_value", ScaleType.INTENSITY_BURDEN, 0.0, 1.0),
    "meta_dependence": GameVariableDefinition("meta_dependence", "execution_value", ScaleType.INTENSITY_BURDEN, 0.0, 1.0),
    "restart_cost": GameVariableDefinition("restart_cost", "execution_value", ScaleType.INTENSITY_BURDEN, 0.0, 1.0),

    # Spectrum Variables (-1.0 to 1.0 or 0.0 to 1.0)
    "power_escalation": GameVariableDefinition("power_escalation", "systemic_depth", ScaleType.SPECTRUM, -1.0, 1.0),
    "focused_structure": GameVariableDefinition("focused_structure", "dynamic_agency", ScaleType.SPECTRUM, 0.0, 1.0),

    # Boolean Tags / Vetoes
    "extreme_competitive_pvp": GameVariableDefinition("extreme_competitive_pvp", "execution_value", ScaleType.BOOLEAN_TAG, 0.0, 1.0, is_vetoable=True),
}


# ============================================================
# 2. RECIPROCAL CONTEXT MATRIX ENGINE
# ============================================================

@dataclass
class MatrixInteraction:
    var_i: str
    var_j: str
    weight: float
    interaction_type: str
    # Supported interaction_type values:
    #   'synergy'          - both variables high -> bonus
    #   'friction_deficit'  - var_i high & var_j low -> penalty (or bonus if weight>0)
    #   'excess_burden'     - var_i exceeding var_j -> penalty proportional to excess
    #   'power_penalty'     - var_i (spectrum, normalized) exceeding a threshold,
    #                         unmitigated by var_j -> penalty
    #   'mismatch_penalty'  - var_i far from (1 - var_j) -> mild symmetric penalty.
    #                         Used for relationships where neither variable is
    #                         inherently "good" or "bad" and only the gap between
    #                         them (e.g. freedom vs. structure) matters. Result is
    #                         clamped to a narrow band so it stays a mild modifier
    #                         rather than dominating the overall multiplier.

    def evaluate(self, vector: Dict[str, float]) -> float:
        val_i = vector.get(self.var_i, 0.0)
        val_j = vector.get(self.var_j, 0.0)

        if self.interaction_type == "synergy":
            delta = val_i * val_j * self.weight
            return 1.0 + delta

        elif self.interaction_type == "friction_deficit":
            delta = val_i * (1.0 - val_j) * abs(self.weight)
            return 1.0 - delta if self.weight < 0 else 1.0 + delta

        elif self.interaction_type == "excess_burden":
            excess = max(0.0, val_i - val_j)
            return 1.0 - (excess * abs(self.weight))

        elif self.interaction_type == "power_penalty":
            power_normalized = clamp((val_i + 1.0) / 2.0, 0.0, 1.0)
            excess_power = max(0.0, power_normalized - 0.75)
            penalty = excess_power * (1.0 - val_j) * abs(self.weight)
            return 1.0 - penalty

        elif self.interaction_type == "mismatch_penalty":
            mismatch = abs(val_i - (1.0 - val_j))
            penalty = mismatch * abs(self.weight)
            return clamp(1.0 - penalty, 0.90, 1.10)

        return 1.0


class ReciprocalContextMatrix:
    def __init__(self, interactions: List[MatrixInteraction]):
        self.interactions = interactions

    def calculate_multiplier(self, game_vector: Dict[str, float]) -> float:
        multiplier = 1.0
        for interaction in self.interactions:
            multiplier *= interaction.evaluate(game_vector)
        return clamp(multiplier, 0.40, 1.50)


DEFAULT_RCM = ReciprocalContextMatrix([
    MatrixInteraction("grind", "meaningful_progression", 0.30, "synergy"),
    MatrixInteraction("grind", "meaningful_progression", -0.20, "friction_deficit"),
    MatrixInteraction("grind", "meaningful_persistence", 0.30, "synergy"),
    MatrixInteraction("grind", "meaningful_persistence", -0.25, "friction_deficit"),
    MatrixInteraction("complexity", "systemic_depth", 0.20, "synergy"),
    MatrixInteraction("complexity", "systemic_depth", -0.25, "friction_deficit"),
    MatrixInteraction("administrative_burden", "systemic_depth", -0.35, "excess_burden"),
    MatrixInteraction("difficulty", "agency_consequence", 0.20, "synergy"),
    MatrixInteraction("difficulty", "agency_consequence", -0.15, "friction_deficit"),
    MatrixInteraction("power_escalation", "meaningful_progression", -0.40, "power_penalty"),
    MatrixInteraction("player_identity", "meaningful_persistence", 0.25, "synergy"),
    MatrixInteraction("difficulty", "meta_dependence", -0.25, "friction_deficit"),

    # --- Ported from v0.2.2, retuned in v0.4.2 for reciprocity ---
    # Difficulty is rewarded when shared (co-op mastery), but now also
    # penalized when faced with no coop mitigation — mirrors the
    # difficulty <-> agency_consequence pair above (+0.20 / -0.15).
    MatrixInteraction("difficulty", "social_coop_integration", 0.18, "synergy"),
    MatrixInteraction("difficulty", "social_coop_integration", -0.12, "friction_deficit"),

    # World feels diluted when scale outpaces content density. Weight
    # raised slightly toward administrative_burden <-> systemic_depth
    # (0.35) since this is the same "unjustified quantity" pattern,
    # kept a bit softer since scale is a milder design sin than busywork.
    MatrixInteraction("scale", "content_density", 0.28, "excess_burden"),

    # Excess convenience cheapens meaningful progression. Kept below the
    # other burden weights — convenience/QoL is a soft tension, not a
    # core axis, and shouldn't be able to dominate a game's score alone.
    MatrixInteraction("convenience", "meaningful_progression", 0.12, "excess_burden"),

    # Freedom vs. structure coherence — neither pole is inherently better;
    # only an unresolved gap between exploratory_autonomy and
    # (1 - focused_structure) is penalized (a game that claims both
    # sandbox freedom and heavy structure, or neither). Slightly higher
    # weight than v0.4.1 since this is what actually differentiates
    # genres (open sandbox vs. linear vs. incoherent middle), still
    # clamped to a narrow 0.90-1.10 band to stay a mild modifier.
    MatrixInteraction("exploratory_autonomy", "focused_structure", 0.12, "mismatch_penalty"),
])


# ============================================================
# 3. GAME PHASES, PLAYER PROFILE & ENGINE CORE
# ============================================================

@dataclass
class GamePhase:
    phase_name: str
    start_hour: int
    end_hour: int
    phase_order: int
    phase_vector: Dict[str, float] = field(default_factory=dict)


@dataclass
class GameProfile:
    title: str
    static_vector: Dict[str, float]
    average_playtime_hours: float
    phases: List[GamePhase] = field(default_factory=list)

    def get_vector_at_hour(self, hour: float) -> Dict[str, float]:
        vector = dict(self.static_vector)
        active_phase: Optional[GamePhase] = None
        for phase in self.phases:
            if phase.start_hour <= hour <= phase.end_hour:
                active_phase = phase
                break
        if active_phase:
            vector.update(active_phase.phase_vector)
        return vector


@dataclass
class Game:
    game_id: str
    title: str
    release_date: Optional[str]
    profile: GameProfile
    current_price: float = 0.0
    currency: str = "EUR"


@dataclass
class PlayerPreference:
    ideal: float = 0.5
    tolerance: float = 0.25
    weight: float = 1.0


@dataclass
class PlayerProfile:
    preferences: Dict[str, PlayerPreference]
    veto_tags: Dict[str, bool] = field(default_factory=dict)
    human_factor: float = 1.0


# ============================================================
# 4. GAUSSIAN FEATURE ALIGNMENT ENGINE
# ============================================================

def gaussian_alignment(game_value: float, preference: PlayerPreference) -> float:
    tolerance = max(preference.tolerance, 0.001)
    alignment = exp(-((game_value - preference.ideal) ** 2) / (2 * (tolerance ** 2)))
    return clamp(alignment, 0.0, 1.0)


def calculate_base_fit(
    game_vector: Dict[str, float],
    player: PlayerProfile,
    variable_definitions: Dict[str, GameVariableDefinition]
) -> float:
    weighted_alignment = 0.0
    total_weight = 0.0

    for variable_id, preference in player.preferences.items():
        if variable_id not in game_vector:
            continue

        value = game_vector[variable_id]
        definition = variable_definitions.get(variable_id)

        if definition is None or not definition.validate(value):
            continue

        if definition.scale_type == ScaleType.QUALITY:
            norm_val = (value - definition.min_value) / max(0.001, (definition.max_value - definition.min_value))
            alignment = clamp(norm_val, 0.0, 1.0)
        elif definition.scale_type in (ScaleType.INTENSITY_BURDEN, ScaleType.SPECTRUM):
            alignment = gaussian_alignment(value, preference)
        elif definition.scale_type == ScaleType.BOOLEAN_TAG:
            alignment = 1.0 if value >= 0.5 else 0.0
        else:
            alignment = 0.0

        weighted_alignment += preference.weight * alignment
        total_weight += preference.weight

    if total_weight == 0:
        return 0.0

    return clamp(weighted_alignment / total_weight, 0.0, 1.0)


# ============================================================
# 5. HAZARD & SURVIVAL FUNCTIONS
# ============================================================

def calculate_rcm_dampened(game_vector, player, raw_rcm: float) -> float:
    if hasattr(game_vector, "profile"):
        vector = game_vector.profile.static_vector
    elif hasattr(game_vector, "static_vector"):
        vector = game_vector.static_vector
    elif isinstance(game_vector, dict):
        vector = game_vector
    else:
        vector = {}

    admin = vector.get("administrative_burden", 0.0)
    meta = vector.get("meta_dependence", 0.0)

    if hasattr(player, "preferences"):
        admin_pref = player.preferences.get("administrative_burden")
        meta_pref = player.preferences.get("meta_dependence")
        admin_ideal = admin_pref.ideal if admin_pref else 0.25
        meta_ideal = meta_pref.ideal if meta_pref else 0.30
    else:
        admin_ideal = 0.25
        meta_ideal = 0.30

    admin_excess = max(0.0, admin - admin_ideal)
    meta_excess = max(0.0, meta - meta_ideal)

    friction_penalty = (admin_excess * 0.40) + (meta_excess * 0.40)
    adjusted_multiplier = raw_rcm - friction_penalty

    return clamp(adjusted_multiplier, 0.40, 1.25)


def calculate_tolerance_stress(
    game_vector: Dict[str, float],
    player: PlayerProfile,
    variable_definitions: Dict[str, GameVariableDefinition]
) -> float:
    stresses = []
    for variable_id, preference in player.preferences.items():
        if variable_id not in game_vector:
            continue
        definition = variable_definitions.get(variable_id)
        if definition is None or definition.scale_type not in (ScaleType.SPECTRUM, ScaleType.INTENSITY_BURDEN):
            continue
        value = game_vector[variable_id]
        tolerance = max(preference.tolerance, 0.001)
        distance = abs(value - preference.ideal)
        stresses.append(clamp(distance / tolerance, 0.0, 1.0))
    return mean(stresses)


def calculate_hazard(
    hour: float,
    enjoyment: float,
    tolerance_stress: float,
    g: Dict[str, float]
) -> float:
    base_hazard = 0.010
    enjoyment_pressure = (1.0 - enjoyment) * 0.04
    tolerance_pressure = (tolerance_stress ** 2.0) * 0.08

    admin_burden = g.get("administrative_burden", 0.0)
    meta_dependence = g.get("meta_dependence", 0.0)

    early_friction_wall = 0.0
    if 4 <= hour <= 12 and (admin_burden > 0.70 or meta_dependence > 0.60):
        early_friction_wall = 0.12

    hazard = base_hazard + enjoyment_pressure + tolerance_pressure + early_friction_wall
    return clamp(hazard, 0.001, 0.95)


def calculate_survival_curve(
    game: Game,
    player: PlayerProfile,
    variable_definitions: Dict[str, GameVariableDefinition],
    rcm: ReciprocalContextMatrix,
    max_hours: Optional[int] = None
) -> List[Dict[str, float]]:
    if max_hours is None:
        max_hours = max(int(game.profile.average_playtime_hours), 1)

    curve = []
    survival = 1.0

    for hour in range(0, max_hours + 1):
        vector = game.profile.get_vector_at_hour(hour)
        base_fit = calculate_base_fit(vector, player, variable_definitions)
        raw_rcm = rcm.calculate_multiplier(vector)
        rcm_multiplier = calculate_rcm_dampened(vector, player, raw_rcm)
        enjoyment = clamp(base_fit * rcm_multiplier, 0.0, 1.0)

        tolerance_stress = calculate_tolerance_stress(vector, player, variable_definitions)
        hazard = calculate_hazard(hour, enjoyment, tolerance_stress, vector)

        curve.append({
            "hour": float(hour),
            "enjoyment": enjoyment,
            "hazard": hazard,
            "survival": survival,
        })
        survival *= exp(-hazard)

    return curve


def calculate_wtp(
    expected_enjoyable_hours: float,
    personal_fit: float,
    commitment_fit: float,
    confidence: float,
    value_density: float,
    human_factor: float = 1.0,
    base_value_per_hour: float = 1.50
) -> Dict[str, float]:
    raw_value = expected_enjoyable_hours * base_value_per_hour
    fit_modifier = 0.70 + 0.30 * personal_fit
    commitment_modifier = 0.75 + 0.25 * commitment_fit
    confidence_modifier = 0.75 + 0.25 * confidence
    value_density_modifier = 0.75 + 0.50 * value_density
    human_factor = clamp(human_factor, 0.85, 1.15)

    central = raw_value * fit_modifier * commitment_modifier * confidence_modifier * value_density_modifier * human_factor

    return {
        "raw_value": round(raw_value, 2),
        "wtp_low": round(central * 0.75, 2),
        "wtp_central": round(central, 2),
        "wtp_high": round(central * 1.25, 2),
    }


def check_vetoes(
    game_vector: Dict[str, float], player: PlayerProfile, variable_definitions: Dict[str, GameVariableDefinition]
) -> bool:
    for variable_id, enabled in player.veto_tags.items():
        if not enabled:
            continue
        definition = variable_definitions.get(variable_id)
        if definition and definition.is_vetoable and game_vector.get(variable_id, 0.0) >= 0.5:
            return True
    return False


# ============================================================
# 5b. PHASE-LEVEL GROUNDING FOR LLM NARRATION
# ============================================================

def summarize_phase_friction(
    vector: Dict[str, float],
    player: PlayerProfile,
    variable_definitions: Dict[str, GameVariableDefinition],
    top_n: int = 2,
) -> Dict[str, List[str]]:
    """
    Ranks each variable present in both the vector and the player's
    preferences by how much it currently helps (hook) or hurts
    (friction) enjoyment, weighted by the player's stated importance
    for that variable. This is the data an LLM narration layer should
    cite -- it should never invent which mechanic is responsible for a
    given phase's friction or appeal.
    """
    scored = []
    for variable_id, preference in player.preferences.items():
        if variable_id not in vector:
            continue
        definition = variable_definitions.get(variable_id)
        if definition is None:
            continue
        value = vector[variable_id]

        if definition.scale_type == ScaleType.QUALITY:
            alignment = clamp(value, 0.0, 1.0)
        elif definition.scale_type in (ScaleType.INTENSITY_BURDEN, ScaleType.SPECTRUM):
            alignment = gaussian_alignment(value, preference)
        elif definition.scale_type == ScaleType.BOOLEAN_TAG:
            alignment = 1.0 if value >= 0.5 else 0.0
        else:
            continue

        contribution = preference.weight * alignment
        friction = preference.weight * (1.0 - alignment)
        scored.append((variable_id, contribution, friction))

    hooks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    frictions = sorted(scored, key=lambda x: x[2], reverse=True)[:top_n]

    return {
        "top_hooks": [h[0] for h in hooks],
        "top_friction": [fr[0] for fr in frictions],
    }


def summarize_phases(
    game: Game,
    player: PlayerProfile,
    variable_definitions: Dict[str, GameVariableDefinition],
    survival_curve: List[Dict[str, float]],
) -> List[Dict]:
    """
    Condenses the hour-by-hour survival curve into one row per declared
    GamePhase (or a single 'Full Game' row if no phases are defined),
    with average enjoyment/hazard and the top hook/friction variables
    active during that phase.
    """
    phases = game.profile.phases or [
        GamePhase("Full Game", 0, int(game.profile.average_playtime_hours), 1, {})
    ]

    summary = []
    for phase in phases:
        points = [p for p in survival_curve if phase.start_hour <= p["hour"] <= phase.end_hour]
        if not points:
            continue

        avg_enjoyment = mean([p["enjoyment"] for p in points])
        avg_hazard = mean([p["hazard"] for p in points])
        midpoint_hour = (phase.start_hour + phase.end_hour) / 2
        vector = game.profile.get_vector_at_hour(midpoint_hour)
        drivers = summarize_phase_friction(vector, player, variable_definitions)

        summary.append({
            "phase_name": phase.phase_name,
            "hour_range": f"{phase.start_hour}-{phase.end_hour}",
            "avg_enjoyment": round(avg_enjoyment, 3),
            "avg_hazard": round(avg_hazard, 4),
            "top_hooks": drivers["top_hooks"],
            "top_friction": drivers["top_friction"],
        })

    return summary


# ============================================================
# 6. EVALUATION ENTRYPOINT
# ============================================================

def evaluate_game(
    game: Game,
    player: PlayerProfile,
    variable_definitions: Dict[str, GameVariableDefinition],
    rcm: ReciprocalContextMatrix = DEFAULT_RCM,
    evidence_quality: float = 0.85,
    phase_completeness: float = 0.80,
    historical_similarity: float = 0.80,
    actual_played: float = 0.0,
    base_value_per_hour: float = 1.50,
    verdict_tolerance: float = 0.10,
) -> Dict:
    static_vector = game.profile.static_vector
    personal_fit = calculate_base_fit(static_vector, player, variable_definitions)
    raw_rcm = rcm.calculate_multiplier(static_vector)
    rcm_multiplier = calculate_rcm_dampened(static_vector, player, raw_rcm)
    adjusted_fit = clamp(personal_fit * rcm_multiplier, 0.0, 1.0)

    veto_triggered = check_vetoes(static_vector, player, variable_definitions)
    survival_curve = calculate_survival_curve(game, player, variable_definitions, rcm)

    expected_enjoyable_hours = sum(p["survival"] * p["enjoyment"] for p in survival_curve)
    expected_survival_hours = sum(p["survival"] for p in survival_curve)

    abandonment_hour = None
    for p in survival_curve:
        if p["survival"] <= 0.50:
            abandonment_hour = int(p["hour"])
            break

    commitment_pressure = mean([
        static_vector.get("scale", 0.5),
        1.0 - static_vector.get("focused_structure", 0.5),
        static_vector.get("complexity", 0.5),
        static_vector.get("restart_cost", 0.5),
    ])
    commitment_fit = clamp(1.0 - commitment_pressure, 0.0, 1.0)
    confidence = clamp(evidence_quality * 0.30 + phase_completeness * 0.25 + historical_similarity * 0.25 + actual_played * 0.20, 0.0, 1.0)
    value_density = clamp(0.50 * static_vector.get("content_density", 0.5) + 0.50 * static_vector.get("effective_variety", 0.5), 0.0, 1.0)

    wtp = calculate_wtp(
        expected_enjoyable_hours, adjusted_fit, commitment_fit, confidence, value_density, player.human_factor, base_value_per_hour
    )

    # wtp_low/wtp_high already model +/-25% uncertainty around wtp_central,
    # but the original cutoff between "BUY / SALE" and "WAIT" was a single
    # exact dollar value with zero tolerance -- $1 over central got the
    # same "WAIT" label as $5 over. near_value_price gives a small band
    # (default 10%) around central that still counts as a buy, so a price
    # that's essentially at fair value isn't bucketed with ones that
    # genuinely aren't.
    near_value_price = wtp["wtp_central"] * (1.0 + verdict_tolerance)

    if veto_triggered:
        verdict = "SKIP"
    elif adjusted_fit >= 0.82 and confidence >= 0.75 and game.current_price <= wtp["wtp_low"]:
        verdict = "BUY — FULL PRICE"
    elif game.current_price <= near_value_price:
        verdict = "BUY / SALE"
    elif game.current_price <= wtp["wtp_high"]:
        verdict = "WAIT"
    else:
        verdict = "DEEP SALE ONLY"

    return {
        "engine_version": "0.4.3",
        "game_id": game.game_id,
        "title": game.title,
        "metrics": {
            "personal_fit_raw": round(personal_fit, 4),
            "rcm_multiplier": round(rcm_multiplier, 4),
            "personal_fit_adjusted": round(adjusted_fit, 4),
            "commitment_fit": round(commitment_fit, 4),
            "confidence": round(confidence, 4),
            "value_density": round(value_density, 4),
        },
        "survival_analysis": {
            "expected_enjoyable_hours": round(expected_enjoyable_hours, 1),
            "expected_survival_hours": round(expected_survival_hours, 1),
            "predicted_50pct_abandonment_hour": abandonment_hour,
        },
        "financial_valuation": {
            "current_price": game.current_price,
            "currency": game.currency,
            "wtp": wtp,
            "veto_triggered": veto_triggered,
            "verdict": verdict,
        },
        "phase_breakdown": summarize_phases(game, player, variable_definitions, survival_curve),
    }


# ============================================================
# 7. EXECUTION WITH ALL VARIABLES (same test case as v0.4.0)
# ============================================================

if __name__ == "__main__":
    pathfinder_vector = {
            "meaningful_progression": 0.45,
            "meaningful_persistence": 0.40,
            "agency_consequence": 0.20,
            "player_identity": 0.70,
            "systemic_depth": 0.65,
            "atmosphere": 0.88,
            "narrative_integration": 0.50,
            "content_density": 0.60,
            "effective_variety": 0.65,
            "exploratory_autonomy": 0.20,
            "social_coop_integration": 0.95,
            "technical_polish": 0.78,
            "convenience": 0.80,
            "scale": 0.50,
            "persistence_stability": 0.80,
            "complexity": 0.55,
            "administrative_burden": 0.25,
            "grind": 0.65,
            "difficulty": 0.80,
            "unintentional_friction": 0.15,
            "meta_dependence": 0.40,
            "restart_cost": 0.05,
            "power_escalation": 0.30,
            "focused_structure": 0.90,
            "extreme_competitive_pvp": 0.0,
    }

    # Convert power_escalation to internal [-1.0, 1.0] spectrum
    pathfinder_vector["power_escalation"] = (pathfinder_vector["power_escalation"] * 2.0) - 1.0

john_player_profile = PlayerProfile(
    preferences={
        "meaningful_progression": PlayerPreference(ideal=0.8, tolerance=0.19, weight=1.1),
        "meaningful_persistence": PlayerPreference(ideal=0.85, tolerance=0.20, weight=1.3),
        "agency_consequence": PlayerPreference(ideal=0.91, tolerance=0.21, weight=1.4),
        "player_identity": PlayerPreference(ideal=0.92, tolerance=0.20, weight=1.2),
        "systemic_depth": PlayerPreference(ideal=0.6, tolerance=0.55, weight=0.8),
        "atmosphere": PlayerPreference(ideal=0.71, tolerance=0.15, weight=1.2),
        "narrative_integration": PlayerPreference(ideal=0.8, tolerance=0.22, weight=0.9),
        "content_density": PlayerPreference(ideal=0.5, tolerance=0.8, weight=0.55),
        "effective_variety": PlayerPreference(ideal=0.65, tolerance=0.7, weight=0.9),
        "exploratory_autonomy": PlayerPreference(ideal=0.7, tolerance=0.6, weight=0.9),
        "social_coop_integration": PlayerPreference(ideal=0.65, tolerance=0.8, weight=0.9),
        "technical_polish": PlayerPreference(ideal=0.8, tolerance=0.8, weight=0.8),
        "convenience": PlayerPreference(ideal=0.75, tolerance=0.25, weight=0.8),
        "scale": PlayerPreference(ideal=0.70, tolerance=0.30, weight=0.7),
        "persistence_stability": PlayerPreference(ideal=0.7, tolerance=0.60, weight=0.9),
        "complexity": PlayerPreference(ideal=0.58, tolerance=0.60, weight=1.0),
        "administrative_burden": PlayerPreference(ideal=0.30, tolerance=0.25, weight=1.1),
        "grind": PlayerPreference(ideal=0.65, tolerance=0.30, weight=0.9),
        "difficulty": PlayerPreference(ideal=0.60, tolerance=0.40, weight=0.8),
        "unintentional_friction": PlayerPreference(ideal=0.3, tolerance=0.5, weight=0.8),
        "meta_dependence": PlayerPreference(ideal=0.25, tolerance=0.23, weight=1.1),
        "restart_cost": PlayerPreference(ideal=0.35, tolerance=0.3, weight=0.8),
        "power_escalation": PlayerPreference(ideal=-0.45, tolerance=0.30, weight=1.1),
        "focused_structure": PlayerPreference(ideal=0.63, tolerance=0.55, weight=0.8),
    },
    veto_tags={
        "extreme_competitive_pvp": True
    },
    human_factor=1.0
)

# ==============================================================================
# STANDALONE EXECUTION CHECK
# ==============================================================================
if __name__ == "__main__":
    print("Evaluation engine loaded successfully.")
    print(f"Active player profile: {john_player_profile}")
