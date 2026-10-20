from __future__ import annotations

from typing import Dict, List, Optional

from difflib import SequenceMatcher


# ============================================================
# PLAY HISTORY DATABASE
# ============================================================
#
# Replaces the old static PLAY_HISTORY_DIGEST (a fixed ~15-game list
# embedded in every prompt regardless of what game was being
# evaluated). That was the actual cause of the "forced weak
# comparison" problem: the evaluator LLM always had *some* games in
# front of it, so it always found *something* to compare to, even
# when nothing was a good match.
#
# This file holds the full history instead, each entry tagged with
# short genre/mechanic keywords. select_relevant_history() matches
# those tags against the CURRENT game's own tags (produced by the
# research LLM -- see first_llm.py) and returns only real matches.
# When nothing matches, it returns an empty list -- there is then
# nothing for the evaluator to force a comparison from.
#
# Entries with commentary (has_commentary=True) come directly from
# your preference PDF, transcribed as close to your own wording as
# possible. Entries without commentary are hours-only -- per your own
# stated rule ("playtime is not equal enjoyment"), these carry NO
# verdict and must never be cited as "loved" or "dropped". Their tags
# are plain public-knowledge genre labels (not a judgment of your
# taste), useful only for a neutral "you've also played N hours of
# this genre" mention if ever needed.
# ============================================================

TAG_VOCABULARY: List[str] = [
    # genre / structural
    "crpg", "arpg", "rpg", "action-rpg", "action-adventure", "open-world", "sandbox",
    "souls-like", "mmo", "survival", "city-builder", "management", "simulation",
    "grand-strategy", "rts", "rts-rpg-hybrid", "turn-based-strategy", "tactical-stealth",
    "tactical-combat", "horde-shooter", "looter-shooter", "fps", "fighting", "puzzle",
    "puzzle-adventure", "puzzle-horror", "narrative-adventure", "narrative",
    "narrative-comedy", "stealth-action", "action", "dialogue-driven", "exploration",
    # multiplayer / world
    "coop", "coop-friendly", "coop-transforms-positive", "pvp", "pvp-team",
    "persistent-world", "social",
    # progression / agency / craft
    "player-identity", "ownership", "grounded-progression", "earned-progression",
    "purposeful-grind", "tangible-progression", "clear-progression", "base-building",
    "creative-freedom", "systemic", "exploration-meaningful", "atmosphere",
    "low-admin-depth", "open-ended-progression", "strong-writing", "character-arc",
    "lucasarts-classic",
    # negative-impact markers (only apply when genuinely true)
    "meaningless-grind-negative", "meta-required-negative", "power-treadmill-negative",
    "admin-heavy-negative", "open-world-empty-negative", "no-long-term-purpose-negative",
    "weak-progression-negative", "slow-pace-negative", "weak-character-connection-negative",
    "static-world-negative", "weak-story-negative", "pacing-negative",
    "power-escalation-negative", "puzzle-adventure-negative",
]
# This is the SINGLE source of truth for tags. first_llm.py imports this list
# directly (`from gaming_his import TAG_VOCABULARY`) instead of keeping its own
# copy -- a second hardcoded copy is exactly how 8 tags actually used below
# (lucasarts-classic, strong-writing, character-arc, dialogue-driven,
# exploration, action, social, rts-rpg-hybrid) ended up unreachable: the
# research LLM was never told they existed, so they could never be produced
# for a freshly-searched game, and any match depending on them silently failed.

CALIBRATION_NOTE = (
    "Hours played is NOT a proxy for enjoyment, and a long playtime does NOT mean a game "
    "was loved start to finish. Baldur's Gate 3: 295 hours played, rated 70/100 overall -- "
    "a strong first act carrying a weaker late-game tail. Witcher 3: 123 hours, an all-time "
    "favorite. Ultima Online: 465 hours, an all-time favorite. Never infer 'loved' or "
    "'dropped' from hours alone -- always read the verdict and notes for each title."
)

GAME_HISTORY: List[Dict] = [
    # --- Commented entries (from the preference PDF) ---
    {
        "title": "Ultima Online", "hours": 465, "verdict": "loved", "has_commentary": True,
        "notes": "Meaningful progression, small improvements feel valuable, long-term character "
                 "development, persistent world, player identity, freedom, social interaction, "
                 "PvE/PvP danger, grinding is enjoyable because it has a purpose, world feels "
                 "alive, ownership matters. Explicitly the strongest example of grind done right.",
        "tags": ["sandbox", "mmo", "persistent-world", "purposeful-grind", "ownership", "social",
                 "pvp", "player-identity", "open-ended-progression"],
    },
    {
        "title": "Witcher 3", "hours": 123, "verdict": "loved", "has_commentary": True,
        "notes": "Immersive world, excellent narrative, meaningful choices, strong atmosphere, "
                 "exploration, side quests, grounded progression, limited but meaningful character "
                 "development, replayability.",
        "tags": ["open-world", "narrative", "atmosphere", "grounded-progression", "rpg",
                 "exploration-meaningful"],
    },
    {
        "title": "Cities: Skylines (1 & 2)", "hours": 359, "verdict": "loved", "has_commentary": True,
        "notes": "Creative freedom, ownership, visible consequences, systems interacting with "
                 "each other, emergent goals, realistic simulation, building something that "
                 "remains yours.",
        "tags": ["city-builder", "management", "systemic", "ownership", "creative-freedom",
                 "simulation"],
    },
    {
        "title": "Heroes of Might and Magic III", "hours": 65, "verdict": "loved", "has_commentary": True,
        "notes": "Strategic depth without administrative bloat, tangible progression (building "
                 "upgrades feel like they matter), excellent turn-based pacing, strong ownership "
                 "over towns and heroes, no distracting micromanagement menus, great for co-op "
                 "or casual hotseat play.",
        "tags": ["turn-based-strategy", "low-admin-depth", "ownership", "coop-friendly",
                 "tangible-progression"],
    },
    {
        "title": "Red Dead Redemption 2", "hours": 65, "verdict": "loved", "has_commentary": True,
        "notes": "World feels incredibly alive and immersive, grounded exploration with "
                 "meaningful discoveries, strong atmosphere and narrative, ownership (customizing "
                 "camp/horse/gear matters), small improvements and hunting feel valuable, avoids "
                 "the large-but-empty open-world trap.",
        "tags": ["open-world", "narrative", "atmosphere", "grounded-progression", "ownership",
                 "exploration-meaningful"],
    },
    {
        "title": "Age of Empires 2 & 4", "hours": 195, "verdict": "loved", "has_commentary": True,
        "notes": "Highly enjoyable in team PvP/co-op skirmishes, clear satisfying progression "
                 "through the ages within a single match, systems interact logically, competitive "
                 "elements are fun with a dedicated team (not solo esports toxicity), base "
                 "building provides temporary but satisfying ownership.",
        "tags": ["rts", "coop", "pvp-team", "clear-progression", "base-building"],
    },
    {
        "title": "Diablo 2 & 4", "hours": 200, "verdict": "mixed", "has_commentary": True,
        "notes": "Classic co-op ARPG experience is great for a while; D2 had strong character "
                 "identity and meaningful loot; D4 had a great initial co-op campaign. NEGATIVE: "
                 "inevitably hits a point where grind becomes a meaningless progression treadmill. "
                 "NEGATIVE: late-game devolves into forced meta builds and artificial playtime "
                 "padding.",
        "tags": ["arpg", "coop", "meaningless-grind-negative", "meta-required-negative",
                 "power-treadmill-negative"],
    },
    {
        "title": "World of Warcraft", "hours": 145, "verdict": "mixed", "has_commentary": True,
        "notes": "Persistent world and strong social interaction (similar to Ultima Online), "
                 "great early-game character development and identity. NEGATIVE: modern endgame "
                 "becomes a meaningless treadmill of gear resets. NEGATIVE: high potential for "
                 "forced meta builds and daily tasks that feel like administrative chores. Best "
                 "enjoyed for grounded leveling, not the competitive endgame.",
        "tags": ["mmo", "persistent-world", "social", "meaningless-grind-negative",
                 "meta-required-negative", "admin-heavy-negative"],
    },
    {
        "title": "No Man's Sky", "hours": 5, "verdict": "dropped", "has_commentary": True,
        "notes": "Massive creative freedom and exploration, base building provides strong "
                 "ownership and persistent progress, excellent relaxed co-op. NEGATIVE: universe "
                 "can feel large but empty without meaningful narrative discovery. NEGATIVE: "
                 "resource farming can cross into repetitive treadmill territory. NEGATIVE: "
                 "sandbox doesn't provide a compelling long-term purpose or reason to return. "
                 "NEGATIVE: multiplayer doesn't create meaningful hours, abandoned quickly.",
        "tags": ["sandbox", "exploration", "open-world-empty-negative", "no-long-term-purpose-negative",
                 "meaningless-grind-negative"],
    },
    {
        "title": "Baldur's Gate 1 & 2", "hours": 113, "verdict": "loved", "has_commentary": True,
        "notes": "Perfect character development; progressing from a low-level squire to a "
                 "god-like hero feels deeply earned.",
        "tags": ["crpg", "narrative", "earned-progression", "rpg"],
    },
    {
        "title": "Baldur's Gate 3", "hours": 295, "verdict": "mixed", "has_commentary": True,
        "notes": "Incredible Act 1: small choices and grounded progression (even finding a "
                 "wooden shield) matter; meaningful choices and rich storytelling. NEGATIVE: "
                 "post-Act 2 pacing suffers badly; power escalation makes you too strong too "
                 "quickly, rendering further progression meaningless. Still a good game overall, "
                 "just one whose late-game design loses the grounded feel that made the first "
                 "act special.",
        "rating_note": "295 hours played, rated 70/100 overall -- strong first act, weaker second half.",
        "tags": ["crpg", "narrative", "grounded-progression", "power-escalation-negative",
                 "pacing-negative", "rpg"],
    },
    {
        "title": "Anno 1800", "hours": 35, "verdict": "mixed", "has_commentary": True,
        "notes": "Deep, satisfying systems interacting with each other, building something that "
                 "remains yours with visible consequences, high creative freedom. NEGATIVE: "
                 "late-game production chains can easily tip into system bloat and bookkeeping; "
                 "requires actively managing complexity so it doesn't become cognitive overhead "
                 "without interesting decisions.",
        "tags": ["city-builder", "management", "systemic", "ownership", "admin-heavy-negative"],
    },
    {
        "title": "Elden Ring (incl. Nightreign)", "hours": 225, "verdict": "conditional", "has_commentary": True,
        "notes": "Gorgeous atmosphere and immersive world. Like Dark Souls 3, solo play is "
                 "unnecessarily tedious for daily gaming life. NEGATIVE: open world can sometimes "
                 "take the structured progression away. NEGATIVE: solo requires specific meta "
                 "builds or guides to avoid frustration. Co-op eliminates this friction and turns "
                 "the difficulty into hours of fun.",
        "tags": ["souls-like", "atmosphere", "meta-required-negative", "coop-transforms-positive",
                 "open-world"],
    },
    {
        "title": "The Elder Scrolls V: Skyrim", "hours": 105, "verdict": "mixed", "has_commentary": True,
        "notes": "Ultimate freedom and player identity, grounded progression where you build your "
                 "own path, world feels alive with reasons to explore and meaningful discoveries, "
                 "no forced meta builds, high ownership over houses/skills/character. NEGATIVE: "
                 "quests don't relate well to each other. NEGATIVE: world feels static and "
                 "disconnected, doesn't react enough, progression doesn't satisfy personal "
                 "achievement scale. NEGATIVE: story/quests not interesting enough to pursue. "
                 "NEGATIVE: overcrowded crafting items with no apparent reason.",
        "tags": ["open-world", "player-identity", "grounded-progression", "static-world-negative",
                 "weak-story-negative", "rpg"],
    },
    {
        "title": "Dark and Light", "hours": 157, "verdict": "mixed", "has_commentary": True,
        "notes": "Blends MMO and survival mechanics in a persistent world, meaningful progression "
                 "through harnessing elemental magic to build bases/craft equipment, ownership "
                 "through taming creatures and constructing homes, highly reliant on cooperation/"
                 "faction support for survival. NEGATIVE: extreme weather/hunger/resource "
                 "gathering easily turns into tedious administrative bookkeeping if the grind "
                 "loses its purpose. NEGATIVE: empty servers kill the point of playing (full "
                 "servers can be genuinely fun).",
        "tags": ["survival", "mmo", "persistent-world", "coop", "ownership",
                 "admin-heavy-negative"],
    },
    {
        "title": "A Plague Tale", "hours": 5.2, "verdict": "dropped", "has_commentary": True,
        "notes": "Great narrative, good storyline, great graphics. NEGATIVE: slow pace. NEGATIVE: "
                 "not getting powerful, mostly running away. NEGATIVE: lack of progression "
                 "feeling, always feeble.",
        "tags": ["narrative", "atmosphere", "weak-progression-negative", "slow-pace-negative"],
    },
    {
        "title": "Hellblade: Senua's Sacrifice", "hours": 1.4, "verdict": "dropped", "has_commentary": True,
        "notes": "Great storyline, good graphics, atmospheric world. NEGATIVE: hard to get into "
                 "the story. NEGATIVE: no real achievement or character connection.",
        "tags": ["narrative", "atmosphere", "weak-character-connection-negative"],
    },
    {
        "title": "Disco Elysium", "hours": 9.5, "verdict": "dropped", "has_commentary": True,
        "notes": "Interesting atmosphere, fun graphics, fun dialogue options. NEGATIVE: on paper "
                 "perfect for this player's taste but feels like an adventure puzzle game after a "
                 "while. NEGATIVE: character attitude and pessimist tone kill the vibe. Wanted "
                 "something closer to Grim Fandango or Sanitarium but it didn't land due to lack "
                 "of atmosphere and character connection.",
        "tags": ["narrative", "atmosphere", "dialogue-driven", "weak-character-connection-negative",
                 "puzzle-adventure-negative"],
    },
    {
        "title": "Grim Fandango", "hours": 24, "verdict": "loved", "has_commentary": True,
        "notes": "LucasArts' narrative peak — brilliant writing, memorable characters, "
                 "a genuinely moving character arc for Manny Calavera, and puzzle design "
                 "that respects the player's intelligence instead of relying on moon logic. "
                 "Replayed regularly, still a favorite. Proves that narrative character "
                 "progression can substitute for mechanical progression when the writing "
                 "and direction are strong enough.",
        "tags": ["puzzle-adventure", "narrative", "atmosphere", "strong-writing",
                 "character-arc", "lucasarts-classic"],
    },
     {
        "title": "Day of the Tentacle", "hours": 12, "verdict": "loved", "has_commentary": True,
        "notes": "Same DNA as Grim Fandango: LucasArts writing, humor that respects the "
                 "player's intelligence, puzzles whose solutions make internal narrative "
                 "sense rather than relying on random trial and error.",
        "tags": ["puzzle-adventure", "narrative", "atmosphere", "strong-writing",
                 "character-arc", "lucasarts-classic"],
    },
    {
        "title": "Monkey Island", "hours": 15, "verdict": "loved", "has_commentary": True,
        "notes": "Classic LucasArts adventure, memorable characters, sharp writing, witty "
                 "dialogue. Worth playing for the tone and writing even where the mechanical "
                 "depth is deliberately light.",
        "tags": ["puzzle-adventure", "narrative", "atmosphere", "strong-writing",
                 "character-arc", "lucasarts-classic"],
    },

    # --- Hours-only entries (no stated verdict -- tags are plain public
    #     genre facts, not a judgment of this player's taste). Per the
    #     player's own rule, these must never be cited as loved/dropped. ---
    {"title": "Divinity: Original Sin 2", "hours": 283, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["crpg", "coop", "tactical-combat"]},
    {"title": "Dark Souls", "hours": 90, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["souls-like"]},
    {"title": "Pillars of Eternity", "hours": 78, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["crpg", "narrative"]},
    {"title": "Witcher 1", "hours": 64, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["rpg", "narrative"]},
    {"title": "Witcher 2", "hours": 41, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["rpg", "narrative"]},
    {"title": "Warhammer Vermintide", "hours": 58, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["coop", "horde-shooter"]},
    {"title": "Medieval II: Total War", "hours": 48, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["grand-strategy", "rts"]},
    {"title": "Middle Earth: Shadow of Mordor", "hours": 33, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["action-rpg", "open-world"]},
    {"title": "Dawn of Man", "hours": 31, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["survival", "city-builder"]},
    {"title": "Far Cry 3", "hours": 27, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["open-world", "fps"]},
    {"title": "Green Hell", "hours": 25, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["survival"]},
    {"title": "Shadow Tactics: Blades of the Shogun", "hours": 24, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["tactical-stealth"]},
    {"title": "Tomb Raider", "hours": 23, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["action-adventure"]},
    {"title": "Manor Lords", "hours": 22, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["city-builder", "management"]},
    {"title": "South Park: The Stick of Truth", "hours": 19, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["rpg", "narrative-comedy"]},
    {"title": "Jurassic World Evolution", "hours": 18.5, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["management", "city-builder"]},
    {"title": "This War of Mine", "hours": 17, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["survival", "management"]},
    {"title": "Stronghold Crusader", "hours": 14, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["rts", "city-builder"]},
    {"title": "Ancestors Legacy", "hours": 11, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["rts"]},
    {"title": "Mortal Kombat X", "hours": 10, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["fighting"]},
    {"title": "Life is Strange", "hours": 9, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["narrative-adventure"]},
    {"title": "Ryse: Son of Rome", "hours": 8.7, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["action-adventure"]},
    {"title": "Batman: Arkham Asylum", "hours": 8.1, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["action-adventure"]},
    {"title": "Metro 2033", "hours": 7.2, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["fps", "narrative"]},
    {"title": "Sanitarium", "hours": 10, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["puzzle-adventure", "atmosphere"]},
    {"title": "Lucius", "hours": 6.7, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["puzzle-horror"]},
    {"title": "Lucius 2", "hours": 6.4, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["puzzle-horror"]},
    {"title": "GTA V", "hours": 76, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["open-world", "action"]},
    {"title": "Warhammer: Darktide", "hours": 56, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["coop", "horde-shooter"]},
    {"title": "Assassin's Creed Valhalla", "hours": 48, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["open-world", "action-rpg"]},
    {"title": "Assassin's Creed Origins", "hours": 35, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["open-world", "action-rpg"]},
    {"title": "Tiny Bang Story", "hours": 5.2, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["puzzle"]},
    {"title": "Dishonored 2", "hours": 45, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["stealth-action", "narrative"]},
    {"title": "Europa Universalis IV", "hours": 35, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["grand-strategy"]},
    {"title": "Crusader Kings 3", "hours": 63, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["grand-strategy"]},
    {"title": "Hearts of Iron 3", "hours": 28, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["grand-strategy"]},
    {"title": "Planescape: Torment", "hours": 45, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["crpg", "narrative"]},
    {"title": "Avatar: Frontiers of Pandora", "hours": 7, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["open-world", "action"]},
    {"title": "Mount & Blade II: Bannerlord", "hours": 59, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["sandbox", "rts-rpg-hybrid"]},
    {"title": "Total War: Warhammer", "hours": 18, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["grand-strategy", "rts"]},
    {"title": "Desperados III", "hours": 18, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["tactical-stealth"]},
    {"title": "Farthest Frontier", "hours": 6.4, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["city-builder", "survival"]},
    {"title": "Dragon's Dogma", "hours": 5.9, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["action-rpg"]},
    {"title": "The Vanishing of Ethan Carter", "hours": 5, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["narrative-adventure"]},
    {"title": "Tom Clancy's The Division", "hours": 4.3, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["looter-shooter", "coop"]},
    {"title": "Portal", "hours": 3.3, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["puzzle"]},
    {"title": "Tiny Glade", "hours": 3.1, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["city-builder"]},
    {"title": "The Darkness 2", "hours": 0.6, "verdict": "unrated", "has_commentary": False, "notes": "", "tags": ["fps", "narrative"]},
]


# Games in the same cluster share player expectations and design DNA,
# even if their extractor tags diverge slightly. Used only as a fallback
# when direct tag overlap yields zero matches, so it never overrides a
# genuine tag match.

GENRE_CLUSTERS = {
    "crpg": {"crpg", "rpg", "arpg", "action-rpg", "turn-based-strategy", "tactical-combat"},
    "management": {"city-builder", "management", "simulation", "grand-strategy"},
    "sandbox": {"sandbox", "open-world", "survival", "mmo"},
    "shooter": {"fps", "horde-shooter", "looter-shooter", "stealth-action"},
    "adventure": {"puzzle-adventure", "narrative-adventure", "narrative", "dialogue-driven"},
}

def _genre_cluster_for(tags):
    """Return the set of all cluster members for any cluster touched by tags."""
    cluster_tags = set()
    normalized = {_normalize(t) for t in tags}
    for cluster_members in GENRE_CLUSTERS.values():
        if normalized & cluster_members:
            cluster_tags |= cluster_members
    return cluster_tags


def _normalize(tag: str) -> str:
    return tag.strip().lower().replace(" ", "-")


def select_relevant_history(
    current_tags: List[str],
    top_n: int = 3,
    min_overlap: int = 1,
    current_title: Optional[str] = None,
) -> List[Dict]:
    """
    Scores every GAME_HISTORY entry by tag overlap with current_tags
    (the CURRENT game's own tags, produced by the research LLM) and
    returns the top_n matches with at least min_overlap shared tags.

    If current_title matches a GAME_HISTORY entry exactly, that entry is
    always included regardless of tag overlap -- without this, a game the
    player has literally played can be dropped from its own evaluation
    just because the extractor's tags differ from the history entry's tags.

    Returns an empty list when nothing matches closely enough.
    """
    current = {_normalize(t) for t in current_tags}
    title_match = find_history_by_title(current_title) if current_title else None

    if not current:
        return [title_match] if title_match else []

    scored = []
    for game in GAME_HISTORY:
        game_tags = {_normalize(t) for t in game.get("tags", [])}
        overlap = len(current & game_tags)
        if overlap >= min_overlap:
            scored.append((overlap, game))

    scored.sort(key=lambda pair: (pair[0], pair[1]["has_commentary"]), reverse=True)
    result = [game for _, game in scored[:top_n]]

    if title_match is not None:
        result = [g for g in result if g is not title_match]
        result = [title_match] + result[:top_n - 1]

    # FALLBACK: if direct tag overlap produced nothing, retry with the
    # broader genre cluster. This handles cases where the extractor
    # produced flavor tags (e.g. "kingdom-management") but missed the
    # anchor genre tag (e.g. "crpg") -- the cluster match recovers the
    # comparison the tag match should have made.
    if not result:
        cluster = _genre_cluster_for(current_tags)
        if cluster:
            fallback = []
            for game in GAME_HISTORY:
                game_tags = {_normalize(t) for t in game.get("tags", [])}
                overlap = len(cluster & game_tags)
                if overlap >= 1:
                    fallback.append((overlap, game))
            fallback.sort(key=lambda pair: (pair[0], pair[1]["has_commentary"]), reverse=True)
            result = [game for _, game in fallback[:top_n]]

    return result


def format_history_digest(matches: List[Dict]) -> str:
    """
    Formats select_relevant_history()'s output into the compact text
    block the evaluator prompt expects. Returns an explicit "no match"
    sentinel when matches is empty, rather than an empty string, so
    the prompt's instructions have something unambiguous to key off.
    """
    if not matches:
        return "No games in the play history closely match this game's genre/mechanics."

    lines = []
    for game in matches:
        if game["has_commentary"]:
            rating = f" [{game['rating_note']}]" if "rating_note" in game else ""
            lines.append(
                f"- {game['title']} ({game['hours']}h, {game['verdict']}){rating}: {game['notes']}"
            )
        else:
            lines.append(
                f"- {game['title']} ({game['hours']}h, same genre, no stated verdict -- "
                f"do not imply this player loved or disliked it)"
            )

    return "\n".join(lines)


def lookup_played_hours(game_title: str) -> float:
    """
    Returns the hours the player has actually logged on this title, if it
    appears in GAME_HISTORY. Case- and punctuation-insensitive title match.
    Zero means the game is unknown to the history.
    """
    def _key(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    target = _key(game_title)
    for game in GAME_HISTORY:
        if _key(game["title"]) == target:
            return float(game.get("hours", 0))
    return 0.0

def best_history_overlap(current_tags: List[str]) -> int:
    """
    Returns the size of the LARGEST tag intersection between current_tags
    and any single game in GAME_HISTORY. This is what the novelty penalty
    in the engine scales on -- a game that superficially shares one generic
    tag (e.g. "narrative") with something you've played should not get the
    same credit as a game that shares three specific tags (e.g. "crpg",
    "grounded-progression", "power-escalation-negative").
    """
    current = {_normalize(t) for t in current_tags}
    if not current:
        return 0
    best = 0
    for game in GAME_HISTORY:
        game_tags = {_normalize(t) for t in game.get("tags", [])}
        overlap = len(current & game_tags)
        if overlap > best:
            best = overlap
    return best

def find_history_by_title(game_title: str) -> Optional[Dict]:
    """
    Match a search query against GAME_HISTORY titles using three tiers:
      1. Exact alphanumeric match (fast path).
      2. Substring containment in either direction, provided the shorter
         string has at least 5 alphanumeric chars (avoids tiny matches
         like "3" matching everything).
      3. SequenceMatcher ratio >= 0.82.
    """
    def _key(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    target = _key(game_title)
    if not target:
        return None

    # Tier 1: exact
    for game in GAME_HISTORY:
        if _key(game["title"]) == target:
            return game

    # Tier 2: substring containment
    best = None
    best_ratio = 0.0
    for game in GAME_HISTORY:
        hist = _key(game["title"])
        if not hist:
            continue
        if hist in target or target in hist:
            shorter = min(len(hist), len(target))
            if shorter < 5:
                continue
            ratio = shorter / max(len(hist), len(target))
            if ratio > best_ratio:
                best = game
                best_ratio = ratio

    if best is not None:
        return best

    # Tier 3: fuzzy
    for game in GAME_HISTORY:
        hist = _key(game["title"])
        if not hist:
            continue
        ratio = SequenceMatcher(None, hist, target).ratio()
        if ratio >= 0.82 and ratio > best_ratio:
            best = game
            best_ratio = ratio

    return best

def find_franchise_match(game_title: str) -> Optional[Dict]:
    """
    Detects whether the current title shares a franchise/root prefix with
    any history entry. E.g. "Heroes of Might and Magic: Olden Era" shares
    the "heroesofmightandmagic" root with "Heroes of Might and Magic III".

    Returns the highest-priority matching history entry (loved > mixed >
    dropped > unrated) or None. Used to boost history_modifier when a
    game is a sequel/prequel/spinoff of something the player already
    has strong opinions on.
    """
    def _key(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    def _root(s: str) -> str:
        # Trim after ":", "-", "—" or roman numerals / edition suffixes
        # so "Heroes of Might and Magic III" and "Heroes of Might and
        # Magic: Olden Era" collapse to the same root.
        import re
        s = re.split(r"[:\-—]", s, maxsplit=1)[0].strip()
        s = re.sub(r"\s+(I{1,3}V?|IV|V|VI{0,3}|IX|X|\d+)\s*$", "", s).strip()
        return _key(s)

    target_root = _root(game_title)
    if len(target_root) < 6:  # avoid weak matches like "The"
        return None

    priority = {"loved": 4, "mixed": 3, "conditional": 2, "dropped": 1, "unrated": 0}
    best = None
    best_rank = -1

    for game in GAME_HISTORY:
        hist_root = _root(game["title"])
        if not hist_root or len(hist_root) < 6:
            continue
        # Root match: one is a prefix of the other and both are >=6 chars
        if hist_root == target_root or hist_root.startswith(target_root) or target_root.startswith(hist_root):
            rank = priority.get(game.get("verdict", "unrated"), 0)
            if rank > best_rank:
                best = game
                best_rank = rank
    return best
