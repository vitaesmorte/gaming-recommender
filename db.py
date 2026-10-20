import os
import requests
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase_client: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_poster_url(game_name: str) -> str:
    """Fetch official poster artwork automatically from Steam Store API."""
    try:
        url = f"https://store.steampowered.com/api/storesearch/?term={requests.utils.quote(game_name)}&l=english&cc=US"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            if items:
                app_id = items[0]["id"]
                # Steam library capsule (600x900 poster aspect ratio)
                return f"https://cdn.akamai.steamstatic.com/steam/apps/{app_id}/library_600x900.jpg"
    except Exception:
        pass
    # High-quality fallback placeholder if game is not on Steam
    return "https://images.unsplash.com/photo-1550745165-9bc0b252726f?auto=format&fit=crop&w=600&q=80"


def get_cached_game(game_id: str) -> dict | None:
    """Retrieve cached Stage 1 vector, tags, and metadata from Supabase."""
    if not supabase_client:
        return None
    try:
        response = supabase_client.table("games").select("*").eq("id", game_id).execute()
        if response.data:
            return response.data[0]
    except Exception as e:
        print(f"[Supabase Read Error]: {e}")
    return None


def save_game(game_id: str, title: str, avg_playtime: float,
              game_vector: dict, poster_url: str,
              tags: list | None = None,
              release_year: int | None = None,
              tagline: str | None = None):
    """Save Stage 1 vector, poster metadata, controlled-vocabulary tags, and metadata."""
    if not supabase_client:
        return
    try:
        payload = {
            "id": game_id,
            "title": title,
            "average_playtime_hours": avg_playtime,
            "game_vector": game_vector,
            "poster_url": poster_url,
            "tags": tags or [],
            "release_year": release_year,
            "tagline": tagline,
        }
        supabase_client.table("games").upsert(payload).execute()
    except Exception as e:
        print(f"[Supabase Write Error]: {e}")


def get_cached_evaluation(game_id: str, price: float) -> str | None:
    """Retrieve cached final evaluation report from Supabase."""
    if not supabase_client:
        return None
    try:
        response = (
            supabase_client.table("evaluations")
            .select("report_markdown")
            .eq("game_id", game_id)
            .eq("price", price)
            .execute()
        )
        if response.data:
            return response.data[0]["report_markdown"]
    except Exception as e:
        print(f"[Supabase Eval Read Error]: {e}")
    return None


def save_evaluation(game_id: str, price: float, report_markdown: str):
    """Save generated evaluation report to Supabase (upsert, not insert)."""
    if not supabase_client:
        return
    try:
        payload = {
            "game_id": game_id,
            "price": price,
            "report_markdown": report_markdown,
        }
        # `insert` created duplicate rows on repeated queries for the same
        # game+price. `upsert` requires a unique constraint on (game_id, price)
        # in Supabase — see the SQL below.
        supabase_client.table("evaluations").upsert(payload).execute()
    except Exception as e:
        print(f"[Supabase Eval Write Error]: {e}")
