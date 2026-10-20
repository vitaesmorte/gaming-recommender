import os
from dotenv import load_dotenv

load_dotenv()

from db import supabase_client

def clear_supabase_cache():
    if not supabase_client:
        print("Error: SUPABASE_URL and SUPABASE_KEY are missing.")
        return

    # Delete evaluations and return count
    eval_res = supabase_client.table("evaluations").delete().neq("game_id", "").execute()
    eval_count = len(eval_res.data) if eval_res.data else 0

    # Delete games and return count
    target_games = [
        "pathfinder-wrath-of-the-righteous",
        "warhammer-40-000-rogue-trader",
        "warhammer-40000-rogue-trader",
        "barbie-project-friendship",
        "the-witcher-3-wild-hunt",
        "grim-fandango"
    ]
    game_res = supabase_client.table("games").delete().in_("id", target_games).execute()
    game_count = len(game_res.data) if game_res.data else 0

    print(f"Deleted {eval_count} evaluation record(s) and {game_count} game vector record(s).")

if __name__ == "__main__":
    clear_supabase_cache()
