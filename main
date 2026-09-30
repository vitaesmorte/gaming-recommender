import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import google.generativeai as genai

app = FastAPI()

# Configure Gemini API Key from environment variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Your personal game preferences & history summary
USER_PREFERENCES = """
Enjoys rich RPG progression (origins from weak to hero), co-op experiences, tactical/PvP strategy.
Dislikes excessive post-game power inflation, heavy sci-fi, Japanese mythological aesthetics, and hyper-competitive shooters.
"""

def evaluate_game_engine(ratings: list) -> dict:
    """Your game engine math logic goes here."""
    # Example placeholder evaluation math:
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    estimated_hours = int(avg_rating * 3.5)
    abandon_hour = max(5, int(estimated_hours * 0.4))
    return {
        "score": avg_rating,
        "estimated_hours": estimated_hours,
        "abandon_hour": abandon_hour
    }

@app.get("/", response_class=HTMLResponse)
def home():
    """Serves the simple website interface with game name and price inputs."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gaming Recommendation Engine</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
            input, button { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; }
            button { background-color: #28a745; color: white; border: none; cursor: pointer; }
            #result { margin-top: 20px; white-space: pre-wrap; background: #f4f4f4; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h2>Game Recommendation Finder</h2>
        <input type="text" id="gameName" placeholder="Enter game name (e.g., Baldur's Gate 3)">
        <input type="number" id="gamePrice" placeholder="Enter current price in USD (e.g., 59.99)">
        <button onclick="getRecommendation()">Analyze Game</button>
        <div id="result"></div>

        <script>
            async function getRecommendation() {
                const name = document.getElementById('gameName').value;
                const price = document.getElementById('gamePrice').value;
                const resultDiv = document.getElementById('result');
                resultDiv.innerText = "Analyzing game and running evaluation...";
                
                const response = await fetch(`/recommend?name=${encodeURIComponent(name)}&price=${encodeURIComponent(price)}`);
                const data = await response.json();
                resultDiv.innerText = data.recommendation;
            }
        </script>
    </body>
    </html>
    """

@app.get("/recommend")
def recommend(name: str, price: float):
    # Step 1: Gemini rates the game across 24 variables
    prompt1 = f"Rate the video game '{name}' across 24 key gameplay and design variables. Return only a list of 24 numbers from 1 to 10 separated by commas."
    resp1 = model.generate_content(prompt1).text
    
    try:
        ratings = [float(x.strip()) for x in resp1.strip().split(",") if x.strip().replace('.', '', 1).isdigit()]
    except Exception:
        ratings = [5.0] * 24

    # Step 2: Pass numbers to your game engine math
    engine_results = evaluate_game_engine(ratings)

    # Step 3: Second Gemini call translates math + price + user preferences into final advice
    prompt2 = f"""
    User Preferences: {USER_PREFERENCES}
    Game Name: {name}
    Current Price: ${price}
    Game Engine Evaluation Output: {engine_results}

    Based on all this data, provide a clear, easy-to-read recommendation:
    1. Expected actual enjoyment time (hours).
    2. Estimated hour they might abandon or finish the game.
    3. Final purchase advice (e.g., Buy now at full price, Wait for deep discount, or Buy at $XX price).
    """
    final_output = model.generate_content(prompt2).text
    return {"recommendation": final_output}
