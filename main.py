from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

import first_llm
import game_eval_claude_v0_4_3
import llm_evaluator

app = FastAPI(title="Game Recommendation Engine")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gaming Recommendation Engine</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 650px; margin: 50px auto; padding: 20px; }
            input, button { width: 100%; padding: 12px; margin: 8px 0; box-sizing: border-box; font-size: 16px; }
            button { background-color: #28a745; color: white; border: none; cursor: pointer; border-radius: 4px; }
            #result { margin-top: 20px; white-space: pre-wrap; background: #f8f9fa; padding: 18px; border-radius: 6px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h2>Game Recommendation Finder</h2>
        <input type="text" id="gameName" placeholder="Enter game name (e.g., Baldur's Gate 3)">
        <input type="number" id="gamePrice" step="0.01" placeholder="Enter current price in USD (e.g., 59.99)">
        <button onclick="getRecommendation()">Analyze Game</button>
        <div id="result">Results will appear here...</div>

        <script>
            async function getRecommendation() {
                const name = document.getElementById('gameName').value;
                const price = document.getElementById('gamePrice').value;
                const resultDiv = document.getElementById('result');
                
                if (!name || !price) {
                    resultDiv.innerText = "Please enter both a game name and a price.";
                    return;
                }

                resultDiv.innerText = "Running evaluation pipeline across Gemini and Game Engine...";
                
                try {
                    const response = await fetch(`/recommend?name=${encodeURIComponent(name)}&price=${encodeURIComponent(price)}`);
                    const data = await response.json();
                    resultDiv.innerText = data.recommendation;
                } catch (error) {
                    resultDiv.innerText = "An error occurred while analyzing the game.";
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/recommend")
def recommend(name: str = "Ultima Online", price: float = 19.99):
    try:
        # Step 1: LLM Research & Extraction
        extracted_data = first_llm.extract_game_data(game_name=name, price=price)

        # Step 2: Build Game Object & Run Evaluation Math Engine
        game_profile_obj = game_eval_claude_v0_4_3.GameProfile(
            title=name,
            static_vector=extracted_data["game_vector"],
            average_playtime_hours=extracted_data["average_playtime_hours"]
        )

        game_obj = game_eval_claude_v0_4_3.Game(
            game_id=name.lower().replace(" ", "-"),
            title=name,
            release_date=None,
            profile=game_profile_obj,
            current_price=price,
            currency="USD"
        )

        engine_results = game_eval_claude_v0_4_3.evaluate_game(
            game=game_obj,
            player=game_eval_claude_v0_4_3.john_player_profile,
            variable_definitions=game_eval_claude_v0_4_3.VARIABLE_DEFINITIONS,
            evidence_quality=extracted_data["evidence_quality"]
        )

        # Step 3: LLM Synthesis Report
        final_narrative = llm_evaluator.generate_final_report(eval_result=engine_results, game_obj=game_obj)

        return {"recommendation": final_narrative}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
