import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Importing functions from your 3 files
import first_llm
import game_eval_claude_v0_4_3
import llm_evaluator

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    """Serves the main website search page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gaming Recommendation Engine</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 650px; margin: 50px auto; padding: 20px; }
            input, button { width: 100%; padding: 12px; margin: 8px 0; box-sizing: border-box; font-size: 16px; }
            button { background-color: #28a745; color: white; border: none; cursor: pointer; border-radius: 4px; }
            button:hover { background-color: #218838; }
            #result { margin-top: 20px; white-space: pre-wrap; background: #f8f9fa; padding: 18px; border-radius: 6px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h2>Game Recommendation Finder</h2>
        <p>Enter a game name and current price to analyze expected enjoyment and buy/wait recommendations.</p>
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
                    resultDiv.innerText = "An error occurred while analyzing the game. Please check your Render logs.";
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/recommend")
def recommend(name: str, price: float):
    # Step 1: Call your first Gemini script to extract the 24 game variables
    ratings_24 = first_llm.get_game_ratings(name)
    
    # Step 2: Pass the 24 numbers into your game engine math script
    engine_results = game_eval_claude_v0_4_3.evaluate_game(ratings_24)
    
    # Step 3: Pass engine math + price to your second Gemini evaluator script
    final_recommendation = llm_evaluator.get_final_evaluation(
        game_name=name,
        price=price,
        engine_output=engine_results
    )
    
    return {"recommendation": final_recommendation}
