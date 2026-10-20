from dotenv import load_dotenv
load_dotenv()

import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

import first_llm
import game_eval_claude_v0_4_3
import llm_evaluator
import db
from gaming_his import select_relevant_history, best_history_overlap

app = FastAPI(title="Game Recommendation Engine")

# In-memory cache of last evaluation per game_id, used by the /ask endpoint.
# Personal single-user tool; if this ever needs multiple users, move to a
# proper session store.
_LAST_CONTEXT: dict = {}

_ASK_MODEL = "gemini-flash-lite-latest"  # separate quota from the evaluator


def _one_line_summary(eval_result: dict) -> str:
    m = eval_result["metrics"]
    s = eval_result["survival_analysis"]
    rec = round(m["recommendation_score"] * 100)
    top = round(m["personal_fit_adjusted"] * 100)
    hours = s["expected_enjoyable_hours"]
    abandon = s["predicted_50pct_abandonment_hour"]

    if abandon is None:
        risk = "holds attention well"
    elif abandon < 12:
        risk = f"likely drop-off around hour {abandon}"
    elif abandon < 30:
        risk = f"might fade after ~{abandon}h"
    else:
        risk = "long tail of attention"

    return f"{top}% on paper / {rec}% practical — ~{hours:.0f}h of enjoyable play, {risk}."


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Game Evaluator</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            :root {
                color-scheme: dark;
                --bg-main: #0b0f19;
                --bg-card: #151c2c;
                --bg-input: #1e293b;
                --border-color: #334155;
                --accent-green: #10b981;
                --accent-green-hover: #059669;
                --accent-yellow: #eab308;
                --accent-red: #ef4444;
                --accent-blue: #38bdf8;
                --accent-purple: #a5b4fc;
                --text-primary: #f8fafc;
                --text-secondary: #cbd5e1;
                --text-tertiary: #94a3b8;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            html { font-size: 16px; }
            body {
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg-main);
                color: var(--text-primary);
                min-height: 100vh;
                padding: 24px 24px 56px;
                display: flex;
                flex-direction: column;
                align-items: center;
                font-size: 16px;
                line-height: 1.55;
            }
            .container { width: 100%; max-width: 1420px; }

            /* HEADER */
            .site-header {
                display: flex; align-items: center; justify-content: space-between;
                margin-bottom: 20px;
            }
            .site-header .brand {
                display: flex; align-items: center; gap: 10px;
                font-size: 20px; font-weight: 800; letter-spacing: -.03em;
            }
            .brand-mark {
                width: 32px; height: 32px; display: grid; place-items: center;
                border-radius: 9px 9px 9px 3px;
                background: var(--accent-green); color: #07140f;
                font-size: 13px; font-weight: 800;
            }
            .site-header p {
                color: var(--text-tertiary); font-size: 13px;
                letter-spacing: .06em; text-transform: uppercase;
            }

            /* HERO BANNER */
            .game-hero {
                display: none; aspect-ratio: 1920 / 620; min-height: unset; position: relative;
                align-items: flex-end; overflow: hidden;
                padding: 32px 36px;
                border: 1px solid var(--border-color); border-radius: 14px;
                background-color: #111827;
                background-position: center; background-size: cover;
            }
            body.has-game-profile .game-hero { display: flex; }
            .game-hero::after {
                content: '';
                position: absolute;
                left: 0; right: 0; bottom: 0;
                height: 42%;
                background: linear-gradient(0deg, var(--bg-main) 0%, transparent 100%);
                pointer-events: none;
            }
            .hero-copy { position: relative; z-index: 1; }
            .hero-eyebrow {
                color: var(--accent-purple); font-size: 12px;
                letter-spacing: .16em; text-transform: uppercase;
            }
            .hero-copy h1 {
                margin: 8px 0 6px; font-size: clamp(1.9rem, 4vw, 3.2rem);
                letter-spacing: -.05em; line-height: 1.05;
            }
            .hero-copy p { color: var(--text-secondary); font-size: 15px; }

            /* SEARCH CARD */
            .search-card {
                position: relative; z-index: 2;
                max-width: 1220px;
                margin: 0 auto 28px;
                padding: 20px;
                border: 1px solid var(--border-color); border-radius: 12px;
                background: var(--bg-card);
                box-shadow: 0 12px 32px rgba(0,0,0,.34);
            }
            body.has-game-profile .search-card { margin-top: -32px; }
            .input-group {
                display: grid;
                grid-template-columns: minmax(0,1.6fr) minmax(160px,.7fr) auto;
                align-items: end; gap: 14px;
            }
            .field { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
            .field label {
                color: var(--text-tertiary); font-size: 12px;
                letter-spacing: .1em; text-transform: uppercase; font-weight: 600;
            }
            input {
                width: 100%; height: 46px; padding: 0 14px;
                border: 1px solid #40516b; border-radius: 8px;
                background: var(--bg-input); color: var(--text-primary);
                font-size: 15px; font-family: inherit; outline: none;
                transition: border-color .2s, box-shadow .2s;
            }
            input:focus { border-color: var(--accent-blue); box-shadow: 0 0 0 3px rgba(56,189,248,.16); }
            .analyze-button {
                min-height: 46px; padding: 0 22px;
                border: 0; border-radius: 8px;
                background: var(--accent-green); color: #04130e;
                font-size: 15px; font-weight: 700; cursor: pointer;
                transition: background-color .2s;
                white-space: nowrap;
            }
            .analyze-button:hover { background: #34d399; }
            @media (max-width: 640px) {
                .input-group { grid-template-columns: 1fr 1fr; }
                .input-group .field:first-child { grid-column: 1 / -1; }
                .analyze-button { grid-column: 1 / -1; width: 100%; }
            }

            /* LOADER + ERROR */
            #loader {
                display: none; margin: 16px auto; padding: 32px;
                text-align: center;
                border: 1px solid var(--border-color); border-radius: 12px;
                background: var(--bg-card);
            }
            .spinner {
                width: 44px; height: 44px; margin: 0 auto 16px;
                border: 4px solid var(--border-color); border-top-color: var(--accent-green);
                border-radius: 50%; animation: spin 1s linear infinite;
            }
            @keyframes spin { 100% { transform: rotate(360deg); } }
            #loader-text { color: var(--text-secondary); font-size: 15px; }
            .ui-error {
                display: none; margin: 0 0 16px; padding: 13px 15px;
                border: 1px solid #713b4b; border-radius: 9px;
                background: #291824; color: #fda4af; font-size: 15px;
            }

            /* MAIN LAYOUT */
            #output-container {
                display: none; width: 100%;
                grid-template-columns: minmax(240px,280px) minmax(0,1fr) minmax(240px,280px);
                align-items: start; gap: 22px;
            }
            @media (max-width: 1100px) {
                #output-container { grid-template-columns: 250px minmax(0,1fr); }
                .side-content { grid-column: 1 / -1; display: grid;
                                grid-template-columns: repeat(2, minmax(0,1fr)); gap: 16px; }
            }
            @media (max-width: 860px) {
                #output-container { grid-template-columns: 220px minmax(0,1fr); }
                .poster-sidebar { position: static; }
            }
            @media (max-width: 640px) {
                #output-container { grid-template-columns: 1fr; gap: 18px; }
                .poster-sidebar { display: grid;
                                  grid-template-columns: minmax(0,200px) minmax(0,1fr);
                                  position: static; gap: 14px; }
                .poster-frame { grid-row: span 2; }
                .tags-block { grid-column: 1 / -1; }
                .side-content { grid-column: auto; grid-template-columns: 1fr; }
            }

            /* SIDEBAR */
            .poster-sidebar {
                min-width: 0; display: flex; flex-direction: column;
                gap: 14px; position: sticky; top: 20px;
            }
            .poster-frame {
                position: relative; width: 100%; overflow: hidden;
                border: 1px solid var(--border-color); border-radius: 10px;
                background: var(--bg-card);
                box-shadow: 0 8px 24px rgba(0,0,0,.28);
            }
            .poster-frame img {
                display: block; width: 100%; height: auto;
                max-height: 500px; object-fit: contain;
            }
            .poster-fallback {
                display: none; position: absolute; inset: 0;
                place-items: center; padding: 16px;
                color: var(--text-tertiary); font-size: 14px; text-align: center;
            }
            .poster-frame.missing .poster-fallback { display: grid; }
            .poster-frame.missing img { display: none; }

            .gauge-card {
                padding: 16px 18px;
                border: 1px solid var(--border-color); border-radius: 10px;
                background: var(--bg-card);
                display: flex; flex-direction: column; align-items: center; text-align: center;
            }
            .score-label {
                color: var(--text-tertiary); font-size: 12px;
                letter-spacing: .1em; text-transform: uppercase; font-weight: 600;
                margin-bottom: 6px;
            }
            .gauge-svg {
                width: 100%; max-width: 220px; height: auto;
                margin: 4px auto 0;
                overflow: visible;
            }
            .score-display {
                color: var(--accent-green);
                font: 800 2.2rem 'Courier New', monospace; line-height: 1;
                margin-top: -4px;
            }
            .score-max { color: var(--text-tertiary); font-size: 1rem; }
            .topical-line {
                display: flex; justify-content: space-between; align-items: baseline;
                margin-top: 8px; padding-top: 10px;
                border-top: 1px solid rgba(148,163,184,.16);
                color: var(--text-secondary); font-size: 14px;
            }
            .topical-line strong { color: var(--accent-blue); font-size: 15px; }

            .tags-card {
                padding: 14px 16px;
                border: 1px solid var(--border-color); border-radius: 10px;
                background: var(--bg-card);
            }
            .tags-card h3 {
                margin: 0 0 10px; font-size: 14px; font-weight: 700;
                letter-spacing: .06em; text-transform: uppercase;
                color: var(--text-tertiary);
            }
            .tags-block {
                display: flex; flex-wrap: wrap; gap: 7px;
            }
            .tag-pill {
                padding: 5px 10px;
                border: 1px solid rgba(165,180,252,.28); border-radius: 16px;
                background: rgba(99,102,241,.14); color: #c7d2fe;
                font-size: 13px;
            }

            /* MAIN COLUMN */
            .main-column { min-width: 0; display: flex; flex-direction: column; gap: 18px; }
            .game-heading { padding: 2px 0 4px; }
            .game-heading h2 {
                margin: 0; font-size: clamp(1.5rem, 3vw, 2.1rem);
                letter-spacing: -.04em; line-height: 1.15;
            }
            .game-heading .meta {
                margin-top: 8px; color: var(--text-secondary); font-size: 15px;
                display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
            }
            .game-heading .meta .year {
                display: inline-block; padding: 2px 8px;
                border: 1px solid var(--border-color); border-radius: 4px;
                background: var(--bg-input); color: var(--text-secondary);
                font: 700 12px 'Courier New', monospace;
            }
            .game-heading .tagline {
                margin-top: 6px; color: var(--accent-purple);
                font-size: 15px; font-style: italic;
            }
            .section-kicker {
                display: block; margin: 0 0 8px;
                color: var(--accent-purple); font-size: 12px;
                letter-spacing: .14em; text-transform: uppercase; font-weight: 600;
            }
            .panel-title {
                margin: 0 0 14px; font-size: 1.3rem; letter-spacing: -.02em;
            }

            /* STAT GRID */
            .stat-grid {
                display: grid; grid-template-columns: repeat(3, minmax(0,1fr));
                gap: 12px;
            }
            @media (max-width: 640px) { .stat-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
            .stat-tile {
                position: relative; min-height: 96px;
                padding: 14px 16px;
                border: 1px solid var(--border-color); border-radius: 10px;
                background: var(--bg-card);
            }
            .stat-tile:before {
                content: ''; position: absolute; top: 0; left: 0; right: 0;
                height: 2px; background: #64748b;
            }
            .stat-tile.tone-violet:before { background: var(--accent-purple); }
            .stat-tile.tone-green:before { background: var(--accent-green); }
            .stat-tile.tone-blue:before { background: var(--accent-blue); }
            .stat-tile.tone-amber:before { background: var(--accent-yellow); }
            .stat-tile.tone-rose:before { background: #fb7185; }
            .stat-label {
                color: var(--text-tertiary); font-size: 12px;
                letter-spacing: .06em; text-transform: uppercase; font-weight: 600;
            }
            .stat-value {
                margin-top: 8px; font-size: 17px; font-weight: 800;
                line-height: 1.3; overflow-wrap: anywhere;
            }
            .tone-violet .stat-value { color: #c4b5fd; }
            .tone-green .stat-value { color: #6ee7b7; }
            .tone-blue .stat-value { color: #7dd3fc; }
            .tone-amber .stat-value { color: #fde047; }
            .tone-rose .stat-value { color: #fda4af; }
            .stat-value.verdict-green { color: var(--accent-green); }
            .stat-value.verdict-yellow { color: var(--accent-yellow); }
            .stat-value.verdict-red { color: var(--accent-red); }
            .stat-value.verdict-blue { color: var(--accent-blue); }
            .stat-sub {
                margin-top: 5px; color: var(--text-tertiary);
                font-size: 13px; line-height: 1.4;
            }

            /* SHORT VERSION */
            .short-version {
                padding: 20px 22px;
                border: 1px solid #3b4b64; border-radius: 12px;
                background: #141d2d;
            }
            .short-version h3 {
                margin: 0 0 10px; color: #6ee7b7; font-size: 1.5rem; line-height: 1.25;
            }
            .short-version h3.verdict-yellow { color: #fde047; }
            .short-version h3.verdict-red { color: #fda4af; }
            .short-version h3.verdict-green { color: #6ee7b7; }
            .short-version h3.verdict-blue { color: #7dd3fc; }
            .one-liner {
                margin: 0 0 14px; color: var(--accent-purple);
                font-size: 15px; font-style: italic;
            }
            .recommendation-summary {
                color: #e2e8f0; font-size: 16px; line-height: 1.7;
            }
            .recommendation-summary p { margin: 0 0 12px; }
            .recommendation-summary p:last-child { margin-bottom: 0; }

            /* FULL REPORT DETAILS */
            .report-details { margin-top: 16px; border-top: 1px solid #344258; }
            .report-details > summary {
                min-height: 46px; display: flex; align-items: center; gap: 10px;
                list-style: none; cursor: pointer;
                color: var(--text-secondary); font-size: 15px; font-weight: 700;
            }
            .report-details > summary::-webkit-details-marker { display: none; }
            .report-details > summary:hover { color: #fff; }
            .expand-icon {
                width: 26px; height: 26px; display: grid; place-items: center;
                border: 1px solid #53647c; border-radius: 50%;
                color: var(--accent-blue); font-size: 16px;
                transition: transform .2s, background .2s;
            }
            .report-details[open] .expand-icon {
                transform: rotate(45deg); background: #26374e;
            }
            .report-content {
                margin: 0; padding: 14px 0 2px;
                border-top: 1px solid #263449;
                color: #d7dee8; font-size: 16px; line-height: 1.75;
            }
            .report-content h1, .report-content h2, .report-content h3 {
                margin: 20px 0 10px; color: var(--accent-purple);
            }
            .report-content h1:first-child,
            .report-content h2:first-child,
            .report-content h3:first-child { margin-top: 0; }
            .report-content h3 {
                border-color: var(--border-color); font-size: 1.15rem;
                padding-bottom: 6px;
            }
            .report-content p { margin: 0 0 12px; }
            .report-content strong { color: #f8fafc; }
            .report-content ul, .report-content ol { margin: 0 0 12px 22px; }
            .report-content li { margin-bottom: 6px; }

            /* ASK BAR */
            .ask-card {
                padding: 20px 22px;
                border: 1px solid var(--border-color); border-radius: 12px;
                background: var(--bg-card);
            }
            .ask-input-group { display: flex; gap: 10px; margin-top: 8px; }
            .ask-input-group input { flex: 1; min-width: 0; }
            .ask-button {
                height: 46px; padding: 0 22px;
                border: 0; border-radius: 8px;
                background: #6366f1; color: #ffffff;
                font-size: 15px; font-weight: 700; cursor: pointer;
                white-space: nowrap;
                transition: background-color .2s;
            }
            .ask-button:hover { background: #7c7fff; }
            .ask-answer {
                display: none; margin-top: 14px; padding: 14px 16px;
                border: 1px solid var(--border-color); border-radius: 10px;
                background: #111827; color: #d7dee8;
                font-size: 16px; line-height: 1.7;
            }
            .ask-answer.thinking { color: var(--text-tertiary); font-style: italic; }

            /* SIDE CARDS */
            .side-card {
                margin: 0 0 14px; padding: 18px 20px;
                border: 1px solid var(--border-color); border-radius: 12px;
                background: var(--bg-card);
            }
            .side-card h3 { margin: 0 0 14px; font-size: 1.05rem; }
            .side-card .section-kicker { color: var(--accent-purple); }
            .side-link {
                display: flex; align-items: center; justify-content: space-between;
                gap: 12px; padding: 12px 14px; margin-bottom: 8px;
                border: 1px solid var(--border-color); border-radius: 8px;
                background: var(--bg-input); color: var(--text-secondary);
                text-decoration: none; font-size: 15px; font-weight: 600;
                transition: background-color .15s, color .15s;
            }
            .side-link:hover { background: #26364b; color: var(--accent-blue); }
            .side-link:last-of-type { margin-bottom: 0; }
            .side-card p.small-note {
                margin: 10px 0 0; color: var(--text-tertiary); font-size: 13px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header class="site-header">
                <div class="brand">
                    <span class="brand-mark">GE</span>
                    <span>Game Evaluator</span>
                </div>
                <p>A second opinion before you press buy</p>
            </header>

            <section class="game-hero" id="game-hero">
                <div class="hero-copy">
                    <span class="hero-eyebrow">Your personal buying guide</span>
                    <h1 id="hero-title">Gaming Recommendation Engine</h1>
                    <p id="hero-subtitle">Personalized deterministic fit analysis and value evaluation</p>
                </div>
            </section>

            <div class="search-card">
                <div class="input-group">
                    <div class="field">
                        <label for="gameName">Find a game</label>
                        <input type="text" id="gameName" placeholder="e.g. Kingdom Come: Deliverance II">
                    </div>
                    <div class="field">
                        <label for="gamePrice">Current price · USD</label>
                        <input type="number" id="gamePrice" step="0.01" placeholder="e.g. 19.99">
                    </div>
                    <button class="analyze-button" onclick="getRecommendation()">Analyse Game</button>
                </div>
            </div>

            <div id="loader">
                <div class="spinner"></div>
                <p id="loader-text">Building your game profile...</p>
            </div>
            <div class="ui-error" id="ui-error" role="alert"></div>

            <div id="output-container">
                <aside class="poster-sidebar">
                    <div class="poster-frame" id="poster-frame">
                        <img id="poster-img" src="" alt="Game poster">
                        <div class="poster-fallback">Game artwork unavailable</div>
                    </div>

                    <div class="gauge-card">
                        <div class="score-label">Recommendation score</div>
                        <svg viewBox="0 0 100 55" class="gauge-svg">
                            <defs>
                                <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                    <stop offset="0%" stop-color="#ef4444" />
                                    <stop offset="50%" stop-color="#eab308" />
                                    <stop offset="100%" stop-color="#10b981" />
                                </linearGradient>
                            </defs>
                            <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#1e293b" stroke-width="8" stroke-linecap="round"/>
                            <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="url(#gaugeGradient)" stroke-width="8" stroke-linecap="round"/>
                            <g id="needleGroup" transform="rotate(-90, 50, 50)" style="transition: transform 0.8s ease-out;">
                                <line x1="50" y1="50" x2="50" y2="18" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round"/>
                                <circle cx="50" cy="50" r="4" fill="#ffffff"/>
                            </g>
                        </svg>
                        <div class="score-display">
                            <span id="score-val">0</span><span class="score-max">/100</span>
                        </div>
                        <div class="topical-line">
                            <span>Topical fit</span>
                            <strong><span id="overall-val">0</span>/100</strong>
                        </div>
                        <div class="topical-line">
                            <span>Confidence</span>
                            <strong id="confidence-val">0%</strong>
                        </div>
                    </div>

                    <section class="tags-card" id="tags-card">
                        <h3>Player tags</h3>
                        <div class="tags-block" id="tags-block"></div>
                    </section>
                </aside>

                <main class="main-column">
                    <div class="game-heading">
                        <span class="section-kicker">Evaluation profile</span>
                        <h2 id="game-title"></h2>
                        <div class="meta" id="game-meta"></div>
                        <div class="tagline" id="game-tagline"></div>
                    </div>

                    <section class="evaluation-panel">
                        <h3 class="panel-title">Your fit at a glance</h3>
                        <div class="stat-grid" id="stat-grid"></div>
                    </section>

                    <section class="short-version">
                        <span class="section-kicker">Short version</span>
                        <h3 id="verdict-val">—</h3>
                        <p class="one-liner" id="one-liner"></p>
                        <div class="recommendation-summary" id="verdict-summary">
                            Your final recommendation will appear here after evaluation.
                        </div>
                        <details class="report-details" id="breakdown-details">
                            <summary>
                                <span class="expand-icon" aria-hidden="true">+</span>
                                <span>Read the full evaluation</span>
                            </summary>
                            <div class="report-content" id="report-box"></div>
                        </details>
                    </section>

                    <section class="ask-card">
                        <span class="section-kicker">Ask a question</span>
                        <div class="ask-input-group">
                            <input type="text" id="ask-input" placeholder="Ask anything about this game…"
                                   onkeydown="if(event.key==='Enter'){event.preventDefault();askQuestion();}">
                            <button class="ask-button" onclick="askQuestion()">Ask</button>
                        </div>
                        <div class="ask-answer" id="ask-answer"></div>
                    </section>
                </main>

                <aside class="side-content">
                    <section class="side-card">
                        <span class="section-kicker">Store page</span>
                        <h3>Go to Steam</h3>
                        <a id="steam-similar-link" class="side-link" href="#" target="_blank" rel="noopener noreferrer">
                            <span>Open on Steam store</span>
                            <span aria-hidden="true">↗</span>
                        </a>
                        <p class="small-note">Steam's store page lists similar games recommended by players.</p>
                    </section>

                    <section class="side-card">
                        <span class="section-kicker">Watch before buying</span>
                        <h3>YouTube</h3>
                        <a id="yt-gameplay-link" class="side-link" href="#" target="_blank" rel="noopener noreferrer">
                            <span>Find gameplay videos</span>
                            <span aria-hidden="true">↗</span>
                        </a>
                        <a id="yt-trailer-link" class="side-link" href="#" target="_blank" rel="noopener noreferrer">
                            <span>Search for a trailer</span>
                            <span aria-hidden="true">↗</span>
                        </a>
                    </section>
                </aside>
            </div>
        </div>

        <script>
            let currentGameId = "";

            function verdictClass(verdict) {
                const v = (verdict || "").toUpperCase();
                if (v.includes("BUY")) return "verdict-green";
                if (v.includes("WAIT")) return "verdict-yellow";
                if (v.includes("DEEP SALE") || v.includes("SKIP")) return "verdict-red";
                return "verdict-blue";
            }

            function escapeHtml(value) {
                return String(value ?? "").replace(/[&<>"']/g, c => ({
                    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
                })[c]);
            }

            function extractFinalRecommendation(markdown) {
                const text = String(markdown || "");
                const lines = text.split(/\\r?\\n/);
                const clean = line => line.trim().replace(/^#{1,6}\s*/, "").replace(/[*_`]/g, "").trim();
                const start = lines.findIndex(line => /^(?:\d+[.)]\s*)?Final Recommendation\\b/i.test(clean(line)));
                if (start < 0) return text;
                const heading = lines[start].match(/^\s*(#{1,6})/);
                const level = heading ? heading[1].length : 1;
                const end = lines.findIndex((line, index) => {
                    if (index <= start) return false;
                    const nextHeading = line.match(/^\s*(#{1,6})\s+/);
                    const numberedSection = /^\s*\d+[.)]\s+[A-Z]/.test(line);
                    return (nextHeading && nextHeading[1].length <= level) || numberedSection;
                });
                return lines.slice(start, end < 0 ? lines.length : end).join("\\n").trim();
            }

            function steamAppId(posterUrl) {
                return String(posterUrl || "").match(/\/apps\/(\d+)\//)?.[1] || "";
            }

            function steamLibraryHeroUrl(appId) {
                return appId
                    ? `https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/${appId}/library_hero.jpg`
                    : "";
            }

            function renderStatGrid(data) {
                const price = Number(data.price ?? 0);
                const wtp = Number(data.wtp_central ?? 0);
                const wtpLow = Number(data.wtp_low ?? 0);
                const wtpHigh = Number(data.wtp_high ?? 0);
                const abandon = data.predicted_50pct_abandonment_hour;
                const abandonStr = (abandon === null || abandon === undefined)
                    ? "Not predicted" : ("Hour " + abandon);

                const tiles = [
                    { label: "Price", value: "$" + price.toFixed(2), sub: "retail" },
                    { label: "Fair Value (WTP)", value: "$" + wtp.toFixed(2),
                      sub: "range $" + wtpLow.toFixed(2) + " – $" + wtpHigh.toFixed(2) },
                    { label: "Verdict", value: String(data.verdict || "—").toUpperCase(),
                      cls: verdictClass(data.verdict) },
                    { label: "Enjoyable Hours",
                      value: (data.expected_enjoyable_hours || 0).toFixed(1) + "h",
                      sub: "out of " + (data.total_playtime || 0).toFixed(0) + "h total" },
                    { label: "Survival Hours",
                      value: (data.expected_survival_hours || 0).toFixed(1) + "h",
                      sub: "expected play window" },
                    { label: "50% Abandonment", value: abandonStr, sub: "predicted drop-off" }
                ];

                document.getElementById("stat-grid").innerHTML = tiles.map(t => `
                    <div class="stat-tile">
                        <div class="stat-label">${escapeHtml(t.label)}</div>
                        <div class="stat-value ${t.cls || ""}">${escapeHtml(t.value)}</div>
                        ${t.sub ? `<div class="stat-sub">${escapeHtml(t.sub)}</div>` : ""}
                    </div>
                `).join("");
            }

                        function renderTags(tags) {
                const block = document.getElementById("tags-block");
                const card = document.getElementById("tags-card");
                if (!tags || tags.length === 0) {
                    block.innerHTML = "";
                    card.style.display = "none";
                    return;
                }
                card.style.display = "block";
                block.innerHTML = tags.map(t => `<span class="tag-pill">${escapeHtml(t)}</span>`).join("");
            }

            function renderExternalLinks(name, posterUrl) {
                const appId = steamAppId(posterUrl);
                const steam = document.getElementById("steam-similar-link");
                if (appId) {
                    steam.href = `https://store.steampowered.com/app/${appId}/`;
                    steam.style.display = "flex";
                } else {
                    steam.style.display = "none";
                }

                const gameplay = document.getElementById("yt-gameplay-link");
                const trailer = document.getElementById("yt-trailer-link");
                gameplay.href = `https://www.youtube.com/results?search_query=${encodeURIComponent(name + " gameplay")}`;
                trailer.href = `https://www.youtube.com/results?search_query=${encodeURIComponent(name + " trailer")}`;
            }

            function showError(message) {
                const box = document.getElementById("ui-error");
                box.textContent = message;
                box.style.display = "block";
            }

            async function askQuestion() {
                const input = document.getElementById("ask-input");
                const answer = document.getElementById("ask-answer");
                const question = input.value.trim();
                if (!question) return;
                if (!currentGameId) {
                    answer.style.display = "block";
                    answer.textContent = "Evaluate a game first, then ask.";
                    return;
                }

                answer.style.display = "block";
                answer.classList.add("thinking");
                answer.innerHTML = "Thinking…";

                try {
                    const url = `/ask?game_id=${encodeURIComponent(currentGameId)}&question=${encodeURIComponent(question)}`;
                    const response = await fetch(url);
                    const data = await response.json().catch(() => ({}));
                    answer.classList.remove("thinking");
                    if (!response.ok) {
                        answer.textContent = "Error: " + (data.detail || "Failed to get answer.");
                        return;
                    }
                    answer.innerHTML = marked.parse(data.answer || "");
                } catch (e) {
                    answer.classList.remove("thinking");
                    answer.textContent = "Network error: " + e.message;
                }
            }

            async function getRecommendation() {
                const name = document.getElementById('gameName').value.trim();
                const price = document.getElementById('gamePrice').value.trim();
                const loader = document.getElementById('loader');
                const outputContainer = document.getElementById('output-container');
                const reportBox = document.getElementById('report-box');
                const posterImg = document.getElementById('poster-img');
                const errorBox = document.getElementById('ui-error');

                if (!name || !price) {
                    showError("Please enter both a game name and a price.");
                    return;
                }

                errorBox.style.display = "none";
                outputContainer.style.display = "none";
                loader.style.display = "block";

                try {
                    const response = await fetch(`/recommend?name=${encodeURIComponent(name)}&price=${encodeURIComponent(price)}`);
                    const data = await response.json().catch(() => ({}));

                    loader.style.display = "none";

                    if (!response.ok) {
                        showError("Error: " + (data.detail || "Failed to generate evaluation."));
                        return;
                    }

                    currentGameId = data.game_id || name.toLowerCase().replace(/\s+/g, "-");
                    document.getElementById('game-title').textContent = name;
                    document.getElementById('hero-title').textContent = name;

                    const yearBit = data.release_year
                        ? `<span class="year">${data.release_year}</span>` : "";
                    const playtimeBit = data.average_playtime_hours
                        ? `${Math.round(data.average_playtime_hours)}h avg to beat` : "";
                    document.getElementById('game-meta').innerHTML = yearBit + playtimeBit;
                    document.getElementById('game-tagline').textContent = data.tagline || "";

                    document.getElementById('verdict-val').textContent = String(data.verdict || "—").toUpperCase();
                    document.getElementById('verdict-val').className = verdictClass(data.verdict);
                    document.getElementById('one-liner').textContent = data.one_line_summary || "";
                    document.getElementById('verdict-summary').innerHTML =
                        marked.parse(extractFinalRecommendation(data.recommendation));

                    if (data.poster_url) {
                        posterImg.src = data.poster_url;
                        posterImg.alt = name + " game poster";
                        document.getElementById("poster-frame").classList.remove("missing");
                        const appId = steamAppId(data.poster_url);
                        const heroUrl = steamLibraryHeroUrl(appId);
                        const heroBg = heroUrl
                            ? `linear-gradient(90deg,rgba(5,9,17,.72),rgba(5,9,17,.15) 74%),linear-gradient(0deg,rgba(5,9,17,.56),transparent 62%),url("${heroUrl}")`
                            : "linear-gradient(135deg,#182233,#0b0f19)";
                        document.getElementById("game-hero").style.backgroundImage = heroBg;
                    } else {
                        posterImg.removeAttribute("src");
                        document.getElementById("poster-frame").classList.add("missing");
                        document.getElementById("game-hero").style.backgroundImage =
                            "linear-gradient(135deg,#182233,#0b0f19)";
                    }
                    document.getElementById("game-hero").style.display = "flex";
                    document.body.classList.add("has-game-profile");

                    reportBox.innerHTML = marked.parse(data.recommendation);

                    const rec = data.recommendation_score || 0;
                    const overall = data.overall_score || 0;
                    const confidence = data.confidence || 0;

                    document.getElementById('score-val').innerText = rec;
                    document.getElementById('needleGroup').setAttribute(
                        "transform",
                        `rotate(${Math.max(0, Math.min(100, rec)) * 1.8 - 90} 50 50)`
                    );
                    document.getElementById('overall-val').innerText = overall;
                    document.getElementById('confidence-val').innerText = `${confidence}%`;

                    renderStatGrid(data);
                    renderTags(data.tags || []);
                    renderExternalLinks(name, data.poster_url);

                    document.getElementById("breakdown-details").open = false;
                    document.getElementById("ask-answer").style.display = "none";
                    document.getElementById("ask-input").value = "";

                    outputContainer.style.display = "grid";
                } catch (error) {
                    loader.style.display = "none";
                    showError("Network error: " + error.message);
                }
            }
        </script>
    </body>
    </html>
    """


@app.get("/ask")
def ask(game_id: str, question: str):
    """Answer a one-off question about the currently-evaluated game, using
    flash-lite so it doesn't consume the evaluator's daily quota."""
    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Question too long (max 500 characters).")

    ctx = _LAST_CONTEXT.get(game_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="No evaluation context for this game. Run /recommend first.")

    eval_result = ctx["eval_result"]
    tags = ctx.get("tags", [])
    title = ctx.get("title", "this game")
    total_playtime = ctx.get("total_playtime", 0)

    m = eval_result["metrics"]
    s = eval_result["survival_analysis"]
    f = eval_result["financial_valuation"]
    wtp = f["wtp"]
    phases = eval_result.get("phase_breakdown", [])

    phase_lines = "\n".join(
        f"  - {p['phase_name']} (hours {p['hour_range']}): "
        f"avg enjoyment {p['avg_enjoyment'] * 100:.0f}%, "
        f"hooks: {', '.join(p['top_hooks']) or 'none'}, "
        f"friction: {', '.join(p['top_friction']) or 'none'}"
        for p in phases
    ) or "  (no phase data)"

    prompt = f"""You are answering a question about **{title}** for a specific player,
using the deterministic engine output below, plus your own knowledge of this game's
general design as needed. Reply with 2-5 sentences of plain prose. Be candid and specific.
You MAY reason from the engine's friction markers, phase data, tags, and hours -- explain
what they imply about the player's experience -- even if the exact answer isn't printed
verbatim in the data. Only refuse if the question genuinely cannot be answered from the
game itself. Never invent precise numbers not given here.

### PLAYER CONTEXT
{llm_evaluator.PLAYER_PERSONA.strip()}

### ENGINE OUTPUT FOR {title}
- Topical fit: {m['personal_fit_adjusted'] * 100:.0f}%
- Recommendation score: {m['recommendation_score'] * 100:.0f}%
- Engine verdict: {f['verdict']}
- Expected enjoyable hours: {s['expected_enjoyable_hours']}h out of ~{total_playtime:.0f}h
- Predicted 50% abandonment: {s['predicted_50pct_abandonment_hour']}
- Current price: {f['current_price']} {f['currency']}
- Central WTP: {wtp['wtp_central']} {f['currency']} (range {wtp['wtp_low']}–{wtp['wtp_high']})
- Tags: {', '.join(tags) or '(none)'}
- Phase breakdown:
{phase_lines}

### QUESTION
{question}

### ANSWER
"""

    api_key = os.getenv("GEMINI_API_KEY") or first_llm.GEMINI_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{_ASK_MODEL}:generateContent?key={api_key}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 400},
    }
    try:
        response = requests.post(url, json=body, timeout=60)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ask model error: {e}")

    answer = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    if answer.startswith("```"):
        answer = answer.replace("```markdown", "").replace("```", "", 1).strip()
    return {"answer": answer}


@app.get("/recommend")
def recommend(name: str = "Kingdom Come Deliverance", price: float = 19.99):
    try:
        game_id = name.lower().strip().replace(" ", "-")
        EVALUATION_VERSION = "v8-2026-01-hero-tagline-ask"
        cache_key = f"{game_id}@{EVALUATION_VERSION}"

        # 1. Retrieve or extract game data (with tags / year / tagline)
        cached_game = db.get_cached_game(game_id)
        if cached_game and cached_game.get("tags"):
            game_vector = cached_game["game_vector"]
            avg_playtime = cached_game["average_playtime_hours"]
            poster_url = cached_game.get("poster_url") or db.fetch_poster_url(name)
            evidence_quality = 0.85
            tags = cached_game["tags"]
            release_year = cached_game.get("release_year") or 0
            tagline = cached_game.get("tagline") or ""
        else:
            extracted_data = first_llm.extract_game_data(game_name=name, price=price)
            game_vector = extracted_data["game_vector"]
            avg_playtime = extracted_data["average_playtime_hours"]
            evidence_quality = extracted_data["evidence_quality"]
            tags = extracted_data.get("tags", [])
            release_year = extracted_data.get("release_year", 0)
            tagline = extracted_data.get("tagline", "")
            poster_url = db.fetch_poster_url(name)

            db.save_game(
                game_id=game_id,
                title=name,
                avg_playtime=avg_playtime,
                game_vector=game_vector,
                poster_url=poster_url,
                tags=tags,
                release_year=release_year or None,
                tagline=tagline or None,
            )

        # 2. History overlap
        best_overlap = best_history_overlap(tags) if tags else 0

        from gaming_his import find_franchise_match
        franchise_match = find_franchise_match(name)
        if franchise_match and franchise_match.get("verdict") in ("loved", "mixed", "conditional"):
            best_overlap = max(best_overlap, 2)

        # 3. Engine math
        game_profile_obj = game_eval_claude_v0_4_3.GameProfile(
            title=name,
            static_vector=game_vector,
            average_playtime_hours=avg_playtime,
        )
        game_obj = game_eval_claude_v0_4_3.Game(
            game_id=game_id,
            title=name,
            release_date=None,
            profile=game_profile_obj,
            current_price=price,
            currency="USD",
        )
        engine_results = game_eval_claude_v0_4_3.evaluate_game(
            game=game_obj,
            player=game_eval_claude_v0_4_3.john_player_profile,
            variable_definitions=game_eval_claude_v0_4_3.VARIABLE_DEFINITIONS,
            evidence_quality=evidence_quality,
            best_overlap=best_overlap,
        )

        # 4. Cache context for /ask
        _LAST_CONTEXT[game_id] = {
            "eval_result": engine_results,
            "tags": tags,
            "title": name,
            "total_playtime": avg_playtime,
        }

        m = engine_results["metrics"]
        s = engine_results["survival_analysis"]
        f = engine_results["financial_valuation"]
        wtp = f["wtp"]

        overall_score = round(m["personal_fit_adjusted"] * 100)
        recommendation_score = round(m["recommendation_score"] * 100)
        confidence = round(m["confidence"] * 100)
        one_liner = _one_line_summary(engine_results)

        # 5. Cached report or generate new one
        cached_report = db.get_cached_evaluation(cache_key, price)
        if cached_report:
            final_narrative = cached_report
        else:
            final_narrative = llm_evaluator.generate_final_report(
                eval_result=engine_results,
                game_obj=game_obj,
                game_tags=tags,
            )
            db.save_evaluation(game_id=cache_key, price=price, report_markdown=final_narrative)

        return {
            "game_id": game_id,
            "recommendation": final_narrative,
            "poster_url": poster_url,
            "overall_score": overall_score,
            "recommendation_score": recommendation_score,
            "confidence": confidence,
            "tags": tags,
            "price": price,
            "verdict": f["verdict"],
            "wtp_central": wtp["wtp_central"],
            "wtp_low": wtp["wtp_low"],
            "wtp_high": wtp["wtp_high"],
            "expected_enjoyable_hours": s["expected_enjoyable_hours"],
            "expected_survival_hours": s["expected_survival_hours"],
            "predicted_50pct_abandonment_hour": s["predicted_50pct_abandonment_hour"],
            "total_playtime": avg_playtime,
            "average_playtime_hours": avg_playtime,
            "release_year": release_year,
            "tagline": tagline,
            "one_line_summary": one_liner,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
