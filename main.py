"""
FastAPI microservice for the AG News classifier.

Run locally after training:
    python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

Test in PowerShell:
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"text":"Apple announced a new AI feature for its latest iPhone."}'
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


MODEL_PATH = Path("outputs/agnews_tfidf_logreg.joblib")

app = FastAPI(
    title="AG News Classification Service",
    description="Classifies news text into World, Sports, Business, or Sci_Tech.",
    version="1.0.0",
)


DEMO_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AG News Classifier</title>
  <style>
    :root {
      --bg: #f7f7f4;
      --ink: #222936;
      --muted: #667085;
      --line: #d9ded8;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warn: #c2410c;
      --blue: #2563eb;
      --green: #15803d;
      --gold: #a16207;
      --shadow: 0 18px 55px rgba(21, 34, 32, 0.10);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .shell {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 38px 0 28px;
    }

    .topline {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 24px;
    }

    h1 {
      margin: 0;
      font-size: clamp(30px, 5vw, 54px);
      line-height: 0.98;
      font-weight: 760;
      letter-spacing: 0;
    }

    .meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 14px;
    }

    .pill {
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.78);
      border-radius: 999px;
      padding: 7px 11px;
      white-space: nowrap;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
      min-height: 560px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
      border-radius: 8px;
      overflow: hidden;
    }

    .compose,
    .results {
      padding: 28px;
    }

    .compose {
      border-right: 1px solid var(--line);
    }

    label {
      display: block;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }

    textarea {
      width: 100%;
      min-height: 270px;
      resize: vertical;
      border: 1px solid #c9d2cc;
      border-radius: 8px;
      padding: 16px;
      color: var(--ink);
      font: inherit;
      line-height: 1.55;
      outline: none;
      background: #fbfcfb;
    }

    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
    }

    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 14px;
    }

    button {
      min-height: 42px;
      border: 0;
      border-radius: 8px;
      padding: 0 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    .primary {
      background: var(--accent);
      color: #fff;
    }

    .primary:hover {
      background: var(--accent-dark);
    }

    .secondary {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
    }

    .samples {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 22px;
    }

    .sample {
      min-height: 36px;
      padding: 0 12px;
      border: 1px solid var(--line);
      background: #f8faf9;
      color: #374151;
      text-align: left;
      font-size: 14px;
      font-weight: 650;
    }

    .result-header {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 18px;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--line);
    }

    .result-label {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .category {
      margin: 0;
      font-size: clamp(32px, 5vw, 50px);
      line-height: 1;
      font-weight: 780;
    }

    .confidence {
      min-width: 104px;
      text-align: right;
    }

    .confidence strong {
      display: block;
      font-size: 31px;
      line-height: 1;
      color: var(--accent);
    }

    .confidence span {
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    .bars {
      display: grid;
      gap: 16px;
      margin-top: 26px;
    }

    .bar-row {
      display: grid;
      gap: 7px;
    }

    .bar-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 14px;
      font-weight: 700;
    }

    .track {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #edf0ed;
    }

    .fill {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      transition: width 240ms ease;
    }

    .Business { background: var(--gold); }
    .Sci_Tech { background: var(--blue); }
    .Sports { background: var(--green); }
    .World { background: var(--warn); }

    .empty {
      margin-top: 44px;
      color: var(--muted);
      line-height: 1.55;
    }

    .error {
      margin-top: 18px;
      color: #991b1b;
      font-weight: 700;
    }

    .links {
      display: flex;
      gap: 14px;
      margin-top: 22px;
      color: var(--muted);
      font-size: 14px;
    }

    .links a {
      color: var(--accent-dark);
      font-weight: 700;
      text-decoration: none;
    }

    @media (max-width: 780px) {
      .topline {
        align-items: start;
        flex-direction: column;
      }

      .meta {
        justify-content: flex-start;
      }

      .workspace {
        grid-template-columns: 1fr;
      }

      .compose {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }

      .samples {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topline">
      <h1>AG News<br>Classifier</h1>
      <div class="meta" aria-label="Model summary">
        <span class="pill">TF-IDF</span>
        <span class="pill">Logistic Regression</span>
        <span class="pill">Test accuracy 92.05%</span>
      </div>
    </header>

    <section class="workspace" aria-label="Prediction workspace">
      <form class="compose" id="predict-form">
        <label for="text">News text</label>
        <textarea id="text" name="text" minlength="3">Apple announced a new AI feature for its latest iPhone.</textarea>
        <div class="actions">
          <button class="primary" type="submit">Predict</button>
          <button class="secondary" type="button" id="clear">Clear</button>
        </div>
        <div class="samples" aria-label="Sample news text">
          <button class="sample" type="button" data-text="Global markets fell after the central bank raised interest rates.">Business</button>
          <button class="sample" type="button" data-text="The team won the championship after scoring in overtime.">Sports</button>
          <button class="sample" type="button" data-text="World leaders met to discuss a new international peace agreement.">World</button>
          <button class="sample" type="button" data-text="A new satellite system will improve mobile internet coverage.">Sci-Tech</button>
        </div>
        <div class="links">
          <a href="/docs">API docs</a>
          <a href="/health">Health</a>
        </div>
      </form>

      <section class="results" aria-live="polite">
        <div class="result-header">
          <div>
            <p class="result-label">Prediction</p>
            <p class="category" id="category">Ready</p>
          </div>
          <div class="confidence">
            <strong id="confidence">--</strong>
            <span>confidence</span>
          </div>
        </div>
        <div class="bars" id="bars"></div>
        <p class="empty" id="empty">Enter a news headline or short article and run a prediction.</p>
        <p class="error" id="error" role="alert"></p>
      </section>
    </section>
  </main>

  <script>
    const form = document.getElementById("predict-form");
    const text = document.getElementById("text");
    const clear = document.getElementById("clear");
    const category = document.getElementById("category");
    const confidence = document.getElementById("confidence");
    const bars = document.getElementById("bars");
    const empty = document.getElementById("empty");
    const error = document.getElementById("error");
    const labels = ["Business", "Sci_Tech", "Sports", "World"];

    function percent(value) {
      return `${Math.round(value * 1000) / 10}%`;
    }

    function renderResult(data) {
      category.textContent = data.predicted_category;
      confidence.textContent = percent(data.confidence);
      empty.hidden = true;
      error.textContent = "";
      bars.innerHTML = "";

      labels.forEach((label) => {
        const value = data.probabilities[label] || 0;
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <div class="bar-top">
            <span>${label.replace("_", " / ")}</span>
            <span>${percent(value)}</span>
          </div>
          <div class="track">
            <div class="fill ${label}" style="width: ${value * 100}%"></div>
          </div>
        `;
        bars.appendChild(row);
      });
    }

    async function predict() {
      const value = text.value.trim();
      if (value.length < 3) {
        error.textContent = "Please enter at least 3 characters.";
        return;
      }

      category.textContent = "Running";
      confidence.textContent = "--";
      error.textContent = "";

      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: value }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Prediction failed.");
      }

      renderResult(await response.json());
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        await predict();
      } catch (err) {
        category.textContent = "Error";
        confidence.textContent = "--";
        empty.hidden = true;
        bars.innerHTML = "";
        error.textContent = err.message;
      }
    });

    clear.addEventListener("click", () => {
      text.value = "";
      text.focus();
    });

    document.querySelectorAll(".sample").forEach((button) => {
      button.addEventListener("click", () => {
        text.value = button.dataset.text;
        text.focus();
      });
    });
  </script>
</body>
</html>
"""


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=3, description="News headline, short description, or article text.")


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run 'python train_agnews_tfidf.py' first."
        )
    return joblib.load(MODEL_PATH)


artifact = None


@app.on_event("startup")
def startup_event() -> None:
    global artifact
    artifact = load_artifact()


@app.get("/", response_class=HTMLResponse)
def demo_page() -> HTMLResponse:
    return HTMLResponse(DEMO_HTML)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "AG News Classification Service",
        "input": "JSON with a text field",
        "output": "predicted_category, confidence, and class probabilities",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if artifact is None:
        raise HTTPException(status_code=500, detail="Model artifact is not loaded.")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="The text field cannot be empty.")

    model = artifact["model"]
    predicted_category = str(model.predict([text])[0])
    proba = model.predict_proba([text])[0]

    # sklearn stores the class order on the fitted classifier / pipeline.
    classes = list(model.classes_)
    probabilities = {
        str(label): round(float(prob), 4)
        for label, prob in zip(classes, proba)
    }
    confidence = round(float(np.max(proba)), 4)

    return PredictionResponse(
        predicted_category=predicted_category,
        confidence=confidence,
        probabilities=probabilities,
    )
