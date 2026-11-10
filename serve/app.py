import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

THRESHOLD = 0.5999

model = joblib.load(Path(__file__).resolve().parent / "model.joblib")
log_path = Path(os.environ.get("LOG_PATH", "/logs/requests.jsonl"))
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(row: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    frame = pd.DataFrame([row])
    probability = float(model.predict_proba(frame)[0, 1])
    label = "yes" if probability >= THRESHOLD else "no"
    return {"label": label, "probability": round(probability, 4)}
