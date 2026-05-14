"""
FastAPI inference endpoint for malicious URL detection.

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

POST /predict
    Body: {"url": "https://example.com"}
    Returns: {"verdict": "SAFE"|"WARN"|"BLOCK", "score": 0.12, "reason": [...]}

POST /feedback
    Body: {"url": "...", "correct_label": 0|1}
    Queues correction for next retraining cycle.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from features import extract_features, feature_names, feature_vector

app = FastAPI(title="URL Threat Classifier", version="1.0.0")

MODEL_DIR = os.environ.get("MODEL_DIR", "models/")
FEEDBACK_LOG = os.environ.get("FEEDBACK_LOG", "data/feedback.jsonl")

_gbm = None
_lr = None
_meta = None


def _load_models():
    global _gbm, _lr, _meta
    if _gbm is None:
        gbm_path = Path(MODEL_DIR) / "gbm.pkl"
        lr_path = Path(MODEL_DIR) / "lr_pipeline.pkl"
        meta_path = Path(MODEL_DIR) / "meta.json"
        if not gbm_path.exists():
            raise RuntimeError(f"Model not found at {gbm_path}. Run src/train.py first.")
        with open(gbm_path, "rb") as f:
            _gbm = pickle.load(f)
        with open(lr_path, "rb") as f:
            _lr = pickle.load(f)
        with open(meta_path) as f:
            _meta = json.load(f)


class PredictRequest(BaseModel):
    url: str


class PredictResponse(BaseModel):
    url: str
    verdict: Literal["SAFE", "WARN", "BLOCK"]
    score: float
    confidence: str
    top_features: List[dict]
    timestamp: str


class FeedbackRequest(BaseModel):
    url: str
    correct_label: int
    note: Optional[str] = None


@app.on_event("startup")
def startup():
    try:
        _load_models()
    except RuntimeError as e:
        print(f"WARNING: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _gbm is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _load_models()

    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    try:
        X = feature_vector(url).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {e}")

    gbm_score = float(_gbm.predict_proba(X)[0, 1])
    lr_score = float(_lr.predict_proba(X)[0, 1])
    score = round(_meta["gbm_weight"] * gbm_score + _meta["lr_weight"] * lr_score, 4)

    block_thresh = _meta["thresholds"]["block"]
    warn_thresh = _meta["thresholds"]["warn"]

    if score >= block_thresh:
        verdict = "BLOCK"
    elif score >= warn_thresh:
        verdict = "WARN"
    else:
        verdict = "SAFE"

    spread = abs(gbm_score - lr_score)
    confidence = "high" if spread < 0.1 else "medium" if spread < 0.25 else "low"

    feats = extract_features(url)
    importances = dict(zip(feature_names(), _gbm.feature_importances_))
    top_features = sorted(
        [{"feature": k, "value": round(float(feats[k]), 4), "importance": round(importances.get(k, 0), 4)} for k in feats],
        key=lambda x: -x["importance"],
    )[:5]

    return PredictResponse(
        url=url,
        verdict=verdict,
        score=score,
        confidence=confidence,
        top_features=top_features,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    Path(FEEDBACK_LOG).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "url": req.url,
        "correct_label": req.correct_label,
        "note": req.note,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "queued", "message": "Correction logged for next retraining cycle."}


@app.get("/stats")
def stats():
    feedback_count = 0
    if Path(FEEDBACK_LOG).exists():
        with open(FEEDBACK_LOG) as f:
            feedback_count = sum(1 for _ in f)
    return {
        "model_auc": _meta["auc"] if _meta else None,
        "thresholds": _meta["thresholds"] if _meta else None,
        "pending_feedback": feedback_count,
    }
