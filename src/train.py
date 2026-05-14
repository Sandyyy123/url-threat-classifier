"""
Train malicious URL classifier.

Usage:
    python src/train.py --data data/urls.csv --output models/

Dataset format (CSV):
    url,label   (label: 1=malicious, 0=safe)

Recommended public datasets:
    - PhishTank: https://www.phishtank.com/developer_info.php
    - URLhaus:   https://urlhaus.abuse.ch/api/
    - Alexa top-1M for benign examples
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import feature_names, feature_vector


def build_features(urls):
    X = np.array([feature_vector(u) for u in urls])
    return X


def train(data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    assert "url" in df.columns and "label" in df.columns, "CSV must have 'url' and 'label' columns"

    print(f"Loaded {len(df)} URLs ({df['label'].sum()} malicious, {(df['label']==0).sum()} safe)")

    print("Extracting features...")
    X = build_features(df["url"].tolist())
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model 1: XGBoost-style gradient boosting
    gbm = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )

    # Model 2: Calibrated logistic regression
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])

    print("Training GBM...")
    gbm.fit(X_train, y_train)

    print("Training LR...")
    lr_pipeline.fit(X_train, y_train)

    # Ensemble: average probabilities
    gbm_probs = gbm.predict_proba(X_test)[:, 1]
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    ensemble_probs = 0.7 * gbm_probs + 0.3 * lr_probs
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    print("\n--- Evaluation ---")
    print(classification_report(y_test, ensemble_preds, target_names=["safe", "malicious"]))
    auc = roc_auc_score(y_test, ensemble_probs)
    print(f"ROC-AUC: {auc:.4f}")

    # Feature importance
    importances = dict(zip(feature_names(), gbm.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 features:")
    for feat, imp in top_features:
        print(f"  {feat:<35} {imp:.4f}")

    # Save models
    with open(os.path.join(output_dir, "gbm.pkl"), "wb") as f:
        pickle.dump(gbm, f)
    with open(os.path.join(output_dir, "lr_pipeline.pkl"), "wb") as f:
        pickle.dump(lr_pipeline, f)

    meta = {
        "feature_names": feature_names(),
        "auc": round(auc, 4),
        "thresholds": {"block": 0.85, "warn": 0.55},
        "gbm_weight": 0.7,
        "lr_weight": 0.3,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/urls.csv")
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()
    train(args.data, args.output)
