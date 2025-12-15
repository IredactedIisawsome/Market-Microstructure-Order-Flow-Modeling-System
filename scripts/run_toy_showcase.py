#!/usr/bin/env python
"""
Toy, interview-safe showcase runner.

Generates synthetic order book + trade data, builds features/labels using the public
pipeline, fits a lightweight classifier, and prints interpretable metrics. All logic
is non-production and designed for demo/discussion only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.feature_engineering import add_basic_features, add_orderflow_features, make_feature_matrix
from src.labels import make_labels


def _generate_synthetic_orderflow(
    symbol: str = "FAKE-USD",
    rows: int = 1400,
    freq: str = "1s",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic Level 1 order book + trade stream for demo purposes.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=rows, freq=freq)

    # Simple microstructure-inspired random walk with modest drift and noise.
    drift = rng.normal(loc=0.0005, scale=0.0002, size=rows)
    shocks = rng.normal(scale=0.04, size=rows)
    mid = 100 + np.cumsum(drift + shocks)

    # Keep spreads positive and bounded.
    spread = np.clip(rng.normal(loc=0.01, scale=0.002, size=rows), 0.002, 0.04)
    bid = mid - spread / 2
    ask = mid + spread / 2

    # Depth and trade sizes are lognormal to mimic skewed liquidity.
    bid_depth = rng.lognormal(mean=1.5, sigma=0.35, size=rows)
    ask_depth = rng.lognormal(mean=1.4, sigma=0.35, size=rows)
    trade_size = rng.lognormal(mean=0.9, sigma=0.5, size=rows)

    df = pd.DataFrame(
        {
            "Time": times,
            "Symbol": symbol,
            "BidPrice1": bid,
            "AskPrice1": ask,
            "BidVolume1": bid_depth,
            "AskVolume1": ask_depth,
            "Volume": trade_size,
        }
    )
    return df.set_index(["Symbol", "Time"]).sort_index()


def _build_feature_label_matrix(df: pd.DataFrame):
    """
    Run feature pipeline and label creation on the synthetic data.
    """
    feat = add_basic_features(df)
    feat = add_orderflow_features(feat)

    # Directional labels a few steps ahead with a small neutral band to simulate slippage.
    y, y_idx = make_labels(feat, price_col="mid", horizon=5, neutral_threshold=0.0005)
    feat = feat.loc[y_idx]

    # Keep all rows; fill missing values since the model and scaler can handle dense numeric input.
    X, X_idx = make_feature_matrix(feat, drop_na=False)
    X = X.fillna(0)
    y = y.loc[X_idx]
    return X, y


def _train_and_report(X: pd.DataFrame, y: pd.Series):
    """
    Fit a lightweight classifier and print interpretable diagnostics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            # Use default multinomial handling; bump iterations to quiet convergence warnings.
            max_iter=2000,
            tol=1e-3,
            class_weight="balanced",
            n_jobs=1,
            solver="lbfgs",
        ),
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = model.classes_

    # Expected-direction proxy: P(up) - P(down)
    proba_df = pd.DataFrame(proba, columns=[f"proba_{c}" for c in classes], index=y_test.index)
    p_up = proba_df.get("proba_1.0", proba_df.get("proba_1", pd.Series(0, index=y_test.index)))
    p_down = proba_df.get("proba_-1.0", proba_df.get("proba_-1", pd.Series(0, index=y_test.index)))
    ev_dir = (p_up - p_down).rename("ev_dir")

    print("\n=== Toy showcase results (synthetic data) ===")
    print(f"Samples: train={len(X_train)}, test={len(X_test)}, feature_dim={X.shape[1]}")
    print("\nDirectional classification report (macro focus):")
    print(classification_report(y_test, preds, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, preds, labels=classes))
    print(
        "\nExpected-direction proxy stats (P(up)-P(down)) on test:\n"
        f"  mean={ev_dir.mean():.4f}, std={ev_dir.std():.4f}, min={ev_dir.min():.4f}, max={ev_dir.max():.4f}"
    )
    print("\nNote: This is a sandboxed illustration. Production feature sets, filters, regimes, and\n"
          "execution logic are redacted; real results are maintained privately.")


def main():
    df = _generate_synthetic_orderflow()
    X, y = _build_feature_label_matrix(df)
    if X.empty or y.empty:
        raise RuntimeError("Synthetic showcase produced no data; check pipeline assumptions.")
    _train_and_report(X, y)


if __name__ == "__main__":
    main()
