"""Train LogisticRegression, RandomForest, and XGBoost winner models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS
from src.preprocess import PROCESSED_DIR

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


def normalize_probs_by_race(
    race_ids: np.ndarray | pd.Series, probs: np.ndarray
) -> np.ndarray:
    """Force predicted win probabilities to sum to 1 within each race."""
    out = np.asarray(probs, dtype=float).copy()
    s = pd.Series(out)
    race = pd.Series(np.asarray(race_ids))
    totals = s.groupby(race).transform("sum").to_numpy()
    totals = np.where(totals <= 0, 1.0, totals)
    out = out / totals
    return out


def _build_lr() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _build_rf() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _build_xgb(scale_pos_weight: float) -> Pipeline:
    if not HAS_XGB:
        raise ImportError("xgboost is required")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_models(df: pd.DataFrame) -> dict:
    train = df[df["split"] == "train"].copy()
    X = train[FEATURE_COLS]
    y = train["won"].astype(int)

    pos = max(int(y.sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    spw = neg / pos

    models = {
        "logreg": _build_lr(),
        "rf": _build_rf(),
    }
    if HAS_XGB:
        models["xgb"] = _build_xgb(spw)

    fitted = {}
    for name, model in models.items():
        model.fit(X, y)
        fitted[name] = model
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"Trained {name} on {len(train)} rows ({pos} wins).")

    meta = {
        "feature_cols": FEATURE_COLS,
        "train_years": [2010, 2019],
        "n_train": int(len(train)),
        "n_wins_train": int(pos),
        "models": list(fitted.keys()),
    }
    (MODELS_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return fitted


def predict_model(
    model, df: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    X = df[FEATURE_COLS]
    raw = model.predict_proba(X)[:, 1]
    normed = normalize_probs_by_race(df["raceId"].to_numpy(), raw)
    out = df[
        [
            c
            for c in (
                "raceId",
                "year",
                "round",
                "date",
                "driverId",
                "driver_name",
                "constructor_name",
                "race_name",
                "grid_pos",
                "won",
                "split",
            )
            if c in df.columns
        ]
    ].copy()
    out[f"prob_{model_name}"] = normed
    out[f"prob_raw_{model_name}"] = raw
    return out


def run(processed_dir: Path | None = None) -> dict:
    proc = Path(processed_dir) if processed_dir else PROCESSED_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feat_path = proc / "features.parquet"
    if not feat_path.exists():
        from src.features import run as features_run

        features_run(processed_dir=proc)

    df = pd.read_parquet(feat_path)
    models = train_models(df)

    # Score all splits for later evaluation / app
    frames = []
    for name, model in models.items():
        scored = predict_model(model, df, name)
        frames.append(scored)

    merged = frames[0]
    for extra in frames[1:]:
        cols = [c for c in extra.columns if c.startswith("prob_")]
        merged = merged.merge(
            extra[["raceId", "driverId"] + cols],
            on=["raceId", "driverId"],
            how="left",
        )

    out_path = proc / "model_preds.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Model predictions → {out_path}")
    return models


if __name__ == "__main__":
    run()
