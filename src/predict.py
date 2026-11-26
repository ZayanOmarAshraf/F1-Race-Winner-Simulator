"""Load saved models/baselines and produce per-race ranked win probabilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.baselines import fit_baseline_params, predict_baselines
from src.features import FEATURE_COLS
from src.preprocess import PROCESSED_DIR
from src.train import normalize_probs_by_race

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

MODEL_NAMES = ("logreg", "rf", "xgb", "uniform", "soft_pole", "champ_leader", "grid_hist")


def load_features(processed_dir: Path | None = None) -> pd.DataFrame:
    proc = Path(processed_dir) if processed_dir else PROCESSED_DIR
    path = proc / "features.parquet"
    if not path.exists():
        from src.features import run as features_run

        features_run(processed_dir=proc)
    return pd.read_parquet(path)


def _load_baseline_params() -> dict:
    pole_path = MODELS_DIR / "baseline_pole_win_rate.txt"
    grid_path = MODELS_DIR / "baseline_grid_rates.csv"
    if pole_path.exists() and grid_path.exists():
        pole = float(pole_path.read_text().strip())
        grid = pd.read_csv(grid_path, index_col=0).squeeze("columns")
        grid.index = pd.to_numeric(grid.index, errors="coerce")
        return {"pole_win_rate": pole, "grid_rates": grid}
    df = load_features()
    return fit_baseline_params(df[df["split"] == "train"])


def predict_race(
    race_id: int,
    model_name: str = "xgb",
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Return ranked win probabilities for one race and one model/baseline."""
    df = load_features(processed_dir)
    race = df[df["raceId"] == race_id].copy()
    if race.empty:
        raise ValueError(f"Unknown raceId={race_id}")

    name = model_name.lower().strip()
    if name in {"logreg", "rf", "xgb"}:
        model_path = MODELS_DIR / f"{name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model {model_path}. Run: python -m src.train"
            )
        model = joblib.load(model_path)
        raw = model.predict_proba(race[FEATURE_COLS])[:, 1]
        probs = normalize_probs_by_race(race["raceId"].to_numpy(), raw)
        race["win_prob"] = probs
    else:
        params = _load_baseline_params()
        scored = predict_baselines(race, params)
        col_map = {
            "uniform": "prob_uniform",
            "soft_pole": "prob_soft_pole",
            "softened_pole": "prob_soft_pole",
            "champ_leader": "prob_champ_leader",
            "grid_hist": "prob_grid_hist",
            "grid": "prob_grid_hist",
        }
        if name not in col_map:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from: {MODEL_NAMES}"
            )
        race["win_prob"] = scored[col_map[name]].to_numpy()

    race = race.sort_values("win_prob", ascending=False).reset_index(drop=True)
    race["rank"] = np.arange(1, len(race) + 1)
    cols = [
        c
        for c in (
            "rank",
            "driver_name",
            "constructor_name",
            "grid_pos",
            "quali_position",
            "win_prob",
            "won",
            "race_name",
            "year",
            "round",
            "raceId",
            "driverId",
        )
        if c in race.columns
    ]
    return race[cols]


def list_races(processed_dir: Path | None = None) -> pd.DataFrame:
    df = load_features(processed_dir)
    races = (
        df.groupby(["raceId", "year", "round", "race_name", "date", "split"], as_index=False)
        .size()
        .sort_values(["year", "round"], ascending=[False, False])
    )
    return races


if __name__ == "__main__":
    races = list_races()
    sample_id = int(races.iloc[0]["raceId"])
    print(predict_race(sample_id, "xgb").head(10).to_string(index=False))
