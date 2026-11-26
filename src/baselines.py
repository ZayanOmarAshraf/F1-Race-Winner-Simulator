"""Simple pre-race baselines with per-race probability normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import PROCESSED_DIR

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _normalize_by_race(df: pd.DataFrame, score_col: str, out_col: str) -> pd.Series:
    clipped = df[score_col].astype(float).clip(lower=0)
    tmp = df[["raceId"]].copy()
    tmp["_score"] = clipped
    race_sum = tmp.groupby("raceId")["_score"].transform("sum")
    counts = tmp.groupby("raceId")["raceId"].transform("count").astype(float)
    probs = np.where(race_sum.to_numpy() > 0, clipped / race_sum, 1.0 / counts)
    return pd.Series(probs, index=df.index, name=out_col)


def baseline_uniform(df: pd.DataFrame) -> pd.Series:
    n = df.groupby("raceId")["raceId"].transform("count").astype(float)
    return (1.0 / n).rename("prob_uniform")


def baseline_softened_pole(df: pd.DataFrame, pole_win_rate: float) -> pd.Series:
    """
    Softened pole: pole (quali_position==1 or grid_pos==1) gets pole_win_rate,
    remaining mass shared uniformly among others.
    """
    is_pole = (df["quali_position"] == 1) | (
        (df["quali_position"].isna()) & (df["grid_pos"] == 1)
    )
    # Prefer quali pole; if multiple/none, use grid==1
    tmp = df[["raceId"]].copy()
    tmp["is_pole"] = is_pole.astype(int)
    # Ensure at most one pole flag per race: prefer quali, else lowest grid
    pole_idx = (
        df.assign(_pole=is_pole.astype(int), _grid=df["grid_pos"].fillna(999))
        .sort_values(["raceId", "_pole", "_grid"], ascending=[True, False, True])
        .groupby("raceId", as_index=False)
        .head(1)
        .index
    )
    flag = pd.Series(0, index=df.index)
    flag.loc[pole_idx] = 1

    p = pole_win_rate
    p = float(np.clip(p, 0.01, 0.99))
    n = df.groupby("raceId")["raceId"].transform("count").astype(float)
    remaining = (1.0 - p) / (n - 1).clip(lower=1)
    probs = np.where(flag.to_numpy() == 1, p, remaining)
    # Fix single-entrant races
    probs = np.where(n.to_numpy() <= 1, 1.0, probs)
    return pd.Series(probs, index=df.index, name="prob_soft_pole")


def baseline_champ_leader(df: pd.DataFrame) -> pd.Series:
    """Put most mass on current championship leader (prev race); rest uniform-ish."""
    is_leader = df["driver_champ_pos_prev"] == 1
    # One leader per race when available
    leader_idx = (
        df.assign(_lead=is_leader.astype(int))
        .sort_values(["raceId", "_lead"], ascending=[True, False])
        .groupby("raceId", as_index=False)
        .head(1)
        .index
    )
    flag = pd.Series(0, index=df.index)
    # Only mark leader if they truly have champ pos 1
    for idx in leader_idx:
        if df.loc[idx, "driver_champ_pos_prev"] == 1:
            flag.loc[idx] = 1

    n = df.groupby("raceId")["raceId"].transform("count").astype(float)
    has_leader = flag.groupby(df["raceId"]).transform("max")
    # Softened: 0.35 to leader, else uniform
    p_lead = 0.35
    remaining = (1.0 - p_lead) / (n - 1).clip(lower=1)
    probs = np.where(flag.to_numpy() == 1, p_lead, remaining)
    # Races without a known leader → uniform
    uniform = 1.0 / n
    probs = np.where(has_leader.to_numpy() == 1, probs, uniform)
    probs = np.where(n.to_numpy() <= 1, 1.0, probs)
    return pd.Series(probs, index=df.index, name="prob_champ_leader")


def baseline_grid_win_rate(df: pd.DataFrame, grid_rates: pd.Series) -> pd.Series:
    """Historical P(win | grid) estimated on train; normalize within race."""
    rates = df["grid_pos"].map(grid_rates).fillna(grid_rates.median() if len(grid_rates) else 0.05)
    tmp = df[["raceId"]].copy()
    tmp["score"] = rates.astype(float).clip(lower=1e-6)
    return _normalize_by_race(tmp, "score", "prob_grid_hist").rename("prob_grid_hist")


def fit_baseline_params(train: pd.DataFrame) -> dict:
    """Estimate baseline parameters from the training split only."""
    # Pole win rate: among races, how often quali/grid pole won
    pole_mask = train["quali_position"] == 1
    if pole_mask.sum() == 0:
        pole_mask = train["grid_pos"] == 1
    pole_rows = train.loc[pole_mask]
    pole_win_rate = float(pole_rows["won"].mean()) if len(pole_rows) else 0.4

    grid_rates = (
        train.groupby("grid_pos")["won"].mean().rename("grid_win_rate")
    )
    return {"pole_win_rate": pole_win_rate, "grid_rates": grid_rates}


def predict_baselines(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    out["prob_uniform"] = baseline_uniform(out)
    out["prob_soft_pole"] = baseline_softened_pole(out, params["pole_win_rate"])
    out["prob_champ_leader"] = baseline_champ_leader(out)
    out["prob_grid_hist"] = baseline_grid_win_rate(out, params["grid_rates"])
    return out


def run(processed_dir: Path | None = None) -> pd.DataFrame:
    proc = Path(processed_dir) if processed_dir else PROCESSED_DIR
    feat_path = proc / "features.parquet"
    if not feat_path.exists():
        from src.features import run as features_run

        features_run(processed_dir=proc)

    df = pd.read_parquet(feat_path)
    train = df[df["split"] == "train"]
    params = fit_baseline_params(train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Persist grid rates
    params["grid_rates"].to_csv(MODELS_DIR / "baseline_grid_rates.csv", header=True)
    with open(MODELS_DIR / "baseline_pole_win_rate.txt", "w", encoding="utf-8") as f:
        f.write(str(params["pole_win_rate"]))

    scored = predict_baselines(df, params)
    out_path = proc / "baseline_preds.parquet"
    scored.to_parquet(out_path, index=False)
    print(f"Baselines written → {out_path} (pole_win_rate={params['pole_win_rate']:.3f})")
    return scored


if __name__ == "__main__":
    run()
