"""Evaluation metrics and calibration plot for winner probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from src.preprocess import PROCESSED_DIR

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _top_k_winner_hit(df: pd.DataFrame, prob_col: str, k: int) -> float:
    """Fraction of races where the true winner is among the top-k predicted."""
    hits = []
    for _, g in df.groupby("raceId"):
        top = g.nlargest(k, prob_col)
        hits.append(int(top["won"].max() == 1))
    return float(np.mean(hits)) if hits else float("nan")


def evaluate_probs(df: pd.DataFrame, prob_col: str) -> dict:
    y = df["won"].astype(int).to_numpy()
    p = df[prob_col].astype(float).clip(1e-15, 1 - 1e-15).to_numpy()
    metrics = {
        "n_rows": int(len(df)),
        "n_races": int(df["raceId"].nunique()),
        "top1_winner": _top_k_winner_hit(df, prob_col, 1),
        "top3_winner": _top_k_winner_hit(df, prob_col, 3),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "prob_sum_mean": float(
            df.groupby("raceId")[prob_col].sum().mean()
        ),
        "prob_sum_max_abs_err": float(
            (df.groupby("raceId")[prob_col].sum() - 1.0).abs().max()
        ),
        "winners_per_race_max": int(df.groupby("raceId")["won"].sum().max()),
        "winners_per_race_min": int(df.groupby("raceId")["won"].sum().min()),
    }
    return metrics


def plot_calibration(
    df: pd.DataFrame,
    prob_cols: dict[str, str],
    out_path: Path,
    n_bins: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    for label, col in prob_cols.items():
        y = df["won"].astype(int).to_numpy()
        p = df[col].astype(float).to_numpy()
        if y.sum() == 0 or len(np.unique(p)) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(
            y, p, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("Mean predicted win probability")
    ax.set_ylabel("Fraction of actual winners")
    ax.set_title("Calibration (test split)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run(processed_dir: Path | None = None) -> dict:
    proc = Path(processed_dir) if processed_dir else PROCESSED_DIR
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    feat = pd.read_parquet(proc / "features.parquet")
    # Ensure baselines + models exist
    base_path = proc / "baseline_preds.parquet"
    model_path = proc / "model_preds.parquet"
    if not base_path.exists():
        from src.baselines import run as baselines_run

        baselines_run(processed_dir=proc)
        base_path = proc / "baseline_preds.parquet"
    if not model_path.exists():
        from src.train import run as train_run

        train_run(processed_dir=proc)
        model_path = proc / "model_preds.parquet"

    baselines = pd.read_parquet(base_path)
    models = pd.read_parquet(model_path)

    merged = feat[
        ["raceId", "driverId", "won", "split", "year"]
    ].merge(
        baselines[
            [
                "raceId",
                "driverId",
                "prob_uniform",
                "prob_soft_pole",
                "prob_champ_leader",
                "prob_grid_hist",
            ]
        ],
        on=["raceId", "driverId"],
        how="left",
    )
    model_prob_cols = [c for c in models.columns if c.startswith("prob_") and not c.startswith("prob_raw_")]
    merged = merged.merge(
        models[["raceId", "driverId"] + model_prob_cols],
        on=["raceId", "driverId"],
        how="left",
    )

    # Verification checks across full data
    winners = merged.groupby("raceId")["won"].sum()
    assert winners.min() == 1 and winners.max() == 1, (
        f"Expected exactly one winner per race; got min={winners.min()} max={winners.max()}"
    )

    all_prob_cols = {
        "uniform": "prob_uniform",
        "soft_pole": "prob_soft_pole",
        "champ_leader": "prob_champ_leader",
        "grid_hist": "prob_grid_hist",
    }
    for c in model_prob_cols:
        all_prob_cols[c.replace("prob_", "")] = c

    results: dict = {"splits": {}, "checks": {}}
    for split in ("train", "val", "test"):
        part = merged[merged["split"] == split]
        results["splits"][split] = {}
        for label, col in all_prob_cols.items():
            if col not in part.columns:
                continue
            results["splits"][split][label] = evaluate_probs(part, col)

    # Prob sum check on test for each model
    test = merged[merged["split"] == "test"]
    results["checks"]["one_winner_per_race"] = True
    results["checks"]["prob_sums"] = {}
    for label, col in all_prob_cols.items():
        if col not in test.columns:
            continue
        sums = test.groupby("raceId")[col].sum()
        results["checks"]["prob_sums"][label] = {
            "mean": float(sums.mean()),
            "max_abs_err": float((sums - 1.0).abs().max()),
        }

    # Calibration plot on test for ML models + soft pole
    plot_cols = {
        k: v
        for k, v in all_prob_cols.items()
        if k in {"soft_pole", "logreg", "rf", "xgb", "grid_hist"} and v in test.columns
    }
    plot_calibration(test, plot_cols, OUTPUTS_DIR / "calibration_test.png")

    out_json = OUTPUTS_DIR / "metrics.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["splits"].get("test", {}), indent=2))
    print(f"Wrote {out_json} and calibration_test.png")
    return results


if __name__ == "__main__":
    run()
