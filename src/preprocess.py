"""Load and clean Ergast-style F1 tables for the winner-prediction MVP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

CORE_TABLES = (
    "races",
    "results",
    "qualifying",
    "drivers",
    "constructors",
    "driver_standings",
    "constructor_standings",
    "circuits",
)


def _replace_null_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Ergast \\N sentinels with NaN."""
    out = df.copy()
    out = out.replace({"\\N": np.nan, r"\N": np.nan})
    # Also catch string literals that survived CSV parsing
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].replace("\\N", np.nan)
    return out


def load_raw(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load core CSV tables from data/raw."""
    root = Path(raw_dir) if raw_dir else RAW_DIR
    tables: dict[str, pd.DataFrame] = {}
    for name in CORE_TABLES:
        path = root / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw table: {path}")
        tables[name] = _replace_null_tokens(pd.read_csv(path))
    return tables


def _parse_lap_time_to_seconds(series: pd.Series) -> pd.Series:
    """Convert qualifying time strings (M:SS.mmm or SS.mmm) to seconds."""

    def one(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if not s or s == "\\N":
            return np.nan
        try:
            if ":" in s:
                mins, rest = s.split(":", 1)
                return float(mins) * 60.0 + float(rest)
            return float(s)
        except ValueError:
            return np.nan

    return series.map(one)


def build_race_driver_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    One row per driver-race for seasons 2010–present.

    - Drop races with no results
    - won = (positionOrder == 1)
    - grid == 0 means pit-lane start
    """
    races = tables["races"].copy()
    results = tables["results"].copy()
    quali = tables["qualifying"].copy()
    drivers = tables["drivers"].copy()
    constructors = tables["constructors"].copy()
    circuits = tables["circuits"].copy()

    races = races[races["year"] >= 2010].copy()
    races_with_results = set(results["raceId"].unique())
    races = races[races["raceId"].isin(races_with_results)].copy()

    # Numeric result fields
    for col in ("grid", "positionOrder", "points", "laps", "rank", "statusId"):
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    df = results.merge(
        races[
            [
                "raceId",
                "year",
                "round",
                "circuitId",
                "name",
                "date",
            ]
        ],
        on="raceId",
        how="inner",
    )
    df = df.rename(columns={"name": "race_name"})

    # Qualifying: position + best available session time
    quali = quali.copy()
    for col in ("q1", "q2", "q3"):
        if col in quali.columns:
            quali[f"{col}_sec"] = _parse_lap_time_to_seconds(quali[col])
    time_cols = [c for c in ("q3_sec", "q2_sec", "q1_sec") if c in quali.columns]
    if time_cols:
        quali["quali_best_sec"] = quali[time_cols].min(axis=1, skipna=True)
    else:
        quali["quali_best_sec"] = np.nan
    quali["quali_position"] = pd.to_numeric(quali["position"], errors="coerce")
    quali_keep = quali[
        ["raceId", "driverId", "quali_position", "quali_best_sec"]
    ].drop_duplicates(["raceId", "driverId"])
    df = df.merge(quali_keep, on=["raceId", "driverId"], how="left")

    # Driver / constructor / circuit labels
    drivers["driver_name"] = (
        drivers["forename"].astype(str) + " " + drivers["surname"].astype(str)
    )
    df = df.merge(
        drivers[["driverId", "driverRef", "driver_name", "code"]],
        on="driverId",
        how="left",
    )
    df = df.merge(
        constructors[["constructorId", "constructorRef", "name"]].rename(
            columns={"name": "constructor_name"}
        ),
        on="constructorId",
        how="left",
    )
    df = df.merge(
        circuits[["circuitId", "circuitRef", "name"]].rename(
            columns={"name": "circuit_name"}
        ),
        on="circuitId",
        how="left",
    )

    # Target + grid / pit-lane
    df["won"] = (df["positionOrder"] == 1).astype(int)
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["pit_lane_start"] = (df["grid"] == 0).astype(int)
    # Keep grid as starting position; 0 stays 0 for pit-lane starters
    df["grid_pos"] = df["grid"]

    # Chronological order within / across seasons
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "raceId", "grid_pos", "driverId"]).reset_index(
        drop=True
    )

    # Sanity: exactly one winner per race (positionOrder==1); keep all rows
    winners = df.groupby("raceId")["won"].sum()
    bad = winners[winners != 1]
    if len(bad):
        # Prefer races that have a clear P1; drop races with zero winners
        zero = bad[bad == 0].index
        df = df[~df["raceId"].isin(zero)].copy()

    return df


def save_processed(df: pd.DataFrame, processed_dir: Path | None = None) -> Path:
    out_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "race_driver_base.parquet"
    df.to_parquet(path, index=False)
    return path


def run(raw_dir: Path | None = None, processed_dir: Path | None = None) -> pd.DataFrame:
    tables = load_raw(raw_dir)
    df = build_race_driver_table(tables)
    save_processed(df, processed_dir)
    n_races = df["raceId"].nunique()
    n_winners = int(df["won"].sum())
    print(f"Processed {len(df)} driver-race rows across {n_races} races ({n_winners} winners).")
    return df


if __name__ == "__main__":
    run()
