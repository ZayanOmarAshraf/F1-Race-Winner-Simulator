"""Leakage-safe pre-race feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import PROCESSED_DIR, RAW_DIR, load_raw

FEATURE_COLS = [
    "grid_pos",
    "pit_lane_start",
    "quali_position",
    "gap_to_pole_sec",
    "driver_avg_finish_5",
    "driver_win_rate_10",
    "constructor_form_5",
    "circuit_win_rate_expanding",
    "driver_champ_pos_prev",
    "constructor_champ_pos_prev",
]

META_COLS = [
    "raceId",
    "year",
    "round",
    "date",
    "circuitId",
    "driverId",
    "constructorId",
    "driver_name",
    "driverRef",
    "constructor_name",
    "race_name",
    "circuit_name",
    "won",
    "split",
]


def _assign_split(year: pd.Series) -> pd.Series:
    split = pd.Series(index=year.index, dtype=object)
    split[year.between(2010, 2019)] = "train"
    split[year.between(2020, 2021)] = "val"
    split[year >= 2022] = "test"
    return split


def _previous_standings(
    race_order: pd.DataFrame,
    standings: pd.DataFrame,
    entity_col: str,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """
    Map championship position from the *previous* race in season order.

    Standings rows are post-race; shifting by one race within each season
    yields pre-race championship position. Season openers get NaN.
    """
    st = standings.copy()
    st["position"] = pd.to_numeric(st["position"], errors="coerce")
    st = st.rename(columns={"position": value_col})

    # Attach season metadata to standings via race_order
    race_meta = race_order[["raceId", "year", "round"]].drop_duplicates()
    st = st.merge(race_meta, on="raceId", how="inner")
    st = st.sort_values(["year", "round", entity_col])

    # Previous raceId within the same season for each entity
    st["prev_raceId"] = st.groupby([entity_col, "year"])["raceId"].shift(1)
    prev = st.dropna(subset=["prev_raceId"]).copy()
    prev["prev_raceId"] = prev["prev_raceId"].astype(int)
    # The standings after prev_race apply going into the *next* race.
    # Build lookup: (next_raceId, entity) -> standing after previous race.
    # After standing at race R, the next race is the following round.
    # Easier approach: for each race R, look up standings at previous raceId.
    # So join keys are (current raceId, entity) <- standings at prev race of current.
    race_prev = race_meta.sort_values(["year", "round"]).copy()
    race_prev["prev_raceId"] = race_prev.groupby("year")["raceId"].shift(1)
    # For each (current_race, entity), take standings from prev_raceId
    keys = race_prev.dropna(subset=["prev_raceId"])[
        ["raceId", "prev_raceId", "year"]
    ].copy()
    keys["prev_raceId"] = keys["prev_raceId"].astype(int)

    st_vals = st[["raceId", entity_col, value_col]].rename(
        columns={"raceId": "prev_raceId"}
    )
    mapped = keys.merge(st_vals, on="prev_raceId", how="left")
    mapped = mapped.rename(columns={value_col: out_col})
    return mapped[["raceId", entity_col, out_col]]


def add_features(
    base: pd.DataFrame,
    tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Add only pre-race features; never current finish/points/FL/status/pits/SC."""
    df = base.copy()
    df = df.sort_values(["date", "raceId", "driverId"]).reset_index(drop=True)

    # --- Gap to pole from qualifying best times (pre-race) ---
    pole = (
        df.groupby("raceId", as_index=False)["quali_best_sec"]
        .min()
        .rename(columns={"quali_best_sec": "pole_sec"})
    )
    df = df.merge(pole, on="raceId", how="left")
    df["gap_to_pole_sec"] = df["quali_best_sec"] - df["pole_sec"]

    # --- Driver form: shift then rolling (exclude current race) ---
    df["driver_avg_finish_5"] = (
        df.groupby("driverId")["positionOrder"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    df["driver_win_rate_10"] = (
        df.groupby("driverId")["won"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    )

    # --- Constructor form: aggregate by constructor-race, THEN shift+rolling ---
    ctor_race = (
        df.groupby(["constructorId", "raceId", "date"], as_index=False)
        .agg(ctor_mean_finish=("positionOrder", "mean"), ctor_win=("won", "max"))
        .sort_values(["constructorId", "date", "raceId"])
    )
    ctor_race["constructor_form_5"] = (
        ctor_race.groupby("constructorId")["ctor_mean_finish"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    df = df.merge(
        ctor_race[["constructorId", "raceId", "constructor_form_5"]],
        on=["constructorId", "raceId"],
        how="left",
    )

    # --- Circuit expanding win rate with shift (pre-race only) ---
    df = df.sort_values(["circuitId", "date", "raceId", "driverId"])
    df["circuit_win_rate_expanding"] = (
        df.groupby("circuitId")["won"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df = df.sort_values(["date", "raceId", "driverId"]).reset_index(drop=True)

    # --- Championship positions from previous race ---
    if tables is None:
        tables = load_raw(RAW_DIR)
    race_order = df[["raceId", "year", "round"]].drop_duplicates()
    dstand = _previous_standings(
        race_order,
        tables["driver_standings"],
        entity_col="driverId",
        value_col="d_pos",
        out_col="driver_champ_pos_prev",
    )
    cstand = _previous_standings(
        race_order,
        tables["constructor_standings"],
        entity_col="constructorId",
        value_col="c_pos",
        out_col="constructor_champ_pos_prev",
    )
    df = df.merge(dstand, on=["raceId", "driverId"], how="left")
    df = df.merge(cstand, on=["raceId", "constructorId"], how="left")

    df["split"] = _assign_split(df["year"])

    keep = [c for c in META_COLS + FEATURE_COLS if c in df.columns]
    # Ensure unique columns
    out = df[keep].copy()
    return out


def run(
    processed_dir: Path | None = None,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    proc = Path(processed_dir) if processed_dir else PROCESSED_DIR
    base_path = proc / "race_driver_base.parquet"
    if not base_path.exists():
        from src.preprocess import run as preprocess_run

        preprocess_run(raw_dir=raw_dir, processed_dir=proc)

    base = pd.read_parquet(base_path)
    tables = load_raw(raw_dir)
    featured = add_features(base, tables=tables)
    out_path = proc / "features.parquet"
    featured.to_parquet(out_path, index=False)
    print(
        f"Wrote features: {len(featured)} rows, "
        f"{featured['raceId'].nunique()} races → {out_path}"
    )
    return featured


if __name__ == "__main__":
    run()
