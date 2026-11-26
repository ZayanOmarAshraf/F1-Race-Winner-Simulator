# F1 Race Winner Predictor (MVP)

Predict the winner of a Formula 1 Grand Prix **before the lights go out**, using only information available pre-race.

This MVP turns Ergast-style historical tables into a leakage-safe driver–race dataset, trains simple baselines and three ML classifiers, evaluates ranking and probability quality, and serves per-race win probabilities in a Streamlit app.

## Problem

For each race from **2010 onward**, assign a win probability to every starter such that:

- probabilities are comparable across the field and **sum to ~1** within a race
- features never peek at that race’s finish, points, fastest lap, status, pit stops, or safety-car events
- models are judged on a **future** time split, not a random shuffle

## Data cleaning

Raw CSVs live in `data/raw/` (Ergast-compatible dumps).

- Restrict to seasons **2010–present**; drop races with **no results**
- Build **one row per driver–race** from `results` joined to `races`, `qualifying`, `drivers`, `constructors`, and `circuits`
- Target: `won = 1` iff `positionOrder == 1`
- Replace Ergast `\N` sentinels with missing values
- Treat **grid = 0** as a **pit-lane start** (`pit_lane_start` flag); keep `grid_pos` as recorded
- Parse qualifying session times to seconds and take the best available session as `quali_best_sec`

## Leakage rules

Anything that is only known **after** the race starts is forbidden as a feature:

- finishing position / points / fastest lap / status
- pit-stop counts or timings
- safety car / VSC / red-flag incident fields for the current race

Championship standings are post-race in the source tables, so we only use standings from the **previous** race in the same season (season openers have missing champ features).

All rolling / expanding history uses **`shift(1)`** so the current race’s outcome is excluded.

## Features (pre-race)

| Feature | Description |
|--------|-------------|
| `grid_pos` | Starting grid (0 = pit lane) |
| `pit_lane_start` | 1 if grid was 0 |
| `quali_position` | Official qualifying position |
| `gap_to_pole_sec` | Best quali time − pole time |
| `driver_avg_finish_5` | Driver mean finish over last 5 races (shift → rolling) |
| `driver_win_rate_10` | Driver win rate over last 10 races (shift → rolling) |
| `constructor_form_5` | Mean finish aggregated to constructor–race, then shift → rolling 5 |
| `circuit_win_rate_expanding` | Expanding circuit win rate with shift |
| `driver_champ_pos_prev` | Driver WDC position after the previous race |
| `constructor_champ_pos_prev` | Constructor WCC position after the previous race |

## Time split

| Split | Years |
|-------|-------|
| Train | 2010–2019 |
| Validation | 2020–2021 |
| Test | 2022–present |

Baselines and models are fit on **train only**. Metrics in `outputs/metrics.json` are reported for all three splits; the calibration plot uses **test**.

## Models

**Baselines**

- **Uniform** — equal probability across the field
- **Softened pole** — train-set pole win rate on the pole sitter; remainder shared
- **Championship leader** — soft mass on the pre-race WDC leader
- **Historical grid win rate** — train `P(win | grid)`, then normalize within race

**Classifiers**

- **Logistic regression** — median impute + standard scale + balanced LR
- **Random forest** — median impute + depth-limited RF
- **XGBoost** — median impute + gradient-boosted trees with `scale_pos_weight`

Raw classifier scores are **renormalized per race** so win probabilities sum to 1.

## Metrics

Reported per model / split in `outputs/metrics.json`:

- **Top-1 winner** — fraction of races where the highest-prob driver actually won
- **Top-3 winner** — true winner inside the top-3 predicted
- **Log loss** and **Brier score** on the binary win label
- **Calibration plot** — `outputs/calibration_test.png`

Sanity checks enforced during evaluation: exactly **one winner per race**, and mean per-race probability sum ≈ **1**.

## Project layout

```
data/raw/           # input CSVs
data/processed/     # parquet intermediates
src/
  preprocess.py
  features.py
  baselines.py
  train.py
  predict.py
  evaluate.py
models/             # joblib models + baseline params
outputs/            # metrics.json, calibration_test.png
app.py              # Streamlit UI
requirements.txt
```

## Run commands

```bash
# Optional venv (if python3-venv is available):
#   python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline
python -m src.preprocess
python -m src.features
python -m src.baselines
python -m src.train
python -m src.evaluate

# Ranked probs for one race (example: latest raceId printed by predict)
python -m src.predict

# Interactive app
streamlit run app.py
```

Place Ergast-style CSVs under `data/raw/` before running preprocess (filenames such as `races.csv`, `results.csv`, `qualifying.csv`, …).
