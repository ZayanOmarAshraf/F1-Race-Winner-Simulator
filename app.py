"""Streamlit app: pick a race + model, see ranked win probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import MODEL_NAMES, list_races, predict_race

st.set_page_config(page_title="F1 Winner Predictor", layout="wide")
st.title("F1 Race Winner Predictor")
st.caption("Pre-race win probabilities — leakage-safe features only.")


@st.cache_data(show_spinner=False)
def _races() -> pd.DataFrame:
    return list_races()


def main() -> None:
    races = _races()
    if races.empty:
        st.error("No features found. Run the pipeline first (see README).")
        return

    races = races.copy()
    races["label"] = (
        races["year"].astype(str)
        + " R"
        + races["round"].astype(str)
        + " — "
        + races["race_name"].astype(str)
        + " ["
        + races["split"].astype(str)
        + "]"
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        choice = st.selectbox("Race", races["label"].tolist(), index=0)
    with c2:
        model = st.selectbox("Model", list(MODEL_NAMES), index=list(MODEL_NAMES).index("xgb") if "xgb" in MODEL_NAMES else 0)

    race_id = int(races.loc[races["label"] == choice, "raceId"].iloc[0])
    try:
        ranked = predict_race(race_id, model_name=model)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return

    # Sanity
    prob_sum = float(ranked["win_prob"].sum())
    st.write(
        f"**{ranked.iloc[0]['race_name']}** ({int(ranked.iloc[0]['year'])}) · "
        f"Σ probabilities = `{prob_sum:.4f}`"
    )

    show = ranked.copy()
    show["win_prob"] = show["win_prob"].map(lambda x: f"{100 * x:.2f}%")
    if "won" in show.columns:
        show["actual_winner"] = show["won"].map(lambda x: "✓" if int(x) == 1 else "")
    st.dataframe(show, use_container_width=True, hide_index=True)

    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        with st.expander("Test-set metrics"):
            st.json(json.loads(metrics_path.read_text()))

    cal = Path("outputs/calibration_test.png")
    if cal.exists():
        st.image(str(cal), caption="Calibration on test split (2022–present)")


if __name__ == "__main__":
    main()
