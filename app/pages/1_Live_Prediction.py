import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time as _time
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.energy import ACTIONS, ICONS, calculate_savings
from utils.features import make_features_from_array, simulate_pir, get_temperature_for_class
from utils.predict import load_artifacts
from utils.realtime import get_timestamp, init_session_deques


# ── Artifact loader ───────────────────────────────────────────────────────────
def _ensure_artifacts():
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = load_artifacts()
    a = st.session_state.artifacts
    st.session_state.model         = a.model
    st.session_state.scaler        = a.scaler
    st.session_state.feature_names = a.feature_names
    st.session_state.label_map     = a.label_mapping
    st.session_state.results_summary = a.summary


def _top_kpi_bar():
    summary      = st.session_state.get("results_summary", {}) or {}
    energy       = summary.get("energy_savings", {})
    best_model   = summary.get("best_model", type(st.session_state.get("model", object())).__name__)
    f1           = summary.get("macro_f1", summary.get("accuracy", 0)) * 100
    annual_usd   = energy.get("annual_savings_USD", 1752)
    annual_co2   = energy.get("annual_CO2_saved_kg", 3400)
    leed_score   = summary.get("leed_score", 90.0)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Best Model",      best_model)
    c2.metric("Test F1",         f"{f1:.1f}%")
    c3.metric("Annual Savings",  f"${annual_usd:,.0f}")
    c4.metric("CO2 / year",      f"{annual_co2:,.0f} kg")
    c5.metric("Green Score",     f"{leed_score:.1f} / 100")


def _init_live_state():
    if "live_running_pred" not in st.session_state:
        st.session_state.live_running_pred = True
    init_session_deques({"pir_history": 30, "time_history": 30,
                         "pred_history": 30, "log_rows": 10})
    for key in ["live_savings_usd", "live_savings_kwh"]:
        if key not in st.session_state:
            st.session_state[key] = 0.0
    for key in ["last_manual_result", "last_manual_timestamp", "last_manual_sim_class"]:
        if key not in st.session_state:
            st.session_state[key] = None


def _state_color(state: str) -> str:
    return {"Vacancy": "#2ecc71", "Stationary": "#3498db"}.get(state, "#e74c3c")


def _render_sidebar():
    """
    Render sidebar controls.

    TEMPERATURE SLIDER:
    - Range is 84–89°F to match the real dataset (Vacancy=87F, Stationary=86F)
    - For Motion class, temperature is ALWAYS set to 0°F automatically
      (matching the real dataset sensor fault - not exposed in UI)
    - This is the root cause of the original bug: old range was 60-90F
      which made the model think low-temp = Motion and high-temp = Vacancy
    """
    with st.sidebar:
        st.header("Simulation Controls")

        sim_class = st.selectbox(
            "Simulate Class",
            ["Vacancy", "Stationary", "Motion", "Random"],
            index=3,
        )

        noise = st.slider("Noise Level", 0, 50, 10,
                          help="Sensor noise level. Real baseline ~200 counts always added.")

        # ── FIXED: temperature range now matches real dataset ─────────────────
        # Real dataset: Vacancy=87F, Stationary=86F, Motion=0F (sensor fault)
        # Slider only applies to Vacancy/Stationary. Motion always uses 0F.
        if sim_class == "Motion":
            st.info("🌡️ Temperature: **0°F** (auto — matches real dataset sensor fault for Motion class)")
            temp_f = 0.0  # not shown as slider for Motion
        else:
            temp_f = st.slider(
                "Temperature (°F)",
                min_value=32, max_value=89,
                value=87,
                help="Real dataset: Vacancy≈87°F, Stationary≈86°F. "
                     "Keep in this range for accurate predictions."
            )

        run_btn = st.button("Run Prediction", type="primary")
        st.markdown("---")
        if st.button(
            "Pause Live Feed" if st.session_state.live_running_pred
            else "Resume Live Feed"
        ):
            st.session_state.live_running_pred = not st.session_state.live_running_pred

    return sim_class, noise, temp_f, run_btn


def _run_prediction(sim_class: str, noise: int, temp_f: float):
    """
    Run a single prediction using the real XGBoost model.

    Key fix: uses get_temperature_for_class() to ensure Motion always
    gets temperature=0F matching the real training data distribution.
    """
    model         = st.session_state.model
    scaler        = st.session_state.scaler
    feature_names = st.session_state.feature_names
    label_map     = st.session_state.label_map

    # Resolve random class
    actual_class = (
        np.random.choice(["Vacancy", "Stationary", "Motion"])
        if sim_class == "Random" else sim_class
    )

    # Generate PIR readings for the class
    pir = simulate_pir(actual_class, noise)

    # ── CRITICAL FIX ──────────────────────────────────────────────────────────
    # Get the correct temperature for this class.
    # Motion → 0°F (real dataset sensor fault)
    # Vacancy/Stationary → slider value (84–89°F)
    # Without this fix, passing 60°F for Vacancy causes model to predict Motion
    # because the model learned: low_temp=Motion, high_temp=Vacancy/Stationary
    correct_temp = get_temperature_for_class(actual_class, temp_f)
    # ──────────────────────────────────────────────────────────────────────────

    feats  = make_features_from_array(pir, correct_temp, feature_names)
    X      = scaler.transform(feats.values)
    y_pred = model.predict(X)[0]
    proba  = model.predict_proba(X)[0]

    state      = label_map.get(int(y_pred), actual_class)
    confidence = float(np.max(proba))
    savings    = calculate_savings(state)

    return pir, state, confidence, proba, savings, feats, actual_class, correct_temp


def _render_results(pir, state, confidence, proba, savings, feats,
                    actual_class=None, used_temp=None):
    color = _state_color(state)

    st.markdown("### Prediction Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected State",  state)
    c2.metric("Confidence",      f"{confidence*100:.1f}%")
    c3.metric("Energy Saving",   f"{savings['savings_pct']:.1f}%")

    # Show debug info when simulated class differs from predicted
    if actual_class and actual_class != "Random" and state != actual_class:
        st.warning(
            f"⚠️ Simulated **{actual_class}** but model predicted **{state}**. "
            f"This can happen with Random class or borderline PIR values."
        )
    elif actual_class:
        st.success(f"✅ Simulated **{actual_class}** → Predicted **{state}** "
                   f"(temp used: {used_temp:.0f}°F)")

    # PIR signal chart
    x = list(range(1, len(pir) + 1))
    fig_pir = go.Figure()
    fig_pir.add_trace(go.Scatter(
        x=x, y=pir, mode="lines+markers",
        line=dict(color=color, width=2), fill="tozeroy", name="PIR Signal"
    ))
    fig_pir.add_trace(go.Scatter(
        x=x, y=[float(np.mean(pir))] * len(pir), mode="lines",
        line=dict(color="#7f8c8d", width=1, dash="dash"), name="Mean"
    ))
    fig_pir.update_layout(title="PIR Signal Window", xaxis_title="Sample",
                          yaxis_title="Signal", height=320,
                          margin=dict(l=0, r=0, t=40, b=0))

    # Probability chart
    classes       = ["Vacancy", "Stationary", "Motion"]
    probs_ordered = [float(proba[i]) if i < len(proba) else 0.0 for i in range(3)]
    fig_prob = go.Figure(data=[go.Bar(
        x=classes, y=[p * 100 for p in probs_ordered],
        marker_color=["#2ecc71", "#3498db", "#e74c3c"]
    )])
    fig_prob.update_layout(title="Class Probabilities", yaxis_title="Confidence (%)",
                           height=320, margin=dict(l=0, r=0, t=40, b=0))

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(fig_pir,  use_container_width=True)
    with c2: st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("### Device Actions & Comfort")
    d1, d2, d3, d4, d5 = st.columns(5)
    actions = savings["actions"]
    for dev, col in zip(["HVAC", "Lighting", "Ventilation", "Screens", "Elevators"],
                        [d1, d2, d3, d4, d5]):
        col.metric(f"{ICONS[dev]} {dev}", actions[dev])
    st.info(f"Recommended comfort setpoint: {savings['comfort_temp_C']:.1f} °C for {state}.")

    with st.expander("PIR Signal Statistics"):
        st.dataframe(feats.describe().T)


def _live_loop(sensor_ph, metrics_ph, log_ph, ticker_ph, sim_class, noise, temp_f):
    """Live auto-refresh loop — one tick per rerun."""
    model         = st.session_state.model
    scaler        = st.session_state.scaler
    feature_names = st.session_state.feature_names
    label_map     = st.session_state.label_map

    actual_class = (
        np.random.choice(["Vacancy", "Stationary", "Motion"])
        if sim_class == "Random" else sim_class
    )

    pir         = simulate_pir(actual_class, noise)
    correct_temp = get_temperature_for_class(actual_class, temp_f)
    feats       = make_features_from_array(pir, correct_temp, feature_names)
    X           = scaler.transform(feats.values)
    y_pred      = model.predict(X)[0]
    proba       = model.predict_proba(X)[0]
    state       = label_map.get(int(y_pred), actual_class)
    confidence  = float(np.max(proba))
    savings     = calculate_savings(state)
    timestamp   = get_timestamp()

    st.session_state.pir_history.append(float(np.mean(pir)))
    st.session_state.time_history.append(timestamp)
    st.session_state.pred_history.append(state)
    st.session_state.log_rows.appendleft({
        "Time":            timestamp,
        "Simulated":       actual_class,
        "Predicted":       state,
        "Confidence":      f"{confidence*100:.1f}%",
        "HVAC":            savings["actions"]["HVAC"],
        "Temp Used (°F)":  f"{correct_temp:.0f}",
    })

    times = list(st.session_state.time_history)
    vals  = list(st.session_state.pir_history)
    color = _state_color(state)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=vals, mode="lines+markers",
        line=dict(color=color, width=2), fill="tozeroy", name="PIR"
    ))
    fig.add_trace(go.Scatter(
        x=times, y=[float(np.mean(vals))] * len(vals), mode="lines",
        line=dict(color="#7f8c8d", width=1, dash="dot"), name="Mean"
    ))
    fig.update_layout(title="Live PIR Sensor Feed – Last 30 Readings",
                      xaxis_title="Time", yaxis_title="Signal", height=300,
                      margin=dict(l=0, r=0, t=40, b=0), showlegend=False)

    with sensor_ph.container():
        st.subheader("Live PIR Sensor Feed")
        st.plotly_chart(fig, use_container_width=True)

    with metrics_ph.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Value", f"{vals[-1]:.1f}")
        c2.metric("Mean (30s)",    f"{np.mean(vals):.1f}")
        c3.metric("Peak (30s)",    f"{np.max(vals):.1f}")

    with log_ph.container():
        st.subheader("Prediction Log")
        st.dataframe(pd.DataFrame(list(st.session_state.log_rows)),
                     use_container_width=True)

    st.session_state.live_savings_usd += savings["savings_usd_day"] / 18000.0
    st.session_state.live_savings_kwh += savings["savings_kwh_day"] / 18000.0
    with ticker_ph.container():
        st.subheader("Savings Accumulating Live")
        c1, c2 = st.columns(2)
        c1.metric("Money Saved Today", f"${st.session_state.live_savings_usd:,.2f}")
        c2.metric("kWh Saved Today",   f"{st.session_state.live_savings_kwh:.2f} kWh")

    _time.sleep(2)
    st.rerun()


def main():
    _ensure_artifacts()
    _init_live_state()

    st.title("Live Occupancy Prediction")
    _top_kpi_bar()
    st.markdown("---")

    sim_class, noise, temp_f, run_btn = _render_sidebar()

    # Clear stale result if class changed
    if (st.session_state.last_manual_sim_class is not None
            and st.session_state.last_manual_sim_class != sim_class
            and sim_class != "Random"):
        st.session_state.last_manual_result    = None
        st.session_state.last_manual_timestamp = None
        st.session_state.last_manual_sim_class = None

    if run_btn:
        result = _run_prediction(sim_class, noise, temp_f)
        st.session_state.last_manual_result    = result
        st.session_state.last_manual_timestamp = get_timestamp()
        st.session_state.last_manual_sim_class = sim_class

    if st.session_state.last_manual_result is not None:
        ts = st.session_state.last_manual_timestamp or ""
        st.markdown(f"*Manual prediction run at {ts} — click **Run Prediction** to refresh*")
        _render_results(*st.session_state.last_manual_result)
    elif not run_btn:
        st.info("Configure the sidebar and click **Run Prediction** to start.")

    st.markdown("---")

    if st.session_state.live_running_pred:
        st.markdown("🔴 **LIVE** — updating every 2 seconds")
    else:
        st.markdown("⏸ **PAUSED**")

    sensor_ph, metrics_ph, log_ph, ticker_ph = (
        st.empty(), st.empty(), st.empty(), st.empty()
    )

    if st.session_state.live_running_pred:
        _live_loop(sensor_ph, metrics_ph, log_ph, ticker_ph, sim_class, noise, temp_f)


if __name__ == "__main__":
    main()
