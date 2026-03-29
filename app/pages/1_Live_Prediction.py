import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time as _time
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.energy import ACTIONS, ICONS, calculate_savings
from utils.features import make_features_from_array, simulate_pir
from utils.predict import load_artifacts
from utils.realtime import get_timestamp, init_session_deques


def _ensure_artifacts():
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = load_artifacts()
    a = st.session_state.artifacts
    st.session_state.model = a.model
    st.session_state.scaler = a.scaler
    st.session_state.feature_names = a.feature_names
    # FIX 4: use consistent key "label_map" matching app.py
    st.session_state.label_map = a.label_mapping
    st.session_state.results_summary = a.summary


def _top_kpi_bar():
    summary = st.session_state.get("results_summary", {}) or {}
    energy = summary.get("energy_savings", {})
    best_model = summary.get("best_model", type(st.session_state.get("model", object())).__name__)
    f1 = summary.get("macro_f1", summary.get("accuracy", 0)) * 100
    annual_savings = energy.get("annual_savings_USD", 1752)
    annual_co2 = energy.get("annual_CO2_saved_kg", 3400)
    leed_score = summary.get("leed_score", 90.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Model", best_model)
    c2.metric("Test F1", f"{f1:.1f}%")
    c3.metric("Annual Savings", f"${annual_savings:,.0f}")
    c4.metric("CO2 Saved / year", f"{annual_co2:,.0f} kg")
    c5.metric("Green Score", f"{leed_score:.1f} / 100")


def _init_live_state():
    if "live_running_pred" not in st.session_state:
        st.session_state.live_running_pred = True
    init_session_deques(
        {
            "pir_history": 30,
            "time_history": 30,
            "pred_history": 30,
            "log_rows": 10,
        }
    )
    if "live_savings_usd" not in st.session_state:
        st.session_state.live_savings_usd = 0.0
    if "live_savings_kwh" not in st.session_state:
        st.session_state.live_savings_kwh = 0.0
    # FIX 2: persist last manual prediction result across reruns
    if "last_manual_result" not in st.session_state:
        st.session_state.last_manual_result = None


def _state_color(state: str) -> str:
    if state == "Vacancy":
        return "#2ecc71"
    if state == "Stationary":
        return "#3498db"
    return "#e74c3c"


def _render_static_section():
    st.title("Live Occupancy Prediction")
    _top_kpi_bar()
    st.markdown("---")

    with st.sidebar:
        st.header("Simulation Controls")
        sim_class = st.selectbox(
            "Simulate Class",
            ["Vacancy", "Stationary", "Motion", "Random"],
            index=3,
        )
        noise = st.slider("Noise Level", 0, 50, 20)
        temp_f = st.slider("Temperature (F)", 60, 90, 72)
        run_btn = st.button("Run Prediction", type="primary")

        st.markdown("---")
        if st.button(
            "Pause Live Feed"
            if st.session_state.live_running_pred
            else "Resume Live Feed"
        ):
            st.session_state.live_running_pred = not st.session_state.live_running_pred

    if st.session_state.live_running_pred:
        st.markdown("LIVE - updating every 2 seconds")
    else:
        st.markdown("PAUSED")

    return sim_class, noise, temp_f, run_btn


def _run_single_prediction(sim_class: str, noise: int, temp_f: float):
    model = st.session_state.model
    scaler = st.session_state.scaler
    feature_names = st.session_state.feature_names
    # FIX 4: use consistent key "label_map"
    label_map = st.session_state.label_map

    if sim_class == "Random":
        sim_class = np.random.choice(["Vacancy", "Stationary", "Motion"])

    pir = simulate_pir(sim_class, noise)
    feats = make_features_from_array(pir, temp_f, feature_names)
    X = scaler.transform(feats.values)
    y_pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    state = label_map.get(int(y_pred), sim_class)
    confidence = float(np.max(proba))
    savings = calculate_savings(state)

    return pir, state, confidence, proba, savings, feats


def _render_prediction_results(pir, state, confidence, proba, savings, feats):
    color = _state_color(state)
    st.markdown("### Prediction Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected State", state)
    c2.metric("Confidence", f"{confidence*100:.1f}%")
    c3.metric("Energy Saving", f"{savings['savings_pct']:.1f}%")

    x = list(range(1, len(pir) + 1))
    fig_pir = go.Figure()
    fig_pir.add_trace(
        go.Scatter(
            x=x,
            y=pir,
            mode="lines+markers",
            line=dict(color=color, width=2),
            fill="tozeroy",
            name="PIR Signal",
        )
    )
    fig_pir.add_trace(
        go.Scatter(
            x=x,
            y=[float(np.mean(pir))] * len(pir),
            mode="lines",
            line=dict(color="#7f8c8d", width=1, dash="dash"),
            name="Mean",
        )
    )
    fig_pir.update_layout(
        title="PIR Signal Window",
        xaxis_title="Sample",
        yaxis_title="Signal",
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    classes = ["Vacancy", "Stationary", "Motion"]
    probs_ordered = [proba[idx] if idx < len(proba) else 0.0 for idx, _ in enumerate(classes)]
    fig_prob = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=[p * 100 for p in probs_ordered],
                marker_color=["#2ecc71", "#3498db", "#e74c3c"],
            )
        ]
    )
    fig_prob.update_layout(
        title="Class Probabilities",
        yaxis_title="Confidence (%)",
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_pir, use_container_width=True)
    with c2:
        st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("### Device Actions & Comfort")
    d1, d2, d3, d4, d5 = st.columns(5)
    actions = savings["actions"]
    devices = ["HVAC", "Lighting", "Ventilation", "Screens", "Elevators"]
    cols = [d1, d2, d3, d4, d5]
    for dev, col in zip(devices, cols):
        col.metric(f"{ICONS[dev]} {dev}", actions[dev])
    st.info(
        f"Recommended comfort setpoint: "
        f"{savings['comfort_temp_C']:.1f} C for {state}."
    )

    with st.expander("PIR Signal Statistics"):
        st.dataframe(feats.describe().T)


def _init_live_placeholders():
    sensor_ph = st.empty()
    metrics_ph = st.empty()
    log_ph = st.empty()
    ticker_ph = st.empty()
    return sensor_ph, metrics_ph, log_ph, ticker_ph


def _live_loop(sensor_ph, metrics_ph, log_ph, ticker_ph, sim_class, noise, temp_f):
    model = st.session_state.model
    scaler = st.session_state.scaler
    feature_names = st.session_state.feature_names
    # FIX 4: use consistent key "label_map"
    label_map = st.session_state.label_map

    # FIX 1: use actual sidebar values — not hardcoded "Motion", noise=20, temp=72
    live_class = sim_class if sim_class != "Random" else np.random.choice(["Vacancy", "Stationary", "Motion"])

    pir = simulate_pir(live_class, noise)
    feats = make_features_from_array(pir, temp_f, feature_names)
    X = scaler.transform(feats.values)
    y_pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    state = label_map.get(int(y_pred), live_class)
    confidence = float(np.max(proba))
    savings = calculate_savings(state)

    timestamp = get_timestamp()
    st.session_state.pir_history.append(float(np.mean(pir)))
    st.session_state.time_history.append(timestamp)
    st.session_state.pred_history.append(state)

    st.session_state.log_rows.appendleft(
        {
            "Time": timestamp,
            "Signal": f"{float(np.mean(pir)):.1f}",
            "Simulated": live_class,
            "Predicted State": state,
            "Confidence": f"{confidence*100:.1f}%",
            "Action": savings["actions"]["HVAC"],
        }
    )

    times = list(st.session_state.time_history)
    vals = list(st.session_state.pir_history)
    color = _state_color(state)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=vals,
            mode="lines+markers",
            line=dict(color=color, width=2),
            fill="tozeroy",
            name="PIR",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=[float(np.mean(vals))] * len(vals),
            mode="lines",
            line=dict(color="#7f8c8d", width=1, dash="dot"),
            name="Mean",
        )
    )
    fig.update_layout(
        title="Live PIR Sensor Feed - Last 30 Readings",
        xaxis_title="Time",
        yaxis_title="Signal",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    with sensor_ph.container():
        st.subheader("Live PIR Sensor Feed")
        st.plotly_chart(fig, use_container_width=True)

    with metrics_ph.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Value", f"{vals[-1]:.1f}")
        c2.metric("Mean (30s)", f"{np.mean(vals):.1f}")
        c3.metric("Peak (30s)", f"{np.max(vals):.1f}")

    with log_ph.container():
        st.subheader("Prediction Log")
        df = pd.DataFrame(list(st.session_state.log_rows))
        st.dataframe(df, use_container_width=True)

    # FIX 3: savings_usd_day covers a 10-hour workday = 18000 two-second ticks
    # Was incorrectly dividing by 1800 (10x too fast)
    st.session_state.live_savings_usd += savings["savings_usd_day"] / 18000.0
    st.session_state.live_savings_kwh += savings["savings_kwh_day"] / 18000.0
    with ticker_ph.container():
        st.subheader("Savings Accumulating Live")
        c1, c2 = st.columns(2)
        c1.metric(
            "Money Saved Today",
            f"${st.session_state.live_savings_usd:,.2f}",
        )
        c2.metric(
            "kWh Saved Today",
            f"{st.session_state.live_savings_kwh:.2f} kWh",
        )

    _time.sleep(2)
    st.rerun()


def main():
    _ensure_artifacts()
    _init_live_state()

    sim_class, noise, temp_f, run_btn = _render_static_section()

    # FIX 2: store result in session_state so it persists across st.rerun() calls
    if run_btn:
        result = _run_single_prediction(sim_class, noise, temp_f)
        st.session_state.last_manual_result = result

    # Always render the last manual result if one exists (survives live loop reruns)
    if st.session_state.last_manual_result is not None:
        st.markdown("---")
        _render_prediction_results(*st.session_state.last_manual_result)
    elif not run_btn:
        st.info("Configure the sidebar and click **Run Prediction** to start.")

    st.markdown("---")

    # Live feed placeholders are rendered below the manual result section
    sensor_ph, metrics_ph, log_ph, ticker_ph = _init_live_placeholders()

    # FIX 1 + 2: pass sidebar values into live loop; one tick per rerun (no while loop)
    if st.session_state.live_running_pred:
        _live_loop(sensor_ph, metrics_ph, log_ph, ticker_ph, sim_class, noise, temp_f)


if __name__ == "__main__":
    main()
