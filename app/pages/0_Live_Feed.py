import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time as _time_module
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.energy import calculate_savings
from utils.predict import load_artifacts
from utils.realtime import (
    compute_live_savings,
    get_timestamp,
    init_session_deques,
)
from utils.simulation import ROOM_NAMES, simulate_building


def _pir_mean_for_state(state: str) -> float:
    """Representative PIR mean per state for analytics."""
    if state == "Vacancy":
        return float(np.random.uniform(2, 8))
    if state == "Stationary":
        return float(np.random.uniform(35, 55))
    return float(np.random.uniform(120, 220))


def _ensure_artifacts():
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = load_artifacts()
    a = st.session_state.artifacts
    st.session_state.model = a.model
    st.session_state.scaler = a.scaler
    st.session_state.feature_names = a.feature_names
    st.session_state.label_mapping = a.label_mapping
    st.session_state.results_summary = a.summary


def _top_kpi_bar():
    summary = st.session_state.get("results_summary", {}) or {}
    energy = summary.get("energy_savings", {})
    best_model = summary.get("best_baseline_model", "Model")
    f1 = summary.get("tuned_xgb_f1", summary.get("best_baseline_f1", 0)) * 100
    annual_savings = energy.get("annual_savings_USD", 0.0)
    annual_co2 = energy.get("annual_CO2_saved_kg", 0.0)
    leed_score = summary.get("leed_score", 90.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Model", best_model)
    c2.metric("Test F1", f"{f1:.1f}%")
    c3.metric("Annual Savings", f"${annual_savings:,.0f}")
    c4.metric("CO₂ Saved / year", f"{annual_co2:,.0f} kg")
    c5.metric("Green Score", f"{leed_score:.1f} / 100")


def _init_state():
    if "live_running" not in st.session_state:
        st.session_state.live_running = True
    init_session_deques(
        {
            "live_log": 20,
            "live_time": 60,
            "live_rooms": 60,
            "live_energy": 60,
        }
    )
    if "pred_count" not in st.session_state:
        st.session_state.pred_count = 0
    if "vacancy_count" not in st.session_state:
        st.session_state.vacancy_count = 0
    if "active_rooms" not in st.session_state:
        st.session_state.active_rooms = 0
    if "saved_today" not in st.session_state:
        st.session_state.saved_today = 0.0
    if "all_predictions" not in st.session_state:
        st.session_state.all_predictions = []
    if "all_pir_values" not in st.session_state:
        st.session_state.all_pir_values = deque(maxlen=500)
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = _time_module.time()
    if "pause_timestamp" not in st.session_state:
        st.session_state.pause_timestamp = get_timestamp()


def _sample_room() -> str:
    return random.choice(ROOM_NAMES)


def _room_state_emoji(state: str) -> str:
    if state == "Vacancy":
        return "🟢"
    if state == "Stationary":
        return "🔵"
    return "🔴"


def _render_layout():
    st.title("🔴 LIVE — Building Intelligence Feed")
    st.subheader("AI processing sensor data from 20 rooms in real time")
    _top_kpi_bar()
    st.markdown("---")

    with st.sidebar:
        if st.button(
            "⏸ Pause Live Feed" if st.session_state.live_running else "▶ Resume Live Feed"
        ):
            st.session_state.live_running = not st.session_state.live_running
            if not st.session_state.live_running:
                st.session_state.pause_timestamp = get_timestamp()
        st.markdown(
            "This view simulates live predictions across the whole building."
        )

    if st.session_state.live_running:
        st.markdown("🔴 **LIVE** — updating every 3 seconds")
    else:
        st.markdown("⏸ **PAUSED**")

    kpi_ph = st.empty()
    log_ph = st.empty()

    # Two-column layout: 60% chart, 40% analytics
    col1_ph, col2_ph = st.columns([0.6, 0.4])
    with col1_ph:
        chart_ph = st.empty()
    with col2_ph:
        analytics_ph = st.empty()

    return kpi_ph, log_ph, chart_ph, analytics_ph


def _update_kpis(kpi_ph):
    summary = st.session_state.get("results_summary", {}) or {}
    energy = summary.get("energy_savings", {})
    annual_usd = energy.get("annual_savings_USD", 0.0)
    live = compute_live_savings(annual_usd)
    st.session_state.saved_today = live["usd_today"]

    with kpi_ph.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predictions Made", f"{st.session_state.pred_count}")
        c2.metric("Vacancies Detected", f"{st.session_state.vacancy_count}")
        c3.metric("Active Rooms", f"{st.session_state.active_rooms}")
        c4.metric("Saved Today", f"${live['usd_today']:,.2f}")


def _update_log(log_ph):
    cols = ["Time", "Room", "State", "Confidence", "Action", "Saving"]
    rows = list(st.session_state.live_log)
    if not rows:
        return

    df = pd.DataFrame(rows, columns=cols)
    with log_ph.container():
        st.markdown("### Live Prediction Feed")
        st.dataframe(df.style.hide(axis="index"), use_container_width=True)


def _render_live_analytics(analytics_ph, building_df):
    """Smaller live stats summary for col2 when LIVE."""
    with analytics_ph.container():
        st.markdown("### Live Stats")
        if not building_df.empty:
            counts = building_df["state"].value_counts()
            labels = counts.index.tolist()
            values = counts.values.tolist()
            colors = ["#2ecc71", "#3498db", "#e74c3c"]
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.5,
                        marker=dict(colors=colors[: len(labels)]),
                    )
                ]
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=220)
            st.plotly_chart(fig, use_container_width=True)

            total_w = building_df["energy_W"].sum()
            active = (building_df["state"] != "Vacancy").sum()
            st.metric("Total Building Watts", f"{total_w:,.0f} W")
            st.metric("Active Rooms", f"{active}")
        else:
            st.info("Collecting data...")


def _render_paused_analytics(analytics_ph):
    """Full analytics panel for col2 when PAUSED."""
    preds = st.session_state.get("all_predictions", [])
    pir_vals = list(st.session_state.get("all_pir_values", []))
    pause_ts = st.session_state.get("pause_timestamp", get_timestamp())
    start_time = st.session_state.get("session_start_time", _time_module.time())
    total_runtime = int(_time_module.time() - start_time) if start_time else 0

    with analytics_ph.container():
        st.markdown(f"### 📊 Analytics — Paused at {pause_ts}")

        if not preds:
            st.info("No prediction data collected yet. Resume the feed to start.")
            st.markdown(
                '<p style="color:#2ecc71;font-weight:bold;">▶ Press Resume to continue live feed</p>',
                unsafe_allow_html=True,
            )
            return

        df = pd.DataFrame(preds)
        total = len(preds)
        vac = (df["state"] == "Vacancy").sum()
        stat = (df["state"] == "Stationary").sum()
        mot = (df["state"] == "Motion").sum()
        vac_pct = (vac / total * 100) if total > 0 else 0
        stat_pct = (stat / total * 100) if total > 0 else 0
        mot_pct = (mot / total * 100) if total > 0 else 0

        # 2x2 summary metrics
        m1, m2 = st.columns(2)
        m1.metric("Total Predictions Made", f"{total}")
        m2.metric("Vacancy Count", f"{vac} ({vac_pct:.1f}%)")
        m3, m4 = st.columns(2)
        m3.metric("Stationary Count", f"{stat} ({stat_pct:.1f}%)")
        m4.metric("Motion Count", f"{mot} ({mot_pct:.1f}%)")

        # Donut chart
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Vacancy", "Stationary", "Motion"],
                    values=[vac, stat, mot],
                    hole=0.5,
                    marker=dict(colors=["#2ecc71", "#3498db", "#e74c3c"]),
                )
            ]
        )
        fig_donut.update_layout(
            title="Distribution by State",
            margin=dict(l=0, r=0, t=30, b=0),
            height=220,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Bar chart: average PIR per state
        avg_pir = df.groupby("state")["pir_mean"].mean().reindex(
            ["Vacancy", "Stationary", "Motion"], fill_value=0
        )
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=avg_pir.index,
                    y=avg_pir.values,
                    marker_color=["#2ecc71", "#3498db", "#e74c3c"],
                )
            ]
        )
        fig_bar.update_layout(
            title="Avg PIR Signal per State",
            yaxis_title="PIR Mean",
            margin=dict(l=0, r=0, t=30, b=0),
            height=200,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Stats table
        pir_arr = np.array(pir_vals) if pir_vals else np.array([0.0])
        n_pir = len(pir_arr)
        stats_data = {
            "Metric": ["Mean Signal", "Peak Signal", "Min Signal", "Std Dev", "Total Runtime"],
            "Value": [
                f"{float(np.mean(pir_arr)):.2f}" if n_pir > 0 else "0.00",
                f"{float(np.max(pir_arr)):.2f}" if n_pir > 0 else "0.00",
                f"{float(np.min(pir_arr)):.2f}" if n_pir > 0 else "0.00",
                f"{float(np.std(pir_arr)):.2f}" if n_pir > 1 else "0.00",
                f"{total_runtime} sec",
            ],
        }
        st.markdown("**Stats**")
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

        # Energy insight (split session saved_today by state contribution)
        saved_today = st.session_state.get("saved_today", 0.0)
        denom = vac_pct + stat_pct if (vac_pct + stat_pct) > 0 else 1
        vac_usd = saved_today * (vac_pct / denom) * 0.8 if denom > 0 else 0
        stat_usd = saved_today * (stat_pct / denom) * 0.2 if denom > 0 else 0
        st.markdown("**Energy insight**")
        st.write(
            f"During this session:\n"
            f"- {vac_pct:.0f}% vacant → saved ${vac_usd:.2f}\n"
            f"- {stat_pct:.0f}% stationary → saved ${stat_usd:.2f}\n"
            f"- {mot_pct:.0f}% motion → $0 saved"
        )

        st.markdown(
            '<p style="color:#2ecc71;font-weight:bold;">▶ Press Resume to continue live feed</p>',
            unsafe_allow_html=True,
        )


def _render_chart(chart_ph):
    """Render chart in col1 (live or frozen)."""
    times = list(st.session_state.live_time)
    rooms = list(st.session_state.live_rooms)
    energy = list(st.session_state.live_energy)
    if not times:
        with chart_ph.container():
            st.info("Collecting data...")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=times,
            y=rooms,
            name="Occupied Rooms",
            marker_color="#3498db",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=energy,
            name="Energy (W)",
            mode="lines",
            line=dict(color="#e74c3c", width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Last 60 Seconds — Occupancy vs Energy",
        xaxis=dict(title="Time", showgrid=False),
        yaxis=dict(title="Occupied Rooms", side="left"),
        yaxis2=dict(
            title="Energy (W)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=320,
        legend=dict(orientation="h"),
    )
    with chart_ph.container():
        st.plotly_chart(fig, use_container_width=True)


def main():
    _ensure_artifacts()
    _init_state()

    kpi_ph, log_ph, chart_ph, analytics_ph = _render_layout()

    # When PAUSED: show frozen chart + full analytics panel
    if not st.session_state.live_running:
        _update_kpis(kpi_ph)
        _update_log(log_ph)
        _render_chart(chart_ph)  # Frozen last state
        _render_paused_analytics(analytics_ph)
        return

    # Live loop
    while st.session_state.live_running:
        # Simulate building states
        building_df = simulate_building(n_rooms=20)
        st.session_state.active_rooms = (building_df["state"] != "Vacancy").sum()

        # Create a synthetic prediction for one random room
        room_row = building_df.sample(1).iloc[0]
        room_name = room_row["room_name"]
        state = room_row["state"]
        emoji = _room_state_emoji(state)
        confidence = float(np.random.uniform(0.9, 1.0))
        savings = calculate_savings(state, rooms=1)
        saved_usd = savings["savings_usd_day"] / 10.0
        pir_mean = _pir_mean_for_state(state)

        st.session_state.pred_count += 1
        if state == "Vacancy":
            st.session_state.vacancy_count += 1
        st.session_state.saved_today += saved_usd

        timestamp = get_timestamp()

        # Store prediction history for analytics
        st.session_state.all_predictions.append({
            "time": timestamp,
            "state": state,
            "pir_mean": pir_mean,
            "confidence": confidence,
            "saving_pct": int(savings["savings_pct"]),
        })
        st.session_state.all_pir_values.append(pir_mean)

        st.session_state.live_log.appendleft(
            [
                timestamp,
                room_name,
                f"{emoji} {state}",
                f"{confidence*100:.1f}%",
                savings["actions"]["HVAC"],
                f"${saved_usd:.2f}",
            ]
        )

        st.session_state.live_time.append(timestamp)
        st.session_state.live_rooms.append(int(st.session_state.active_rooms))
        st.session_state.live_energy.append(int(building_df["energy_W"].sum()))

        _update_kpis(kpi_ph)
        _update_log(log_ph)
        _render_chart(chart_ph)
        _render_live_analytics(analytics_ph, building_df)

        _time_module.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()

