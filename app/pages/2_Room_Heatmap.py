import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import deque

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.predict import load_artifacts
from utils.realtime import get_timestamp, init_session_deques
from utils.simulation import simulate_building


def _ensure_artifacts():
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = load_artifacts()
    a = st.session_state.artifacts
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
    c4.metric("CO₂ Saved / year", f"{annual_co2:,.0f} kg")
    c5.metric("Green Score", f"{leed_score:.1f} / 100")


def _init_state():
    if "live_running_heatmap" not in st.session_state:
        st.session_state.live_running_heatmap = True
    init_session_deques({"energy_hist": 12})


def _render_controls():
    with st.sidebar:
        st.header("Heatmap Controls")
        rooms = st.slider("Number of rooms", 5, 20, 12)
        day = st.selectbox(
            "Day of week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )
        hour = st.slider("Hour of day", 0, 23, 9)
        st.caption(f"Current simulated time: {hour:02d}:00")
        if st.button(
            "⏸ Pause Live Heatmap"
            if st.session_state.live_running_heatmap
            else "▶ Resume Live Heatmap"
        ):
            st.session_state.live_running_heatmap = not st.session_state.live_running_heatmap

    if st.session_state.live_running_heatmap:
        st.markdown("🔴 **LIVE** — auto-refreshing every 5 seconds")
    else:
        st.markdown("⏸ **PAUSED**")
    return rooms, day, hour


def _render_room_grid(grid_ph, df):
    n = len(df)
    cols = 5
    rows = int(np.ceil(n / cols))
    with grid_ph.container():
        st.markdown("### Live Room Status")
        for r in range(rows):
            row = st.columns(cols)
            for c in range(cols):
                idx = r * cols + c
                if idx >= n:
                    continue
                room = df.iloc[idx]
                bg = room["color"]
                state = room["state"]
                emoji = "🟢" if state == "Vacancy" else "🔵" if state == "Stationary" else "🔴"
                with row[c]:
                    st.markdown(
                        f"<div style='padding:0.75rem;border-radius:0.5rem;"
                        f"background-color:{bg}22;'>"
                        f"<strong>{room['room_name']}</strong><br/>"
                        f"{emoji} {state}<br/>"
                        f"{room['energy_W']:.0f} W"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


def _render_summary(summary_ph, df):
    vacant = (df["state"] == "Vacancy").sum()
    stationary = (df["state"] == "Stationary").sum()
    motion = (df["state"] == "Motion").sum()
    total_w = df["energy_W"].sum()
    waste_w = df.loc[df["state"] == "Vacancy", "energy_W"].sum()
    cost_hour = (total_w / 1000.0) * 0.12

    with summary_ph.container():
        st.markdown("### Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vacant Rooms", f"{vacant}")
        c2.metric("Stationary Rooms", f"{stationary}")
        c3.metric("Motion Rooms", f"{motion}")
        c4.metric("Energy Now", f"{total_w:,.0f} W")
        st.metric("Estimated Cost / hour", f"${cost_hour:,.2f}")

        pct_vacant_on = (
            (vacant / len(df)) * 100.0 if len(df) > 0 else 0.0
        )
        if pct_vacant_on > 30:
            st.warning(
                f"⚠️ {vacant} rooms detected vacant — recommend shutting HVAC "
                f"to save approximately ${cost_hour:.2f} per hour."
            )
        elif motion == len(df):
            st.success(
                "✅ Building fully utilized — AI running at peak efficiency."
            )


def _render_waste_gauge(gauge_ph, df):
    total_w = df["energy_W"].sum()
    waste_w = df.loc[df["state"] == "Vacancy", "energy_W"].sum()
    waste_pct = (waste_w / total_w) * 100.0 if total_w > 0 else 0.0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=waste_pct,
            title={"text": "Energy Waste (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c"},
                "steps": [
                    {"range": [0, 30], "color": "#2ecc71"},
                    {"range": [30, 60], "color": "#f1c40f"},
                    {"range": [60, 100], "color": "#e74c3c"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0))
    with gauge_ph.container():
        st.markdown("### Live Energy Waste Meter")
        st.plotly_chart(fig, use_container_width=True)


def _render_energy_history(history_ph):
    times = list(st.session_state.energy_hist)
    if not times:
        return
    labels = [t["time"] for t in times]
    vals = [t["energy_W"] for t in times]
    fig = go.Figure(
        go.Scatter(
            x=labels,
            y=vals,
            mode="lines+markers",
            line=dict(color="#3498db", width=2),
            fill="tozeroy",
        )
    )
    fig.update_layout(
        title="Energy History (last ~1 minute)",
        xaxis_title="Time",
        yaxis_title="Energy (W)",
        height=260,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    with history_ph.container():
        st.plotly_chart(fig, use_container_width=True)


def main():
    _ensure_artifacts()
    _init_state()

    st.title("Room Heatmap — Live Building Monitor")
    _top_kpi_bar()
    st.markdown("---")

    rooms, day, hour = _render_controls()

    grid_ph = st.empty()
    summary_ph = st.empty()
    gauge_ph = st.empty()
    hist_ph = st.empty()

    while st.session_state.live_running_heatmap:
        df = simulate_building(n_rooms=rooms, time_of_day=hour, day_of_week=day)

        st.session_state.energy_hist.append(
            {"time": get_timestamp(), "energy_W": float(df["energy_W"].sum())}
        )

        _render_room_grid(grid_ph, df)
        _render_summary(summary_ph, df)
        _render_waste_gauge(gauge_ph, df)
        _render_energy_history(hist_ph)

        import time as _time

        _time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
