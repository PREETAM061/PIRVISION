import time
from collections import deque
from datetime import datetime
from typing import Dict, Iterable, Any


def get_timestamp() -> str:
    """Return current time as HH:MM:SS string."""
    return datetime.now().strftime("%H:%M:%S")


def seconds_since_9am() -> float:
    """Return seconds elapsed since 9:00 AM today (business start)."""
    now = datetime.now()
    start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    delta = (now - start).total_seconds()
    return max(0.0, delta)


def compute_live_savings(annual_usd: float) -> Dict[str, float]:
    """
    Compute live savings counters based on time elapsed since 9AM.
    Returns today's accumulated savings in $, kWh, and CO2 kg.
    """
    # Assume 10 hour working day in annual calculation
    seconds = seconds_since_9am()
    per_second = annual_usd / (365 * 10 * 3600) if annual_usd > 0 else 0.0
    usd_today = per_second * seconds
    kwh_today = usd_today / 0.12 if usd_today > 0 else 0.0
    co2_today = kwh_today * 0.233
    return {
        "usd_today": round(usd_today, 3),
        "kwh_today": round(kwh_today, 3),
        "co2_today": round(co2_today, 3),
        "per_second": round(per_second, 6),
    }


def compute_live_co2(state: str, rooms: int = 1) -> Dict[str, float]:
    """
    Compute live CO2 rates for carbon tracker.
    Returns current production rate and saved rate.
    """
    power_W = {
        "Vacancy": 150,
        "Stationary": 1350,
        "Motion": 2600,
        "Baseline": 2600,
    }
    current_W = power_W.get(state, 1350) * max(rooms, 1)
    baseline_W = power_W["Baseline"] * max(rooms, 1)

    current_kwh_hr = current_W / 1000.0
    baseline_kwh_hr = baseline_W / 1000.0

    return {
        "co2_rate_kg_hr": round(current_kwh_hr * 0.233, 4),
        "baseline_rate_kg_hr": round(baseline_kwh_hr * 0.233, 4),
        "saved_rate_kg_hr": round((baseline_kwh_hr - current_kwh_hr) * 0.233, 4),
        "co2_per_second": round(current_kwh_hr * 0.233 / 3600.0, 6),
        "saved_per_second": round(
            (baseline_kwh_hr - current_kwh_hr) * 0.233 / 3600.0, 6
        ),
    }


def init_session_deques(keys_maxlen: Dict[str, int]) -> None:
    """
    Initialize session_state deques if not already present.
    keys_maxlen = {'pir_history': 30, 'time_history': 30, ...}
    """
    import streamlit as st

    for key, maxlen in keys_maxlen.items():
        if key not in st.session_state:
            st.session_state[key] = deque(maxlen=maxlen)


def make_live_chart(
    x_data: Iterable[Any],
    y_data: Iterable[float],
    color: str,
    title: str,
    y_label: str,
    height: int = 300,
):
    """
    Build a Plotly live scrolling line chart.
    Returns a go.Figure ready for st.plotly_chart().
    """
    import plotly.graph_objects as go

    x_list = list(x_data)
    y_list = list(y_data)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_list,
            y=y_list,
            mode="lines+markers",
            line=dict(color=color, width=2),
            fill="tozeroy",
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_label,
        height=height,
        margin=dict(l=0, r=0, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig

