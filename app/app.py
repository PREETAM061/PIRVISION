import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from utils.predict import load_artifacts


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def _init_session_state():
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = get_artifacts()
    artifacts = st.session_state.artifacts
    st.session_state.model = artifacts.model
    st.session_state.scaler = artifacts.scaler
    st.session_state.feature_names = artifacts.feature_names
    st.session_state.label_mapping = artifacts.label_mapping
    st.session_state.results_summary = artifacts.summary


def _top_kpi_bar():
    summary = st.session_state.get("results_summary", {}) or {}
    energy = summary.get("energy_savings", {})
    best_model = summary.get("best_baseline_model", "Model")
    f1 = summary.get("tuned_xgb_f1", summary.get("best_baseline_f1", 0)) * 100
    annual_savings = energy.get("annual_savings_USD", 0.0)
    annual_co2 = energy.get("annual_CO2_saved_kg", 0.0)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Best Model", best_model)
    col2.metric("Test F1", f"{f1:.1f}%")
    col3.metric("Annual Savings", f"${annual_savings:,.0f}")
    col4.metric("CO₂ Saved / year", f"{annual_co2:,.0f} kg")

    leed_score = summary.get("leed_score", 90.0)
    col5.metric("Green Score", f"{leed_score:.1f} / 100")


def main():
    _init_session_state()

    st.set_page_config(
        page_title="PIRvision Smart Occupancy",
        page_icon="🌱",
        layout="wide",
    )

    st.title("PIRvision Smart Occupancy & Sustainability System")
    st.subheader("AI-powered building intelligence for a greener future")

    _top_kpi_bar()
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What this system does")
        st.write(
            "PIRvision ingests rich PIR sensor and temperature data to detect "
            "room occupancy in real time, optimize energy use across HVAC, "
            "lighting, and other devices, and surface actionable insights."
        )
        st.markdown("### How the AI works")
        st.write(
            "A tuned gradient-boosted model learns nuanced motion patterns "
            "from 55 PIR channels and engineered features, then classifies "
            "each window as Vacancy, Stationary, or Motion."
        )
    with col2:
        st.markdown("### Real-world impact")
        st.write(
            "By aligning device operation with true occupancy, buildings can "
            "cut wasted energy hours, reduce CO₂ emissions, and improve "
            "comfort for occupants."
        )
        st.markdown("### Sustainability commitment")
        st.write(
            "PIRvision aligns with LEED and ESG frameworks, exposing ROI, "
            "carbon savings, and green building scores ready for reporting."
        )

    st.markdown("---")
    st.markdown("### Navigation")
    

    pages = [
        ("1 - Live Feed", "Live building intelligence feed"),
        ("2 - Live Prediction", "Real-time occupancy prediction simulation"),
        ("3 - Room Heatmap", "Floor plan room occupancy heatmap"),
    ]

    for name, desc in pages:
        st.markdown(f"- **{name}**: {desc}")

    st.markdown("---")
    st.markdown("### Key Features")
    st.write(
        "Real-time occupancy detection"
        "55-channel PIR sensor input"
        "AI-based classification"
        "High accuracy prediction"
        "Privacy-safe (no camera)"
        "Low-light operation"
        "Energy-saving automation support"
        "Live data visualization"
    )


if __name__ == "__main__":
    main()

