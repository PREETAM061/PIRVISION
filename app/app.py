# app/app.py
import streamlit as st
from utils.predict import load_artifacts

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="PIRVision — Smart Occupancy",
    page_icon="🏢",
    layout="wide"
)

# ── Load artifacts (MODEL + SCALER + SUMMARY) ───────────────
@st.cache_resource
def get_artifacts():
    return load_artifacts()

art = get_artifacts()

# ── Store in session (for all pages) ────────────────────────
st.session_state["model"] = art.model
st.session_state["scaler"] = art.scaler
st.session_state["feature_names"] = art.feature_names
st.session_state["label_map"] = art.label_mapping
st.session_state["summary"] = art.summary

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("🏢 PIRVision")
    st.caption("Smart Occupancy Detection")

    model_name = type(art.model).__name__

    if model_name == "XGBClassifier":
        st.success(f"✅ Model: {model_name}")
    else:
        st.warning(f"⚠️ Model: {model_name}")

    st.write(f"Features: {len(art.feature_names)}")
    st.write(f"Classes: {list(art.label_mapping.values())}")

# ── MAIN TITLE ─────────────────────────────────────────────
st.title("🏢 PIRVision Smart Occupancy System")
st.subheader("AI-powered real-time building intelligence")

# ── DESCRIPTION ────────────────────────────────────────────
st.markdown("""
PIRVision uses 55-channel PIR sensor data and machine learning to detect occupancy states:

- 🟢 **Vacancy** → Systems OFF  
- 🔵 **Stationary** → Eco Mode  
- 🔴 **Motion** → Full Power  
""")

# ── KPI DASHBOARD (REAL DATA) ──────────────────────────────
summary = art.summary or {}
energy = summary.get("energy_savings", {})

best_model = summary.get("best_baseline_model", type(art.model).__name__)
f1_score = summary.get("tuned_xgb_f1", 0.0) * 100
annual_savings = energy.get("annual_savings_USD", 0.0)
annual_co2 = energy.get("annual_CO2_saved_kg", 0.0)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Best Model", best_model)
col2.metric("F1 Score", f"{f1_score:.1f}%")
col3.metric("Annual Savings", f"${annual_savings:,.0f}")
col4.metric("CO₂ Saved / year", f"{annual_co2:,.0f} kg")

# ── EXTRA INFO ─────────────────────────────────────────────
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### What this system does")
    st.write(
        "Detects occupancy using PIR sensors and AI, enabling smart control of "
        "HVAC, lighting, and building systems."
    )

with col2:
    st.markdown("### Real-world impact")
    st.write(
        "Reduces energy waste, lowers carbon emissions, and improves building efficiency."
    )

st.divider()

st.info("👉 Use sidebar pages for live prediction, heatmaps, and analytics.")
