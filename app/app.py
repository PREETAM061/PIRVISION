# app/app.py
import streamlit as st
import joblib
import json
import os

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="PIRVision — Smart Occupancy",
    page_icon="🏢",
    layout="wide"
)

# ── Path resolution ─────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# ── Load models ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    required_files = [
        "best_xgb_model.pkl",
        "feature_scaler.pkl",
        "feature_names.json",
        "label_mapping.json"
    ]

    missing = [f for f in required_files if not os.path.exists(os.path.join(MODELS_DIR, f))]

    if missing:
        st.error(f"❌ Missing files: {missing}")
        st.error(f"Expected in: {MODELS_DIR}")
        st.stop()

    model = joblib.load(os.path.join(MODELS_DIR, "best_xgb_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    with open(os.path.join(MODELS_DIR, "label_mapping.json")) as f:
        raw = json.load(f)
        label_map = {int(k): v for k, v in raw["decode"].items()}

    return model, scaler, feature_names, label_map


# ── Load into session ───────────────────────────────────────
model, scaler, feature_names, label_map = load_models()

st.session_state["model"] = model
st.session_state["scaler"] = scaler
st.session_state["feature_names"] = feature_names
st.session_state["label_map"] = label_map

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("🏢 PIRVision")
    st.caption("Smart Occupancy Detection")

    model_name = type(model).__name__
    if model_name == "XGBClassifier":
        st.success(f"✅ Model: {model_name}")
    else:
        st.warning(f"⚠️ Model: {model_name}")

    st.write(f"Features: {len(feature_names)}")
    st.write(f"Classes: {list(label_map.values())}")

# ── Main UI ────────────────────────────────────────────────
st.title("🏢 PIRVision Smart Occupancy System")
st.subheader("AI-based real-time occupancy detection")

st.markdown("""
- 🟢 Vacancy → All systems OFF  
- 🔵 Stationary → Eco Mode  
- 🔴 Motion → Full Power  
""")

# KPI
c1, c2, c3 = st.columns(3)
c1.metric("Sensors", "55")
c2.metric("Model", "XGBoost")
c3.metric("Status", "Active")

st.divider()
st.info("Go to prediction page to test real model.")
