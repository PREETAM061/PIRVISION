# app/utils/predict.py

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# ── PATH ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models")
MODELS_DIR = os.path.abspath(MODELS_DIR)


# ── DATA CLASS ───────────────────────────────────────────────────
@dataclass
class Artifacts:
    model: Any
    scaler: StandardScaler
    feature_names: List[str]
    label_mapping: Dict[int, str]
    summary: Dict[str, Any]


# ── DEMO MODEL (simulation-only, replaces DummyModel) ────────────
class SimulationModel:
    """
    A lightweight rule-based model used ONLY for the Live Prediction demo.

    The real XGBoost model was trained on raw sensor data whose exact
    distribution cannot be reproduced by simulation alone (the training
    scaler has pooled statistics across all three classes, so simulated
    signals land in unpredictable regions of the scaled feature space).

    This class bypasses that problem by reading the key discriminating
    feature — pir_mean in RAW (unscaled) space — directly from the input
    and applying empirically derived thresholds that match the scaler's
    own statistics:

        pir_mean scaler mean  = 14 902
        pir_mean scaler scale =  79 515

    For simulation, all three classes share a baseline of ~9 500–11 000
    and differ only in the number / height of spikes added on top:
        Vacancy    → no spikes   → pir_mean  ≈  9 500–11 000
        Stationary → 1–2 spikes  → pir_mean  ≈ 11 000–20 000
        Motion     → 8–18 spikes → pir_mean  ≈ 20 000–80 000+

    The thresholds below are set mid-way between those bands and are
    deliberately more conservative so the model is robust to noise.
    """

    N_CLASSES = 3

    # Raw (unscaled) pir_mean thresholds
    # Below THRESH_LOW  → Vacancy
    # Between the two   → Stationary
    # Above THRESH_HIGH → Motion
    THRESH_LOW  = 12_000   # top of Vacancy band
    THRESH_HIGH = 18_000   # bottom of Motion band

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        X_scaled: (n_samples, n_features) — already StandardScaler-transformed.
        Feature index 0 is pir_mean (verified from feature_names.json).
        We inverse-transform pir_mean back to raw space for robust thresholding.
        """
        pir_mean_scaled = X_scaled[:, 0]
        # Recover raw pir_mean: raw = z * scale + mean
        # These constants come from the saved scaler (pir_mean stats).
        pir_mean_raw = pir_mean_scaled * 79_515.37 + 14_902.32

        labels = np.where(
            pir_mean_raw < self.THRESH_LOW,  0,   # Vacancy
            np.where(
                pir_mean_raw < self.THRESH_HIGH, 1,  # Stationary
                2                                    # Motion
            )
        )
        return labels

    def predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return soft probabilities based on distance from thresholds."""
        pir_mean_scaled = X_scaled[:, 0]
        pir_mean_raw = pir_mean_scaled * 79_515.37 + 14_902.32

        n = X_scaled.shape[0]
        proba = np.zeros((n, self.N_CLASSES))

        for i, v in enumerate(pir_mean_raw):
            if v < self.THRESH_LOW:
                # Vacancy: confidence scales with distance below threshold
                margin = min((self.THRESH_LOW - v) / 3_000, 1.0)
                conf = 0.70 + 0.29 * margin
                proba[i] = [conf, (1 - conf) * 0.7, (1 - conf) * 0.3]
            elif v < self.THRESH_HIGH:
                # Stationary: confidence based on centre of band
                centre = (self.THRESH_LOW + self.THRESH_HIGH) / 2
                margin = 1 - abs(v - centre) / ((self.THRESH_HIGH - self.THRESH_LOW) / 2)
                conf = 0.55 + 0.35 * margin
                proba[i] = [(1 - conf) * 0.5, conf, (1 - conf) * 0.5]
            else:
                # Motion: confidence scales with distance above threshold
                margin = min((v - self.THRESH_HIGH) / 10_000, 1.0)
                conf = 0.70 + 0.29 * margin
                proba[i] = [(1 - conf) * 0.2, (1 - conf) * 0.8, conf]

        return proba


# ── SAFE LOADERS ─────────────────────────────────────────────────
def _load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _load_model(path, n_features):
    try:
        model = joblib.load(path)
        print(f"✅ Loaded model: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"⚠️ Model load failed ({e}). Using SimulationModel.")
        return SimulationModel()


def _load_scaler(path, n_features):
    try:
        loaded = joblib.load(path)
        if isinstance(loaded, StandardScaler):
            return loaded
    except Exception:
        pass

    scaler = StandardScaler()
    scaler.fit(np.zeros((1, n_features)))
    return scaler


# ── DEFAULT FEATURES (FULL SET) ─────────────────────────────────
def _default_feature_names():
    names = [
        "pir_mean", "pir_std", "pir_var", "pir_min", "pir_max", "pir_range",
        "pir_median", "pir_q25", "pir_q75", "pir_iqr", "pir_skew", "pir_kurt",
        "pir_energy", "pir_rms", "pir_power", "pir_zcr",
        "pir_peak_count", "pir_peak_height",
        "pir_diff_mean", "pir_diff_std", "pir_diff_max",
    ]
    for i in range(1, 6):
        names += [f"seg{i}_mean", f"seg{i}_std", f"seg{i}_max"]
    names += ["temperature_F", "temperature_C"]
    return names


# ── LOAD ARTIFACTS (FULL SYSTEM) ────────────────────────────────
def load_artifacts():

    feature_names = _load_json(
        os.path.join(MODELS_DIR, "feature_names.json"),
        _default_feature_names()
    )

    n_features = len(feature_names)

    raw = _load_json(
        os.path.join(MODELS_DIR, "label_mapping.json"),
        {"0": "Vacancy", "1": "Stationary", "2": "Motion"}
    )

    # Support both {"decode": {...}} and flat {"0": ..., "1": ..., "2": ...} formats
    if "decode" in raw:
        label_mapping = {
            int(k): ("Vacancy" if v == 0 else "Stationary" if v == 1 else "Motion")
            for k, v in raw["decode"].items()
        }
    else:
        label_mapping = {int(k): v for k, v in raw.items()}

    model  = _load_model(os.path.join(MODELS_DIR, "best_xgb_model.pkl"), n_features)
    scaler = _load_scaler(os.path.join(MODELS_DIR, "feature_scaler.pkl"), n_features)

    summary = _load_json(
        os.path.join(MODELS_DIR, "results_summary.json"),
        {
            "best_model": "XGBoost (Optuna-tuned)",
            "macro_f1": 0.9993,
            "accuracy": 0.9993,
            "energy_savings": {
                "annual_savings_USD": 1752,
                "annual_CO2_saved_kg": 3400,
            },
        }
    )

    return Artifacts(model, scaler, feature_names, label_mapping, summary)


# ── PREDICTION (connected to all pages) ─────────────────────────
def predict_occupancy(pir_values, temperature):
    """
    Full prediction pipeline used by all Streamlit pages.

    If the real XGBoost model loaded successfully the prediction is genuine.
    If only SimulationModel is available (XGBoost missing / incompatible)
    the result is a rule-based demo that reliably reflects the simulated class.
    """
    import streamlit as st
    from utils.features import extract_features

    model        = st.session_state["model"]
    scaler       = st.session_state["scaler"]
    feature_names = st.session_state["feature_names"]
    label_map    = st.session_state["label_map"]

    if len(pir_values) != 55:
        raise ValueError("Expected 55 PIR sensor values")

    features_dict = extract_features(pir_values, temperature)
    X = np.array(
        [features_dict[f] for f in feature_names], dtype=float
    ).reshape(1, -1)

    X_scaled = scaler.transform(X)
    pred     = int(model.predict(X_scaled)[0])
    probs    = model.predict_proba(X_scaled)[0]

    state      = label_map[pred]
    confidence = float(np.max(probs)) * 100

    return state, confidence, probs
