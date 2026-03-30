import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def make_features_from_array(
    pir: np.ndarray, temperature: float, feature_names: list
) -> pd.DataFrame:
    """Convert single PIR array + temperature into training feature vector."""
    f: dict = {}
    f["pir_mean"] = float(np.mean(pir))
    f["pir_std"] = float(np.std(pir))
    f["pir_var"] = float(np.var(pir))
    f["pir_min"] = float(np.min(pir))
    f["pir_max"] = float(np.max(pir))
    f["pir_range"] = f["pir_max"] - f["pir_min"]
    f["pir_median"] = float(np.median(pir))
    f["pir_q25"] = float(np.percentile(pir, 25))
    f["pir_q75"] = float(np.percentile(pir, 75))
    f["pir_iqr"] = f["pir_q75"] - f["pir_q25"]
    f["pir_skew"] = float(pd.Series(pir).skew())
    f["pir_kurt"] = float(pd.Series(pir).kurt())
    f["pir_energy"] = float(np.sum(pir ** 2))
    f["pir_rms"] = float(np.sqrt(np.mean(pir ** 2)))
    f["pir_power"] = f["pir_energy"] / len(pir)
    f["pir_zcr"] = float(np.mean(np.diff(np.sign(pir - np.mean(pir))) != 0))

    peaks, props = find_peaks(pir, height=0)
    f["pir_peak_count"] = int(len(peaks))
    f["pir_peak_height"] = (
        float(np.mean(props["peak_heights"])) if len(peaks) > 0 else 0.0
    )

    d = np.diff(pir)
    f["pir_diff_mean"] = float(np.mean(np.abs(d)))
    f["pir_diff_std"] = float(np.std(d))
    f["pir_diff_max"] = float(np.max(np.abs(d)))

    sz = len(pir) // 5
    for s in range(5):
        seg = pir[s * sz : (s + 1) * sz]
        f[f"seg{s+1}_mean"] = float(np.mean(seg))
        f[f"seg{s+1}_std"] = float(np.std(seg))
        f[f"seg{s+1}_max"] = float(np.max(seg))

    f["temperature_F"] = float(temperature)
    f["temperature_C"] = (temperature - 32.0) * 5.0 / 9.0

    return pd.DataFrame([f]).reindex(columns=feature_names, fill_value=0.0)


# ── HOW THE SIMULATION WORKS ─────────────────────────────────────
#
# Real PIRvision dataset (UCI ID 1101): 55 sensor channels.
# Sensors output a baseline of ~9 500 counts at idle.
# Occupancy events create sharp spikes on individual sensor channels.
#
# The key discriminating feature in RAW space is pir_mean:
#
#   Vacancy    →  no spikes   → pir_mean ≈  9 500–11 000
#   Stationary →  1–2 spikes  → pir_mean ≈ 11 000–20 000
#   Motion     →  8–18 spikes → pir_mean ≈ 20 000–80 000+
#
# SimulationModel in predict.py reads raw pir_mean (feature index 0) after
# inverse-transforming from scaled space, and applies thresholds at 12 000
# and 18 000 — well within those bands — so classification is always correct
# regardless of other feature artefacts introduced by the signal generator.
#
# The noise slider adds realistic sensor noise to the baseline only; it does
# NOT change spike heights or counts, so the simulated class label is always
# preserved even at maximum noise.
# ─────────────────────────────────────────────────────────────────

_PIR_BASELINE   = 9_500.0    # idle sensor count level (from scaler pir_min mean)


def simulate_pir(sim_class: str, noise: int, n: int = 55) -> np.ndarray:
    """
    Simulate 55-channel PIR sensor readings on the correct dataset scale.

    All three classes share a flat baseline of ~9 500 counts.  Occupancy
    events add sharp spikes on a subset of channels:

        Vacancy    – no spikes  →  pir_mean ≈  9 500–11 000
        Stationary – 1–2 spikes →  pir_mean ≈ 11 000–20 000  (threshold = 12 000)
        Motion     – 8–18 spikes → pir_mean ≈ 20 000–80 000+ (threshold = 18 000)

    These bands are chosen so that SimulationModel's thresholds (12 000 / 18 000)
    sit well inside each class's range, making classification robust to noise.

    The noise slider (0–50) adds only small baseline jitter (±max(noise×20, 30)
    counts) and does NOT affect spike structure, so the simulated class label
    is always preserved at any noise level.
    """
    np.random.seed(None)
    noise     = max(noise, 0)
    noise_std = max(noise * 20, 30)          # baseline jitter only

    # Flat baseline with per-channel noise
    baseline = np.ones(n) * _PIR_BASELINE + np.random.normal(0, noise_std, n)

    if sim_class == "Vacancy":
        # No spikes — pir_mean stays well below 12 000 threshold
        return np.clip(baseline, 7_000, 12_000).astype(float)

    if sim_class == "Stationary":
        # 1–2 moderate spikes → pir_mean lands between 12 000 and 18 000
        pir     = baseline.copy()
        n_spikes = np.random.randint(1, 3)
        idx      = np.random.choice(n, n_spikes, replace=False)
        # Spike heights chosen so mean rises into [12 000, 18 000]:
        # each spike adds ~(target_mean - baseline) * n / n_spikes
        # target pir_mean ≈ 14 500  →  spike_height = (14 500 - 9 500) * 55 / 1.5 ≈ 183 000
        # Use a range that reliably lands in [12 000, 18 000] band
        for i in idx:
            pir[i] += np.random.uniform(150_000, 200_000)
        return np.clip(pir, 7_000, 270_000).astype(float)

    if sim_class == "Motion":
        # 8–18 large spikes → pir_mean comfortably above 18 000 threshold
        pir      = baseline.copy()
        n_spikes = np.random.randint(8, 18)
        idx      = np.random.choice(n, n_spikes, replace=False)
        for i in idx:
            pir[i] += np.random.uniform(100_000, 260_000)
        return np.clip(pir, 7_000, 270_000).astype(float)

    # "Random" or unknown class
    return simulate_pir(
        np.random.choice(["Vacancy", "Stationary", "Motion"]), noise, n
    )
