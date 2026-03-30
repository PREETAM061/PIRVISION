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


# ---------------------------------------------------------------------------
# Real PIRvision dataset (UCI ID 1101) sensor characteristics (from scaler):
#   pir_min    mean =   8,561  scale =     770   → all sensors idle at ~8500-10000
#   pir_median mean =  10,395  scale =     113   → median always near baseline
#   pir_max    mean = 263,028  scale = 4,373,648 → key discriminator: spike height
#   pir_peak_count  mean = 6.0 scale = 1.2       → exactly 6 gentle baseline peaks
#   pir_peak_height mean = 10,893 scale = 388    → Vacancy peaks at ~10,893
#
# Pattern: 55 sensors all output ~9,000-10,500 at idle (baseline).
# Occupancy adds sharp spikes on individual sensors:
#   Vacancy:    no spikes  → pir_max ≈ 11,000
#   Stationary: 1-2 spikes → pir_max ≈ 60,000-150,000
#   Motion:     8-18 spikes → pir_max ≈ 100,000-262,000
#
# The noise slider controls only baseline sensor noise (not spike structure),
# so class identity is always determined by sim_class, not noise level.
# ---------------------------------------------------------------------------

_PEAK_POSITIONS = None   # computed once on first call


def _make_baseline(n: int = 55, noise_std: float = 30.0) -> np.ndarray:
    """
    Smooth baseline with exactly 6 Gaussian bumps that match the real dataset:
      - peak_count  ≈ 6   (real mean = 6.0, scale = 1.2)
      - peak_height ≈ 10,893  (real mean = 10,893, scale = 388)
    """
    global _PEAK_POSITIONS
    if _PEAK_POSITIONS is None or len(_PEAK_POSITIONS) != 6:
        _PEAK_POSITIONS = np.round(np.linspace(4, n - 5, 6)).astype(int)

    base = np.ones(n) * 9200.0
    for pos in _PEAK_POSITIONS:
        bump = 1700.0 * np.exp(-0.5 * ((np.arange(n) - pos) / 3.0) ** 2)
        base += bump
    return base + np.random.normal(0, noise_std, n)


def simulate_pir(sim_class: str, noise: int, n: int = 55) -> np.ndarray:
    """
    Simulate PIR readings on the correct real-dataset scale.

    The noise slider (0-50) adds only small baseline sensor noise (~20-95 counts).
    This ensures the class label — not noise — always determines the prediction.

    Previous (broken) implementation used pir_mean=2/40/180, which is orders of
    magnitude below the real sensor range (~9,000-262,000). After StandardScaler
    transform all three classes produced z-scores within 0.001 of each other →
    the model could not distinguish them at all.
    """
    np.random.seed(None)
    noise = max(noise, 0)
    # Noise slider (0-50) → baseline std-dev of 20-95 counts only
    noise_std = max(noise * 1.5, 20.0)

    if sim_class == "Vacancy":
        # No occupancy: all 55 sensors stay near idle baseline, no spikes.
        pir = _make_baseline(n, noise_std)
        return np.clip(pir, 7_000, 13_000).astype(float)

    if sim_class == "Stationary":
        # One person sitting: 1-2 sensors spike moderately (single detection events).
        pir = _make_baseline(n, noise_std)
        n_spikes = np.random.randint(1, 3)
        positions = np.random.choice(n, n_spikes, replace=False)
        for pos in positions:
            pir[pos] += np.random.uniform(60_000, 150_000)
        return np.clip(pir, 7_000, 270_000).astype(float)

    if sim_class == "Motion":
        # Active movement: 8-18 sensors spiked to high counts repeatedly.
        pir = _make_baseline(n, noise_std)
        n_spikes = np.random.randint(8, 18)
        positions = np.random.choice(n, n_spikes, replace=False)
        for pos in positions:
            pir[pos] += np.random.uniform(100_000, 260_000)
        return np.clip(pir, 7_000, 270_000).astype(float)

    # "Random" or unknown: pick a class at random
    return simulate_pir(
        np.random.choice(["Vacancy", "Stationary", "Motion"]), noise, n
    )
