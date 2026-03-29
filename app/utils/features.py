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


# Real PIRvision dataset (UCI ID 1101): 55 sensors with baseline ~9500 counts.
# Occupancy creates spikes on individual sensors (up to ~262,000 counts).
# The scaler was fit on this real scale. Correct ranges derived from scaler params:
#   pir_min  mean=8,561  → sensors idle at ~8,500-10,000
#   pir_max  mean=263,028 → peak spikes up to ~262,000
#   pir_median mean=10,395 → most sensors stay near baseline even during motion
_PIR_BASELINE = 9_500.0   # idle sensor count level


def simulate_pir(sim_class: str, noise: int, n: int = 55) -> np.ndarray:
    """Simulate PIR readings on the correct real-dataset scale.

    Real sensors output ~9,500 counts at baseline.  Occupancy adds spikes on
    a subset of sensors.  Old code used mean=2/40/180 which is orders-of-
    magnitude wrong → all three classes produced identical z-scores after
    StandardScaler transform → model always predicted the same class.
    """
    np.random.seed(None)
    noise = max(noise, 0)
    noise_counts = max(noise * 50, 100)   # noise slider → real-unit std-dev

    if sim_class == "Vacancy":
        # No occupancy: all 55 sensors stay near idle baseline, no spikes.
        pir = np.ones(n) * _PIR_BASELINE + np.random.normal(0, noise_counts, n)
        return np.clip(pir, 7_000, 12_000)

    if sim_class == "Stationary":
        # 1 person sitting still: 2-5 sensors pick up a moderate heat spike.
        pir = np.ones(n) * _PIR_BASELINE + np.random.normal(0, noise_counts, n)
        n_spikes = np.random.randint(2, 6)
        idx = np.random.choice(n, n_spikes, replace=False)
        pir[idx] += np.random.uniform(15_000, 70_000, n_spikes)
        return np.clip(pir, 7_000, 200_000)

    if sim_class == "Motion":
        # Active movement: 10-30 sensors spiked to high counts.
        pir = np.ones(n) * _PIR_BASELINE + np.random.normal(0, noise_counts * 2, n)
        n_spikes = np.random.randint(10, 30)
        idx = np.random.choice(n, n_spikes, replace=False)
        pir[idx] += np.random.uniform(80_000, 260_000, n_spikes)
        return np.clip(pir, 7_000, 270_000)

    return simulate_pir(
        np.random.choice(["Vacancy", "Stationary", "Motion"]), noise, n
    )
