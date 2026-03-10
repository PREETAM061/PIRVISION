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


def simulate_pir(sim_class: str, noise: int, n: int = 55) -> np.ndarray:
    """Simulate PIR readings calibrated to match real dataset distributions."""
    np.random.seed(None)
    noise = max(noise, 0)

    if sim_class == "Vacancy":
        pir = np.ones(n) * 2 + np.random.normal(0, max(noise * 0.05, 0.1), n)
        return np.clip(pir, 0, 10)
    if sim_class == "Stationary":
        pir = np.ones(n) * 40 + np.random.normal(0, max(noise * 0.3, 2), n)
        return np.clip(pir, 20, 80)
    if sim_class == "Motion":
        pir = np.ones(n) * 180 + np.random.normal(0, max(noise * 1.5, 30), n)
        return np.clip(np.abs(pir), 80, 300)

    return simulate_pir(
        np.random.choice(["Vacancy", "Stationary", "Motion"]), noise, n
    )

