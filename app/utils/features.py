import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def make_features_from_array(
    pir: np.ndarray, temperature: float, feature_names: list
) -> pd.DataFrame:
    """Convert single PIR array + temperature into training feature vector."""
    f: dict = {}
    f["pir_mean"]   = float(np.mean(pir))
    f["pir_std"]    = float(np.std(pir))
    f["pir_var"]    = float(np.var(pir))
    f["pir_min"]    = float(np.min(pir))
    f["pir_max"]    = float(np.max(pir))
    f["pir_range"]  = f["pir_max"] - f["pir_min"]
    f["pir_median"] = float(np.median(pir))
    f["pir_q25"]    = float(np.percentile(pir, 25))
    f["pir_q75"]    = float(np.percentile(pir, 75))
    f["pir_iqr"]    = f["pir_q75"] - f["pir_q25"]
    f["pir_skew"]   = float(pd.Series(pir).skew())
    f["pir_kurt"]   = float(pd.Series(pir).kurt())
    f["pir_energy"] = float(np.sum(pir ** 2))
    f["pir_rms"]    = float(np.sqrt(np.mean(pir ** 2)))
    f["pir_power"]  = f["pir_energy"] / len(pir)
    f["pir_zcr"]    = float(np.mean(np.diff(np.sign(pir - np.mean(pir))) != 0))

    peaks, props = find_peaks(pir, height=0)
    f["pir_peak_count"]  = int(len(peaks))
    f["pir_peak_height"] = (
        float(np.mean(props["peak_heights"])) if len(peaks) > 0 else 0.0
    )

    d = np.diff(pir)
    f["pir_diff_mean"] = float(np.mean(np.abs(d)))
    f["pir_diff_std"]  = float(np.std(d))
    f["pir_diff_max"]  = float(np.max(np.abs(d)))

    sz = len(pir) // 5
    for s in range(5):
        seg = pir[s * sz : (s + 1) * sz]
        f[f"seg{s+1}_mean"] = float(np.mean(seg))
        f[f"seg{s+1}_std"]  = float(np.std(seg))
        f[f"seg{s+1}_max"]  = float(np.max(seg))

    f["temperature_F"] = float(temperature)
    f["temperature_C"] = (temperature - 32.0) * 5.0 / 9.0

    return pd.DataFrame([f]).reindex(columns=feature_names, fill_value=0.0)


_PIR_BASELINE = 9_500.0


def simulate_pir(sim_class: str, noise: int, n: int = 55) -> np.ndarray:
    """
    Simulate PIR readings matching REAL dataset distributions.

    Real dataset facts (from actual CSV):
      Vacancy    mean=10,347 std=667   max=16,186  temp=87F
      Stationary mean=10,315 std=486   max=12,494  temp=86F
      Motion     mean=49,947 std=293k  max=111M    temp=0F (sensor fault)

    IMPORTANT: Call get_temperature_for_class() to get the correct
    temperature to pass alongside this PIR array.
    """
    np.random.seed(None)
    noise = max(noise, 0)
    # Always add at least 200 baseline noise counts (real Vacancy std ~667)
    noise_counts = 200 + noise * 50

    if sim_class == "Vacancy":
        pir = np.ones(n) * _PIR_BASELINE + np.random.normal(0, noise_counts, n)
        return np.clip(pir, 7_000, 16_000)

    if sim_class == "Stationary":
        pir = np.ones(n) * _PIR_BASELINE + np.random.normal(0, noise_counts, n)
        n_spikes = np.random.randint(1, 4)
        idx = np.random.choice(n, n_spikes, replace=False)
        pir[idx] += np.random.uniform(15_000, 50_000, n_spikes)
        return np.clip(pir, 7_000, 70_000)

    if sim_class == "Motion":
        pir = np.ones(n) * _PIR_BASELINE + np.random.normal(0, noise_counts * 3, n)
        # PIR_1 (entrance sensor) always has the biggest spike
        pir[0] += np.random.uniform(500_000, 5_000_000)
        n_extra = np.random.randint(5, 15)
        idx = np.random.choice(range(1, n), n_extra, replace=False)
        pir[idx] += np.random.uniform(50_000, 200_000, n_extra)
        return np.clip(pir, 7_000, 6_000_000)

    return simulate_pir(
        np.random.choice(["Vacancy", "Stationary", "Motion"]), noise, n
    )


def get_temperature_for_class(sim_class: str, slider_temp_f: float) -> float:
    """
    Return the correct temperature for make_features_from_array().

    The model was trained on real data where ALL Motion recordings
    have temperature = 0F (sensor fault). So:
        Motion     -> always return 0.0
        Vacancy    -> use slider value (range 84-89F)
        Stationary -> use slider value (range 84-89F)
    """
    if sim_class == "Motion":
        return 0.0
    return float(slider_temp_f)
