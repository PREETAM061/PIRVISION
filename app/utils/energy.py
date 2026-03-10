DEVICE_PROFILES = {
    "HVAC": {"Vacancy": 0, "Stationary": 800, "Motion": 1200, "Baseline": 1200},
    "Lighting": {"Vacancy": 0, "Stationary": 100, "Motion": 200, "Baseline": 200},
    "Ventilation": {"Vacancy": 50, "Stationary": 200, "Motion": 400, "Baseline": 400},
    "Screens": {"Vacancy": 0, "Stationary": 150, "Motion": 300, "Baseline": 300},
    "Elevators": {"Vacancy": 100, "Stationary": 100, "Motion": 500, "Baseline": 500},
}

ACTIONS = {
    "Vacancy": {
        "HVAC": "OFF",
        "Lighting": "OFF",
        "Ventilation": "15%",
        "Screens": "Sleep",
        "Elevators": "Standby",
    },
    "Stationary": {
        "HVAC": "Eco 67%",
        "Lighting": "Dim 50%",
        "Ventilation": "50%",
        "Screens": "Low",
        "Elevators": "Standby",
    },
    "Motion": {
        "HVAC": "Full",
        "Lighting": "Full",
        "Ventilation": "Full",
        "Screens": "Active",
        "Elevators": "Active",
    },
}

ICONS = {
    "HVAC": "🌡️",
    "Lighting": "💡",
    "Ventilation": "💨",
    "Screens": "🖥️",
    "Elevators": "🛗",
}

COMFORT_TEMP = {"Vacancy": 28, "Stationary": 23, "Motion": 21}


def calculate_savings(
    state: str, rooms: int = 1, hours: float = 10.0, price_kwh: float = 0.12
) -> dict:
    baseline = sum(p["Baseline"] for p in DEVICE_PROFILES.values())
    optimized = sum(p.get(state, p["Baseline"]) for p in DEVICE_PROFILES.values())
    rooms = max(rooms, 1)

    baseline_total = baseline * rooms
    optimized_total = optimized * rooms
    savings_w = baseline_total - optimized_total
    kwh_day = (savings_w / 1000.0) * hours

    return {
        "baseline_W": baseline_total,
        "optimized_W": optimized_total,
        "savings_W": savings_w,
        "savings_pct": round(
            (savings_w / baseline_total) * 100.0, 1
        )
        if baseline_total > 0
        else 0.0,
        "savings_kwh_day": round(kwh_day, 3),
        "savings_usd_day": round(kwh_day * price_kwh, 2),
        "savings_usd_year": round(kwh_day * price_kwh * 365.0, 2),
        "co2_year_kg": round(kwh_day * 0.233 * 365.0, 1),
        "actions": ACTIONS.get(state, ACTIONS["Vacancy"]),
        "comfort_temp_C": COMFORT_TEMP.get(state, COMFORT_TEMP["Stationary"]),
        "device_breakdown": {
            dev: DEVICE_PROFILES[dev].get(state, DEVICE_PROFILES[dev]["Baseline"])
            for dev in DEVICE_PROFILES
        },
    }

