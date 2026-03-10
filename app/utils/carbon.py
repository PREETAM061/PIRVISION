CO2_PER_KWH = 0.233  # kg CO2 per kWh (US average)
CO2_PER_TREE = 21.77  # kg CO2 absorbed per tree per year
CO2_PER_KM_CAR = 0.250  # kg CO2 per km driven (average car)
CO2_PER_FLIGHT_LON_NYC = 790  # kg CO2 per round trip London-NYC
CO2_HOME_MONTHLY = 567  # kg CO2 per month for average home


def kwh_to_co2(kwh: float) -> float:
    """Convert kWh to kg CO2."""
    return round(kwh * CO2_PER_KWH, 3)


def co2_to_equivalents(co2_kg: float) -> dict:
    """
    Convert CO2 saved into human-understandable equivalents.
    Used by CO2 Impact Visualizer page.
    """
    return {
        "trees_planted": round(co2_kg / CO2_PER_TREE, 1) if CO2_PER_TREE > 0 else 0.0,
        "car_km_avoided": round(co2_kg / CO2_PER_KM_CAR, 0)
        if CO2_PER_KM_CAR > 0
        else 0.0,
        "flights_avoided": round(co2_kg / CO2_PER_FLIGHT_LON_NYC, 2)
        if CO2_PER_FLIGHT_LON_NYC > 0
        else 0.0,
        "home_months_saved": round(co2_kg / CO2_HOME_MONTHLY, 2)
        if CO2_HOME_MONTHLY > 0
        else 0.0,
        "lightbulb_hours": round(co2_kg / (0.01 * CO2_PER_KWH), 0)
        if CO2_PER_KWH > 0
        else 0.0,
        "co2_kg": round(co2_kg, 1),
    }


def get_carbon_status(co2_rate_kg_hr: float, target_kg_hr: float = 0.5) -> dict:
    """
    Determine carbon status color and label vs target.
    Used by real-time Carbon Tracker.
    """
    if target_kg_hr <= 0:
        pct = 100.0
    else:
        pct = (co2_rate_kg_hr / target_kg_hr) * 100.0

    pct_rounded = round(pct, 1)

    if pct <= 70:
        return {
            "status": "Below Target",
            "color": "#2ecc71",
            "emoji": "🟢",
            "pct": pct_rounded,
        }
    if pct <= 100:
        return {
            "status": "Near Target",
            "color": "#f39c12",
            "emoji": "🟡",
            "pct": pct_rounded,
        }
    return {
        "status": "Over Budget",
        "color": "#e74c3c",
        "emoji": "🔴",
        "pct": pct_rounded,
    }


def project_co2_savings(daily_savings_kwh: float, years: int = 10) -> list:
    """
    Project cumulative CO2 savings over N years.
    Returns list of (year, cumulative_co2_kg) dicts.
    """
    annual = daily_savings_kwh * 365.0 * CO2_PER_KWH
    return [
        {"year": y, "cumulative_co2_kg": round(annual * y, 1)}
        for y in range(1, max(years, 1) + 1)
    ]

