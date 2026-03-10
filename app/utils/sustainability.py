LEED_THRESHOLDS = {
    "Platinum": 90,
    "Gold": 75,
    "Silver": 60,
    "Certified": 40,
}

ESG_WEIGHTS = {
    "Environmental": 0.50,
    "Social": 0.30,
    "Governance": 0.20,
}


def calculate_leed_score(
    occupancy_accuracy: float,
    energy_waste_pct: float,
    avg_savings_pct: float,
    uptime_pct: float = 99.0,
) -> dict:
    """
    Calculate a LEED-style Green Building Score 0-100.
    """
    accuracy_score = min(max(occupancy_accuracy, 0.0), 100.0) * 0.35
    waste_score = max(0.0, (100.0 - max(energy_waste_pct, 0.0))) * 0.30
    savings_score = min(max(avg_savings_pct, 0.0), 100.0) * 0.25
    uptime_score = min(max(uptime_pct, 0.0), 100.0) * 0.10
    total = accuracy_score + waste_score + savings_score + uptime_score

    rating = "Not Certified"
    for level, threshold in LEED_THRESHOLDS.items():
        if total >= threshold:
            rating = f"LEED {level}"
            break

    next_level = None
    for level, threshold in LEED_THRESHOLDS.items():
        if total < threshold:
            next_level = {
                "level": f"LEED {level}",
                "points_needed": round(threshold - total, 1),
            }
            break

    return {
        "total_score": round(total, 1),
        "rating": rating,
        "breakdown": {
            "Occupancy Accuracy (35%)": round(accuracy_score, 1),
            "Energy Efficiency (30%)": round(waste_score, 1),
            "Savings Achievement (25%)": round(savings_score, 1),
            "System Uptime (10%)": round(uptime_score, 1),
        },
        "next_level": next_level,
    }


def _grade(score: float) -> str:
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B+"
    if score >= 60:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def calculate_esg_score(
    energy_efficiency: float,
    carbon_reduction: float,
    comfort_score: float,
    compliance_pct: float,
) -> dict:
    """
    Calculate ESG (Environmental, Social, Governance) scores.
    """
    E_score = energy_efficiency * 0.5 + carbon_reduction * 0.5
    S_score = comfort_score
    G_score = compliance_pct

    overall = (
        E_score * ESG_WEIGHTS["Environmental"]
        + S_score * ESG_WEIGHTS["Social"]
        + G_score * ESG_WEIGHTS["Governance"]
    )

    return {
        "Environmental": {
            "score": round(E_score, 1),
            "grade": _grade(E_score),
            "components": {
                "Energy Efficiency": energy_efficiency,
                "Carbon Reduction": carbon_reduction,
            },
        },
        "Social": {
            "score": round(S_score, 1),
            "grade": _grade(S_score),
            "components": {"Occupant Comfort": comfort_score},
        },
        "Governance": {
            "score": round(G_score, 1),
            "grade": _grade(G_score),
            "components": {"Policy Compliance": compliance_pct},
        },
        "Overall": {"score": round(overall, 1), "grade": _grade(overall)},
    }


def calculate_goals_progress(
    current_savings_pct: float,
    current_co2_kg: float,
    current_accuracy: float,
    targets: dict | None = None,
) -> list:
    """
    Calculate progress toward sustainability goals.
    Returns list of goal dicts with progress bars.
    """
    if targets is None:
        targets = {
            "Energy Reduction": {
                "target": 30.0,
                "unit": "%",
                "current": current_savings_pct,
            },
            "CO2 Saved": {
                "target": 500.0,
                "unit": "kg",
                "current": current_co2_kg,
            },
            "Prediction Accuracy": {
                "target": 95.0,
                "unit": "%",
                "current": current_accuracy,
            },
        }

    goals: list = []
    for name, data in targets.items():
        target_val = max(data.get("target", 0.0), 1e-6)
        current_val = max(data.get("current", 0.0), 0.0)
        pct_done = min((current_val / target_val) * 100.0, 100.0)
        goals.append(
            {
                "name": name,
                "current": current_val,
                "target": target_val,
                "unit": data.get("unit", ""),
                "pct_done": round(pct_done, 1),
                "achieved": pct_done >= 100.0,
                "status": "Achieved"
                if pct_done >= 100.0
                else f"{pct_done:.0f}% complete",
            }
        )
    return goals

