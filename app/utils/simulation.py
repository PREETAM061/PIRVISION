import numpy as np
import pandas as pd

ROOM_NAMES = [
    "Lobby",
    "Reception",
    "Meeting A",
    "Meeting B",
    "Meeting C",
    "Office 1",
    "Office 2",
    "Office 3",
    "Office 4",
    "Office 5",
    "Kitchen",
    "Server Rm",
    "HR Dept",
    "Finance",
    "Marketing",
    "Exec Suite",
    "Board Rm",
    "IT Dept",
    "Warehouse",
    "Parking",
]

STATES = ["Vacancy", "Stationary", "Motion"]
STATE_COLOR = {"Vacancy": "#2ecc71", "Stationary": "#3498db", "Motion": "#e74c3c"}
STATE_CODE = {"Vacancy": 0, "Stationary": 1, "Motion": 2}


def simulate_building(
    n_rooms: int = 20, time_of_day: int = 9, day_of_week: str = "Monday"
) -> pd.DataFrame:
    """
    Simulate occupancy states for all rooms in the building.
    Patterns vary by time of day and day of week.
    """
    np.random.seed(int(time_of_day * 7) % (2**32 - 1))

    is_weekend = day_of_week in ["Saturday", "Sunday"]
    is_business = 8 <= time_of_day <= 18 and not is_weekend

    rows: list[dict] = []
    for i in range(min(n_rooms, len(ROOM_NAMES))):
        name = ROOM_NAMES[i]

        if "Lobby" in name or "Reception" in name:
            probs = [0.1, 0.3, 0.6] if is_business else [0.8, 0.1, 0.1]
        elif "Server" in name or "Parking" in name:
            probs = [0.3, 0.2, 0.5] if is_business else [0.5, 0.3, 0.2]
        elif "Meeting" in name or "Board" in name:
            probs = [0.3, 0.2, 0.5] if is_business else [0.9, 0.1, 0.0]
        elif not is_business:
            probs = [0.85, 0.1, 0.05]
        else:
            probs = [0.3, 0.4, 0.3]

        state = np.random.choice(STATES, p=probs)

        rows.append(
            {
                "room_id": i,
                "room_name": name,
                "state": state,
                "state_code": STATE_CODE[state],
                "color": STATE_COLOR[state],
                "energy_W": {"Vacancy": 150, "Stationary": 1350, "Motion": 2600}[
                    state
                ],
            }
        )

    return pd.DataFrame(rows)


def simulate_weekly_timeline(n_hours: int = 24) -> pd.DataFrame:
    """
    Simulate hourly occupancy counts across a full week.
    Used by Occupancy Timeline page.
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rows: list[dict] = []
    for day in days:
        is_weekend = day in ["Saturday", "Sunday"]
        for hour in range(n_hours):
            if is_weekend:
                base = max(0, int(np.random.normal(2, 1)))
            elif 9 <= hour <= 18:
                base = max(0, int(np.random.normal(15, 4)))
            elif 7 <= hour <= 9 or 18 <= hour <= 20:
                base = max(0, int(np.random.normal(8, 3)))
            else:
                base = max(0, int(np.random.normal(1, 1)))

            rows.append(
                {
                    "day": day,
                    "hour": hour,
                    "occupied": base,
                    "stationary": max(0, int(base * 0.4 + np.random.normal(0, 1))),
                    "motion": max(0, int(base * 0.6 + np.random.normal(0, 1))),
                    "vacancy": max(0, 20 - base),
                }
            )

    return pd.DataFrame(rows)

