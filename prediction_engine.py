import math
from pathlib import Path
import re
import joblib
import pandas as pd

MODEL_DIR = Path("models")
SPECIAL_DIR = MODEL_DIR / "specialized"
FEATURES = [
    "building_type", "location", "room",
    "hour", "minute", "day_of_week", "weekend", "holiday", "event",
    "occupancy", "temperature", "ac_hours", "floor_area_sqft",
    "operating_hours"
]

def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

def _load(target, building_type):
    specific = SPECIAL_DIR / f"{_slug(building_type)}_{target}_model.pkl"
    if specific.exists():
        return joblib.load(specific), f"specialized:{building_type}"
    universal = MODEL_DIR / f"universal_{target}_model.pkl"
    if not universal.exists():
        raise FileNotFoundError(
            f"{universal} not found. Train models using train_hybrid_models.py first."
        )
    return joblib.load(universal), "universal"

def predict_resources(payload):
    missing = [c for c in FEATURES if c not in payload]
    if missing:
        raise ValueError("Missing fields: " + ", ".join(missing))

    try:
        occupancy = float(payload["occupancy"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Occupancy must be a valid number.")

    if not math.isfinite(occupancy) or not 0 <= occupancy <= 500:
        raise ValueError("Occupancy must be between 0 and 500.")

    try:
        temperature = float(payload["temperature"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Temperature must be a valid number.")

    if not math.isfinite(temperature) or not 0 <= temperature <= 50:
        raise ValueError("Temperature must be between 0 and 50°C.")

    try:
        ac_hours = float(payload["ac_hours"])
        operating_hours = float(payload["operating_hours"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("AC hours and operating hours must be valid numbers.")

    if (
        not math.isfinite(ac_hours)
        or not math.isfinite(operating_hours)
        or not 0 <= ac_hours <= operating_hours <= 24
    ):
        raise ValueError(
            "AC hours must be between 0 and operating hours, "
            "and operating hours must not exceed 24."
        )

    row = {c: payload[c] for c in FEATURES}
    row["occupancy"] = occupancy
    row["temperature"] = temperature
    row["ac_hours"] = ac_hours
    row["operating_hours"] = operating_hours

    for c in ["building_type", "location", "room"]:
        row[c] = str(row[c])

    X = pd.DataFrame([row], columns=FEATURES)

    e_model, e_type = _load("electricity", row["building_type"])
    w_model, w_type = _load("water", row["building_type"])

    e = max(0.0, float(e_model.predict(X)[0]))
    w = max(0.0, float(w_model.predict(X)[0]))

    return {
        "predicted_electricity": round(e, 3),
        "predicted_water": round(w, 3),
        "electricity_model_used": e_type,
        "water_model_used": w_type
    }