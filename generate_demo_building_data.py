"""
CampusTwinAI synthetic demonstration data generator.

IMPORTANT:
- This creates SYNTHETIC demo records, not real measured building data.
- Use only to test the software/training pipeline.
- Do not present its model accuracy or predictions as real-world validated.
- Replace/augment with permissioned, measured data before real deployment.

Run from the project root:
    python3 generate_demo_building_data.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT = Path("data/building_consumption.csv")
RNG = np.random.default_rng(42)

BUILDING_PROFILES = {
    "College":   {"area": (15000, 90000), "occupancy": (20, 900), "hours": (8, 14), "base_e": 5.0, "e_person": 0.012, "base_w": 12, "w_person": 0.10, "ac": (0, 12)},
    "Hospital":  {"area": (20000, 180000), "occupancy": (30, 1200), "hours": (20, 24), "base_e": 12.0, "e_person": 0.018, "base_w": 25, "w_person": 0.18, "ac": (6, 24)},
    "Home":      {"area": (500, 3500), "occupancy": (1, 8), "hours": (10, 24), "base_e": 0.25, "e_person": 0.08, "base_w": 0.5, "w_person": 1.2, "ac": (0, 10)},
    "Apartment": {"area": (5000, 80000), "occupancy": (10, 500), "hours": (12, 24), "base_e": 2.0, "e_person": 0.035, "base_w": 5, "w_person": 0.45, "ac": (0, 14)},
    "Office":    {"area": (5000, 120000), "occupancy": (10, 1000), "hours": (8, 16), "base_e": 4.0, "e_person": 0.025, "base_w": 4, "w_person": 0.08, "ac": (0, 12)},
    "Hotel":     {"area": (10000, 150000), "occupancy": (10, 700), "hours": (18, 24), "base_e": 8.0, "e_person": 0.030, "base_w": 15, "w_person": 0.65, "ac": (4, 24)},
    "Factory":   {"area": (10000, 250000), "occupancy": (10, 600), "hours": (8, 24), "base_e": 15.0, "e_person": 0.020, "base_w": 10, "w_person": 0.12, "ac": (0, 12)},
}

def make_rows(building_type: str, profile: dict, n: int = 700) -> pd.DataFrame:
    area = RNG.integers(profile["area"][0], profile["area"][1] + 1, n)
    occupancy = RNG.integers(profile["occupancy"][0], profile["occupancy"][1] + 1, n)
    operating_hours = RNG.integers(profile["hours"][0], profile["hours"][1] + 1, n)
    hour = RNG.integers(0, 24, n)
    minute = RNG.choice([0, 15, 30, 45], n)
    day_of_week = RNG.integers(0, 7, n)
    weekend = (day_of_week >= 5).astype(int)
    holiday = RNG.binomial(1, 0.04, n)
    event = RNG.binomial(1, 0.10 if building_type in {"College", "Hotel"} else 0.04, n)
    temperature = np.clip(RNG.normal(29, 5, n), 15, 45).round(1)
    ac_hours = np.minimum(RNG.uniform(*profile["ac"], n), operating_hours).round(1)

    # Synthetic usage patterns for a software demonstration only.
    active_factor = np.where((hour < operating_hours) | (operating_hours >= 20), 1.0, 0.55)
    if building_type in {"College", "Office"}:
        active_factor *= np.where(weekend == 1, 0.65, 1.0)
    if building_type == "Factory":
        active_factor *= np.where(operating_hours >= 16, 1.15, 1.0)

    electricity = (
        profile["base_e"]
        + area * 0.00008
        + occupancy * profile["e_person"]
        + ac_hours * 0.12
        + np.maximum(temperature - 24, 0) * 0.08
        + event * 0.8
    ) * active_factor
    electricity *= RNG.normal(1.0, 0.12, n)
    electricity = np.maximum(electricity, 0.05).round(3)

    water = (
        profile["base_w"]
        + occupancy * profile["w_person"]
        + event * 2.0
        + np.maximum(temperature - 28, 0) * 0.15
    ) * active_factor
    water *= RNG.normal(1.0, 0.15, n)
    water = np.maximum(water, 0.05).round(3)

    locations = RNG.choice(
        ["Andhra Pradesh", "Tamil Nadu", "Karnataka", "Telangana", "Maharashtra"],
        n
    )
    rooms = [f"{building_type}_Zone_{i:02d}" for i in RNG.integers(1, 16, n)]

    return pd.DataFrame({
        "building_type": building_type,
        "location": locations,
        "room": rooms,
        "hour": hour,
        "minute": minute,
        "day_of_week": day_of_week,
        "weekend": weekend,
        "holiday": holiday,
        "event": event,
        "occupancy": occupancy,
        "temperature": temperature,
        "ac_hours": ac_hours,
        "floor_area_sqft": area,
        "operating_hours": operating_hours,
        "electricity": electricity,
        "water": water,
        "data_source": "SYNTHETIC_DEMO_NOT_REAL_MEASUREMENTS",
    })

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        make_rows(name, profile, n=700)
        for name, profile in BUILDING_PROFILES.items()
    ]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Created {OUTPUT} with {len(df):,} synthetic demo rows.")
    print("Rows per building type:")
    print(df["building_type"].value_counts().sort_index().to_string())
    print("\nWARNING: Synthetic demo data only; not valid evidence of real-world accuracy.")

if __name__ == "__main__":
    main()
