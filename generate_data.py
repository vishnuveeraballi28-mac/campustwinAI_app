import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)



locations = [
    "Engineering Block",
    "Computer Science Block",
    "Library",
    "Canteen",
    "Administration Block"
]

rooms = [
    "Room 101",
    "Room 102",
    "Room 201",
    "Room 202",
    "Room 301"
]

start_date = datetime(2026, 1, 1)

records = []

for i in range(10000):

    date_time = start_date + timedelta(
        minutes=15 * i
    )

    date = date_time.date()

    hour = date_time.hour

    minute = date_time.minute

    day_of_week = date_time.weekday()

    location = random.choice(locations)

    room = random.choice(rooms)

    # Weekend
    weekend = 1 if day_of_week >= 5 else 0

    # Holiday
    holiday = random.choice([0, 0, 0, 0, 1])

    # College event
    event = random.choice([0, 0, 0, 0, 0, 1])

    # ------------------------------------------------
    # Occupancy
    # ------------------------------------------------

    if weekend or holiday:
        occupancy = random.randint(2, 20)

    elif 9 <= hour <= 17:
        occupancy = random.randint(20, 60)

    else:
        occupancy = random.randint(5, 25)

    # Events increase occupancy
    if event:
        occupancy += random.randint(10, 30)

    occupancy = min(80, occupancy)

    # ------------------------------------------------
    # AC usage
    # ------------------------------------------------

    if 10 <= hour <= 17 and occupancy > 15:
        ac_hours = random.uniform(0.5, 1.0)
    else:
        ac_hours = random.uniform(0, 0.4)

    # ------------------------------------------------
    # Temperature
    # ------------------------------------------------

    temperature = np.random.normal(30, 3)

    if temperature > 32:
        ac_hours += 0.2

    # ------------------------------------------------
    # Electricity consumption
    # ------------------------------------------------

    base_electricity = 3

    occupancy_effect = occupancy * 0.08

    ac_effect = ac_hours * 5

    random_noise = np.random.normal(0, 1)

    electricity = (
        base_electricity
        + occupancy_effect
        + ac_effect
        + random_noise
    )

    electricity = max(0.5, electricity)

    # ------------------------------------------------
    # Water consumption
    # ------------------------------------------------

    water = (
        20
        + occupancy * 1.5
        + np.random.normal(0, 5)
    )

    water = max(5, water)

    # ------------------------------------------------
    # Electricity cost
    # ------------------------------------------------

    electricity_rate = 8.0

    electricity_cost = electricity * electricity_rate

    # ------------------------------------------------
    # Water cost
    # ------------------------------------------------

    water_rate = 0.05

    water_cost = water * water_rate

    # ------------------------------------------------
    # Waste condition
    # ------------------------------------------------

    waste = 0

    if occupancy < 10 and electricity > 7:
        waste = 1

    if water > 120 and occupancy < 30:
        waste = 1

    records.append([
        date,
        hour,
        minute,
        day_of_week,
        location,
        room,
        weekend,
        holiday,
        event,
        occupancy,
        temperature,
        ac_hours,
        electricity,
        water,
        electricity_cost,
        water_cost,
        waste
    ])


columns = [
    "date",
    "hour",
    "minute",
    "day_of_week",
    "location",
    "room",
    "weekend",
    "holiday",
    "event",
    "occupancy",
    "temperature",
    "ac_hours",
    "electricity",
    "water",
    "electricity_cost",
    "water_cost",
    "waste"
]

df = pd.DataFrame(records, columns=columns)

df.to_csv(
    "data/campus_data.csv",
    index=False
)

print("Dataset generated successfully!")
print("Rows:", len(df))
print()
print(df.head())