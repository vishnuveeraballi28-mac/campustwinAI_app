from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="CampusTwinAI API")


# =========================================================
# LOAD AI MODELS
# =========================================================

electricity_model = joblib.load(
    "models/electricity_model.pkl"
)

water_model = joblib.load(
    "models/water_model.pkl"
)


# =========================================================
# INPUT DATA
# =========================================================

class PredictionInput(BaseModel):

    hour: int
    minute: int
    day_of_week: int
    location: str
    room: str
    occupancy: int
    temperature: float
    ac_hours: float
    holiday: int
    event: int


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "CampusTwinAI API is running!"
    }


# =========================================================
# AI PREDICTION
# =========================================================

@app.post("/predict")
def predict(data: PredictionInput):

    # Weekend calculation
    weekend = (
        1 if data.day_of_week >= 5 else 0
    )

    # Create exactly the same features
    # used when training the models

    input_data = pd.DataFrame([{

        "hour": data.hour,

        "minute": data.minute,

        "day_of_week": data.day_of_week,

        "location": data.location,

        "room": data.room,

        "weekend": weekend,

        "holiday": data.holiday,

        "event": data.event,

        "occupancy": data.occupancy,

        "temperature": data.temperature,

        "ac_hours": data.ac_hours
    }])


    # =====================================================
    # PREDICTIONS
    # =====================================================

    electricity = electricity_model.predict(
        input_data
    )[0]

    water = water_model.predict(
        input_data
    )[0]


    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "electricity": round(
            float(electricity),
            2
        ),

        "water": round(
            float(water),
            2
        )
    }