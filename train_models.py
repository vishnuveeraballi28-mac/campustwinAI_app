import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



df = pd.read_csv("data/campus_data.csv")


# ==================================================
# ELECTRICITY MODEL
# ==================================================

features = [
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
    "ac_hours"
]

X = df[features]

y = df["electricity"]


categorical_features = [
    "location",
    "room"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


electricity_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


electricity_model.fit(
    X_train,
    y_train
)


predictions = electricity_model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("ELECTRICITY MODEL")
print("-----------------")
print("MAE:", round(mae, 3))
print("R2:", round(r2, 3))


joblib.dump(
    electricity_model,
    "models/electricity_model.pkl"
)


# ==================================================
# WATER MODEL
# ==================================================

y_water = df["water"]


water_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_water,
    test_size=0.2,
    random_state=42
)


water_model.fit(
    X_train,
    y_train
)


water_predictions = water_model.predict(
    X_test
)


water_mae = mean_absolute_error(
    y_test,
    water_predictions
)

water_r2 = r2_score(
    y_test,
    water_predictions
)


print()
print("WATER MODEL")
print("-----------")
print("MAE:", round(water_mae, 3))
print("R2:", round(water_r2, 3))


joblib.dump(
    water_model,
    "models/water_model.pkl"
)


print()
print("Both AI models saved successfully!")