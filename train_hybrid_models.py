from pathlib import Path
import json, re
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA = Path("data/building_consumption.csv")
MODEL_DIR = Path("models")
SPECIAL_DIR = MODEL_DIR / "specialized"
MIN_SPECIALIZED_ROWS = 500

FEATURES = [
    "building_type", "location", "room",
    "hour", "minute", "day_of_week", "weekend", "holiday", "event",
    "occupancy", "temperature", "ac_hours", "floor_area_sqft",
    "operating_hours"
]
CATEGORICAL = ["building_type", "location", "room"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
TARGETS = ["electricity", "water"]

def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

def make_model():
    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ], remainder="passthrough")
    return Pipeline([
        ("preprocessor", prep),
        ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ])

def fit_model(df, target, path, label):
    data = df[FEATURES + [target]].copy()
    data = data.dropna(subset=[target])
    for c in CATEGORICAL:
        data[c] = data[c].fillna("Unknown").astype(str)
    for c in NUMERIC:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=NUMERIC)
    if len(data) < 20:
        print(f"Skipping {label}: only {len(data)} clean rows (minimum 20).")
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[target], test_size=0.2, random_state=42
    )
    model = make_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    result = {
        "model": label,
        "rows": int(len(data)),
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "r2": round(float(r2_score(y_test, pred)), 4)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(result, "saved:", path)
    return result

def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Create {DATA} with measured/verified multi-building records first. "
            "Do not relabel campus-only synthetic data as hospital/home/factory data."
        )
    df = pd.read_csv(DATA)
    required = set(FEATURES + TARGETS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Missing dataset columns: " + ", ".join(missing))
    df["building_type"] = df["building_type"].astype(str).str.strip().str.lower()
    df = df[df["building_type"].ne("") & df["building_type"].ne("nan")]
    MODEL_DIR.mkdir(exist_ok=True)
    SPECIAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = []
    # Universal models trained on all available building types
    for target in TARGETS:
        r = fit_model(df, target, MODEL_DIR / f"universal_{target}_model.pkl",
                      f"universal_{target}")
        if r: metrics.append(r)
    # Specialized models only when enough rows exist for that building type
    for btype, subset in df.groupby("building_type"):
        clean_rows = subset.dropna(subset=TARGETS + NUMERIC).shape[0]
        if clean_rows < MIN_SPECIALIZED_ROWS:
            print(f"Skipping specialized model for {btype}: {clean_rows} rows; "
                  f"need {MIN_SPECIALIZED_ROWS}.")
            continue
        name = slug(btype)
        for target in TARGETS:
            r = fit_model(subset, target, SPECIAL_DIR / f"{name}_{target}_model.pkl",
                          f"specialized_{name}_{target}")
            if r: metrics.append(r)
    (MODEL_DIR / "hybrid_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("Done. Metrics saved to models/hybrid_model_metrics.json")

if __name__ == "__main__":
    main()
