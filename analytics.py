import pandas as pd
import plotly.express as px


def load_data():
    return pd.read_csv("data/campus_data.csv")


def basic_summary(df):
    return {
        "Total Electricity": round(df["electricity"].sum(), 2),
        "Total Water": round(df["water"].sum(), 2),
        "Electricity Cost": round(df["electricity"].sum() * 8, 2),
        "Water Cost": round(df["water"].sum() * 0.05, 2),
    }


def location_analysis(df):
    aggregations = {
        "electricity": ("electricity", "sum"),
        "water": ("water", "sum"),
    }

    if "electricity_cost" in df.columns:
        aggregations["electricity_cost"] = ("electricity_cost", "sum")

    if "water_cost" in df.columns:
        aggregations["water_cost"] = ("water_cost", "sum")

    return (
        df.groupby("location")
        .agg(**aggregations)
        .reset_index()
    )


def hourly_analysis(df):
    return (
        df.groupby("hour")
        .agg(
            electricity=("electricity", "mean"),
            water=("water", "mean"),
        )
        .reset_index()
    )


def daily_analysis(df):
    if "date" not in df.columns:
        return pd.DataFrame(
            columns=["date", "electricity", "water"]
        )

    daily_df = df.copy()
    daily_df["date"] = pd.to_datetime(
        daily_df["date"], errors="coerce"
    )
    daily_df = daily_df.dropna(subset=["date"])

    if daily_df.empty:
        return pd.DataFrame(
            columns=["date", "electricity", "water"]
        )

    return (
        daily_df.groupby(daily_df["date"].dt.date)
        .agg(
            electricity=("electricity", "sum"),
            water=("water", "sum"),
        )
        .reset_index(names="date")
    )


def create_location_chart(df):
    fig = px.bar(
        df,
        x="location",
        y="electricity",
        title="Electricity Consumption by Location",
        labels={
            "electricity": "Electricity Consumption",
            "location": "Campus Location",
        },
    )

    return fig


def create_water_chart(df):
    fig = px.bar(
        df,
        x="location",
        y="water",
        title="Water Consumption by Location",
        labels={
            "water": "Water Consumption",
            "location": "Campus Location",
        },
    )

    return fig


def create_hourly_chart(df):
    fig = px.line(
        df,
        x="hour",
        y=["electricity", "water"],
        markers=True,
        title="Average Resource Consumption by Hour",
    )

    return fig


def create_daily_chart(df):
    fig = px.line(
        df,
        x="date",
        y=["electricity", "water"],
        title="Daily Resource Consumption",
    )

    return fig