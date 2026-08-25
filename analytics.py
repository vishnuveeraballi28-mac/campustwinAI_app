import pandas as pd
import plotly.express as px


def load_data():
    return pd.read_csv("data/campus_data.csv")


def basic_summary(df):
    return {
        "Total Electricity": round(df["electricity"].sum(), 2),
        "Total Water": round(df["water"].sum(), 2),
        "Electricity Cost": round(df["electricity_cost"].sum(), 2),
        "Water Cost": round(df["water_cost"].sum(), 2),
    }


def location_analysis(df):
    return (
        df.groupby("location")
        .agg(
            electricity=("electricity", "sum"),
            water=("water", "sum"),
            electricity_cost=("electricity_cost", "sum"),
            water_cost=("water_cost", "sum"),
        )
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
    return (
        df.groupby("date")
        .agg(
            electricity=("electricity", "sum"),
            water=("water", "sum"),
        )
        .reset_index()
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