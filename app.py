import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from analytics import (
    basic_summary,
    location_analysis,
    hourly_analysis,
    daily_analysis
)

st.set_page_config(
    page_title="CampusTwinAI",
    page_icon="🏫",
    layout="wide"
)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "predicted_electricity" not in st.session_state:
    st.session_state.predicted_electricity = None

if "predicted_water" not in st.session_state:
    st.session_state.predicted_water = None
from analytics import (
    basic_summary,
    location_analysis,
    hourly_analysis,
    daily_analysis
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CampusTwinAI",
    page_icon="🏫",
    layout="wide"
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/campus_data.csv")


@st.cache_resource
def load_models():
    electricity_model = joblib.load(
        "models/electricity_model.pkl"
    )

    water_model = joblib.load(
        "models/water_model.pkl"
    )

    return electricity_model, water_model


df = load_data()
electricity_model, water_model = load_models()

# =========================================================
# TITLE
# =========================================================

st.title("🏫 CampusTwinAI")

st.subheader(
    "AI-Powered Campus Energy & Water Analytics"
)

st.write(
    "CampusTwinAI uses Artificial Intelligence and "
    "Data Analytics to understand campus resource "
    "consumption and generate predictions."
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎛️ Campus Controls")

selected_location = st.sidebar.selectbox(
    "Select Campus Location",
    ["All"] + sorted(
        df["location"].dropna().unique().tolist()
    ),
    key="sidebar_location"
)

# =========================================================
# FILTER DATA
# =========================================================

if selected_location == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df["location"] == selected_location
    ]
# =========================================================
# DATE FILTER
# =========================================================

st.sidebar.subheader("📅 Date Filter")

df["date"] = pd.to_datetime(df["date"],
errors="coerce")
available_dates = sorted(
    df["date"].dt.date.unique()
)

selected_date = st.sidebar.selectbox(
    "Select Date",
    ["All"] + available_dates,
    key="selected_date"
)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
if selected_date != "All":
    filtered_df = filtered_df[
        filtered_df["date"].dt.date == selected_date
    ]
# =========================================================
# SUMMARY
# =========================================================

summary = basic_summary(filtered_df)
# =========================================================
# DASHBOARD KPIs
# =========================================================

st.header("📊 Campus Dashboard")

avg_occupancy = filtered_df["occupancy"].mean()

dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = st.columns(4)

with dashboard_col1:
    st.metric(
        "⚡ Total Electricity",
        f"{summary['Total Electricity']:.2f}"
    )

with dashboard_col2:
    st.metric(
        "💧 Total Water",
        f"{summary['Total Water']:.2f}"
    )

with dashboard_col3:
    st.metric(
        "👥 Average Occupancy",
        f"{avg_occupancy:.1f}"
    )

with dashboard_col4:

    if avg_occupancy >= 50:
        campus_status = "⚠️ High Usage"
    elif avg_occupancy >= 25:
        campus_status = "🟡 Moderate"
    else:
        campus_status = "🟢 Good"

    st.metric(
        "🌱 Campus Status",
        campus_status
    )
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "⚡ Electricity",
        f"{summary['Total Electricity']:.2f}"
    )

with col2:
    st.metric(
        "💧 Water",
        f"{summary['Total Water']:.2f}"
    )

with col3:
    st.metric(
        "💰 Electricity Cost",
        f"₹{summary['Electricity Cost']:.2f}"
    )

with col4:
    st.metric(
        "💰 Water Cost",
        f"₹{summary['Water Cost']:.2f}"
    )
# =========================================================
# CAMPUS PERFORMANCE
# =========================================================

st.header("🌱 Campus Performance")

electricity_per_person = (
    summary["Total Electricity"] / avg_occupancy
    if avg_occupancy > 0 else 0
)

water_per_person = (
    summary["Total Water"] / avg_occupancy
    if avg_occupancy > 0 else 0
)

performance_col1, performance_col2, performance_col3 = st.columns(3)

with performance_col1:
    st.metric(
        "⚡ Electricity / Person",
        f"{electricity_per_person:.2f}"
    )

with performance_col2:
    st.metric(
        "💧 Water / Person",
        f"{water_per_person:.2f}"
    )

with performance_col3:
    if electricity_per_person < 500:
        performance_status = "🟢 Efficient"
    elif electricity_per_person < 1000:
        performance_status = "🟡 Moderate"
    else:
        performance_status = "🔴 High Usage"

    st.metric(
        "🌍 Performance",
        performance_status
    )
    # =========================================================
# RESOURCE CONSUMPTION COMPARISON
# =========================================================

st.header("📊 Resource Consumption Comparison")

comparison_df = pd.DataFrame({
    "Resource": [
        "Electricity",
        "Water"
    ],
    "Consumption": [
        summary["Total Electricity"],
        summary["Total Water"]
    ]
})

fig_comparison = px.bar(
    comparison_df,
    x="Resource",
    y="Consumption",
    text_auto=".2f",
    title="⚡ Electricity vs 💧 Water Consumption"
)

st.plotly_chart(
    fig_comparison,
    width="stretch"
)
# =========================================================
# LOCATION ANALYSIS
# =========================================================

st.header("📍 Location Analysis")

location_df = location_analysis(filtered_df)

col1, col2 = st.columns(2)

with col1:

    fig1 = px.bar(
        location_df,
        x="location",
        y="electricity",
        title="⚡ Electricity Consumption"
    )

    st.plotly_chart(
        fig1,
        width="stretch"
    )

with col2:

    fig2 = px.bar(
        location_df,
        x="location",
        y="water",
        title="💧 Water Consumption"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )
# =========================================================
# OCCUPANCY ANALYSIS
# =========================================================

st.header("👥 Occupancy Analysis")

occupancy_df = (
    filtered_df.groupby("hour")["occupancy"]
    .mean()
    .reset_index()
)

fig_occupancy = px.line(
    occupancy_df,
    x="hour",
    y="occupancy",
    markers=True,
    title="Average Occupancy by Hour"
)

st.plotly_chart(
    fig_occupancy,
    width="stretch"
)
# =========================================================
# PEAK USAGE DETECTION
# =========================================================

st.header("🔥 Peak Usage Detection")

peak_df = (
    filtered_df.groupby("hour")["electricity"]
    .mean()
    .reset_index()
)

peak_row = peak_df.loc[
    peak_df["electricity"].idxmax()
]

peak_hour = int(peak_row["hour"])
peak_electricity = peak_row["electricity"]

peak_col1, peak_col2, peak_col3 = st.columns(3)

with peak_col1:
    st.metric(
        "🔥 Peak Usage Hour",
        f"{peak_hour}:00"
    )

with peak_col2:
    st.metric(
        "⚡ Peak Electricity",
        f"{peak_electricity:.2f}"
    )

with peak_col3:
    st.info(
        "💡 Consider reducing AC usage during peak hours."
    )
# =========================================================
# HOURLY ANALYSIS
# =========================================================

st.header("⏰ Hourly Consumption")

hourly_df = hourly_analysis(filtered_df)

fig3 = px.line(
    hourly_df,
    x="hour",
    y=["electricity", "water"],
    markers=True,
    title="Average Consumption by Hour"
)

st.plotly_chart(
    fig3,
    width="stretch"
)

# =========================================================
# DAILY ANALYSIS
# =========================================================

st.header("📅 Daily Consumption")

daily_df = daily_analysis(filtered_df)

fig4 = px.line(
    daily_df,
    x="date",
    y=["electricity", "water"],
    title="Daily Resource Consumption"
)

st.plotly_chart(
    fig4,
    width="stretch"
)
# =========================================================
# DAILY CONSUMPTION SUMMARY
# =========================================================

st.subheader("📊 Daily Consumption Summary")

daily_total_electricity = daily_df["electricity"].sum()
daily_total_water = daily_df["water"].sum()

daily_col1, daily_col2 = st.columns(2)

with daily_col1:
    st.metric(
        "⚡ Daily Electricity",
        f"{daily_total_electricity:.2f}"
    )

with daily_col2:
    st.metric(
        "💧 Daily Water",
        f"{daily_total_water:.2f}"
    )
# =========================================================
# AI RESOURCE PREDICTION
# =========================================================

st.header("🤖 AI Resource Prediction")

st.write(
    "Enter campus conditions and let the trained AI "
    "models predict electricity and water consumption."
)

pred_col1, pred_col2 = st.columns(2)
# =========================================================
# AI RECOMMENDATIONS
# =========================================================

st.header("🤖 AI Recommendations")

recommendations = []

# Electricity recommendation
if summary["Total Electricity"] > 60000:
    recommendations.append(
        "⚡ Electricity usage is high. Reduce AC usage during peak hours."
    )
else:
    recommendations.append(
        "✅ Electricity consumption is within a reasonable range."
    )

# Water recommendation
if summary["Total Water"] > 500000:
    recommendations.append(
        "💧 Water consumption is high. Check for leakage and unnecessary water usage."
    )
else:
    recommendations.append(
        "✅ Water consumption is under control."
    )

# Occupancy recommendation
if avg_occupancy > 25:
    recommendations.append(
        "👥 Occupancy is high. Optimize room usage and avoid running empty rooms."
    )
else:
    recommendations.append(
        "🏫 Occupancy is moderate. Continue optimizing room usage."
    )

# Display recommendations
for recommendation in recommendations:
    st.info(recommendation)

st.success(
    "🌱 Following these recommendations can help the campus reduce "
    "electricity and water consumption."
)
# =========================================================
# LEFT SIDE INPUTS
# =========================================================

with pred_col1:

    prediction_hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=12,
        key="prediction_hour"
    )

    prediction_minute = st.selectbox(
        "Minute",
        [0, 15, 30, 45],
        index=0,
        key="prediction_minute"
    )

    prediction_day = st.slider(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=2,
        help="0 = Monday, 6 = Sunday",
        key="prediction_day"
    )

    prediction_location = st.selectbox(
        "Location",
        sorted(
            df["location"].dropna().unique().tolist()
        ),
        key="prediction_location"
    )

    prediction_room = st.selectbox(
        "Room",
        sorted(
            df["room"].dropna().unique().tolist()
        ),
        key="prediction_room"
    )

# =========================================================
# RIGHT SIDE INPUTS
# =========================================================

with pred_col2:

    prediction_occupancy = st.number_input(
        "Occupancy / Number of People",
        min_value=0,
        max_value=500,
        value=50,
        key="prediction_occupancy"
    )

    prediction_temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=30.0,
        step=0.1,
        key="prediction_temperature"
    )

    prediction_ac_hours = st.number_input(
        "AC Usage Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.1,
        key="prediction_ac_hours"
    )

    prediction_holiday = st.checkbox(
        "Holiday",
        key="prediction_holiday"
    )

    prediction_event = st.checkbox(
        "Event",
        key="prediction_event"
    )


# WEEKEND CALCULATION


prediction_weekend = (
    1 if prediction_day >= 5 else 0
)

# =========================================================
# CREATE MODEL INPUT
# =========================================================

prediction_input = pd.DataFrame({
    "hour": [prediction_hour],
    "minute": [prediction_minute],
    "day_of_week": [prediction_day],
    "location": [prediction_location],
    "room": [prediction_room],
    "weekend": [prediction_weekend],
    "holiday": [int(prediction_holiday)],
    "event": [int(prediction_event)],
    "occupancy": [prediction_occupancy],
    "temperature": [prediction_temperature],
    "ac_hours": [prediction_ac_hours]
})

# =========================================================
# GENERATE AI PREDICTION
# =========================================================

if st.button(
    "🔮 Generate AI Prediction",
    key="generate_ai_prediction"
):

    try:

        # -------------------------------------------------
        # AI MODEL PREDICTION
        # -------------------------------------------------

        predicted_electricity = electricity_model.predict(
            prediction_input
        )[0]

        predicted_water = water_model.predict(
            prediction_input
        )[0]
        # SAVE PREDICTION HISTORY
        
        st.session_state.prediction_history.append({
    "Location": prediction_location,
    "Room": prediction_room,
    "Occupancy": prediction_occupancy,
    "Temperature": prediction_temperature,
    "AC Hours": prediction_ac_hours,
    "Predicted Electricity": round(predicted_electricity, 2),
    "Predicted Water": round(predicted_water, 2)
})
        st.session_state.predicted_electricity = predicted_electricity
        st.session_state.predicted_water = predicted_water
        # -------------------------------------------------
        # SUCCESS MESSAGE
        # -------------------------------------------------

        st.success(
            "✅ AI Prediction Generated Successfully!"
        )

        # -------------------------------------------------
        # PREDICTION RESULT
        # -------------------------------------------------

        st.subheader("🔮 AI Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "⚡ Predicted Electricity",
                f"{predicted_electricity:.2f}"
            )

        with result_col2:

            st.metric(
                "💧 Predicted Water",
                f"{predicted_water:.2f}"
            )

        # -------------------------------------------------
        # PREDICTION ANALYSIS
        # -------------------------------------------------

        st.subheader("📊 Prediction Analysis")

        analysis_col1, analysis_col2 = st.columns(2)

        # Electricity
        with analysis_col1:

            if predicted_electricity >= 10:

                st.error(
                    "⚡ Electricity Usage: HIGH"
                )

            elif predicted_electricity >= 5:

                st.warning(
                    "⚡ Electricity Usage: MEDIUM"
                )

            else:

                st.success(
                    "⚡ Electricity Usage: LOW"
                )

        # Water
        with analysis_col2:

            if predicted_water >= 80:

                st.error(
                    "💧 Water Usage: HIGH"
                )

            elif predicted_water >= 50:

                st.warning(
                    "💧 Water Usage: MEDIUM"
                )

            else:

                st.success(
                    "💧 Water Usage: LOW"
                )

        # -------------------------------------------------
        # ESTIMATED COST
        # -------------------------------------------------

        st.subheader("💰 Estimated Resource Cost")

        electricity_rate = 8.0
        water_rate = 0.05

        estimated_electricity_cost = (
            predicted_electricity
            * electricity_rate
        )

        estimated_water_cost = (
            predicted_water
            * water_rate
        )

        cost_col1, cost_col2 = st.columns(2)

        with cost_col1:

            st.metric(
                "⚡ Estimated Electricity Cost",
                f"₹{estimated_electricity_cost:.2f}"
            )

        with cost_col2:

            st.metric(
                "💧 Estimated Water Cost",
                f"₹{estimated_water_cost:.2f}"
            )

        # -------------------------------------------------
        # VISUAL COMPARISON
        # -------------------------------------------------

        st.subheader("📈 AI Prediction Comparison")

        prediction_chart = pd.DataFrame({
            "Resource": [
                "Electricity",
                "Water"
            ],
            "Predicted Consumption": [
                predicted_electricity,
                predicted_water
            ]
        })

        fig_prediction = px.bar(
            prediction_chart,
            x="Resource",
            y="Predicted Consumption",
            text_auto=".2f",
            title="Predicted Resource Consumption"
        )

        st.plotly_chart(
            fig_prediction,
            width="stretch"
        )

        # -------------------------------------------------
        # AI RECOMMENDATION
        # -------------------------------------------------

        st.subheader("💡 AI Recommendation")

        recommendations = []

        if predicted_electricity >= 10:

            recommendations.append(
                "Reduce AC usage during peak hours "
                "to lower electricity consumption."
            )

        if predicted_water >= 80:

            recommendations.append(
                "Monitor water usage and check for "
                "unnecessary water consumption."
            )

        if prediction_occupancy >= 50:

            recommendations.append(
                "Consider optimizing room usage "
                "when occupancy is high."
            )

        if not recommendations:

            recommendations.append(
                "Resource consumption is within "
                "an acceptable range."
            )

        for recommendation in recommendations:

            st.info(
                "💡 " + recommendation
            )

    except Exception as e:

        st.error(
            f"❌ Prediction Error: {e}"
        )
        
predicted_electricity = st.session_state.predicted_electricity
predicted_water = st.session_state.predicted_water

if predicted_electricity is None or predicted_water is None:
    st.info("👆 Generate an AI Prediction first.")
    st.stop()


electricity_rate = 8.0
water_rate = 0.05 


# -----------------------------
# RESOURCE SAVINGS ESTIMATOR
# -----------------------------

st.subheader("🌱 Resource Savings Estimator")

electricity_saving_percent = 15
water_saving_percent = 10

recommended_electricity = (
    predicted_electricity *
    (1 - electricity_saving_percent / 100)
)

recommended_water = (
    predicted_water *
    (1 - water_saving_percent / 100)
)

electricity_saved = (
    predicted_electricity -
    recommended_electricity
)

water_saved = (
    predicted_water -
    recommended_water
)

electricity_saving_cost = (
    electricity_saved * electricity_rate
)

water_saving_cost = (
    water_saved * water_rate
)

save_col1, save_col2 = st.columns(2)

with save_col1:

    st.metric(
        "⚡ Electricity Savings",
        f"{electricity_saved:.2f}"
    )

    st.metric(
        "💰 Electricity Money Saved",
        f"₹{electricity_saving_cost:.2f}"
    )

with save_col2:

    st.metric(
        "💧 Water Savings",
        f"{water_saved:.2f}"
    )

    st.metric(
        "💰 Water Money Saved",
        f"₹{water_saving_cost:.2f}"
    )

st.info(
    f"🌱 By following AI recommendations, the campus "
    f"could reduce electricity consumption by approximately "
    f"{electricity_saving_percent}% and water consumption by "
    f"{water_saving_percent}%."
)
# -----------------------------
# CAMPUS SUSTAINABILITY SCORE
# -----------------------------

st.subheader("🌍 Campus Sustainability Score")

# Start with 100 points
sustainability_score = 100

# Electricity usage penalty
if predicted_electricity >= 10:
    sustainability_score -= 20
elif predicted_electricity >= 5:
    sustainability_score -= 10

# Water usage penalty
if predicted_water >= 80:
    sustainability_score -= 20
elif predicted_water >= 50:
    sustainability_score -= 10

# AC usage penalty
if prediction_ac_hours >= 8:
    sustainability_score -= 15
elif prediction_ac_hours >= 5:
    sustainability_score -= 5

# Occupancy consideration
if prediction_occupancy >= 100:
    sustainability_score -= 10

# Keep score between 0 and 100
sustainability_score = max(
    0,
    min(100, sustainability_score)
)

# Display score
st.metric(
    "🌱 Sustainability Score",
    f"{sustainability_score}/100"
)

# Score interpretation
if sustainability_score >= 80:
    st.success(
        "🌱 Excellent! Campus resource usage is highly sustainable."
    )
elif sustainability_score >= 60:
    st.warning(
        "🌿 Good! Some resource-saving improvements are recommended."
    )
else:
    st.error(
        "⚠️ Resource consumption is high. Immediate conservation "
        "measures are recommended."
    )
    # -----------------------------
# DOWNLOAD AI REPORT
# -----------------------------

st.subheader("📥 Download AI Report")

report = f"""
CampusTwinAI
AI-Powered Campus Energy & Water Analytics
===========================================

AI PREDICTION
-------------
Predicted Electricity : {predicted_electricity:.2f}
Predicted Water       : {predicted_water:.2f}

ESTIMATED COST
--------------
Electricity Cost : ₹{estimated_electricity_cost:.2f}
Water Cost       : ₹{estimated_water_cost:.2f}
RESOURCE SAVINGS
----------------
Electricity Saved : {electricity_saved:.2f}
Water Saved       : {water_saved:.2f}

Money Saved
-----------
Electricity Money Saved : ₹{electricity_saving_cost:.2f}
Water Money Saved       : ₹{water_saving_cost:.2f}

SUSTAINABILITY
--------------
Sustainability Score : {sustainability_score}/100

AI RECOMMENDATIONS
------------------
"""

for recommendation in recommendations:
    report += f"- {recommendation}\n"

report += """
===========================================
CampusTwinAI | AI + Data Analytics Project
"""

st.download_button(
    label="📥 Download AI Report",
    data=report,
    file_name="CampusTwinAI_AI_Report.txt",
    mime="text/plain"
)
# =========================================================
# DATA TABLE
# =========================================================

st.header("📊 Campus Dataset")

st.dataframe(
    filtered_df.head(100),
    width="stretch"
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CampusTwinAI | AI + Data Analytics Mini Project"
)
# -----------------------------
# PREDICTION HISTORY
# -----------------------------

st.header("📜 AI Prediction History")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

history_data = pd.DataFrame(
    st.session_state.prediction_history
)

if history_data.empty:
    st.info("No prediction history yet.")
else:
    st.dataframe(
        history_data,
        use_container_width=True,
        hide_index=True
    )