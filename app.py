import streamlit as st
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.capacity import forecast_week
from src import eda
from src.capacity import check_availability
from src.predict import predict_demand

st.set_page_config(page_title="AquaPredict", page_icon="💧", layout="wide")
@st.cache_data
def load_eda_data():
    return eda.load_data()


@st.cache_data
def load_model_files():
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    preds = pd.read_csv("models/test_predictions.csv", parse_dates=["date"])
    imp = pd.read_csv("models/feature_importance.csv")
    return metrics, preds, imp
st.title("💧 AquaPredict")
st.caption("Residential water demand forecast and availability check. "
           "Trained on synthetic, pattern-calibrated data.")

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Inputs")
date = st.sidebar.date_input("Date")
capacity = st.sidebar.number_input("Tank capacity (L)", 5_000, 200_000, 40_000, step=1_000)
tank_level = st.sidebar.slider("Current tank level (L)", 0, int(capacity), min(25_000, int(capacity)), step=500)
supply = st.sidebar.slider("Expected supply today (L)", 0, 40_000, 20_000, step=500)
occupancy = st.sidebar.slider("Occupancy (%)", 30, 100, 85)
temp = st.sidebar.slider("Temperature (°C)", 10, 45, 30)
festival_choice = st.sidebar.selectbox("Festival today?", ["Auto-detect", "Yes", "No"])
is_festival = {"Auto-detect": None, "Yes": 1, "No": 0}[festival_choice]
st.sidebar.caption("Auto-detect only knows fixed-date festivals. "
                   "Use Yes/No for moving ones (Diwali, Holi, Ganesh Chaturthi).")

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "7-Day Forecast", "Usage Patterns", "Model Performance"])

with tab1:
    demand = predict_demand(date, temp, occupancy, is_festival=is_festival)
    res = check_availability(demand, tank_level, supply, capacity)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted consumption", f"{demand:,.0f} L")
    c2.metric("Available water", f"{res['available']:,.0f} L", help="Tank level + expected supply")
    c3.metric("Status", f"{res['icon']} {res['status']}")

    if res["status"] == "Safe":
        st.success(res["message"])
    elif res["status"] == "Tight":
        st.warning(res["message"])
    else:
        st.error(res["message"])
        st.info(f"Suggested action: request an extra tanker of about "
                f"{res['shortfall']:,.0f} L and reduce non-essential use.")

with tab2:
    st.subheader("Next 7 days")
    st.caption("Uses the sidebar's tank level, supply, occupancy and temperature. "
               "Tank level is carried forward day by day.")
    fest_days = st.multiselect(
        "Mark festival dates (optional)",
        [pd.Timestamp(date) + pd.Timedelta(days=i) for i in range(7)],
        format_func=lambda d: d.strftime("%a %d %b"),
    )
    table, first_shortage = forecast_week(date, tank_level, supply, capacity, temp, occupancy,
                                          festival_dates=fest_days)

    fig = go.Figure()
    fig.add_bar(x=table["date"], y=table["demand"], name="Predicted demand")
    fig.add_scatter(x=table["date"], y=table["available"], name="Available water",
                    mode="lines+markers")
    fig.update_layout(yaxis_title="Litres", xaxis_title="Date", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    if first_shortage:
        st.error(f"First expected shortage: {first_shortage}")
    else:
        st.success("No shortage expected in the next 7 days.")
    st.dataframe(table, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Usage patterns in 3 years of data")
    data = load_eda_data()
    col_a, col_b = st.columns(2)
    col_a.plotly_chart(eda.weekday_weekend_chart(data), use_container_width=True)
    col_b.plotly_chart(eda.festival_chart(data), use_container_width=True)
    st.plotly_chart(eda.monthly_chart(data), use_container_width=True)
    st.plotly_chart(eda.timeseries_chart(data), use_container_width=True)
    st.markdown(
        "- Weekends use about 16% more water than weekdays.\n"
        "- Festival days use about 19% more than normal days.\n"
        "- Occupancy is the strongest driver; temperature and rainfall barely matter.\n"
        "- No long-term trend: the yearly average stays near 19,000 L/day."
    )

with tab4:
    st.subheader("Model performance (last 182 days, held out)")
    metrics, preds, imp = load_model_files()
    rf = next(m for m in metrics if m["Model"] == "RandomForest")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{rf['MAE']:,} L")
    m2.metric("MAE %", f"{rf['MAE %']}%")
    m3.metric("RMSE", f"{rf['RMSE']:,} L")
    m4.metric("R²", rf["R2"])
    st.caption("R² is capped near 0.66 by the random noise built into the synthetic data. "
               "MAE as a share of daily use is the more meaningful number here.")

    fig = px.line(preds, x="date", y=["actual", "predicted"], labels={"value": "Litres/day", "variable": ""},
                  title="Actual vs predicted")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                 title="Feature importance")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(metrics), hide_index=True)