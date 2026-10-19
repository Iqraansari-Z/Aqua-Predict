import pandas as pd
import plotly.express as px

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LABELS = {"consumption_l": "Avg consumption (L/day)"}


def load_data(path="data/water_data.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["day_name"] = df["date"].dt.day_name()
    return df


def weekday_weekend_chart(df):
    d = (
        df.assign(day_type=df["is_weekend"].map({0: "Weekday", 1: "Weekend"}))
        .groupby("day_type", as_index=False)["consumption_l"].mean()
    )
    return px.bar(d, x="day_type", y="consumption_l", text_auto=".0f",
                  title="Weekday vs Weekend", labels={**LABELS, "day_type": ""})


def festival_chart(df):
    d = (
        df.assign(day_type=df["is_festival"].map({0: "Normal day", 1: "Festival day"}))
        .groupby("day_type", as_index=False)["consumption_l"].mean()
    )
    return px.bar(d, x="day_type", y="consumption_l", text_auto=".0f",
                  title="Festival vs Normal Day", labels={**LABELS, "day_type": ""})


def monthly_chart(df):
    d = df.groupby(["month", "month_name"], as_index=False)["consumption_l"].mean()
    fig = px.line(d, x="month_name", y="consumption_l", markers=True,
                  title="Monthly Average Consumption",
                  labels={**LABELS, "month_name": "Month"})
    fig.update_yaxes(rangemode="tozero")
    return fig


def timeseries_chart(df):
    d = df.sort_values("date").copy()
    d["30-day average"] = d["consumption_l"].rolling(30).mean()
    fig = px.line(d, x="date", y=["consumption_l", "30-day average"],
                  title="Daily Consumption Over 3 Years",
                  labels={"value": "Litres/day", "date": "Date", "variable": ""})
    return fig


if __name__ == "__main__":
    data = load_data()
    for make_chart in (weekday_weekend_chart, festival_chart, monthly_chart, timeseries_chart):
        make_chart(data).show()