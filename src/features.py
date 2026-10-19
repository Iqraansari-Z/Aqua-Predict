import pandas as pd

TARGET = "demand_l"

FEATURE_COLS = [
    "day_of_week",
    "is_weekend",
    "is_festival",
    "month",
    "season",
    "temp_c",
    "rainfall_mm",
    "occupancy_pct",
]


def get_season(month):
    """0 = winter (Dec-Feb), 1 = summer (Mar-May), 2 = monsoon (Jun-Sep), 3 = post-monsoon (Oct-Nov)"""
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8, 9):
        return 2
    return 3


def build_features(df):
    """Takes a dataframe with a 'date' column and returns it with model-ready feature columns added."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(get_season)
    return df


def load_and_prepare(path="data/water_data.csv"):
    df = pd.read_csv(path)
    # True demand = what was used + what could not be supplied (shortage)
    df["demand_l"] = df["consumption_l"] + df.get("shortage_l", 0)
    df = build_features(df)
    return df.sort_values("date").reset_index(drop=True)