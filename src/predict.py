import joblib
import pandas as pd
from functools import lru_cache

from src.features import FEATURE_COLS, get_season

_MODEL = None


def _load_model(path="models/water_model.joblib"):
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(path)
    return _MODEL

@lru_cache(maxsize=1)
def _festival_lookup(path="data/water_data.csv"):
    """Return {(month, day): festival_name} from the historical data."""
    df = pd.read_csv(path)
    df = df[df["is_festival"] == 1]
    dates = pd.to_datetime(df["date"])
    return {(d.month, d.day): n for d, n in zip(dates, df["festival_name"])}


def build_row(date, temp_c, occupancy_pct, rainfall_mm=0.0, is_festival=None):
    """Turn user inputs into one model-ready row. is_festival=None means auto-detect."""
    date = pd.Timestamp(date)
    if is_festival is None:
        is_festival = int((date.month, date.day) in _festival_lookup())
    dow = date.dayofweek
    return pd.DataFrame([{
        "day_of_week": dow,
        "is_weekend": int(dow >= 5),
        "is_festival": int(is_festival),
        "month": date.month,
        "season": get_season(date.month),
        "temp_c": temp_c,
        "rainfall_mm": rainfall_mm,
        "occupancy_pct": occupancy_pct,
    }])[FEATURE_COLS]


def predict_demand(date, temp_c, occupancy_pct, rainfall_mm=0.0, is_festival=None):
    """Predicted water demand in litres for one day."""
    row = build_row(date, temp_c, occupancy_pct, rainfall_mm, is_festival)
    return float(_load_model().predict(row)[0])


if __name__ == "__main__":
    # Same weather and occupancy; only the day type changes
    tue = predict_demand("2026-10-20", 30, 85, is_festival=0)
    sun = predict_demand("2026-10-25", 30, 85, is_festival=0)
    sun_fest = predict_demand("2026-10-25", 30, 85, is_festival=1)
    print(f"Normal Tuesday:    {tue:,.0f} L")
    print(f"Normal Sunday:     {sun:,.0f} L")
    print(f"Festival Sunday:   {sun_fest:,.0f} L")
    print("Check (festival Sunday > normal Tuesday):", sun_fest > tue)