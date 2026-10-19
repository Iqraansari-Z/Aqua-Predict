"""
AquaPredict - synthetic dataset generator
=========================================
Creates 3 years (1,095 days) of daily water data for ONE residential building
(40 flats, ~150 residents, Kolhapur / Maharashtra-style climate and festivals).

Patterns built in (the model must rediscover them):
  - Weekends use ~15-20% more water than weekdays
  - Festival days use ~25% more on average (main days higher) (guests, cooking, cleaning); Ganeshotsav and
    Diwali span several days
  - Summer (Mar-May) uses more water, heavy-rain days slightly less
  - Slow yearly growth (more residents / more appliances)
  - Municipal supply is weaker in summer and has random outage days
  - The storage tank is simulated day by day, so shortages emerge naturally

Run:  python data/generate_data.py
Out:  data/water_data.csv

Real data? If you get a real building CSV, keep the same column names and the
rest of the project works unchanged.
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- settings
SEED = 42
END_DATE = date(2026, 10, 18)          # last day of data (yesterday)
N_DAYS = 1095                          # 3 years
START_DATE = END_DATE - timedelta(days=N_DAYS - 1)

FLATS = 40
RESIDENTS_PER_FLAT = 3.8               # ~152 residents
PER_CAPITA_L = 135                     # litres/person/day (Indian urban norm)
TANK_CAPACITY_L = 40_000               # total storage (~2 days of use)
BASE_SUPPLY_L = 20_500                 # average daily inflow (municipal + borewell)

OUT_FILE = Path(__file__).parent / "water_data.csv"

# Approximate festival dates (edit freely).
# (name, date, days_before, days_after)
FESTIVALS = [
    # 2023
    ("Dussehra", date(2023, 10, 24), 0, 0),
    ("Diwali", date(2023, 11, 12), 2, 2),
    ("Christmas", date(2023, 12, 25), 0, 0),
    # 2024
    ("Makar Sankranti", date(2024, 1, 15), 0, 0),
    ("Republic Day", date(2024, 1, 26), 0, 0),
    ("Holi", date(2024, 3, 25), 1, 0),
    ("Gudi Padwa", date(2024, 4, 9), 0, 0),
    ("Eid", date(2024, 4, 11), 0, 0),
    ("Ram Navami", date(2024, 4, 17), 0, 0),
    ("Independence Day", date(2024, 8, 15), 0, 0),
    ("Raksha Bandhan", date(2024, 8, 19), 0, 0),
    ("Ganeshotsav", date(2024, 9, 7), 0, 9),
    ("Dussehra", date(2024, 10, 12), 0, 0),
    ("Diwali", date(2024, 11, 1), 2, 2),
    ("Christmas", date(2024, 12, 25), 0, 0),
    # 2025
    ("Makar Sankranti", date(2025, 1, 14), 0, 0),
    ("Republic Day", date(2025, 1, 26), 0, 0),
    ("Holi", date(2025, 3, 14), 1, 0),
    ("Gudi Padwa", date(2025, 3, 30), 0, 0),
    ("Eid", date(2025, 3, 31), 0, 0),
    ("Ram Navami", date(2025, 4, 6), 0, 0),
    ("Independence Day", date(2025, 8, 15), 0, 0),
    ("Raksha Bandhan", date(2025, 8, 9), 0, 0),
    ("Ganeshotsav", date(2025, 8, 27), 0, 9),
    ("Dussehra", date(2025, 10, 2), 0, 0),
    ("Diwali", date(2025, 10, 20), 2, 2),
    ("Christmas", date(2025, 12, 25), 0, 0),
    # 2026
    ("Makar Sankranti", date(2026, 1, 14), 0, 0),
    ("Republic Day", date(2026, 1, 26), 0, 0),
    ("Holi", date(2026, 3, 4), 1, 0),
    ("Gudi Padwa", date(2026, 3, 19), 0, 0),
    ("Eid", date(2026, 3, 21), 0, 0),
    ("Ram Navami", date(2026, 3, 26), 0, 0),
    ("Independence Day", date(2026, 8, 15), 0, 0),
    ("Raksha Bandhan", date(2026, 8, 28), 0, 0),
    ("Ganeshotsav", date(2026, 9, 14), 0, 9),
    ("Dussehra", date(2026, 10, 20), 0, 0),   # falls just after the data ends
    # 2023 early-range leftovers are covered above; 2023 Holi etc. are before START_DATE
]
SIDE_DAY_INTENSITY = 0.6   # festival "side" days are a bit weaker than the main day
LOW_KEY = {"Republic Day", "Independence Day", "Christmas"}  # smaller bump

# Kolhapur-like monthly climate (Jan..Dec)
MONTH_TEMP = [23, 25, 28, 30, 30, 26, 24.5, 24.5, 25, 26, 24.5, 23]
RAIN_PROB = [0.03, 0.02, 0.04, 0.08, 0.15, 0.65, 0.85, 0.85, 0.65, 0.30, 0.10, 0.04]
RAIN_SCALE = [3, 3, 4, 6, 8, 18, 22, 20, 16, 10, 5, 3]


def build_calendar() -> pd.DataFrame:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    df = pd.DataFrame({"date": dates})
    df["day_of_week"] = df["date"].dt.dayofweek          # 0 = Mon ... 6 = Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # festival flags
    names, intensity = {}, {}
    for name, d0, before, after in FESTIVALS:
        for off in range(-before, after + 1):
            d = pd.Timestamp(d0 + timedelta(days=off))
            w = 1.0 if off == 0 else SIDE_DAY_INTENSITY
            if name in LOW_KEY:
                w *= 0.5
            if d not in intensity or w > intensity[d]:
                intensity[d], names[d] = w, name
    df["festival_name"] = df["date"].map(names).fillna("")
    df["is_festival"] = (df["festival_name"] != "").astype(int)
    df["_fest_intensity"] = df["date"].map(intensity).fillna(0.0)   # hidden driver
    return df


def add_weather(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    m = df["date"].dt.month.values - 1
    noise = pd.Series(rng.normal(0, 2.2, len(df))).rolling(3, min_periods=1).mean().values
    df["temp_c"] = np.round(np.array(MONTH_TEMP)[m] + noise, 1)

    rainy = rng.random(len(df)) < np.array(RAIN_PROB)[m]
    amount = rng.gamma(1.3, np.array(RAIN_SCALE)[m])
    df["rainfall_mm"] = np.round(np.where(rainy, amount, 0.0), 1)
    return df


def add_occupancy(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    month = df["date"].dt.month
    day = df["date"].dt.day
    occ = (
        88
        + 6 * df["is_weekend"]
        + 5 * df["_fest_intensity"]
        - np.where(month == 5, 6, 0)                                   # May holidays
        - np.where(((month == 12) & (day >= 25)) | ((month == 1) & (day <= 2)), 4, 0)
        + rng.normal(0, 2, len(df))
    )
    df["occupancy_pct"] = np.round(np.clip(occ, 70, 100), 1)
    return df


def add_demand(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    residents = FLATS * RESIDENTS_PER_FLAT
    t_years = np.arange(len(df)) / 365.0

    direct = 1 + 0.10 * df["is_weekend"] + 0.22 * df["_fest_intensity"]
    temp = 1 + 0.008 * (df["temp_c"] - 27)
    rain = np.where(df["rainfall_mm"] > 10, 0.97, 1.0)
    trend = 1 + 0.015 * t_years
    noise = rng.lognormal(0, 0.05, len(df))

    demand = (
        residents * df["occupancy_pct"] / 100 * PER_CAPITA_L
        * direct * temp * rain * trend * noise
    )
    return demand.values


def add_supply(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    month = df["date"].dt.month.values
    season = np.where(np.isin(month, [3, 4, 5]), 0.93,        # summer: weaker supply
             np.where(np.isin(month, [6, 7, 8, 9]), 1.05, 1.0))
    supply = BASE_SUPPLY_L * season * rng.normal(1.0, 0.07, len(df))

    # random outages (pipeline repair, power cut): ~4% of days, sometimes 2 days in a row
    n = len(df)
    outage = np.zeros(n, dtype=bool)
    for i in np.where(rng.random(n) < 0.04)[0]:
        outage[i] = True
        if i + 1 < n and rng.random() < 0.4:
            outage[i + 1] = True
    supply = np.where(outage, supply * rng.uniform(0.0, 0.3, n), supply)
    df["supply_litres"] = np.round(supply).astype(int)
    return df


def simulate_tank(df: pd.DataFrame, demand: np.ndarray) -> pd.DataFrame:
    """Day-by-day tank balance. Shortage = demand the tank could not meet."""
    opening, consumption, shortage, level = [], [], [], TANK_CAPACITY_L * 0.7
    for i in range(len(df)):
        opening.append(level)
        available = level + df["supply_litres"].iloc[i]
        used = min(demand[i], available)
        shortage.append(demand[i] - used)
        consumption.append(used)
        level = min(TANK_CAPACITY_L, available - used)         # overflow is wasted

    df["tank_capacity_l"] = TANK_CAPACITY_L
    df["opening_level_l"] = np.round(opening).astype(int)
    df["consumption_l"] = np.round(consumption).astype(int)
    df["shortage_l"] = np.round(shortage).astype(int)
    df["water_available"] = (df["shortage_l"] <= 0).astype(int)
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = build_calendar()
    df = add_weather(df, rng)
    df = add_occupancy(df, rng)
    demand = add_demand(df, rng)
    df = add_supply(df, rng)
    df = simulate_tank(df, demand)

    cols = [
        "date", "day_of_week", "is_weekend", "is_festival", "festival_name",
        "temp_c", "rainfall_mm", "occupancy_pct", "supply_litres",
        "tank_capacity_l", "opening_level_l", "consumption_l",
        "shortage_l", "water_available",
    ]
    df = df[cols]
    df.to_csv(OUT_FILE, index=False)

    # ---- quick sanity report
    normal = df[(df.is_weekend == 0) & (df.is_festival == 0)].consumption_l.mean()
    weekend = df[(df.is_weekend == 1) & (df.is_festival == 0)].consumption_l.mean()
    festival = df[df.is_festival == 1].consumption_l.mean()
    print(f"Saved {len(df)} rows -> {OUT_FILE}")
    print(f"Date range            : {df.date.min().date()} to {df.date.max().date()}")
    print(f"Avg normal weekday    : {normal:,.0f} L")
    print(f"Avg weekend           : {weekend:,.0f} L  ({(weekend / normal - 1) * 100:+.1f}%)")
    print(f"Avg festival day      : {festival:,.0f} L  ({(festival / normal - 1) * 100:+.1f}%)")
    print(f"Shortage days         : {(df.water_available == 0).sum()} "
          f"({(df.water_available == 0).mean() * 100:.1f}%)")
    print(f"Nulls                 : {int(df.drop(columns='festival_name').isna().sum().sum())}")


if __name__ == "__main__":
    main()
