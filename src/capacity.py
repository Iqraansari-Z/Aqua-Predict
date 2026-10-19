from datetime import timedelta

import pandas as pd

from src.predict import predict_demand

SAFE_RATIO = 1.2


def check_availability(predicted_demand, tank_level, supply_expected, tank_capacity=40_000):
    """Compare predicted demand with the water we will have (tank + supply)."""
    available = tank_level + supply_expected
    ratio = available / predicted_demand
    shortfall = max(0.0, predicted_demand - available)

    if ratio >= SAFE_RATIO:
        status, icon = "Safe", "🟢"
        message = f"Water is sufficient. Coverage is {ratio:.2f}x predicted demand."
    elif ratio >= 1.0:
        status, icon = "Tight", "🟡"
        message = f"Water is tight. Only {available - predicted_demand:,.0f} L of buffer is left."
    else:
        status, icon = "Shortage", "🔴"
        message = f"Expected shortage of {shortfall:,.0f} L."

    # Tank level at the start of the next day (capped at capacity, never below 0)
    next_level = min(tank_capacity, max(0.0, available - predicted_demand))

    return {"status": status, "icon": icon, "message": message,
            "demand": predicted_demand, "available": available,
            "ratio": ratio, "shortfall": shortfall, "next_level": next_level}


def forecast_week(start_date, tank_level, supply_expected, tank_capacity,
                  temp_c, occupancy_pct, days=7, festival_dates=()):
    """Predict each day, carry the tank level forward, flag the first shortage day."""
    festival_dates = {pd.Timestamp(d).date() for d in festival_dates}
    start = pd.Timestamp(start_date)
    rows, level = [], tank_level

    for i in range(days):
        day = start + timedelta(days=i)
        is_fest = 1 if day.date() in festival_dates else None   # None = auto-detect
        demand = predict_demand(day, temp_c, occupancy_pct, is_festival=is_fest)
        res = check_availability(demand, level, supply_expected, tank_capacity)
        rows.append({"date": day.date(), "weekday": day.strftime("%a"),
                     "opening_level": round(level), "supply": round(supply_expected),
                     "demand": round(demand), "available": round(res["available"]),
                     "status": f"{res['icon']} {res['status']}",
                     "shortfall": round(res["shortfall"])})
        level = res["next_level"]

    df = pd.DataFrame(rows)
    shortage_days = df[df["status"].str.contains("Shortage")]
    first_shortage = shortage_days["date"].iloc[0] if len(shortage_days) else None
    return df, first_shortage


if __name__ == "__main__":
    # --- 1. Replay the real dataset: does our rule reproduce shortage_l? ---
    d = pd.read_csv("data/water_data.csv")
    demand = d["consumption_l"] + d["shortage_l"]
    mine = (demand - (d["opening_level_l"] + d["supply_litres"])).clip(lower=0)
    diff = (mine - d["shortage_l"]).abs()
    print(f"Replay check: max difference vs dataset shortage_l = {diff.max():.0f} L "
          f"({(diff > 5).sum()} rows differ by more than 5 L)")

    # --- 2. Three scenarios ---
    print("\nNormal weekday, healthy tank:")
    print(check_availability(18_300, 25_000, 20_000)["icon"], check_availability(18_300, 25_000, 20_000)["message"])
    print("Weekend, low tank:")
    r = check_availability(20_500, 2_000, 19_000); print(r["icon"], r["message"])
    print("Festival, low tank, weak supply:")
    r = check_availability(21_000, 1_000, 12_000); print(r["icon"], r["message"])

    # --- 3. 7-day forecast with a low tank ---
    table, first = forecast_week("2026-10-20", 8_000, 19_000, 40_000, 30, 85)
    print("\n", table.to_string(index=False))
    print("First shortage day:", first)