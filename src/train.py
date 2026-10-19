import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import FEATURE_COLS, TARGET, load_and_prepare

TEST_DAYS = 182  # last ~6 months


def time_split(df, test_days=TEST_DAYS):
    """Time-based split: earlier rows train, last rows test. Never shuffle time data."""
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    return train, test


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mae_pct = mae / y_true.mean() * 100
    return {"Model": name, "MAE": round(mae), "MAE %": round(mae_pct, 1),
            "RMSE": round(rmse), "R2": round(r2, 3)}


def main():
    df = load_and_prepare()
    train, test = time_split(df)

    print("Train:", train["date"].min().date(), "to", train["date"].max().date(), f"({len(train)} rows)")
    print("Test: ", test["date"].min().date(), "to", test["date"].max().date(), f"({len(test)} rows)")
    print()

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test, y_test = test[FEATURE_COLS], test[TARGET]

    baseline = LinearRegression().fit(X_train, y_train)
    forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
    forest.fit(X_train, y_train)

    results = [
        evaluate("LinearRegression (baseline)", y_test, baseline.predict(X_test)),
        evaluate("RandomForest", y_test, forest.predict(X_test)),
    ]
    print(pd.DataFrame(results).to_string(index=False))
    print()
    print("Avg daily consumption in test period:", round(y_test.mean()), "L")


if __name__ == "__main__":
    main()