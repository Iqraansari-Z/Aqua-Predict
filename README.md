# 💧 AquaPredict

**Residential water consumption and availability prediction.**

AquaPredict learns from 3 years of daily water usage in a residential building and answers two questions:

1. **How much water will the building need?** (regression, litres/day)
2. **Will water be available, or will we run short?** (tank level + supply vs predicted demand)

🔗 **Live app:** https://aqua-predict-bmih9xvauuauebgmg277ta.streamlit.app/

## How it works

```
Date (+ weather, occupancy)
        ↓
Feature engineering  → weekday/weekend, festival, month, season, temp, rain, occupancy
        ↓
Random Forest model  → predicted demand (litres)
        ↓
Capacity check       → predicted demand vs (tank level + supply)
        ↓
🟢 Safe  |  🟡 Tight  |  🔴 Shortage  (+ tanker suggestion)
```

**Design point:** the model predicts *demand* only. Availability is decided by a simple capacity rule, not by a second classifier, so the answer can't leak into the model.

| Status | Rule (coverage = available water / predicted demand) |
|---|---|
| 🟢 Safe | coverage ≥ 1.2 |
| 🟡 Tight | 1.0 ≤ coverage < 1.2 |
| 🔴 Shortage | coverage < 1.0 |

## Data

- 1,095 daily rows (2023-10-20 to 2026-10-18), one residential building.
- **Synthetic but pattern-calibrated**: weekend, festival and occupancy effects with random noise. No public dataset has daily tank levels, municipal supply and Indian festivals together.
- The pipeline works unchanged with a real building's CSV that uses the same column names.
- The target is **demand = consumption + shortage**, because recorded consumption is capped by availability on empty-tank days.

## Results

Time-based split: train on the first ~2.5 years, test on the last 182 days (no shuffling).

| Model | MAE | MAE % | RMSE | R² |
|---|---|---|---|---|
| Linear Regression (baseline) | 869 L | 4.4% | 1,115 L | 0.667 |
| **Random Forest** | 889 L | 4.5% | 1,125 L | 0.661 |

- MAE is under 5% of average daily demand.
- R² is limited by the random noise built into the synthetic data. Lag features were tested and gave no real gain.
- Top driver: occupancy (importance 0.63). Weekends use about 16% more water, festival days about 19% more.

## App tabs

1. **Prediction:** predicted demand, available water, status, tanker suggestion on 🔴
2. **7-Day Forecast:** demand vs available water per day, first shortage day
3. **Usage Patterns:** weekday vs weekend, festival vs normal, monthly trend
4. **Model Performance:** metrics, actual vs predicted, feature importance

## Run locally

```bash
git clone https://github.com/Iqraansari-Z/Aqua-Predict.git
cd Aqua-Predict
python -m venv .venv
.venv\Scripts\activate        # Windows (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Optional: regenerate the data and retrain the model.

```bash
python data/generate_data.py
python -m src.train
```

## Project structure

```
app.py               Streamlit app
data/                dataset + generator
src/features.py      feature engineering
src/train.py         training and evaluation
src/predict.py       predict_demand()
src/capacity.py      availability check + 7-day forecast
src/eda.py           usage-pattern charts
models/              saved model + metrics
docs/demo_script.md  demo walkthrough
```

## Known limitations

- Festival auto-detection matches by (month, day), so moving festivals (Diwali, Holi, Ganesh Chaturthi) can be mislabeled for future dates. The app has a festival override for this.
- The data is synthetic, so reported accuracy shows the pipeline works, not real-world performance.

## Tech stack

Python, pandas, scikit-learn, Plotly, Streamlit. Deployed on Streamlit Community Cloud.