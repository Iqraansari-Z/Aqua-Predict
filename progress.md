# AquaPredict — Project Progress & Build Guide

> **Residential Water Consumption & Availability Prediction System**
> *Deadline: tomorrow. Concepts first, but keep it fast. Working demo beats perfect code.*

---

## 🧠 Concepts First

### What is AquaPredict?

A system for a **residential building** that looks at **3 years of past water usage** and answers two questions:

1. **How much water will the building consume?** (Regression: litres/day)
2. **Will water be available or will we run short?** (Capacity check: tank + supply vs. predicted demand)

It learns **patterns**: weekdays vs weekends, festival days (guests, cooking, cleaning), seasons (summer = more use), and temperature/rain.

### Key Concepts You Must Understand

| Concept | Plain English |
|---|---|
| **Time-series data** | Data recorded day by day (here: 3 years × 365 days ≈ 1,095 rows) |
| **Feature** | An input the model learns from (day of week, is_festival, temperature...) |
| **Target** | What we want to predict (`consumption_l`) |
| **Feature engineering** | Turning raw dates into useful signals (weekend flag, festival flag, month, season) |
| **Regression model** | Predicts a number (litres of water tomorrow) |
| **Train/test split (time-based)** | Train on first 2.5 years, test on the last 6 months. Never shuffle time data. |
| **MAE / RMSE / R²** | How wrong the model is on average (lower MAE/RMSE = better, higher R² = better) |
| **Capacity check** | Available water = tank level + expected supply inflow. If predicted demand > available → **shortage alert** |
| **Risk level** | 🟢 Safe / 🟡 Tight / 🔴 Shortage, based on the demand vs availability ratio |
| **Synthetic data** | If we don't have the real dataset, we generate realistic fake data with built-in patterns |

### The Core Logic

```
Date (+ weather, occupancy)
        ↓
[ Feature Engineering ]  → weekday/weekend, festival, month, season, temp
        ↓
[ ML Model (Random Forest / XGBoost) ] → predicted consumption (litres)
        ↓
[ Capacity Check ]  predicted demand  vs  (tank level + supply inflow)
        ↓
   🟢 Safe  |  🟡 Tight  |  🔴 Shortage
        ↓
[ Streamlit Dashboard ] → charts, forecast, alerts, "what-if" inputs
```

> **Important design point:** "Water available or not" = **predicted consumption vs. tank capacity + supply**. We do NOT train a separate classifier on the same data (that would leak the answer). The model predicts demand; the capacity rule decides availability.

---

## 📁 Folder Structure (keep it small!)

```
aquapredict/
│
├── README.md
├── progress.md                    ← YOU ARE HERE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── generate_data.py           ← Creates synthetic 3-year dataset (if no real data)
│   └── water_data.csv             ← The dataset (real or generated)
│
├── src/
│   ├── features.py                ← Feature engineering (weekend, festival, season...)
│   ├── train.py                   ← Train + evaluate + save model
│   ├── predict.py                 ← Load model → predict demand for any date(s)
│   └── capacity.py                ← Demand vs supply → Safe / Tight / Shortage
│
├── models/
│   └── water_model.joblib         ← Saved trained model (commit this so deployment works)
│
├── app.py                         ← Streamlit app (entire frontend)
│
└── docs/
    └── demo_script.md             ← What to show/say in the demo
```

> No FastAPI, no database, no React. **Streamlit loads the model directly.** One less thing to break and deploy.

---

## 🛠️ Free Stack (everything is free, no card needed)

| Component | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | Open source |
| Data handling | pandas, numpy | Open source |
| ML | scikit-learn (RandomForest / GradientBoosting) | Open source, fast to train on 1k rows |
| Optional ML | XGBoost | Open source (only if time permits) |
| Festival/holiday calendar | `holidays` library (India / Maharashtra) | Open source |
| Charts | Plotly | Open source, interactive in Streamlit |
| **Frontend** | **Streamlit** | Open source |
| Model saving | joblib | Open source |
| Version control | GitHub | Free |
| **Deployment** | **Streamlit Community Cloud** | Free tier, deploys straight from GitHub |

`requirements.txt`:
```
pandas
numpy
scikit-learn
holidays
plotly
streamlit
joblib
```

---

## ⏱️ Time Plan (assume ~12–14 working hours total)

| Phase | Goal | Time |
|---|---|---|
| 0 | Setup | 0.5 h |
| 1 | Dataset | 1.5 h |
| 2 | Explore patterns (EDA) | 1 h |
| 3 | Features + Model | 2.5 h |
| 4 | Capacity / availability logic | 1.5 h |
| 5 | Streamlit app | 3 h |
| 6 | Deploy | 1 h |
| 7 | Polish + demo prep | 1.5 h |

> **Rule:** If a phase runs over, cut the optional items (marked 🔸) and move on. Never skip deployment or the demo prep.

---

## ✅ Build Checklist — Phase by Phase

### Phase 0 — Setup (0.5 h) ✅ DONE
- [x] Create GitHub repo `Aqua-Predict` → https://github.com/Iqraansari-Z/Aqua-Predict
- [x] Create virtual env: `uv venv` then `.venv\Scripts\Activate.ps1`
- [x] Create `requirements.txt` (above) and install: `uv pip install -r requirements.txt`
- [x] Create folder structure (empty files are fine)
- [x] Verify: all imports OK, `streamlit --version` → 1.66.0
- [x] Git initialised, first commit made, pushed to `origin/main`

### Phase 1 — Dataset (1.5 h) ✅ DONE
**Concept:** The model is only as good as the data. We need one row per day for 3 years.

**Columns:**

| Column | Meaning |
|---|---|
| `date` | Day |
| `day_of_week` | 0 = Mon … 6 = Sun |
| `is_weekend` | 1 if Sat/Sun |
| `is_festival` | 1 if a festival/holiday |
| `festival_name` | e.g. Diwali, Ganesh Chaturthi, Holi |
| `temp_c` | Average temperature |
| `rainfall_mm` | Rainfall |
| `occupancy_pct` | % of flats occupied that day |
| `supply_litres` | Water supplied by municipality/borewell that day |
| `tank_capacity_l` | Total storage capacity (fixed, e.g. 50,000 L) |
| `opening_level_l` | Tank level at start of the day |
| `consumption_l` | **TARGET:** total litres used |

**Dataset search result (checked 19-Oct-2026):** No public dataset matches this problem exactly (3 years + daily + one residential building + Indian festivals + tank level/supply). Closest real ones:

| Dataset | What it has | Why it doesn't fit alone |
|---|---|---|
| IEEE DataPort — *Water Consumption Dataset of Smart City Users* | Daily litres per house/flat, 365 days, 50 users | Only 1 year, per household, no festivals, may need a free IEEE login |
| Mendeley — *Hot-water consumption, 10 apartment buildings, Kaunas* | Hourly, residential buildings, multi-month | Hot water only, Lithuania |
| Danish smart meter dataset (*Data in Brief*, 2023) | Hourly water meters, up to 5 years, building info | Mostly commercial buildings, Denmark |
| HydroShare / Mendeley (Utah, Sweden) | High-resolution residential data | Only 3–4 weeks |
| Flume / HydroShare US city-level | Daily residential use, 39 US cities | City level, US habits |

None has tank level, municipal supply or Indian festivals, so the "will water be available" part must be simulated anyway.

**Decision: hybrid approach (fastest and best fit)**
1. Generate the 3-year dataset ourselves (`generate_data.py`), tuned to Indian usage (~135 L/person/day) with weekend, festival and season effects.
2. 🔸 Optionally download the IEEE DataPort daily dataset to sanity-check that our weekly/seasonal shapes look realistic, and mention it as a validation source in the demo.
3. In the README and demo, state clearly that the data is synthetic but pattern-calibrated, and that the pipeline works unchanged with a real building's CSV (use the same column names).

**Tasks:**
- [x] If the user has a real building CSV → load it, rename columns to match the above, fill gaps
- [x] Otherwise (default) → write `data/generate_data.py` with realistic built-in patterns:
  - Base usage ≈ (flats × people × ~135 L/person/day)
  - **Weekends:** +15–20% (people at home)
  - **Festivals:** +25–40% (guests, cooking, cleaning). Use the `holidays` library or a hand-made festival list
  - **Summer (Apr–Jun):** +10–15%; monsoon slightly lower
  - Add random noise (±5–8%) so it isn't perfectly clean
  - Simulate tank level day by day: `next_opening = min(capacity, opening + supply - consumption)`
- [x] Save to `data/water_data.csv`
- [x] Verify: `df.shape` ≈ (1095, 12), no nulls

### Phase 2 — Explore the Patterns (1 h)
**Concept:** EDA = confirm the story the data tells *before* modelling. These charts also go straight into the demo.
- [ ] Average consumption: weekday vs weekend (bar chart)
- [ ] Average consumption: festival vs normal day
- [ ] Monthly average consumption (line chart, shows seasons)
- [ ] Consumption over time for 3 years (line chart)
- [ ] Write 3–4 one-line insights, e.g. *"Weekends use ~18% more water"*
- [ ] 🔸 Heatmap: day of week × month

### Phase 3 — Features + Model (2.5 h)
**Concept:** Convert raw columns into signals the model can use, then train and check honestly.
- [ ] `src/features.py`: build features — `day_of_week`, `is_weekend`, `is_festival`, `month`, `season`, `temp_c`, `rainfall_mm`, `occupancy_pct`
  - 🔸 Lag features: yesterday's consumption, 7-day rolling average (large accuracy boost)
  - 🔸 `days_to_next_festival`
- [ ] **Time-based split:** train = first ~2.5 years, test = last ~6 months (no shuffling)
- [ ] Train a baseline: **LinearRegression** (to compare)
- [ ] Train main model: **RandomForestRegressor** (or GradientBoosting)
- [ ] Evaluate with **MAE, RMSE, R²** — print a comparison table
- [ ] Plot **actual vs predicted** on the test period
- [ ] Plot **feature importance** (shows weekend/festival matter)
- [ ] Save model: `joblib.dump(model, "models/water_model.joblib")`
- [ ] `src/predict.py`: function `predict_demand(date, temp, occupancy, ...) → litres`
- [ ] Verify: a Sunday-festival prediction > a normal Tuesday prediction

> **Target:** R² > 0.85 and MAE within ~8% of average daily use. If worse, add lag features before anything else.

### Phase 4 — Capacity & Availability Logic (1.5 h)
**Concept:** This is the "will water be available?" part — it's simple rules on top of the prediction.
- [ ] `src/capacity.py`: function `check_availability(predicted_demand, tank_level, supply_expected, tank_capacity)`
- [ ] Compute:
  - `available = tank_level + supply_expected`
  - `coverage_ratio = available / predicted_demand`
- [ ] Classify:
  - 🟢 **Safe:** ratio ≥ 1.2
  - 🟡 **Tight:** 1.0 ≤ ratio < 1.2
  - 🔴 **Shortage:** ratio < 1.0
- [ ] Output: status, shortfall in litres (if any), and a plain message ("Expected shortage of 4,200 L on Sunday")
- [ ] **7-day forecast loop:** predict each of the next 7 days, carry the tank level forward day by day, flag the first day of shortage
- [ ] 🔸 Suggest an action: "Request an extra tanker / reduce non-essential use"
- [ ] Verify: lowering supply or raising festival demand flips the status to 🔴

### Phase 5 — Streamlit Frontend (3 h)
**Concept:** One `app.py`, sidebar for inputs, tabs for content. Don't over-design.

- [ ] Page config + title + short description
- [ ] **Sidebar inputs:** date picker, current tank level, expected supply today, tank capacity, occupancy %, temperature
- [ ] **Tab 1 — Prediction:**
  - Big metric cards: *Predicted consumption*, *Available water*, *Status (🟢/🟡/🔴)*
  - Message banner (`st.success` / `st.warning` / `st.error`)
- [ ] **Tab 2 — 7-Day Forecast:**
  - Plotly chart: predicted demand vs available water per day
  - Table with the status per day
- [ ] **Tab 3 — Usage Patterns (from EDA):**
  - Weekday vs weekend chart
  - Festival vs normal chart
  - Monthly trend
  - Insight text lines
- [ ] **Tab 4 — Model Performance:**
  - Actual vs predicted chart
  - MAE / RMSE / R² metrics
  - Feature importance
- [ ] Use `@st.cache_resource` for the model and `@st.cache_data` for the CSV (keeps the app fast)
- [ ] 🔸 "What-if" toggle: *"What if tomorrow is a festival?"*
- [ ] 🔸 Upload your own CSV tab
- [ ] Verify: `streamlit run app.py` works end-to-end with no errors

### Phase 6 — Deploy (1 h)
- [ ] Make sure `models/water_model.joblib` and `data/water_data.csv` are **committed** (Streamlit Cloud reads from the repo)
- [ ] Pin versions in `requirements.txt` if you hit a version error
- [ ] Push to GitHub (public repo)
- [ ] Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → New app → select repo, branch, `app.py` → Deploy
- [ ] Verify the live URL works on your phone too
- [ ] Keep a local backup (screen recording) in case the demo Wi-Fi fails

### Phase 7 — Polish & Demo (1.5 h)
- [ ] Write `README.md`: problem, approach, how to run, screenshots
- [ ] Prepare 3 demo scenarios (`docs/demo_script.md`):
  1. **Normal weekday** → 🟢 Safe
  2. **Weekend + low tank** → 🟡 Tight
  3. **Festival (e.g. Ganesh Chaturthi / Diwali) + low supply** → 🔴 Shortage alert + suggestion
- [ ] Prepare a 1-minute pitch: *problem → data → model → live demo → impact*
- [ ] Be ready to explain: why time-based split, what features matter most, how availability is decided
- [ ] Final check: fresh clone runs with `pip install -r requirements.txt` and `streamlit run app.py`

---

## 🔥 If Time Runs Out — Cut in This Order

1. Drop all 🔸 optional items
2. Drop Tab 4 (Model Performance) — keep metrics in the README
3. Drop the 7-day forecast loop → keep single-day prediction only
4. **Never cut:** dataset, working model, capacity status, deployed app

---

## 📚 Learning Log

> One short entry per session.

### Template
```
### [Date] — [Phase]
**What I learned:**
**Where I got stuck:**
**How I solved it:**
**Next goal:**
```

### Log
*(start here)*

---

## 🚦 Quick Start Commands

```bash
# 1. Create & activate environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset (skip if using real data)
python data/generate_data.py

# 4. Train the model
python src/train.py

# 5. Run the app
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 🔑 Key Files to Know

| File | Purpose |
|---|---|
| `data/generate_data.py` | Builds the 3-year dataset with weekend/festival patterns |
| `src/features.py` | Turns dates into model-ready features |
| `src/train.py` | Trains, evaluates, saves the model |
| `src/capacity.py` | The "water available or not" decision logic |
| `app.py` | The entire Streamlit frontend |
| `models/water_model.joblib` | Trained model (must be committed for deployment) |

---

*Last updated: 19-Oct-2026 — Plan created. Start at Phase 0.*
