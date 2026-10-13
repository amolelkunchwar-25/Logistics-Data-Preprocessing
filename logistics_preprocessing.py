"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Simulated shipment-level dataset modelled on a typical public logistics /
supply-chain shipment dataset (e.g. Kaggle "Supply Chain Shipment Pricing" /
"Logistics and Supply Chain" style records).

Run:  python logistics_preprocessing.py
Outputs: raw_shipments.csv, clean_shipments.csv, figures/*.png, report_stats.json
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
os.makedirs("figures", exist_ok=True)
stats = {}

# ---------------------------------------------------------------------------
# STEP 1: DATA COLLECTION (SIMULATION)
# ---------------------------------------------------------------------------
N = 5000
cities = ["Mumbai", "Delhi", "Chennai", "Kolkata", "Bengaluru", "Nagpur", "Pune", "Hyderabad"]
modes = ["Road", "Rail", "Air", "Sea"]
carriers = ["BlueDart", "Delhivery", "DTDC", "VRL", "SafeExpress"]

origin = rng.choice(cities, N)
dest = rng.choice(cities, N)
mode = rng.choice(modes, N, p=[0.55, 0.20, 0.15, 0.10])
distance = rng.gamma(shape=4, scale=250, size=N).round(1)               # km
weight = rng.lognormal(mean=6, sigma=0.8, size=N).round(1)             # kg
speed = pd.Series(mode).map({"Road": 45, "Rail": 35, "Air": 600, "Sea": 25}).values   # avg km/h
transit = (distance / (speed * 24) + rng.normal(1.5, 0.5, N)).clip(0.3)   # days = travel + handling
cost_rate = pd.Series(mode).map({"Road": 9, "Rail": 5, "Air": 38, "Sea": 3}).values
cost = (distance * cost_rate * (weight / 500) ** 0.5 + rng.normal(0, 400, N)).clip(100)
order_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
inventory = rng.poisson(120, N).astype(float)
delivered = np.where(rng.random(N) < 0.92, "Delivered", "Delayed")

raw = pd.DataFrame({
    "shipment_id": [f"SH{100000 + i}" for i in range(N)],
    "order_date": order_date.strftime("%Y-%m-%d"),
    "origin": origin, "destination": dest, "mode": mode, "carrier": rng.choice(carriers, N),
    "distance_km": distance, "weight_kg": weight, "transit_days": transit.round(2),
    "freight_cost_inr": cost.round(2), "stock_on_hand": inventory, "status": delivered,
})

# ---- inject realistic data-quality problems ----
def idx(frac): return rng.choice(N, int(N * frac), replace=False)

# missing values (MCAR-style + one column missing more often)
for col, frac in [("weight_kg", 0.06), ("transit_days", 0.04), ("freight_cost_inr", 0.03),
                  ("stock_on_hand", 0.08), ("carrier", 0.02)]:
    raw.loc[idx(frac), col] = np.nan
# extreme outliers / data-entry errors
raw.loc[idx(0.006), "weight_kg"] *= 100
raw.loc[idx(0.005), "freight_cost_inr"] *= 40
raw.loc[idx(0.004), "transit_days"] = rng.uniform(60, 200, int(N * 0.004))
raw.loc[idx(0.003), "distance_km"] = -abs(raw["distance_km"].sample(int(N * 0.003), random_state=1).values)
# inconsistent categorical labels + whitespace
raw.loc[idx(0.05), "mode"] = raw["mode"].str.lower()
raw.loc[idx(0.03), "origin"] = raw["origin"].str.upper() + " "
raw.loc[idx(0.02), "status"] = "delivered"
# duplicate rows
dups = raw.sample(int(N * 0.02), random_state=7)
raw = pd.concat([raw, dups], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
# wrong dtype: dates/numbers stored as text in places
raw.loc[idx(0.01), "order_date"] = "13/05/2025"
raw.to_csv("raw_shipments.csv", index=False)

# ---------------------------------------------------------------------------
# STEP 2: DATA INSPECTION / PROFILING
# ---------------------------------------------------------------------------
df = pd.read_csv("raw_shipments.csv")
stats["raw_shape"] = list(df.shape)
stats["missing_before"] = df.isna().sum()[df.isna().sum() > 0].to_dict()
stats["duplicates"] = int(df.duplicated().sum())
stats["raw_describe"] = df[["distance_km", "weight_kg", "transit_days", "freight_cost_inr", "stock_on_hand"]] \
    .describe().round(2).to_dict()
stats["mode_labels_before"] = df["mode"].value_counts().to_dict()
stats["negative_distance"] = int((df["distance_km"] < 0).sum())

# ---------------------------------------------------------------------------
# STEP 3: CLEANING
# ---------------------------------------------------------------------------
# 3.1 remove exact duplicates
df = df.drop_duplicates().reset_index(drop=True)
stats["after_dedup_rows"] = len(df)

# 3.2 standardise text + types
for c in ["origin", "destination", "mode", "carrier", "status"]:
    df[c] = df[c].astype("string").str.strip().str.title()
df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", dayfirst=False, errors="coerce")
stats["mode_labels_after"] = df["mode"].value_counts().to_dict()
stats["bad_dates"] = int(df["order_date"].isna().sum())

# 3.3 impossible values -> NaN (negative distance is physically impossible)
df.loc[df["distance_km"] <= 0, "distance_km"] = np.nan

# 3.4 outlier detection (IQR rule + visual check); done BEFORE imputation
num_cols = ["distance_km", "weight_kg", "transit_days", "freight_cost_inr"]

fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, c in zip(ax, num_cols):
    a.boxplot(df[c].dropna(), flierprops=dict(marker="o", markersize=3, alpha=.5))
    a.set_title(c); a.set_yscale("log")
plt.suptitle("Boxplots BEFORE outlier treatment (log scale)")
plt.tight_layout(); plt.savefig("figures/boxplots_before.png", dpi=150); plt.close()

def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

outlier_counts = {}
for c in ["weight_kg", "transit_days", "freight_cost_inr"]:
    # work on log scale for right-skewed variables so the rule is not over-aggressive
    s = np.log1p(df[c].dropna())
    lo, hi = iqr_bounds(s, k=3.0)          # k=3 -> "extreme" outliers only
    lo_o, hi_o = np.expm1(lo), np.expm1(hi)
    mask = (df[c] > hi_o) | (df[c] < lo_o)
    outlier_counts[c] = {"n": int(mask.sum()), "upper_bound": round(float(hi_o), 1)}
    # winsorise: cap to bound instead of dropping -> preserves the row
    df[c] = df[c].clip(lower=max(lo_o, 0), upper=hi_o)
stats["outliers"] = outlier_counts

# 3.5 missing values
stats["missing_pct_before_impute"] = (df.isna().mean() * 100).round(2)[df.isna().sum() > 0].to_dict()
# numeric: median imputation within transport mode (groups differ strongly)
for c in ["distance_km", "weight_kg", "transit_days", "freight_cost_inr"]:
    df[c] = df[c].fillna(df.groupby("mode")[c].transform("median"))
# inventory: Poisson-like count -> overall median, rounded
df["stock_on_hand"] = df["stock_on_hand"].fillna(df["stock_on_hand"].median()).round().astype(int)
# categorical: mode (most frequent) / explicit 'Unknown'
df["carrier"] = df["carrier"].fillna("Unknown")
df["order_date"] = df["order_date"].fillna(df["order_date"].median())
stats["missing_after"] = int(df.isna().sum().sum())

# ---------------------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING + NORMALISATION
# ---------------------------------------------------------------------------
df["cost_per_km"] = (df["freight_cost_inr"] / df["distance_km"]).round(3)
df["order_month"] = df["order_date"].dt.month

# right-skew fix: log transform
for c in ["weight_kg", "freight_cost_inr"]:
    df[f"log_{c}"] = np.log1p(df[c])

feat = ["distance_km", "weight_kg", "transit_days", "freight_cost_inr", "stock_on_hand"]
stats["skew_before"] = df[feat].skew().round(2).to_dict()
stats["skew_log"] = df[["log_weight_kg", "log_freight_cost_inr"]].skew().round(2).to_dict()

# min-max (0-1) and z-score, implemented with pandas so no extra dependency
for c in feat:
    mn, mx = df[c].min(), df[c].max()
    df[f"{c}_minmax"] = (df[c] - mn) / (mx - mn)
    df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std()
stats["minmax_range"] = {c: [round(df[f"{c}_minmax"].min(), 2), round(df[f"{c}_minmax"].max(), 2)] for c in feat}
stats["z_mean_std"] = {c: [round(df[f"{c}_z"].mean(), 3), round(df[f"{c}_z"].std(), 3)] for c in feat}

# one-hot encode categoricals for modelling
model_df = pd.get_dummies(df, columns=["mode", "carrier"], prefix=["mode", "carrier"], dtype=int)
stats["final_shape_encoded"] = list(model_df.shape)

# ---------------------------------------------------------------------------
# STEP 5: VALIDATION + BEFORE/AFTER VISUALS
# ---------------------------------------------------------------------------
assert df["distance_km"].gt(0).all()
assert df.duplicated().sum() == 0
assert df.isna().sum().sum() == 0

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
raw_w = pd.read_csv("raw_shipments.csv")["weight_kg"].dropna()
ax[0].hist(raw_w.clip(upper=raw_w.quantile(.995)), bins=50, color="#c0504d"); ax[0].set_title("weight_kg - raw (clipped for display)")
ax[1].hist(df["weight_kg"], bins=50, color="#4f81bd"); ax[1].set_title("weight_kg - cleaned")
ax[2].hist(df["log_weight_kg"], bins=50, color="#9bbb59"); ax[2].set_title("log1p(weight_kg)")
plt.tight_layout(); plt.savefig("figures/weight_distributions.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 3.8))
miss_before = pd.Series(stats["missing_before"])
miss_before.plot(kind="barh", color="#c0504d", ax=ax)
ax.set_title("Missing values per column (raw data)"); ax.set_xlabel("count")
plt.tight_layout(); plt.savefig("figures/missing_before.png", dpi=150); plt.close()

stats["clean_sample"] = df[["shipment_id", "mode", "distance_km", "weight_kg", "transit_days",
                            "freight_cost_inr", "cost_per_km"]].head(5).round(2).astype(str).values.tolist()
stats["raw_sample"] = pd.read_csv("raw_shipments.csv")[["shipment_id", "mode", "distance_km", "weight_kg",
                                                        "transit_days", "freight_cost_inr"]].head(5).astype(str).values.tolist()
stats["clean_describe"] = df[feat].describe().round(2).to_dict()
stats["avg_cost_by_mode"] = df.groupby("mode")["freight_cost_inr"].mean().round(0).to_dict()

df.to_csv("clean_shipments.csv", index=False)
with open("report_stats.json", "w") as f:
    json.dump(stats, f, indent=2, default=str)
print(json.dumps(stats, indent=2, default=str))
