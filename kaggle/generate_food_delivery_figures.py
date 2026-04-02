"""
Generate publication-quality figures for the Food Delivery Time Prediction project.
Outputs 4 PNG files to ../figures/ with prefix fig_food_

Models benchmarked:
  Ridge (baseline), Random Forest, XGBoost,
  HistGradientBoosting, LightGBM, CatBoost, Stacking Ensemble
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8")
sns.set_context("talk", font_scale=0.88)
plt.rcParams["axes.grid"] = False

DARK_BG  = "#0f1a2a"
BLUE     = "#2563eb"
SKY      = "#0ea5e9"
TEAL     = "#14b8a6"
AMBER    = "#f59e0b"
ROSE     = "#f43f5e"
GREEN    = "#22c55e"
TEXT     = "#e6f0ff"
TEXT_DIM = "#9fb0c8"
CARD_BG  = "#162236"

# ── Colors also used for the new models ──────────────────────────────────────
PURPLE   = "#7c3aed"
LIME     = "#84cc16"
ORANGE   = "#f97316"

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
KAGGLE_DIR  = Path(__file__).resolve().parent / "food_delivery"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load & Clean Data
# ══════════════════════════════════════════════════════════════════════════════
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_and_clean(path):
    df = pd.read_csv(path)

    # Strip leading/trailing whitespace from all string columns
    str_cols = df.select_dtypes("object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Numeric conversions
    df["Delivery_person_Age"]     = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
    df["multiple_deliveries"]     = pd.to_numeric(df["multiple_deliveries"], errors="coerce")

    # Target: strip "(min) " prefix  ── only present in train
    if "Time_taken(min)" in df.columns:
        df["Time_taken_min"] = (
            df["Time_taken(min)"].str.extract(r"(\d+)").astype(float)
        )

    # ── Feature engineering ───────────────────────────────────────────────
    # Distance: straight-line km from restaurant → delivery location
    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"],  df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    )

    # Pickup wait time: minutes between order accepted and rider pick-up
    # Some rows have NaN Time_Orderd — fill with 0
    def to_minutes(t):
        try:
            h, m, s = t.split(":")
            return int(h) * 60 + int(m) + int(s) / 60
        except Exception:
            return np.nan

    df["ordered_min"]  = df["Time_Orderd"].apply(to_minutes)
    df["picked_min"]   = df["Time_Order_picked"].apply(to_minutes)
    df["pickup_wait"]  = (df["picked_min"] - df["ordered_min"]).clip(lower=0)

    # Order hour (0–23)
    df["order_hour"] = df["ordered_min"].apply(
        lambda x: int(x // 60) % 24 if not np.isnan(x) else np.nan
    )

    # Clean categoricals — map "NaN" strings to proper NaN
    for col in ["Weatherconditions", "Road_traffic_density", "Festival", "City"]:
        df[col] = df[col].replace("NaN", np.nan)

    # Remove "conditions " prefix from Weatherconditions
    df["Weatherconditions"] = df["Weatherconditions"].str.replace(
        r"^conditions\s+", "", regex=True
    )

    return df


df_raw = load_and_clean(KAGGLE_DIR / "train.csv")

# Drop rows where target is missing (shouldn't happen, but safety first)
df = df_raw.dropna(subset=["Time_taken_min"]).copy()

print(f"Training rows after cleaning: {len(df):,}")
print(f"Target mean: {df['Time_taken_min'].mean():.1f} min  |  std: {df['Time_taken_min'].std():.1f} min")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Feature Matrix
# ══════════════════════════════════════════════════════════════════════════════
NUM_FEATURES = [
    "Delivery_person_Age", "Delivery_person_Ratings",
    "Vehicle_condition", "multiple_deliveries",
    "distance_km", "pickup_wait", "order_hour",
]
CAT_FEATURES = [
    "Weatherconditions", "Road_traffic_density",
    "Type_of_order", "Type_of_vehicle", "Festival", "City",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET       = "Time_taken_min"

X = df[ALL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pre-processor: impute + scale numerics, impute + OHE categoricals
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, NUM_FEATURES),
    ("cat", cat_pipe, CAT_FEATURES),
], remainder="drop")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Train Models
# ══════════════════════════════════════════════════════════════════════════════

# HistGradientBoosting handles NaN natively — no imputer needed for numerics.
# CatBoost handles categoricals natively — we pass raw string columns to it.
# For a fair pipeline comparison, all other models use the full preprocessor.

# Shared preprocessor (for Ridge, RF, XGBoost, LightGBM, Stacking)
num_pipe_shared = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe_shared = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", num_pipe_shared, NUM_FEATURES),
    ("cat", cat_pipe_shared, CAT_FEATURES),
], remainder="drop")

# HistGBRT preprocessor: numeric only imputed, categoricals label-encoded
# (HistGBRT accepts NaN natively so we omit the numeric imputer too)
preprocessor_hist = ColumnTransformer([
    ("num", "passthrough", NUM_FEATURES),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CAT_FEATURES),
], remainder="drop")

# CatBoost: handle NaN in numerics via imputer; pass categoricals as strings.
# We build a separate X matrix where categoricals stay as object dtype.
X_cb = df[ALL_FEATURES].copy()
for col in CAT_FEATURES:
    X_cb[col] = X_cb[col].fillna("Unknown").astype(str)
for col in NUM_FEATURES:
    X_cb[col] = pd.to_numeric(X_cb[col], errors="coerce")

X_train_cb, X_test_cb = X_cb.iloc[X_train.index], X_cb.iloc[X_test.index]
cat_feature_indices = [X_cb.columns.get_loc(c) for c in CAT_FEATURES]

models = {
    "Ridge\n(baseline)": Pipeline([
        ("prep", preprocessor),
        ("model", Ridge(alpha=1.0)),
    ]),
    "Random\nForest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=5, n_jobs=-1, random_state=42
        )),
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=0,
        )),
    ]),
    "HistGBRT": Pipeline([
        ("prep", preprocessor_hist),
        ("model", HistGradientBoostingRegressor(
            max_iter=400, max_depth=6, learning_rate=0.05,
            l2_regularization=0.1, random_state=42,
        )),
    ]),
    "LightGBM": Pipeline([
        ("prep", preprocessor),
        ("model", LGBMRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=-1,
        )),
    ]),
}

results = {}
for name, pipe in models.items():
    print(f"Training {name.replace(chr(10), ' ')} …")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results[name] = {"pipe": pipe, "y_pred": y_pred, "rmse": rmse, "mae": mae, "r2": r2}
    print(f"  RMSE={rmse:.2f} min | MAE={mae:.2f} min | R²={r2:.3f}")

# CatBoost — trained on its own X matrix with native categorical support
print("Training CatBoost …")
cb_model = CatBoostRegressor(
    iterations=400, depth=6, learning_rate=0.05,
    l2_leaf_reg=3, random_seed=42, verbose=0,
    cat_features=cat_feature_indices,
)
cb_model.fit(X_train_cb, y_train)
y_pred_cb = cb_model.predict(X_test_cb)
results["CatBoost"] = {
    "pipe": cb_model, "y_pred": y_pred_cb,
    "rmse": np.sqrt(mean_squared_error(y_test, y_pred_cb)),
    "mae":  mean_absolute_error(y_test, y_pred_cb),
    "r2":   r2_score(y_test, y_pred_cb),
}
print(f"  RMSE={results['CatBoost']['rmse']:.2f} min | MAE={results['CatBoost']['mae']:.2f} min | R²={results['CatBoost']['r2']:.3f}")

# Stacking ensemble: RF + XGBoost + LightGBM base learners, Ridge meta-learner
print("Training Stacking Ensemble …")
stacking = StackingRegressor(
    estimators=[
        ("rf", Pipeline([("prep", preprocessor),
                         ("model", RandomForestRegressor(n_estimators=200, max_depth=12,
                                                         min_samples_leaf=5, n_jobs=-1, random_state=42))])),
        ("xgb", Pipeline([("prep", preprocessor),
                           ("model", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                                   subsample=0.8, colsample_bytree=0.8,
                                                   n_jobs=-1, random_state=42, verbosity=0))])),
        ("lgb", Pipeline([("prep", preprocessor),
                           ("model", LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                                    subsample=0.8, colsample_bytree=0.8,
                                                    n_jobs=-1, random_state=42, verbosity=-1))])),
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1,
)
stacking.fit(X_train, y_train)
y_pred_st = stacking.predict(X_test)
results["Stacking\nEnsemble"] = {
    "pipe": stacking, "y_pred": y_pred_st,
    "rmse": np.sqrt(mean_squared_error(y_test, y_pred_st)),
    "mae":  mean_absolute_error(y_test, y_pred_st),
    "r2":   r2_score(y_test, y_pred_st),
}
print(f"  RMSE={results['Stacking'+chr(10)+'Ensemble']['rmse']:.2f} min | MAE={results['Stacking'+chr(10)+'Ensemble']['mae']:.2f} min | R²={results['Stacking'+chr(10)+'Ensemble']['r2']:.3f}")

best_name = min(results, key=lambda n: results[n]["rmse"])
print(f"\nBest model: {best_name.replace(chr(10), ' ')}  (RMSE={results[best_name]['rmse']:.2f} min  R²={results[best_name]['r2']:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# 4a.  FIGURE 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Food Delivery Time — Exploratory Data Analysis",
             color=TEXT, fontsize=18, fontweight="bold", y=0.98)

for ax in axes.flat:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_DIM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3f5f")

# Panel 1: Target distribution
ax = axes[0, 0]
ax.hist(df["Time_taken_min"], bins=45, color=BLUE, edgecolor=DARK_BG, linewidth=0.4)
ax.axvline(df["Time_taken_min"].mean(), color=AMBER,  linewidth=2, linestyle="--", label=f"Mean: {df['Time_taken_min'].mean():.1f} min")
ax.axvline(df["Time_taken_min"].median(), color=TEAL, linewidth=2, linestyle=":",  label=f"Median: {df['Time_taken_min'].median():.0f} min")
ax.set_title("Distribution of Delivery Time", color=TEXT, fontsize=13, pad=8)
ax.set_xlabel("Delivery Time (minutes)", color=TEXT_DIM, fontsize=10)
ax.set_ylabel("Count", color=TEXT_DIM, fontsize=10)
ax.legend(fontsize=9, facecolor=CARD_BG, labelcolor=TEXT, edgecolor="#2a3f5f")

# Panel 2: Traffic density vs time
ax = axes[0, 1]
traffic_order = ["Low", "Medium", "High", "Jam"]
traffic_df = df[df["Road_traffic_density"].isin(traffic_order)].copy()
colors_traffic = [GREEN, AMBER, ROSE, "#7c3aed"]
medians = [traffic_df[traffic_df["Road_traffic_density"] == t]["Time_taken_min"].median()
           for t in traffic_order]
bars = ax.bar(traffic_order, medians, color=colors_traffic,
              edgecolor=DARK_BG, linewidth=0.5, width=0.6)
for bar, val in zip(bars, medians):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{val:.0f} min", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
ax.set_title("Median Delivery Time by Traffic Density", color=TEXT, fontsize=13, pad=8)
ax.set_xlabel("Traffic Density", color=TEXT_DIM, fontsize=10)
ax.set_ylabel("Median Delivery Time (min)", color=TEXT_DIM, fontsize=10)
ax.set_ylim(0, max(medians) * 1.25)

# Panel 3: Weather conditions vs time
ax = axes[1, 0]
weather_order = ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"]
weather_df = df[df["Weatherconditions"].isin(weather_order)].copy()
weather_palette = [SKY, "#64748b", TEAL, AMBER, ROSE, "#f97316"]
w_medians = [weather_df[weather_df["Weatherconditions"] == w]["Time_taken_min"].median()
             for w in weather_order]
bars = ax.bar(weather_order, w_medians, color=weather_palette,
              edgecolor=DARK_BG, linewidth=0.5, width=0.6)
for bar, val in zip(bars, w_medians):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.0f}", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
ax.set_title("Median Delivery Time by Weather Condition", color=TEXT, fontsize=13, pad=8)
ax.set_xlabel("Weather Condition", color=TEXT_DIM, fontsize=10)
ax.set_ylabel("Median Delivery Time (min)", color=TEXT_DIM, fontsize=10)
ax.set_ylim(0, max(w_medians) * 1.25)
ax.tick_params(axis="x", labelrotation=25)

# Panel 4: Distance vs delivery time (scatter + trend)
ax = axes[1, 1]
sample = df.sample(n=3000, random_state=42)
ax.scatter(sample["distance_km"], sample["Time_taken_min"],
           alpha=0.25, s=12, color=BLUE, rasterized=True)
# Trend line
z = np.polyfit(sample["distance_km"].dropna(), sample["Time_taken_min"].dropna(), 1)
p = np.poly1d(z)
xline = np.linspace(sample["distance_km"].min(), sample["distance_km"].max(), 200)
ax.plot(xline, p(xline), color=AMBER, linewidth=2.5, label=f"Trend (slope={z[0]:.2f} min/km)")
ax.set_title("Delivery Time vs. Distance to Customer", color=TEXT, fontsize=13, pad=8)
ax.set_xlabel("Straight-line Distance (km)", color=TEXT_DIM, fontsize=10)
ax.set_ylabel("Delivery Time (min)", color=TEXT_DIM, fontsize=10)
ax.legend(fontsize=9, facecolor=CARD_BG, labelcolor=TEXT, edgecolor="#2a3f5f")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = FIGURES_DIR / "fig_food_01_eda.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4b.  FIGURE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Model Performance Comparison — All 7 Models",
             color=TEXT, fontsize=18, fontweight="bold", y=0.98)

for ax in axes.flat:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3f5f")

model_colors = {
    "Ridge\n(baseline)": "#64748b",
    "Random\nForest":    SKY,
    "XGBoost":           AMBER,
    "HistGBRT":          TEAL,
    "LightGBM":          LIME,
    "CatBoost":          ORANGE,
    "Stacking\nEnsemble": ROSE,
}

names_clean = [n.replace("\n", " ") for n in results]
rmse_vals   = [results[n]["rmse"] for n in results]
r2_vals     = [results[n]["r2"]   for n in results]
mae_vals    = [results[n]["mae"]  for n in results]
bar_cols    = [model_colors[n]    for n in results]

# Panel 1: RMSE comparison
ax = axes[0, 0]
bars = ax.bar(names_clean, rmse_vals, color=bar_cols, edgecolor=DARK_BG, linewidth=0.5, width=0.6)
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f"{val:.2f}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
ax.set_title("RMSE — lower is better", color=TEXT, fontsize=13, pad=8)
ax.set_ylabel("Root Mean Squared Error (min)", color=TEXT_DIM, fontsize=10)
ax.set_ylim(0, max(rmse_vals) * 1.3)
ax.tick_params(axis="x", labelrotation=15)

# Panel 2: R² comparison
ax = axes[0, 1]
bars = ax.bar(names_clean, r2_vals, color=bar_cols, edgecolor=DARK_BG, linewidth=0.5, width=0.6)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
            f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
ax.set_title("R² Score — higher is better", color=TEXT, fontsize=13, pad=8)
ax.set_ylabel("R² Score", color=TEXT_DIM, fontsize=10)
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color="#2a3f5f", linewidth=1, linestyle="--")
ax.tick_params(axis="x", labelrotation=15)

# Panel 3: Actual vs Predicted (best model)
ax = axes[1, 0]
best_pred  = results[best_name]["y_pred"]
best_label = best_name.replace("\n", " ")
best_color = model_colors[best_name]
sample_idx = np.random.default_rng(42).choice(len(y_test), size=2000, replace=False)
ax.scatter(y_test.values[sample_idx], best_pred[sample_idx],
           alpha=0.3, s=10, color=best_color, rasterized=True)
lo, hi = y_test.min() - 1, y_test.max() + 1
ax.plot([lo, hi], [lo, hi], color=ROSE, linewidth=2, linestyle="--", label="Perfect prediction")
ax.set_title(f"Actual vs. Predicted — {best_label}", color=TEXT, fontsize=13, pad=8)
ax.set_xlabel("Actual Delivery Time (min)", color=TEXT_DIM, fontsize=10)
ax.set_ylabel("Predicted Delivery Time (min)", color=TEXT_DIM, fontsize=10)
ax.legend(fontsize=9, facecolor=CARD_BG, labelcolor=TEXT, edgecolor="#2a3f5f")

# Panel 4: Residuals (best model)
ax = axes[1, 1]
residuals = y_test.values - best_pred
ax.hist(residuals, bins=50, color=best_color, edgecolor=DARK_BG, linewidth=0.4)
ax.axvline(0,  color=ROSE, linewidth=2, linestyle="--", label="Zero error")
ax.axvline(residuals.mean(), color=AMBER, linewidth=2, linestyle=":",
           label=f"Mean error: {residuals.mean():.2f} min")
ax.set_title(f"Residual Distribution — {best_label}", color=TEXT, fontsize=13, pad=8)
ax.set_xlabel("Residual (Actual − Predicted, min)", color=TEXT_DIM, fontsize=10)
ax.set_ylabel("Count", color=TEXT_DIM, fontsize=10)
ax.legend(fontsize=9, facecolor=CARD_BG, labelcolor=TEXT, edgecolor="#2a3f5f")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = FIGURES_DIR / "fig_food_02_model_performance.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4c.  FIGURE 3 — Feature Importance (RF, LightGBM, CatBoost)
# ══════════════════════════════════════════════════════════════════════════════

# Get OHE feature names (shared preprocessor)
ohe_names = list(
    results["Random\nForest"]["pipe"].named_steps["prep"]
    .named_transformers_["cat"]
    .named_steps["ohe"]
    .get_feature_names_out(CAT_FEATURES)
)
feature_names = NUM_FEATURES + ohe_names

TOP_N = 12

def top_n_importance(importance, names, n=TOP_N):
    idx = np.argsort(importance)[::-1][:n]
    return np.array(names)[idx], importance[idx]

def shorten(name):
    return (name.replace("Road_traffic_density_", "Traffic: ")
                .replace("Weatherconditions_", "Weather: ")
                .replace("Type_of_vehicle_", "Vehicle: ")
                .replace("Type_of_order_", "Order: ")
                .replace("Festival_", "Festival: ")
                .replace("City_", "City: ")
                .replace("Delivery_person_", "Driver "))

# RF importances
rf_imp   = results["Random\nForest"]["pipe"].named_steps["model"].feature_importances_
rf_n, rf_v = top_n_importance(rf_imp, feature_names)
rf_n = [shorten(n) for n in rf_n]

# LightGBM importances
lgb_imp  = results["LightGBM"]["pipe"].named_steps["model"].feature_importances_
lgb_n, lgb_v = top_n_importance(lgb_imp, feature_names)
lgb_n = [shorten(n) for n in lgb_n]

# CatBoost importances — returns one value per raw input feature
cb_imp_raw   = cb_model.get_feature_importance()
cb_feat_names = list(X_train_cb.columns)
cb_n_raw, cb_v_raw = top_n_importance(cb_imp_raw, cb_feat_names)
cb_n = [shorten(n).replace("_", " ").title() for n in cb_n_raw]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Feature Importance — What Drives Delivery Time?",
             color=TEXT, fontsize=18, fontweight="bold", y=1.01)

for ax in (ax1, ax2, ax3):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_DIM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3f5f")

ax1.barh(rf_n[::-1],  rf_v[::-1],  color=SKY,    edgecolor=DARK_BG, linewidth=0.5)
ax1.set_title("Random Forest — Top 12 Features",  color=TEXT, fontsize=13, pad=8)
ax1.set_xlabel("Feature Importance (Gini impurity reduction)", color=TEXT_DIM, fontsize=10)

ax2.barh(lgb_n[::-1], lgb_v[::-1], color=LIME,   edgecolor=DARK_BG, linewidth=0.5)
ax2.set_title("LightGBM — Top 12 Features",        color=TEXT, fontsize=13, pad=8)
ax2.set_xlabel("Feature Importance (split gain)",   color=TEXT_DIM, fontsize=10)

ax3.barh(cb_n[::-1],  cb_v_raw[::-1], color=ORANGE, edgecolor=DARK_BG, linewidth=0.5)
ax3.set_title("CatBoost — Top 12 Features",        color=TEXT, fontsize=13, pad=8)
ax3.set_xlabel("Feature Importance (PredictionValuesChange)", color=TEXT_DIM, fontsize=10)

plt.tight_layout()
out = FIGURES_DIR / "fig_food_03_feature_importance.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4d.  FIGURE 4 — Business Impact (SLA Analysis)
# ══════════════════════════════════════════════════════════════════════════════
SLA = 30  # minutes

naive_pred = np.full_like(y_test.values, y_train.mean(), dtype=float)

def sla_metrics(y_true, y_pred, sla=SLA):
    actual_yes = y_true <= sla
    pred_yes   = y_pred <= sla
    tp  = (actual_yes &  pred_yes).sum()
    fp  = (~actual_yes &  pred_yes).sum()
    fn  = (actual_yes & ~pred_yes).sum()
    tn  = (~actual_yes & ~pred_yes).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy  = (tp + tn) / len(y_true)
    return {"precision": precision, "recall": recall, "accuracy": accuracy}

model_order_sla = [
    "Naive\n(mean)", "Ridge\n(baseline)", "Random\nForest",
    "XGBoost", "HistGBRT", "LightGBM", "CatBoost", "Stacking\nEnsemble",
]
model_colors_sla = {
    "Naive\n(mean)":       "#475569",
    "Ridge\n(baseline)":   "#64748b",
    "Random\nForest":      SKY,
    "XGBoost":             AMBER,
    "HistGBRT":            TEAL,
    "LightGBM":            LIME,
    "CatBoost":            ORANGE,
    "Stacking\nEnsemble":  ROSE,
}

sla_results = {"Naive\n(mean)": sla_metrics(y_test.values, naive_pred)}
for name in list(results.keys()):
    sla_results[name] = sla_metrics(y_test.values, results[name]["y_pred"])

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(f"Business Impact — SLA Compliance (Promise: ≤{SLA} min Delivery)",
             color=TEXT, fontsize=18, fontweight="bold", y=1.01)

for ax in axes:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3f5f")

metric_defs = [
    ("precision", "Precision\n(fewer false promises)"),
    ("recall",    "Recall\n(catch all on-time orders)"),
    ("accuracy",  "Accuracy"),
]

for i, (metric_key, label) in enumerate(metric_defs):
    ax = axes[i]
    vals   = [sla_results[n][metric_key] for n in model_order_sla]
    cols   = [model_colors_sla[n] for n in model_order_sla]
    labels = [n.replace("\n", " ") for n in model_order_sla]
    bars   = ax.bar(labels, vals, color=cols, edgecolor=DARK_BG, linewidth=0.5, width=0.65)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{val:.1%}", ha="center", va="bottom", color=TEXT, fontsize=8.5, fontweight="bold")
    ax.set_title(label, color=TEXT, fontsize=13, pad=8)
    ax.set_ylabel("Score", color=TEXT_DIM, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.tick_params(axis="x", labelrotation=30)

plt.tight_layout()
out = FIGURES_DIR / "fig_food_04_business_impact.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"Saved {out.name}")

print("\nAll figures saved.")
print("\nFinal model metrics:")
for name, r in results.items():
    marker = " ← best" if name == best_name else ""
    print(f"  {name.replace(chr(10), ' '):25s}  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  R²={r['r2']:.3f}{marker}")
best = results[best_name]
print(f"\nSLA ({SLA} min) — {best_name.replace(chr(10), ' ')} accuracy:  {sla_results[best_name]['accuracy']:.1%}")
print(f"SLA ({SLA} min) — {best_name.replace(chr(10), ' ')} precision: {sla_results[best_name]['precision']:.1%}")
print(f"SLA ({SLA} min) — {best_name.replace(chr(10), ' ')} recall:    {sla_results[best_name]['recall']:.1%}")
