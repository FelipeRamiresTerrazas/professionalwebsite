"""
DoorDash ETA Prediction
Predicts delivery duration (minutes) from order-time operational signals.
Dataset: ~197K historical orders with dasher supply/demand features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#cbd5e1",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#e2e8f0",
    "grid.color":       "#334155",
    "grid.alpha":       0.5,
    "axes.titlecolor":  "#f1f5f9",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.facecolor": "#1e293b",
    "legend.edgecolor": "#334155",
    "legend.labelcolor":"#cbd5e1",
})
ACCENT   = "#38bdf8"
ACCENT2  = "#818cf8"
ACCENT3  = "#34d399"
WARN     = "#fb923c"
PALETTE  = [ACCENT, ACCENT2, ACCENT3, WARN, "#f472b6", "#a78bfa", "#fbbf24"]

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 1. Load & clean ────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "historical_data.csv"),
    parse_dates=["created_at", "actual_delivery_time"],
)

# Target: delivery duration in minutes
df["delivery_minutes"] = (
    df["actual_delivery_time"] - df["created_at"]
).dt.total_seconds() / 60

# Drop impossible / extreme values (< 5 min or > 180 min)
df = df[(df["delivery_minutes"] >= 5) & (df["delivery_minutes"] <= 180)].copy()
print(f"  Rows after cleaning: {len(df):,}")

# ── 2. Feature engineering ─────────────────────────────────────────────────────
# Temporal features from order timestamp
df["order_hour"]    = df["created_at"].dt.hour
df["order_dow"]     = df["created_at"].dt.dayofweek  # 0=Mon, 6=Sun
df["is_weekend"]    = (df["order_dow"] >= 5).astype(int)
df["is_peak_hour"]  = df["order_hour"].apply(
    lambda h: 1 if (11 <= h <= 14) or (18 <= h <= 22) else 0
)

# Dasher supply / demand pressure
df["dasher_utilization"]  = (
    df["total_busy_dashers"] / df["total_onshift_dashers"].replace(0, np.nan)
).clip(0, 2)
df["available_dashers"]   = (
    df["total_onshift_dashers"] - df["total_busy_dashers"]
).clip(lower=0)
df["orders_per_dasher"]   = (
    df["total_outstanding_orders"] / (df["available_dashers"] + 1)
)

# Order complexity
df["avg_item_price"]  = df["subtotal"] / df["total_items"].replace(0, np.nan)
df["price_range"]     = df["max_item_price"] - df["min_item_price"]

# Estimate features (convert seconds → minutes)
df["est_order_min"]  = df["estimated_order_place_duration"] / 60
df["est_drive_min"]  = df["estimated_store_to_consumer_driving_duration"] / 60
df["est_total_min"]  = df["est_order_min"] + df["est_drive_min"]

# market_id as string for categorical treatment
df["market_id"]    = df["market_id"].astype(str)
df["order_protocol"] = df["order_protocol"].astype(str)

print(f"  Features engineered. Target mean: {df['delivery_minutes'].mean():.1f} min")

# ── 3. Train / test split ──────────────────────────────────────────────────────
NUM_FEATURES = [
    "total_items", "subtotal", "num_distinct_items",
    "min_item_price", "max_item_price",
    "total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders",
    "est_order_min", "est_drive_min", "est_total_min",
    "order_hour", "order_dow", "is_weekend", "is_peak_hour",
    "dasher_utilization", "available_dashers", "orders_per_dasher",
    "avg_item_price", "price_range",
]
CAT_FEATURES = ["market_id", "store_primary_category", "order_protocol"]

X = df[NUM_FEATURES + CAT_FEATURES]
y = df["delivery_minutes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── 4. Preprocessing pipeline ──────────────────────────────────────────────────
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, NUM_FEATURES),
    ("cat", cat_pipe, CAT_FEATURES),
])

# ── 5. Model training ──────────────────────────────────────────────────────────
print("\nTraining models...")

models = {
    "Naive Mean": None,
    "Ridge": Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0))]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)),
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
preds   = {}

naive_pred = np.full(len(y_test), y_train.mean())
results["Naive Mean"] = {
    "RMSE": np.sqrt(mean_squared_error(y_test, naive_pred)),
    "MAE":  mean_absolute_error(y_test, naive_pred),
    "R2":   r2_score(y_test, naive_pred),
}
preds["Naive Mean"] = naive_pred

for name, pipe in models.items():
    if pipe is None:
        continue
    print(f"  Fitting {name}...")
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    preds[name] = yp
    results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, yp)),
        "MAE":  mean_absolute_error(y_test, yp),
        "R2":   r2_score(y_test, yp),
    }
    print(f"    RMSE={results[name]['RMSE']:.2f}  MAE={results[name]['MAE']:.2f}  R²={results[name]['R2']:.3f}")

best_model_name = max(
    [k for k in results if k != "Naive Mean"],
    key=lambda k: results[k]["R2"],
)
print(f"\nBest model: {best_model_name}")

# ── 6. ETA accuracy metrics ────────────────────────────────────────────────────
def eta_accuracy(y_true, y_pred, tolerance_min=5):
    """Fraction of predictions within ±tolerance_min of actual delivery time."""
    return np.mean(np.abs(y_true - y_pred) <= tolerance_min)

for name in results:
    results[name]["Within5"] = eta_accuracy(y_test, preds[name], 5)
    results[name]["Within10"] = eta_accuracy(y_test, preds[name], 10)

# ── 7. Feature importance ──────────────────────────────────────────────────────
best_pipe  = models[best_model_name]
best_model_obj = best_pipe.named_steps["model"]
ohe_cats   = (best_pipe.named_steps["prep"]
              .named_transformers_["cat"]
              .named_steps["ohe"]
              .get_feature_names_out(CAT_FEATURES))
feat_names = NUM_FEATURES + list(ohe_cats)

if hasattr(best_model_obj, "feature_importances_"):
    importances = best_model_obj.feature_importances_
    feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    top_features = feat_imp.head(15)
else:
    top_features = None

# LightGBM importance for comparison
lgb_pipe  = models["LightGBM"]
lgb_model = lgb_pipe.named_steps["model"]
lgb_imp   = pd.Series(lgb_model.feature_importances_, index=feat_names).sort_values(ascending=False).head(15)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1: EDA
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DoorDash ETA Prediction — Exploratory Data Analysis", fontsize=16, y=0.98)
plt.subplots_adjust(hspace=0.40, wspace=0.32)

# 1a. Delivery time distribution
ax = axes[0, 0]
ax.hist(df["delivery_minutes"], bins=60, color=ACCENT, edgecolor="none", alpha=0.85)
ax.axvline(df["delivery_minutes"].mean(), color=WARN, linestyle="--", linewidth=1.8, label=f"Mean {df['delivery_minutes'].mean():.1f} min")
ax.axvline(df["delivery_minutes"].median(), color=ACCENT3, linestyle="--", linewidth=1.8, label=f"Median {df['delivery_minutes'].median():.1f} min")
ax.set_title("Delivery Time Distribution")
ax.set_xlabel("Delivery Duration (minutes)")
ax.set_ylabel("Order Count")
ax.legend(fontsize=9)

# 1b. Average delivery time by hour
hourly = df.groupby("order_hour")["delivery_minutes"].mean()
ax = axes[0, 1]
bars = ax.bar(hourly.index, hourly.values, color=ACCENT2, edgecolor="none", alpha=0.85)
peak_hours = [h for h in hourly.index if (11 <= h <= 14) or (18 <= h <= 22)]
for bar, h in zip(bars, hourly.index):
    if h in peak_hours:
        bar.set_color(WARN)
ax.set_title("Avg Delivery Time by Hour of Day")
ax.set_xlabel("Hour (0–23)")
ax.set_ylabel("Avg Delivery (min)")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=ACCENT2, label="Off-peak"),
    Patch(color=WARN, label="Peak (11–14, 18–22)"),
], fontsize=9)

# 1c. Dasher utilization vs delivery time
ax = axes[1, 0]
sample = df.sample(min(5000, len(df)), random_state=42)
sc = ax.scatter(
    sample["dasher_utilization"], sample["delivery_minutes"],
    c=sample["orders_per_dasher"].clip(0, 10),
    cmap="plasma", alpha=0.35, s=10,
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Orders per Available Dasher", color="#94a3b8", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="#94a3b8")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8")
ax.set_title("Dasher Utilization vs Delivery Time")
ax.set_xlabel("Dasher Utilization (busy / on-shift)")
ax.set_ylabel("Delivery Duration (min)")
ax.set_xlim(0, 2)

# 1d. Delivery time by food category (top 8)
top_cats = df["store_primary_category"].value_counts().head(8).index
cat_data = df[df["store_primary_category"].isin(top_cats)]
cat_order = cat_data.groupby("store_primary_category")["delivery_minutes"].median().sort_values()
ax = axes[1, 1]
bp = ax.boxplot(
    [cat_data[cat_data["store_primary_category"] == c]["delivery_minutes"].values for c in cat_order.index],
    vert=True, patch_artist=True, showfliers=False,
    medianprops=dict(color=WARN, linewidth=2),
    boxprops=dict(facecolor="#1e4b7a", alpha=0.8),
    whiskerprops=dict(color="#94a3b8"),
    capprops=dict(color="#94a3b8"),
)
for patch, color in zip(bp["boxes"], [ACCENT, ACCENT2, ACCENT3, WARN, "#f472b6", "#a78bfa", "#fbbf24", "#e2e8f0"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks(range(1, len(cat_order) + 1))
ax.set_xticklabels([c.title() for c in cat_order.index], rotation=35, ha="right", fontsize=9)
ax.set_title("Delivery Time by Food Category")
ax.set_ylabel("Delivery Duration (min)")

plt.savefig(os.path.join(FIGURES_DIR, "fig_doordash_01_eda.png"),
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  fig_doordash_01_eda.png saved")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Model Performance
# ──────────────────────────────────────────────────────────────────────────────
model_names_ordered = ["Naive Mean", "Ridge", "Random Forest", "XGBoost", "LightGBM"]
rmse_vals = [results[m]["RMSE"] for m in model_names_ordered]
r2_vals   = [results[m]["R2"]   for m in model_names_ordered]
colors_bar = [ACCENT if m != best_model_name else ACCENT3 for m in model_names_ordered]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DoorDash ETA Prediction — Model Performance", fontsize=16, y=0.98)
plt.subplots_adjust(hspace=0.40, wspace=0.32)

# 2a. RMSE comparison
ax = axes[0, 0]
bars = ax.barh(model_names_ordered, rmse_vals, color=colors_bar, edgecolor="none", alpha=0.85)
for bar, v in zip(bars, rmse_vals):
    ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}", va="center", fontsize=10, color="#e2e8f0")
ax.set_title("RMSE Comparison (lower is better)")
ax.set_xlabel("RMSE (minutes)")
ax.invert_yaxis()

# 2b. R² comparison
ax = axes[0, 1]
bars = ax.barh(model_names_ordered, r2_vals, color=colors_bar, edgecolor="none", alpha=0.85)
for bar, v in zip(bars, r2_vals):
    ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=10, color="#e2e8f0")
ax.set_title("R² Comparison (higher is better)")
ax.set_xlabel("R²")
ax.invert_yaxis()

# 2c. Actual vs Predicted (best model)
ax = axes[1, 0]
best_pred = preds[best_model_name]
ax.scatter(y_test[:3000], best_pred[:3000], alpha=0.25, s=6, color=ACCENT)
lims = [max(0, min(y_test.min(), best_pred.min()) - 2),
        max(y_test.max(), best_pred.max()) + 2]
ax.plot(lims, lims, color=WARN, linewidth=1.5, linestyle="--", label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title(f"Actual vs Predicted — {best_model_name}")
ax.set_xlabel("Actual Delivery Time (min)")
ax.set_ylabel("Predicted ETA (min)")
ax.legend(fontsize=9)

# 2d. Residuals
ax = axes[1, 1]
residuals = y_test.values - best_pred
ax.hist(residuals, bins=80, color=ACCENT2, edgecolor="none", alpha=0.85)
ax.axvline(0, color=WARN, linestyle="--", linewidth=1.8)
ax.axvline(residuals.mean(), color=ACCENT3, linestyle="--", linewidth=1.5,
           label=f"Mean {residuals.mean():.2f} min")
ax.set_title(f"Residuals — {best_model_name}")
ax.set_xlabel("Residual (Actual − Predicted, min)")
ax.set_ylabel("Count")
ax.legend(fontsize=9)

plt.savefig(os.path.join(FIGURES_DIR, "fig_doordash_02_model_performance.png"),
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  fig_doordash_02_model_performance.png saved")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Feature Importance
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("DoorDash ETA Prediction — Feature Importance", fontsize=16, y=1.01)
plt.subplots_adjust(wspace=0.5)

def clean_feat_name(n):
    n = n.replace("market_id_", "market=").replace("store_primary_category_", "cat=")
    n = n.replace("order_protocol_", "protocol=").replace("_", " ")
    return n.title()

# XGBoost / best model
ax = axes[0]
top15 = feat_imp.head(15) if top_features is not None else pd.Series(dtype=float)
if len(top15):
    names_clean = [clean_feat_name(n) for n in top15.index]
    bars = ax.barh(names_clean[::-1], top15.values[::-1], color=ACCENT, edgecolor="none", alpha=0.85)
    ax.set_title(f"{best_model_name} — Top 15 Features")
    ax.set_xlabel("Feature Importance")

# LightGBM
ax = axes[1]
names_lgb = [clean_feat_name(n) for n in lgb_imp.index]
bars = ax.barh(names_lgb[::-1], lgb_imp.values[::-1], color=ACCENT2, edgecolor="none", alpha=0.85)
ax.set_title("LightGBM — Top 15 Features")
ax.set_xlabel("Feature Importance")

plt.savefig(os.path.join(FIGURES_DIR, "fig_doordash_03_feature_importance.png"),
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  fig_doordash_03_feature_importance.png saved")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Business Impact — ETA Accuracy
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("DoorDash ETA Prediction — Business Impact", fontsize=16, y=1.01)
plt.subplots_adjust(wspace=0.38)

model_names_biz = ["Naive Mean", "Ridge", "Random Forest", "XGBoost", "LightGBM"]
colors_biz = [ACCENT if m != best_model_name else ACCENT3 for m in model_names_biz]

# 4a. Within ±5 min accuracy
ax = axes[0]
vals5 = [results[m]["Within5"] * 100 for m in model_names_biz]
bars = ax.bar(model_names_biz, vals5, color=colors_biz, edgecolor="none", alpha=0.85)
for bar, v in zip(bars, vals5):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=10, color="#e2e8f0")
ax.set_title("Within ±5 min Accuracy")
ax.set_ylabel("% of Orders")
ax.set_ylim(0, 105)
ax.tick_params(axis="x", rotation=35)
ax.tick_params(axis="x", labelsize=9)

# 4b. Within ±10 min accuracy
ax = axes[1]
vals10 = [results[m]["Within10"] * 100 for m in model_names_biz]
bars = ax.bar(model_names_biz, vals10, color=colors_biz, edgecolor="none", alpha=0.85)
for bar, v in zip(bars, vals10):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=10, color="#e2e8f0")
ax.set_title("Within ±10 min Accuracy")
ax.set_ylabel("% of Orders")
ax.set_ylim(0, 105)
ax.tick_params(axis="x", rotation=35)
ax.tick_params(axis="x", labelsize=9)

# 4c. Error distribution at different tolerances (best vs naive)
ax = axes[2]
tolerances = [1, 2, 3, 5, 7, 10, 15, 20]
acc_best  = [np.mean(np.abs(y_test.values - preds[best_model_name]) <= t) * 100 for t in tolerances]
acc_naive = [np.mean(np.abs(y_test.values - preds["Naive Mean"]) <= t) * 100 for t in tolerances]
acc_ridge = [np.mean(np.abs(y_test.values - preds["Ridge"]) <= t) * 100 for t in tolerances]

ax.plot(tolerances, acc_best,  color=ACCENT3, linewidth=2.5, marker="o", markersize=5, label=best_model_name)
ax.plot(tolerances, acc_ridge, color=ACCENT2, linewidth=1.8, marker="s", markersize=4, linestyle="--", label="Ridge")
ax.plot(tolerances, acc_naive, color=WARN,    linewidth=1.8, marker="^", markersize=4, linestyle=":", label="Naive Mean")
ax.axvline(5,  color="#475569", linestyle="--", linewidth=1, alpha=0.7)
ax.axvline(10, color="#475569", linestyle="--", linewidth=1, alpha=0.7)
ax.set_title("Accuracy vs Tolerance Threshold")
ax.set_xlabel("Tolerance (±minutes)")
ax.set_ylabel("% Orders Within Tolerance")
ax.legend(fontsize=9)
ax.set_ylim(0, 105)

plt.savefig(os.path.join(FIGURES_DIR, "fig_doordash_04_business_impact.png"),
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  fig_doordash_04_business_impact.png saved")

# ──────────────────────────────────────────────────────────────────────────────
# Print summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== RESULTS SUMMARY ===")
for m in model_names_ordered:
    r = results[m]
    print(f"  {m:<20}  RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f}  R²={r['R2']:.3f}  "
          f"W±5={r['Within5']*100:.1f}%  W±10={r['Within10']*100:.1f}%")
print(f"\nBest: {best_model_name}")
print("Done!")
