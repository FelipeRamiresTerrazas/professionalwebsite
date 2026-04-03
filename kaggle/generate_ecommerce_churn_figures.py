"""
Generate publication-quality figures for the Food Delivery Customer Intelligence project.
Outputs 3 PNG files to ../figures/ with prefix fig_ecom_

Pipeline:
  Feature Engineering (AOV, Churn, RFM)
  → Descriptive Analytics (Tier KPIs, Coupon Efficiency)
  → Churn Propensity Model (Random Forest, 5-fold CV)
  → Coupon Engine (Next Best Action)

Dataset reframed as food delivery:
  Items Purchased        → Order Frequency
  Total Spend            → Lifetime GMV
  Total Spend / Items    → AOV (Ticket Médio)
  Membership Type        → Subscription Tier (Clube iFood equivalent)
  Discount Applied       → Coupon Usage
  Days Since Last Purch. → Recency  (> 30 days = Churned)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay, classification_report

# ── Dark-theme style (mirrors the portfolio) ─────────────────────────────────
DARK_BG      = "#0f1a2a"
CARD_BG      = "#162236"
BLUE         = "#2563eb"
SKY          = "#0ea5e9"
TEAL         = "#14b8a6"
AMBER        = "#f59e0b"
ROSE         = "#f43f5e"
GREEN        = "#22c55e"
PURPLE       = "#7c3aed"
TEXT         = "#e6f0ff"
DIM          = "#9fb0c8"
TIER_PALETTE = {"Gold": AMBER, "Silver": "#9b9b9b", "Bronze": "#cd7f32"}

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DATA_FILE = Path(__file__).resolve().parent / "e-commerce" / "E-commerce Customer Behavior - Sheet1.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load & Rename
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_FILE)

# Remap column names to food-delivery domain language so every downstream
# transformation reads like a real business analytics brief.
df = df.rename(columns={
    "Items Purchased":          "order_frequency",
    "Total Spend":              "lifetime_gmv",
    "Average Rating":           "avg_rating",
    "Membership Type":          "subscription_tier",
    "Discount Applied":         "coupon_used",
    "Days Since Last Purchase": "recency_days",
    "Satisfaction Level":       "satisfaction",
})

# Normalise string columns — strip whitespace introduced during CSV export
str_cols = ["Gender", "City", "subscription_tier", "satisfaction"]
for c in str_cols:
    df[c] = df[c].str.strip()

print(f"Dataset loaded: {len(df):,} customers  ·  {df.shape[1]} columns")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Feature Engineering & KPI Creation
# ══════════════════════════════════════════════════════════════════════════════

# ── AOV (Average Order Value / Ticket Médio) ──────────────────────────────────
# Primary marketplace health metric: if AOV drops, customers are ordering
# smaller meals or migrating to cheaper items — a margin warning signal.
df["aov"] = df["lifetime_gmv"] / df["order_frequency"]

# ── Churn Target ─────────────────────────────────────────────────────────────
# Business rule: customer not seen in > 30 days is churned.
# In production this would also integrate explicit cancellations and
# refund patterns, but recency is the strongest leading indicator.
df["churn"] = (df["recency_days"] > 30).astype(int)

churn_rate = df["churn"].mean()
print(f"Overall churn rate : {churn_rate:.1%}")
print(f"Active customers   : {(df['churn'] == 0).sum()}")
print(f"Churned customers  : {(df['churn'] == 1).sum()}\n")

# ── RFM Segmentation ─────────────────────────────────────────────────────────
# Industry-standard behavioural segmentation for food delivery.
# Tertile scoring (1–3) is appropriate here given dataset size (~450 rows).
# Production systems with millions of users use quintiles (1–5).
def rfm_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    labels = [1, 2, 3] if ascending else [3, 2, 1]
    return pd.qcut(series, q=3, labels=labels, duplicates="drop").astype(int)

df["R"] = rfm_score(df["recency_days"],    ascending=False)  # lower days = better
df["F"] = rfm_score(df["order_frequency"], ascending=True)
df["M"] = rfm_score(df["lifetime_gmv"],    ascending=True)
df["rfm_score"] = df["R"] + df["F"] + df["M"]

def rfm_segment(row: pd.Series) -> str:
    if row["rfm_score"] >= 8:   return "Champions"
    elif row["rfm_score"] >= 6: return "Loyal"
    elif row["rfm_score"] >= 4: return "At Risk"
    else:                        return "Churned"

df["rfm_segment"] = df.apply(rfm_segment, axis=1)

seg_counts = df["rfm_segment"].value_counts()
print("RFM segment distribution:")
print(seg_counts.to_string(), "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Descriptive Analytics Prep
# ══════════════════════════════════════════════════════════════════════════════

tier_order = ["Gold", "Silver", "Bronze"]
tier_df = df.groupby("subscription_tier").agg(
    avg_aov    =("aov",             "mean"),
    avg_freq   =("order_frequency", "mean"),
    avg_gmv    =("lifetime_gmv",    "mean"),
    churn_rate =("churn",           "mean"),
).reindex(tier_order).reset_index()

print("Subscription Tier KPIs:")
print(tier_df.to_string(), "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIGURE 1 — Subscription Tier Performance Dashboard
# ══════════════════════════════════════════════════════════════════════════════

fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
fig1.patch.set_facecolor(DARK_BG)
fig1.suptitle(
    "Subscription Tier Performance Dashboard",
    color=TEXT, fontsize=17, fontweight="bold", y=0.98,
)

for ax in axes.flat:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=DIM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3f5f")

panels = [
    (axes[0, 0], "avg_aov",    "Avg. Ticket Médio (AOV)",  "AOV by Subscription Tier"),
    (axes[0, 1], "avg_freq",   "Avg. Orders",               "Order Frequency by Tier"),
    (axes[1, 0], "avg_gmv",    "Avg. Lifetime GMV",         "Lifetime GMV by Tier"),
    (axes[1, 1], "churn_rate", "Churn Rate",                "Churn Rate by Tier"),
]

bar_colors = [TIER_PALETTE[t] for t in tier_df["subscription_tier"]]

for ax, col, ylabel, title in panels:
    bars = ax.bar(tier_df["subscription_tier"], tier_df[col],
                  color=bar_colors, width=0.5, edgecolor=DARK_BG, linewidth=0.5)

    # Annotate bar tops for executive readability
    for bar, val in zip(bars, tier_df[col]):
        fmt = f"{val:.1%}" if col == "churn_rate" else f"{val:,.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + tier_df[col].max() * 0.025,
            fmt, ha="center", va="bottom", fontsize=11, color=TEXT, fontweight="bold",
        )

    ax.set_title(title, fontsize=12, color=TEXT, pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=DIM)
    ax.tick_params(axis="x", labelsize=10)

    if col == "churn_rate":
        ax.axhline(
            churn_rate, color=BLUE, linestyle="--", linewidth=1.5,
            label=f"Fleet avg {churn_rate:.1%}",
        )
        ax.legend(fontsize=9, facecolor=CARD_BG, labelcolor=TEXT, edgecolor="#2a3f5f")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

plt.tight_layout(rect=[0, 0, 1, 0.96])
out1 = FIGURES_DIR / "fig_ecom_01_tier_kpis.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig1)
print(f"Saved {out1.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURE 2 — Coupon Efficiency Analysis
# ══════════════════════════════════════════════════════════════════════════════

coupon_labels = {True: "Coupon Used", False: "No Coupon"}
sat_order     = ["Satisfied", "Neutral", "Unsatisfied"]
sat_palette   = {"Satisfied": GREEN, "Neutral": AMBER, "Unsatisfied": ROSE}

coupon_churn = df.groupby("coupon_used")["churn"].mean().rename(coupon_labels)
coupon_aov   = df.groupby("coupon_used")["aov"].mean().rename(coupon_labels)
print("\nChurn rate by coupon usage:")
print(coupon_churn.to_string())
print("\nAOV by coupon usage:")
print(coupon_aov.to_string(), "\n")

# Satisfaction breakdown (dropna for the 2 missing rows)
sat_counts = (
    df.dropna(subset=["satisfaction"])
    .groupby(["coupon_used", "satisfaction"])
    .size()
    .unstack("satisfaction")
    .reindex(columns=sat_order, fill_value=0)
)
sat_pct        = sat_counts.div(sat_counts.sum(axis=1), axis=0)
sat_pct.index  = sat_pct.index.map(coupon_labels)

fig2, axes = plt.subplots(1, 3, figsize=(17, 5))
fig2.patch.set_facecolor(DARK_BG)
fig2.suptitle("Coupon Efficiency Analysis", color=TEXT, fontsize=17, fontweight="bold", y=1.02)

for ax in axes:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=DIM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3f5f")

# Panel A — Churn rate by coupon
colors_c = [ROSE if v > churn_rate else GREEN for v in coupon_churn]
axes[0].bar(coupon_churn.index, coupon_churn.values, color=colors_c, width=0.4, edgecolor=DARK_BG)
for i, (_, val) in enumerate(coupon_churn.items()):
    axes[0].text(i, val + 0.012, f"{val:.1%}", ha="center", fontsize=12, color=TEXT, fontweight="bold")
axes[0].axhline(churn_rate, color=BLUE, linestyle="--", linewidth=1.5, label=f"Fleet avg {churn_rate:.1%}")
axes[0].set_title("Churn Rate by Coupon Usage", fontsize=11, color=TEXT, pad=8)
axes[0].set_ylabel("Churn Rate", fontsize=9, color=DIM)
axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes[0].legend(fontsize=9, facecolor=CARD_BG, labelcolor=TEXT, edgecolor="#2a3f5f")

# Panel B — Satisfaction stacked % bar
bottom = np.zeros(len(sat_pct))
for sat in sat_order:
    axes[1].bar(sat_pct.index, sat_pct[sat], bottom=bottom,
                color=sat_palette[sat], label=sat, width=0.4, edgecolor=DARK_BG)
    for j, val in enumerate(sat_pct[sat]):
        if val > 0.07:
            axes[1].text(
                j, bottom[j] + val / 2, f"{val:.0%}",
                ha="center", va="center", fontsize=10, color=DARK_BG, fontweight="bold",
            )
    bottom += sat_pct[sat].values
axes[1].set_title("Satisfaction Mix by Coupon Usage", fontsize=11, color=TEXT, pad=8)
axes[1].set_ylabel("% of Customers", fontsize=9, color=DIM)
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes[1].legend(fontsize=9, facecolor=CARD_BG, labelcolor=TEXT, edgecolor="#2a3f5f", loc="upper right")

# Panel C — AOV comparison (key margin test: are coupons attracting low-AOV orders?)
axes[2].bar(coupon_aov.index, coupon_aov.values, color=[SKY, TEAL], width=0.4, edgecolor=DARK_BG)
for i, (_, val) in enumerate(coupon_aov.items()):
    axes[2].text(i, val + 0.5, f"R$ {val:.1f}", ha="center", fontsize=12, color=TEXT, fontweight="bold")
axes[2].set_title("Avg. Ticket Médio by Coupon Usage", fontsize=11, color=TEXT, pad=8)
axes[2].set_ylabel("AOV", fontsize=9, color=DIM)

fig2.tight_layout()
out2 = FIGURES_DIR / "fig_ecom_02_coupon_efficiency.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig2)
print(f"Saved {out2.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Churn Propensity Model — Random Forest
# ══════════════════════════════════════════════════════════════════════════════

# Critical design choice: recency_days is EXCLUDED from features.
# churn IS recency_days > 30. Including it gives ~100% accuracy but the model
# learns nothing — it would just threshold the same column it defined churn from.
# Excluding recency forces the model to learn from behavioural signals
# (spend patterns, subscription tier, satisfaction, coupon sensitivity)
# which is what a real production churn model does.

FEATURES = [
    "Age", "Gender", "City",
    "subscription_tier",
    "order_frequency",
    "lifetime_gmv",
    "avg_rating",
    "coupon_used",
    "satisfaction",
    "aov",
]
TARGET = "churn"

X = df[FEATURES].copy()
y = df[TARGET]
X["coupon_used"] = X["coupon_used"].astype(int)

cat_features = ["Gender", "City", "subscription_tier", "satisfaction"]
num_features  = [c for c in FEATURES if c not in cat_features]

# Ordinal encoder with explicit rank ordering.
# Using ordinal (not OHE) so the model can leverage the known ordering of
# tier (Bronze < Silver < Gold) and satisfaction (Unsatisfied < Neutral < Satisfied).
cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),   # fills the 2 missing satisfaction rows
    ("encode", OrdinalEncoder(
        categories=[
            ["Female", "Male"],
            sorted(df["City"].dropna().unique().tolist()),
            ["Bronze", "Silver", "Gold"],
            ["Unsatisfied", "Neutral", "Satisfied"],
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])
num_pipeline  = Pipeline([("impute", SimpleImputer(strategy="median"))])
preprocessor  = ColumnTransformer([
    ("cat", cat_pipeline, cat_features),
    ("num", num_pipeline, num_features),
])

# Random Forest with shallow depth to avoid overfitting on a ~450-row dataset.
# max_depth=4 and min_samples_leaf=10 ensure each split has generalizable signal.
clf = RandomForestClassifier(
    n_estimators     = 200,
    max_depth        = 4,
    min_samples_leaf = 10,
    class_weight     = "balanced",
    random_state     = 42,
)
full_pipeline = Pipeline([("prep", preprocessor), ("model", clf)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_proba_cv = cross_val_predict(full_pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
y_pred_cv  = (y_proba_cv >= 0.5).astype(int)

roc_auc = roc_auc_score(y, y_proba_cv)
print(f"\n── Churn Model Performance (5-fold Stratified CV) ──")
print(classification_report(y, y_pred_cv, target_names=["Active", "Churned"]))
print(f"ROC-AUC: {roc_auc:.4f}\n")

# Fit on full data to extract feature importances for the business brief
full_pipeline.fit(X, y)
importances  = full_pipeline.named_steps["model"].feature_importances_
feature_names = cat_features + num_features
fi_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=True)
)
print("Feature importance (all features):")
print(fi_df.to_string(), "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FIGURE 3 — Churn Model Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

fig3, axes = plt.subplots(1, 2, figsize=(15, 6))
fig3.patch.set_facecolor(DARK_BG)
fig3.suptitle(
    "Churn Propensity Model — Random Forest",
    color=TEXT, fontsize=17, fontweight="bold",
)

for ax in axes:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=DIM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3f5f")

# Panel A — Feature Importance (What drives churn?)
axes[0].barh(fi_df["feature"], fi_df["importance"], color=TEAL, height=0.6, edgecolor=DARK_BG)
axes[0].set_title("Feature Importance — What Drives Churn?", fontsize=12, color=TEXT, pad=8)
axes[0].set_xlabel("Mean Decrease in Impurity", fontsize=9, color=DIM)
axes[0].tick_params(axis="y", labelsize=9)

# Panel B — ROC Curve (5-fold CV out-of-fold probabilities)
RocCurveDisplay.from_predictions(
    y, y_proba_cv,
    name=f"Random Forest (5-fold CV)  AUC = {roc_auc:.3f}",
    ax=axes[1],
    color=AMBER,
)
axes[1].plot([0, 1], [0, 1], linestyle="--", color=DIM, linewidth=1, label="Random baseline")
axes[1].set_title("ROC Curve", fontsize=12, color=TEXT, pad=8)
axes[1].set_xlabel("False Positive Rate", fontsize=9, color=DIM)
axes[1].set_ylabel("True Positive Rate", fontsize=9, color=DIM)
axes[1].legend(fontsize=9, facecolor=CARD_BG, labelcolor=TEXT, edgecolor="#2a3f5f")

fig3.tight_layout()
out3 = FIGURES_DIR / "fig_ecom_03_churn_model.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig3)
print(f"Saved {out3.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Prescriptive — Coupon Engine Summary
# ══════════════════════════════════════════════════════════════════════════════

df["churn_proba"] = cross_val_predict(full_pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

def next_best_action(row: pd.Series) -> str:
    p    = row["churn_proba"]
    tier = row["subscription_tier"]
    used = row["coupon_used"]

    if p >= 0.70:
        return "SEND_AGGRESSIVE_COUPON" if used else "UPSELL_TO_SUBSCRIPTION"
    elif p >= 0.40:
        return "UPSELL_SUBSCRIPTION" if tier == "Bronze" else "SEND_LIGHT_COUPON"
    else:
        return "DO_NOT_SEND"

df["nba_action"] = df.apply(next_best_action, axis=1)

print("\n── Next Best Action Distribution ──")
nba_dist = df["nba_action"].value_counts()
print(nba_dist.to_string())

nba_summary = (
    df.groupby("nba_action").agg(
        customers      =("Customer ID",  "count"),
        avg_churn_prob =("churn_proba",  "mean"),
        churn_rate     =("churn",        "mean"),
        avg_gmv        =("lifetime_gmv", "mean"),
    ).sort_values("avg_churn_prob", ascending=False)
)
print("\nNBA Summary:")
print(nba_summary.to_string())

# Export enriched file for CRM ingestion demo
out_csv = FIGURES_DIR / "customers_enriched.csv"
df.to_csv(out_csv, index=False)
print(f"\nExported enriched customer table: {out_csv.name}")
print("\nDone. All 3 figures saved to figures/")
