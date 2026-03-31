"""
Generate publication-quality figures for the Customer Churn project.
Outputs 4 PNG files to ../figures/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, f1_score, accuracy_score
)
from xgboost import XGBClassifier

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8")
sns.set_context("talk", font_scale=0.88)
plt.rcParams["axes.grid"] = False

DARK_BG   = "#0f1a2a"
BLUE      = "#2563eb"
SKY       = "#0ea5e9"
TEAL      = "#14b8a6"
AMBER     = "#f59e0b"
ROSE      = "#f43f5e"
TEXT      = "#e6f0ff"
TEXT_DIM  = "#9fb0c8"
CARD_BG   = "#162236"

PALETTE   = {0: "#2ecc71", 1: "#e74c3c"}   # No Churn / Churn

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
KAGGLE_DIR = Path(__file__).resolve().parent

# ── Load & Prep Data ─────────────────────────────────────────────────────────
df = pd.read_csv(KAGGLE_DIR / "train.csv")

# Fix TotalCharges (blank strings for new customers)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Binary target
df["ChurnBin"] = (df["Churn"] == "Yes").astype(int)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# Full feature encoding for model — store encoders fitted on train
cat_cols = df.select_dtypes("object").columns.drop(["Churn"])
encoders = {}
df_enc = df.copy()
for c in cat_cols:
    le = LabelEncoder()
    df_enc[c] = le.fit_transform(df_enc[c].astype(str))
    encoders[c] = le

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
df_enc[num_cols] = scaler.fit_transform(df_enc[num_cols].fillna(0))

FEATURE_COLS = [c for c in df_enc.columns
                if c not in ("id", "Churn", "ChurnBin")]
X = df_enc[FEATURE_COLS].fillna(0)
y = df_enc["ChurnBin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scale_pos  = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X_train, y_train)
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Also train a full model on all training data for submission
model_full = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model_full.fit(X, y)


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Exploratory Data Analysis
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(DARK_BG)
for ax in axes.flat:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_DIM)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3f55")

# Panel 1 — Class Imbalance
ax = axes[0, 0]
counts = df["Churn"].value_counts()
pcts   = counts / counts.sum() * 100
bars   = ax.bar(counts.index, counts.values,
                color=[PALETTE[0], PALETTE[1]], width=0.5, edgecolor="none")
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30, f"{pct:.1f}%",
            ha="center", va="bottom", color=TEXT, fontsize=13, fontweight="bold")
ax.set_title("Churn Class Distribution", color=TEXT, fontsize=14, pad=10)
ax.set_xlabel("")
ax.set_ylabel("Customers", color=TEXT_DIM, fontsize=11)
ax.tick_params(labelcolor=TEXT_DIM)
ax.set_facecolor(CARD_BG)

# Panel 2 — Churn Rate by Contract Type
ax = axes[0, 1]
contract_churn = (
    df.groupby("Contract")["ChurnBin"]
    .agg(["sum", "count"])
    .assign(rate=lambda d: d["sum"] / d["count"] * 100)
    .sort_values("rate", ascending=False)
)
bars = ax.barh(contract_churn.index, contract_churn["rate"],
               color=[ROSE, AMBER, TEAL], edgecolor="none", height=0.5)
for bar, val in zip(bars, contract_churn["rate"]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", color=TEXT, fontsize=12)
ax.set_title("Churn Rate by Contract Type", color=TEXT, fontsize=14, pad=10)
ax.set_xlabel("Churn Rate (%)", color=TEXT_DIM, fontsize=11)
ax.tick_params(labelcolor=TEXT_DIM)
ax.set_facecolor(CARD_BG)
ax.set_xlim(0, contract_churn["rate"].max() * 1.25)

# Panel 3 — Tenure Distribution by Churn
ax = axes[1, 0]
for label, grp in df.groupby("Churn"):
    color = ROSE if label == "Yes" else TEAL
    ax.hist(grp["tenure"], bins=30, alpha=0.72, color=color,
            label=f"Churn: {label}", edgecolor="none", density=True)
ax.set_title("Tenure Distribution by Churn", color=TEXT, fontsize=14, pad=10)
ax.set_xlabel("Tenure (months)", color=TEXT_DIM, fontsize=11)
ax.set_ylabel("Density", color=TEXT_DIM, fontsize=11)
ax.tick_params(labelcolor=TEXT_DIM)
ax.set_facecolor(CARD_BG)
leg = ax.legend(fontsize=11, facecolor=DARK_BG, labelcolor=TEXT, framealpha=0.8)

# Panel 4 — Monthly Charges Distribution by Churn
ax = axes[1, 1]
for label, grp in df.groupby("Churn"):
    color = ROSE if label == "Yes" else TEAL
    ax.hist(grp["MonthlyCharges"], bins=30, alpha=0.72, color=color,
            label=f"Churn: {label}", edgecolor="none", density=True)
ax.set_title("Monthly Charges Distribution", color=TEXT, fontsize=14, pad=10)
ax.set_xlabel("Monthly Charges ($)", color=TEXT_DIM, fontsize=11)
ax.set_ylabel("Density", color=TEXT_DIM, fontsize=11)
ax.tick_params(labelcolor=TEXT_DIM)
ax.set_facecolor(CARD_BG)
ax.legend(fontsize=11, facecolor=DARK_BG, labelcolor=TEXT, framealpha=0.8)

fig.suptitle("Customer Churn — Exploratory Data Analysis",
             color=TEXT, fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_churn_01_eda.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("✓ fig_churn_01_eda.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(DARK_BG)
for ax in axes.flat:
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_DIM)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3f55")

# Panel 1 — ROC Curve
ax = axes[0, 0]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score   = roc_auc_score(y_test, y_proba)
ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"XGBoost  (AUC = {auc_score:.3f})")
ax.plot([0, 1], [0, 1], color=TEXT_DIM, lw=1.2, linestyle="--", label="Random baseline")
ax.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
ax.set_title("ROC Curve", color=TEXT, fontsize=14, pad=10)
ax.set_xlabel("False Positive Rate", color=TEXT_DIM, fontsize=11)
ax.set_ylabel("True Positive Rate", color=TEXT_DIM, fontsize=11)
ax.tick_params(labelcolor=TEXT_DIM)
ax.legend(fontsize=11, facecolor=DARK_BG, labelcolor=TEXT, framealpha=0.8)
ax.set_facecolor(CARD_BG)

# Panel 2 — Precision-Recall Curve
ax = axes[0, 1]
precision, recall, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
baseline = y_test.mean()
ax.plot(recall, precision, color=SKY, lw=2.5, label=f"XGBoost  (AP = {ap:.3f})")
ax.axhline(baseline, color=TEXT_DIM, lw=1.2, linestyle="--",
           label=f"Random baseline ({baseline:.2f})")
ax.fill_between(recall, precision, alpha=0.12, color=SKY)
ax.set_title("Precision-Recall Curve", color=TEXT, fontsize=14, pad=10)
ax.set_xlabel("Recall", color=TEXT_DIM, fontsize=11)
ax.set_ylabel("Precision", color=TEXT_DIM, fontsize=11)
ax.tick_params(labelcolor=TEXT_DIM)
ax.legend(fontsize=11, facecolor=DARK_BG, labelcolor=TEXT, framealpha=0.8)
ax.set_facecolor(CARD_BG)

# Panel 3 — Confusion Matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: No Churn", "Pred: Churn"],
            yticklabels=["True: No Churn", "True: Churn"],
            ax=ax, cbar=False,
            annot_kws={"size": 15, "weight": "bold", "color": TEXT})
ax.set_title("Confusion Matrix", color=TEXT, fontsize=14, pad=10)
ax.tick_params(labelcolor=TEXT_DIM, labelsize=10)
ax.set_facecolor(CARD_BG)

# Panel 4 — Metrics Bar Chart (CV)
ax = axes[1, 1]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba",
                             n_jobs=-1)[:, 1]
cv_pred  = (cv_proba >= 0.5).astype(int)
metrics = {
    "ROC AUC":   roc_auc_score(y, cv_proba),
    "Avg. Prec.":average_precision_score(y, cv_proba),
    "F1":        f1_score(y, cv_pred),
    "Accuracy":  accuracy_score(y, cv_pred),
}
colors = [BLUE, SKY, TEAL, AMBER]
bars = ax.barh(list(metrics.keys()), list(metrics.values()),
               color=colors, edgecolor="none", height=0.55)
for bar, val in zip(bars, metrics.values()):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", color=TEXT, fontsize=12)
ax.set_title("5-Fold CV Metrics", color=TEXT, fontsize=14, pad=10)
ax.set_xlabel("Score", color=TEXT_DIM, fontsize=11)
ax.set_xlim(0, 1.12)
ax.tick_params(labelcolor=TEXT_DIM)
ax.set_facecolor(CARD_BG)

fig.suptitle("Customer Churn — Model Performance (XGBoost)",
             color=TEXT, fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_churn_02_model_performance.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("✓ fig_churn_02_model_performance.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — Feature Importance
# ════════════════════════════════════════════════════════════════════════════
importance_df = (
    pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    })
    .sort_values("importance", ascending=True)
    .tail(18)
)

# Map raw encoded column names back to readable labels
label_map = {
    "MonthlyCharges":     "Monthly Charges",
    "TotalCharges":       "Total Charges",
    "tenure":             "Tenure (months)",
    "Contract":           "Contract Type",
    "InternetService":    "Internet Service",
    "PaymentMethod":      "Payment Method",
    "OnlineSecurity":     "Online Security",
    "TechSupport":        "Tech Support",
    "OnlineBackup":       "Online Backup",
    "DeviceProtection":   "Device Protection",
    "StreamingTV":        "Streaming TV",
    "StreamingMovies":    "Streaming Movies",
    "MultipleLines":      "Multiple Lines",
    "PaperlessBilling":   "Paperless Billing",
    "SeniorCitizen":      "Senior Citizen",
    "Partner":            "Has Partner",
    "Dependents":         "Has Dependents",
    "PhoneService":       "Phone Service",
    "gender":             "Gender",
}
importance_df["label"] = importance_df["feature"].map(
    lambda x: label_map.get(x, x)
)

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)

norm_imp = importance_df["importance"] / importance_df["importance"].max()
bar_colors = [
    BLUE if v >= 0.5 else (SKY if v >= 0.25 else TEXT_DIM)
    for v in norm_imp
]
bars = ax.barh(importance_df["label"], importance_df["importance"],
               color=bar_colors, edgecolor="none", height=0.7)
ax.axvline(importance_df["importance"].iloc[importance_df["importance"].values.argmax()],
           color=BLUE, lw=0, alpha=0)   # invisible anchor for scale

for bar, val in zip(bars, importance_df["importance"]):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", color=TEXT_DIM, fontsize=10)

ax.set_title("Feature Importance — XGBoost Churn Model (Top 18)",
             color=TEXT, fontsize=14, pad=12)
ax.set_xlabel("Importance Score (F-score)", color=TEXT_DIM, fontsize=11)
ax.tick_params(labelcolor=TEXT_DIM, labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor("#2a3f55")

fig.suptitle("Key Drivers of Customer Churn",
             color=TEXT, fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_churn_03_feature_importance.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("✓ fig_churn_03_feature_importance.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Business Impact
# ════════════════════════════════════════════════════════════════════════════
# Business assumptions
AVG_CLV        = 500    # USD — estimated customer lifetime value
RETENTION_COST = 50     # USD — cost of a proactive retention offer
FP_COST        = 10     # USD — cost of offering a discount to a loyal customer

# Compute test-set financials
monthly_charges_test = df.loc[X_test.index, "MonthlyCharges"].values
clv_test = monthly_charges_test * 12   # proxy: 12 months revenue

cm_flat = cm.ravel()
tn, fp, fn, tp = cm_flat

money_saved  = clv_test[(y_test.values == 1) & (y_pred == 1)].sum()
money_lost   = clv_test[(y_test.values == 1) & (y_pred == 0)].sum()
retention_spend = tp * RETENTION_COST
fp_spend     = fp * FP_COST
net_benefit  = money_saved - retention_spend - fp_spend

fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor(DARK_BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.38)

ax_bar  = fig.add_subplot(gs[0, :2])
ax_pie  = fig.add_subplot(gs[0, 2])
ax_seg  = fig.add_subplot(gs[1, :2])
ax_kpi  = fig.add_subplot(gs[1, 2])

for ax in [ax_bar, ax_pie, ax_seg, ax_kpi]:
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3f55")

# Panel 1 — Revenue Saved vs Lost vs Costs
labels  = ["Revenue\nSaved", "Revenue\nLost", "Retention\nCost", "FP\nCost", "Net\nBenefit"]
values  = [money_saved, -money_lost, -retention_spend, -fp_spend, net_benefit]
colors_ = [TEAL, ROSE, AMBER, TEXT_DIM, BLUE]
bars    = ax_bar.bar(labels, values, color=colors_, edgecolor="none", width=0.6)
ax_bar.axhline(0, color=TEXT_DIM, lw=0.8)
for bar, val in zip(bars, values):
    ypos = bar.get_height() + (abs(val) * 0.02) if val >= 0 else bar.get_height() - (abs(val) * 0.07)
    ax_bar.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"${abs(val):,.0f}", ha="center", color=TEXT, fontsize=12, fontweight="bold")
ax_bar.set_title("Business Impact on Test Set", color=TEXT, fontsize=13, pad=10)
ax_bar.set_ylabel("USD ($)", color=TEXT_DIM, fontsize=11)
ax_bar.tick_params(labelcolor=TEXT_DIM)

# Panel 2 — Correctly caught vs missed pie
ax_pie.pie(
    [tp, fn],
    labels=["Caught\n(TP)", "Missed\n(FN)"],
    colors=[TEAL, ROSE],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"color": TEXT, "fontsize": 11},
    wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
)
ax_pie.set_title(f"Churners Identified\n(out of {tp+fn})",
                 color=TEXT, fontsize=13, pad=6)

# Panel 3 — Churn rate by Internet Service + Contract (segment heatmap)
seg = (
    df.groupby(["InternetService", "Contract"])["ChurnBin"]
    .mean()
    .unstack()
    .fillna(0) * 100
)
sns.heatmap(seg, ax=ax_seg, cmap="RdYlGn_r",
            annot=True, fmt=".1f", annot_kws={"size": 11, "color": TEXT},
            linewidths=0.5, linecolor=DARK_BG, cbar=False,
            xticklabels=seg.columns, yticklabels=seg.index)
ax_seg.set_title("Churn Rate (%) by Service & Contract",
                 color=TEXT, fontsize=13, pad=10)
ax_seg.tick_params(labelcolor=TEXT_DIM, labelsize=10, rotation=0)
ax_seg.set_xlabel("Contract Type", color=TEXT_DIM, fontsize=10)
ax_seg.set_ylabel("Internet Service", color=TEXT_DIM, fontsize=10)

# Panel 4 — KPI summary
ax_kpi.axis("off")
kpis = [
    ("ROC AUC",         f"{auc_score:.3f}"),
    ("Avg. Precision",  f"{ap:.3f}"),
    ("F1 Score",        f"{f1_score(y_test, y_pred):.3f}"),
    ("Churners Found",  f"{tp}/{tp+fn}  ({tp/(tp+fn)*100:.0f}%)"),
    ("Net Benefit",     f"${net_benefit:,.0f}"),
]
y_pos = 0.95
for label, value in kpis:
    ax_kpi.text(0.05, y_pos, label, transform=ax_kpi.transAxes,
                color=TEXT_DIM, fontsize=11, va="top")
    ax_kpi.text(0.95, y_pos, value, transform=ax_kpi.transAxes,
                color=TEXT, fontsize=12, fontweight="bold", va="top", ha="right")
    y_pos -= 0.17
ax_kpi.set_title("Key Metrics", color=TEXT, fontsize=13, pad=10)

fig.suptitle("Customer Churn — Business Impact Analysis",
             color=TEXT, fontsize=16, fontweight="bold")
fig.savefig(FIGURES_DIR / "fig_churn_04_business_impact.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("✓ fig_churn_04_business_impact.png")

print("\nAll 4 figures saved to", FIGURES_DIR)


# ════════════════════════════════════════════════════════════════════════════
# SUBMISSION — predict on test.csv using full-data model
# ════════════════════════════════════════════════════════════════════════════
print("\n--- Generating Kaggle submission ---")

test_raw = pd.read_csv(KAGGLE_DIR / "test.csv")
test_raw["TotalCharges"] = pd.to_numeric(test_raw["TotalCharges"], errors="coerce")
test_raw["SeniorCitizen"] = test_raw["SeniorCitizen"].map({0: "No", 1: "Yes"})

test_enc = test_raw.copy()
for c, le in encoders.items():
    if c in test_enc.columns:
        # handle unseen labels by mapping to the most frequent class (index 0)
        test_enc[c] = test_enc[c].astype(str).map(
            lambda v, le=le: le.transform([v])[0]
            if v in le.classes_ else 0
        )

test_enc[num_cols] = scaler.transform(test_enc[num_cols].fillna(0))

X_submit = test_enc[FEATURE_COLS].fillna(0)
churn_proba = model_full.predict_proba(X_submit)[:, 1]

submission = pd.DataFrame({"id": test_raw["id"], "Churn": churn_proba})

submission_path = KAGGLE_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"✓ submission.csv  ({len(submission):,} rows, proba range "
      f"[{churn_proba.min():.4f}, {churn_proba.max():.4f}])")
print(f"  Saved to: {submission_path}")
