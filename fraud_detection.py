"""
=============================================================================
CREDIT CARD FRAUD DETECTION â€” PORTFOLIO PROJECT
=============================================================================
Author : Felipe Ramires Terrazas

Visualization philosophy:
  Every figure uses plt.subplots() to create a structured panel grid.
  plt.tight_layout() + subplots_adjust() guarantee zero overlap between
  titles, axes, and annotations regardless of screen size.
  Output: 4 large multi-panel PNGs, each telling a complete story.

Key technical decisions (commented inline):
  - RobustScaler  : handles outliers in 'Amount' better than StandardScaler
  - AUPRC         : gold standard for imbalanced classification (99.83/0.17%)
  - SMOTE on train only : test set stays imbalanced = real-world conditions
  - class_weight  : alternative strategy, no synthetic data generated
=============================================================================
"""

import warnings
import os
import sys

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows terminals (avoids cp1252 encoding errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, average_precision_score,
    roc_auc_score, f1_score, precision_recall_curve,
    classification_report,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# =============================================================================
# GLOBAL AESTHETICS
# =============================================================================
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Professional colour palette
C_NORMAL  = "#2ecc71"   # green   â€” normal transactions
C_FRAUD   = "#e74c3c"   # red     â€” fraudulent transactions
C_BLUE    = "#3498db"   # blue    â€” XGBoost / secondary
C_PURPLE  = "#9b59b6"   # purple  â€” accent
C_GRAY    = "#95a5a6"   # gray    â€” baseline / neutral
C_ORANGE  = "#e67e22"   # orange  â€” third series

MODEL_COLORS = [C_NORMAL, C_BLUE, C_ORANGE, C_PURPLE]

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f5f6fa",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.titlepad":    12,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "legend.framealpha": 0.85,
})


def save_fig(fig, name: str, dpi: int = 180):
    """Save figure as high-res PNG and close it."""
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  [SAVED] {path}")
    plt.close(fig)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 65)
print("1. LOADING DATA")
print("=" * 65)

df = pd.read_csv("creditcard.csv")

n_fraud  = df["Class"].sum()
n_total  = len(df)
n_normal = n_total - n_fraud

print(f"  Shape       : {df.shape}")
print(f"  Null values : {df.isnull().sum().sum()}")
print(f"  Normal      : {n_normal:,}  ({n_normal/n_total*100:.3f}%)")
print(f"  Fraud       : {n_fraud:,}   ({n_fraud/n_total*100:.3f}%)")
print(f"  Imbalance   : {n_normal/n_fraud:.0f}:1")


# =============================================================================
# 2. FIGURE 1 â€” EXPLORATORY DATA ANALYSIS  (2 Ã— 2 panel grid)
# =============================================================================
print("\n" + "=" * 65)
print("2. FIGURE 1 â€” EDA")
print("=" * 65)

# Why plt.subplots() instead of separate figures?
# A single figure object with a constrained layout makes it impossible for
# panel titles and axis labels to overlap â€” the layout engine handles spacing
# automatically. Saving as one PNG also means one card on the website.

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Credit Card Fraud Detection - Exploratory Data Analysis",
    fontsize=15, fontweight="bold"
)

# â”€â”€ Panel A: Class imbalance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax = axes[0, 0]
counts = df["Class"].value_counts()
pcts   = df["Class"].value_counts(normalize=True) * 100
bars   = ax.bar(
    ["Normal (0)", "Fraud (1)"], counts.values,
    color=[C_NORMAL, C_FRAUD], edgecolor="white", linewidth=1.5, width=0.5
)
for bar, cnt, pct in zip(bars, counts.values, pcts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2500,
        f"{cnt:,}\n({pct:.2f}%)",
        ha="center", fontweight="bold", fontsize=10
    )
ax.set_ylabel("Number of Transactions")
ax.set_title("Class Distribution - Extreme 578:1 Imbalance")
ax.set_ylim(0, counts.max() * 1.18)
# Key annotation explaining why accuracy is a useless metric here
ax.annotate(
    "A model that always predicts\n'Normal' scores 99.83% accuracy\n"
    "but catches ZERO fraud.\n-> We use AUPRC instead.",
    xy=(1, counts[1] + 8000), xytext=(0.5, counts.max() * 0.45),
    fontsize=8.5, color="#444",
    arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
)

# â”€â”€ Panel B: Transaction amount â€” log scale â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax = axes[0, 1]
# np.log1p(x) = log(1+x) avoids log(0) for zero-amount transactions
# Log scale reveals the full distribution including rare high-value fraud
for cls, label, color in [(0, "Normal", C_NORMAL), (1, "Fraud", C_FRAUD)]:
    vals = np.log1p(df[df["Class"] == cls]["Amount"])
    ax.hist(vals, bins=70, alpha=0.65, label=label,
            color=color, edgecolor="white", linewidth=0.3, density=True)
ax.set_xlabel("log(1 + Amount)  [USD]")
ax.set_ylabel("Density")
ax.set_title("Transaction Amount Distribution (Log Scale)")
ax.legend()
ax.annotate(
    "Log scale reveals high-value\nfraud outliers invisible\nin linear scale",
    xy=(7.5, 0.08), xytext=(4.5, 0.35),
    fontsize=8.5, color="#444",
    arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
)

# â”€â”€ Panel C: Temporal distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax = axes[1, 0]
for cls, label, color in [(0, "Normal", C_NORMAL), (1, "Fraud", C_FRAUD)]:
    vals = df[df["Class"] == cls]["Time"] / 3600   # seconds â†’ hours
    ax.hist(vals, bins=48, alpha=0.65, label=label,
            color=color, edgecolor="white", linewidth=0.3, density=True)
ax.set_xlabel("Hours since start of data collection")
ax.set_ylabel("Density")
ax.set_title("Temporal Distribution - Fraud Spikes at Specific Hours")
ax.legend()

# â”€â”€ Panel D: Most discriminative PCA features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax = axes[1, 1]
pca_cols = [f"V{i}" for i in range(1, 29)]
# Measure how much each feature's absolute value differs between classes.
# A large positive difference â†’ feature is elevated in fraud transactions.
diff = (
    df[df["Class"] == 1][pca_cols].abs().mean()
    - df[df["Class"] == 0][pca_cols].abs().mean()
).sort_values(ascending=False).head(12)
bar_colors = [C_FRAUD if v >= 0 else C_BLUE for v in diff.values[::-1]]
diff.sort_values().plot(kind="barh", ax=ax, color=bar_colors, edgecolor="white")
ax.set_xlabel("Mean |value| difference  (Fraud - Normal)")
ax.set_title("Top 12 Most Discriminative PCA Features")
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

plt.tight_layout()
plt.subplots_adjust(hspace=0.44, wspace=0.34)
save_fig(fig, "fig_01_eda")


# =============================================================================
# 3. PREPROCESSING
# =============================================================================
print("\n" + "=" * 65)
print("3. PREPROCESSING")
print("=" * 65)

# Why RobustScaler instead of StandardScaler?
# StandardScaler computes mean Â± std. A single $25,000 fraud transaction
# skews both statistics heavily. RobustScaler uses median + IQR, which are
# resistant to extreme outliers â€” exactly what fraud data contains.
scaler = RobustScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
df.drop(["Amount", "Time"], axis=1, inplace=True)

X = df.drop("Class", axis=1)
y = df["Class"]

# stratify=y ensures the 0.17% fraud rate is preserved in both splits â€”
# without this, the test set could randomly contain zero fraud cases.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE: generate synthetic minority samples by interpolating between
# real fraud examples. CRUCIAL: applied ONLY on the training set.
# The test set stays imbalanced to simulate real-world deployment.
smote = SMOTE(random_state=42)
X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)

print(f"  Train (original): {X_train.shape[0]:,}  |  fraud: {y_train.sum()}")
print(f"  Train (SMOTE)   : {X_tr_sm.shape[0]:,}  |  fraud: {int(y_tr_sm.sum())}")
print(f"  Test  (original): {X_test.shape[0]:,}  |  fraud: {y_test.sum()}")


# =============================================================================
# 4. MODEL TRAINING
# =============================================================================
print("\n" + "=" * 65)
print("4. MODEL TRAINING")
print("=" * 65)

results = {}


def train_eval(model, X_tr, y_tr, name: str):
    """Train model on (X_tr, y_tr), evaluate against the held-out test set."""
    print(f"\n  -- {name}")
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # AUPRC: focuses exclusively on the minority class (fraud).
    # ROC-AUC can look inflated because the huge normal class dilutes FPR.
    auprc = average_precision_score(y_test, y_proba)
    roc   = roc_auc_score(y_test, y_proba)
    f1    = f1_score(y_test, y_pred)
    cm    = confusion_matrix(y_test, y_pred)

    print(f"     AUPRC={auprc:.4f}  ROC={roc:.4f}  F1={f1:.4f}  "
          f"TP={cm[1,1]}  FP={cm[0,1]}  FN={cm[1,0]}")

    results[name] = {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "auprc": auprc, "roc": roc, "f1": f1, "cm": cm,
    }


neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())

# â”€â”€ Random Forest with class_weight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# class_weight='balanced' automatically weights the loss inversely proportional
# to class frequencies: weight_fraud = n_samples / (2 * n_fraud)
train_eval(
    RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    X_train, y_train, "Random Forest (Weights)"
)

# â”€â”€ XGBoost with scale_pos_weight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# scale_pos_weight = n_negative / n_positive â‰ˆ 578,
# equivalent to class_weight='balanced' in sklearn.
train_eval(
    XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=neg_count / pos_count,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="aucpr",
        random_state=42, n_jobs=-1
    ),
    X_train, y_train, "XGBoost (Weights)"
)

# â”€â”€ Random Forest with SMOTE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# No class_weight needed â€” SMOTE already balanced the training distribution.
train_eval(
    RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    ),
    X_tr_sm, y_tr_sm, "Random Forest (SMOTE)"
)

# â”€â”€ XGBoost with SMOTE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train_eval(
    XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="aucpr",
        random_state=42, n_jobs=-1
    ),
    X_tr_sm, y_tr_sm, "XGBoost (SMOTE)"
)

best_name = max(results, key=lambda k: results[k]["auprc"])
runner_up = sorted(results, key=lambda k: results[k]["auprc"], reverse=True)[1]
SHORT_NAMES = ["RF\nWeights", "XGB\nWeights", "RF\nSMOTE", "XGB\nSMOTE"]


# =============================================================================
# 5. FIGURE 2 â€” MODEL PERFORMANCE  (2 Ã— 2 panel grid)
# =============================================================================
print("\n" + "=" * 65)
print("5. FIGURE 2 â€” MODEL PERFORMANCE")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Credit Card Fraud Detection - Model Performance",
    fontsize=15, fontweight="bold"
)

# â”€â”€ Panel A: Precision-Recall Curves â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax = axes[0, 0]
for (name, res), color in zip(results.items(), MODEL_COLORS):
    prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
    label = f"{name}  (AUPRC={res['auprc']:.4f})"
    ax.plot(rec, prec, color=color, linewidth=2.2, label=label)
baseline = y_test.sum() / len(y_test)
ax.axhline(y=baseline, color=C_GRAY, linestyle="--", linewidth=1.2,
           label=f"Random baseline: {baseline:.4f}")
ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves\n(Gold standard for imbalanced classification)")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

# â”€â”€ Panel B: Grouped metrics comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax = axes[0, 1]
x = np.arange(len(results))
w = 0.25
auprc_v = [results[n]["auprc"] for n in results]
roc_v   = [results[n]["roc"]   for n in results]
f1_v    = [results[n]["f1"]    for n in results]

b1 = ax.bar(x - w, auprc_v, w, label="AUPRC",   color=C_FRAUD,  edgecolor="white")
b2 = ax.bar(x,     roc_v,   w, label="ROC AUC", color=C_BLUE,   edgecolor="white")
b3 = ax.bar(x + w, f1_v,    w, label="F1",      color=C_NORMAL, edgecolor="white")

for b_grp in [b1, b2, b3]:
    for b in b_grp:
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.005,
                f"{b.get_height():.3f}",
                ha="center", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(SHORT_NAMES, fontsize=9)
ax.set_ylim(0, 1.13)
ax.set_ylabel("Score")
ax.set_title("AUPRC / ROC AUC / F1 - Four-Model Comparison")
ax.legend()

# â”€â”€ Panel C: Confusion matrix â€” best model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def draw_cm(ax, name):
    cm   = results[name]["cm"]
    cm_p = cm.astype(float) / cm.sum() * 100
    annot = np.array([
        [f"TN\n{cm[0,0]:,}\n({cm_p[0,0]:.2f}%)",  f"FP\n{cm[0,1]}\n({cm_p[0,1]:.3f}%)"],
        [f"FN\n{cm[1,0]}\n({cm_p[1,0]:.3f}%)",   f"TP\n{cm[1,1]}\n({cm_p[1,1]:.3f}%)"],
    ])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"],
                cbar=False, linewidths=2, linecolor="white")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

draw_cm(axes[1, 0], best_name)
axes[1, 0].set_title(
    f"Confusion Matrix - {best_name}\n"
    f"Best AUPRC: {results[best_name]['auprc']:.4f}"
)

draw_cm(axes[1, 1], runner_up)
axes[1, 1].set_title(
    f"Confusion Matrix - {runner_up}\n"
    f"Runner-up AUPRC: {results[runner_up]['auprc']:.4f}"
)

plt.tight_layout()
plt.subplots_adjust(hspace=0.44, wspace=0.34)
save_fig(fig, "fig_02_model_performance")


# =============================================================================
# 6. FIGURE 3 â€” FEATURE IMPORTANCE  (1 Ã— 2 panel grid)
# =============================================================================
print("\n" + "=" * 65)
print("6. FIGURE 3 â€” FEATURE IMPORTANCE")
print("=" * 65)

feature_names = X_train.columns.tolist()


def feat_imp_df(model_name, top_n=12):
    imp = results[model_name]["model"].feature_importances_
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": imp})
        .sort_values("Importance", ascending=False)
        .head(top_n)
        .sort_values("Importance")
    )


fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    "Feature Importance - Which PCA Components Detect Fraud?",
    fontsize=15, fontweight="bold"
)

for idx, (name, color) in enumerate([
    ("Random Forest (Weights)", C_NORMAL),
    ("XGBoost (Weights)",       C_BLUE),
]):
    ax  = axes[idx]
    imp = feat_imp_df(name)
    ax.barh(imp["Feature"], imp["Importance"],
            color=color, edgecolor="white", alpha=0.85)
    for val, feat in zip(imp["Importance"], imp["Feature"]):
        ax.text(val + imp["Importance"].max() * 0.01,
                feat, f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Importance (Gini / Gain)")
    title = "Random Forest" if "Forest" in name else "XGBoost"
    ax.set_title(f"{title} - Top 12 Features\n"
                 "(V14 dominant in both -> strongest fraud signal)")
    sns.despine(ax=ax)

plt.tight_layout()
plt.subplots_adjust(wspace=0.38)
save_fig(fig, "fig_03_feature_importance")


# =============================================================================
# 7. FIGURE 4 â€” BUSINESS IMPACT  (1 Ã— 2 panel grid)
# =============================================================================
print("\n" + "=" * 65)
print("7. FIGURE 4 â€” BUSINESS IMPACT")
print("=" * 65)

# Reload original amounts for the test indices to compute dollar values
df_orig = pd.read_csv("creditcard.csv")
_, test_idx = train_test_split(
    df_orig.index, test_size=0.2, random_state=42,
    stratify=df_orig["Class"]
)
amount_test = df_orig.loc[test_idx, "Amount"].values

# Assumption: a false positive costs $10 (SMS alert + potential call-centre time).
# This is conservative; blocking a legitimate transaction can also cause
# customer churn, but $10 provides a defensible lower bound.
COST_FP = 10

biz_rows = []
for name, res in results.items():
    tp_mask = (y_test.values == 1) & (res["y_pred"] == 1)
    fn_mask = (y_test.values == 1) & (res["y_pred"] == 0)
    saved   = amount_test[tp_mask].sum()
    lost    = amount_test[fn_mask].sum()
    fp_cost = res["cm"][0, 1] * COST_FP
    biz_rows.append({
        "Model": name, "Saved": saved, "Lost": lost,
        "FP_cost": fp_cost, "Net": saved - fp_cost
    })
    print(f"  {name:<32s}  saved=${saved:,.0f}  lost=${lost:,.0f}  net=${saved-fp_cost:,.0f}")

bdf = pd.DataFrame(biz_rows)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "Business Impact - Financial Value of the Fraud Detection Model",
    fontsize=15, fontweight="bold"
)

# â”€â”€ Panel A: Saved vs. Lost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax = axes[0]
x  = np.arange(len(bdf))
w  = 0.35
ax.bar(x - w/2, bdf["Saved"], w,
       label="$ Saved  (TP - fraud blocked)", color=C_NORMAL, edgecolor="white")
ax.bar(x + w/2, bdf["Lost"],  w,
       label="$ Lost   (FN - fraud missed)",  color=C_FRAUD,  edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(SHORT_NAMES, fontsize=10)
ax.set_ylabel("Amount (USD)")
ax.set_title("Fraud Blocked vs. Fraud Missed per Model")
ax.legend()

# â”€â”€ Panel B: Net benefit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax = axes[1]
net_colors = [C_NORMAL if v > 0 else C_FRAUD for v in bdf["Net"]]
ax.bar(x, bdf["Net"], color=net_colors, edgecolor="white", width=0.5)
for i, val in enumerate(bdf["Net"]):
    ax.text(i, val + bdf["Net"].max() * 0.015,
            f"${val:,.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(SHORT_NAMES, fontsize=10)
ax.set_ylabel("Net Benefit (USD)")
ax.set_title(f"Net Benefit = $ Saved - FP Cost (${COST_FP} per alert)")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
save_fig(fig, "fig_04_business_impact")


# =============================================================================
# 8. RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("RESULTS SUMMARY")
print("=" * 65)

summary = pd.DataFrame({
    "Model":   list(results.keys()),
    "AUPRC":   [results[n]["auprc"] for n in results],
    "ROC AUC": [results[n]["roc"]   for n in results],
    "F1":      [results[n]["f1"]    for n in results],
    "TP":      [int(results[n]["cm"][1, 1]) for n in results],
    "FP":      [int(results[n]["cm"][0, 1]) for n in results],
    "FN":      [int(results[n]["cm"][1, 0]) for n in results],
}).sort_values("AUPRC", ascending=False)

print(f"\n{summary.to_string(index=False)}\n")
print(f"  Best model  : {best_name}")
print(f"  AUPRC       : {results[best_name]['auprc']:.4f}")
print(f"  Fraud caught: {results[best_name]['cm'][1,1]} / {int(y_test.sum())}")

joblib.dump(results[best_name]["model"], "fraud_detection_model.joblib")
print(f"\n  Model saved : fraud_detection_model.joblib")
print(f"  Figures in  : {FIG_DIR}/  (4 multi-panel PNGs)")
print("\n  PIPELINE COMPLETE")

