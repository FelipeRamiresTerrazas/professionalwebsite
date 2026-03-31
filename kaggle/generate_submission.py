"""
Generates kaggle/submission.csv for the Customer Churn competition.
Trains XGBoost on the full training set and predicts churn probabilities
on test.csv using the same preprocessing encoders fitted on train.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

KAGGLE_DIR = Path(__file__).resolve().parent

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv(KAGGLE_DIR / "train.csv")
test  = pd.read_csv(KAGGLE_DIR / "test.csv")

# ── Clean ─────────────────────────────────────────────────────────────────────
def clean(df):
    df = df.copy()
    df["TotalCharges"]  = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df

train = clean(train)
test  = clean(test)

y = (train["Churn"] == "Yes").astype(int)

# ── Encode — fit on train, transform both ─────────────────────────────────────
cat_cols = [c for c in train.select_dtypes("object").columns if c != "Churn"]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    train[c] = le.fit_transform(train[c].astype(str))
    encoders[c] = le

for c, le in encoders.items():
    if c in test.columns:
        classes_map = {v: i for i, v in enumerate(le.classes_)}
        test[c] = test[c].astype(str).map(classes_map).fillna(0).astype(int)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols].fillna(0))
test[num_cols]  = scaler.transform(test[num_cols].fillna(0))

FEATURE_COLS = [c for c in train.columns if c not in ("id", "Churn", "ChurnBin")]
X_train = train[FEATURE_COLS].fillna(0)
X_test  = test[FEATURE_COLS].fillna(0)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training XGBoost on full training set …")
model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    eval_metric="logloss",
    random_state=42,
    n_jobs=1,          # single-threaded — avoids Windows parallelism stalls
    verbosity=0,
)
model.fit(X_train, y)
print("Training complete.")

# ── Predict & write ───────────────────────────────────────────────────────────
churn_proba = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"id": test["id"], "Churn": churn_proba})

out_path = KAGGLE_DIR / "submission.csv"
submission.to_csv(out_path, index=False)

print(f"\n✓ submission.csv written ({len(submission):,} rows)")
print(f"  Churn probability range: [{churn_proba.min():.4f}, {churn_proba.max():.4f}]")
print(f"  Mean predicted churn probability: {churn_proba.mean():.4f}")
print(f"  Saved to: {out_path}")
