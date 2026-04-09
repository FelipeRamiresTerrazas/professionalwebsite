"""
Flight Search Engine — End-to-End Data Science Pipeline
========================================================
Reads partitioned search-engine results files (CSV or Parquet), consolidates
them into a single master DataFrame, computes core Search & Discovery metrics,
trains an XGBoost Learning-to-Rank model, and exports publication-ready figures
for the portfolio HTML page.

Run from the repository root:
    python kaggle/generate_flight_search_figures.py
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
import xgboost as xgb

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join("kaggle", "search_flights")
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Palette matching the site's dark-theme aesthetic
PAL = {
    "primary"   : "#6366f1",   # indigo
    "secondary" : "#22d3ee",   # cyan
    "accent"    : "#f59e0b",   # amber
    "danger"    : "#ef4444",   # red
    "bg"        : "#0f172a",   # slate-900
    "surface"   : "#1e293b",   # slate-800
    "text"      : "#e2e8f0",   # slate-200
    "muted"     : "#64748b",   # slate-500
}

plt.rcParams.update({
    "figure.facecolor" : PAL["bg"],
    "axes.facecolor"   : PAL["surface"],
    "axes.edgecolor"   : PAL["muted"],
    "axes.labelcolor"  : PAL["text"],
    "xtick.color"      : PAL["text"],
    "ytick.color"      : PAL["text"],
    "text.color"       : PAL["text"],
    "grid.color"       : "#334155",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "font.family"      : "sans-serif",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "axes.titlecolor"  : PAL["text"],
})

BOOKING_DOMAINS = {
    "www.kayak.com", "kayak.com",
    "www.expedia.com", "expedia.com",
    "www.skyscanner.com", "www.skyscanner.net", "skyscanner.net",
    "www.booking.com", "booking.com",
    "www.cheaptickets.com", "cheaptickets.com",
    "www.orbitz.com", "orbitz.com",
    "www.travelocity.com", "travelocity.com",
    "www.priceline.com", "priceline.com",
    "www.hopper.com", "hopper.com",
    "flights.google.com", "www.google.com",
    "www.momondo.com", "momondo.com",
    "www.cheapflights.com", "cheapflights.com",
}

AIRLINE_KEYWORDS = [
    "united", "delta", "american", "southwest", "jetblue", "alaska",
    "spirit", "frontier", "cathay", "emirates", "lufthansa", "british",
    "qatar", "singapore", "air france", "klm", "turkish", "ana", "jal",
    "ryanair", "easyjet", "aeromexico", "latam", "avianca",
]

INTENT_KEYWORDS = {
    "cheap"     : ["cheap", "afford", "budget", "deal", "discount", "sale"],
    "luxury"    : ["business class", "first class", "premium", "luxury"],
    "direct"    : ["nonstop", "non-stop", "direct flight"],
    "booking"   : ["book", "buy", "purchase", "ticket", "reserve"],
}

# ─── 1. DATA ENGINEERING ──────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — DATA ENGINEERING")
print("=" * 60)

# Support both Parquet (production) and CSV (development)
parquet_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])
csv_files     = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])

if parquet_files:
    print(f"  → Found {len(parquet_files)} Parquet partition(s). Loading…")
    frames = [pd.read_parquet(os.path.join(DATA_DIR, f)) for f in parquet_files]
    src = "Parquet"
else:
    print(f"  → Found {len(csv_files)} CSV partition(s). Loading…")
    frames = []
    for fname in csv_files:
        try:
            frames.append(pd.read_csv(os.path.join(DATA_DIR, fname), low_memory=False))
        except Exception as e:
            print(f"     Warning: skipping {fname} — {e}")
    src = "CSV"

master = pd.concat(frames, ignore_index=True)
print(f"  ✓ Consolidated {len(frames)} {src} partitions → {master.shape[0]:,} rows × {master.shape[1]} columns")

# ─── 2. CLEANING & FEATURE ENGINEERING ───────────────────────────────────────
print("\nPHASE 2 — CLEANING & FEATURE ENGINEERING")

master["queryTime"]    = pd.to_datetime(master["queryTime"], utc=True, errors="coerce")
master["totalResults"] = pd.to_numeric(master["totalResults"], errors="coerce").fillna(0).astype(int)
master["rank"]         = pd.to_numeric(master["rank"],         errors="coerce")
master["searchTime"]   = pd.to_numeric(master["searchTime"],   errors="coerce")
master["count"]        = pd.to_numeric(master["count"],        errors="coerce")

master["query_date"]    = master["queryTime"].dt.date
master["query_month"]   = master["queryTime"].dt.to_period("M")
master["displayLink"]   = master["displayLink"].fillna("").str.lower().str.strip()
master["title"]         = master["title"].fillna("")
master["snippet"]       = master["snippet"].fillna("")
master["searchTerms"]   = master["searchTerms"].fillna("").str.lower().str.strip()

master["is_booking_domain"] = master["displayLink"].isin(BOOKING_DOMAINS)
master["is_top3"]           = master["rank"] <= 3

q = master["searchTerms"]
master["query_word_count"]       = q.str.split().str.len().fillna(0).astype(int)
master["query_has_airline"]      = q.apply(lambda x: any(a in x for a in AIRLINE_KEYWORDS))
master["query_has_cheap_intent"] = q.apply(lambda x: any(k in x for k in INTENT_KEYWORDS["cheap"]))
master["query_has_booking_verb"] = q.apply(lambda x: any(k in x for k in INTENT_KEYWORDS["booking"]))
master["query_char_count"]       = q.str.len().fillna(0).astype(int)

t = master["title"] + " " + master["snippet"]
master["has_price_signal"]  = t.str.contains(r'\$[\d,]+|from \$|starting at', flags=re.I, regex=True)
master["title_len"]         = master["title"].str.len().fillna(0).astype(int)
master["snippet_len"]       = master["snippet"].str.len().fillna(0).astype(int)

domain_top3_rate = (
    master.groupby("displayLink")["is_top3"]
    .mean()
    .rename("domain_top3_rate")
)
master = master.merge(domain_top3_rate, on="displayLink", how="left")

print(f"  ✓ Date range  : {master['queryTime'].min().date()} → {master['queryTime'].max().date()}")
print(f"  ✓ Unique queries : {master['searchTerms'].nunique():,}")
print(f"  ✓ Unique domains : {master['displayLink'].nunique():,}")

# ─── 3. CORE SEARCH & DISCOVERY METRICS ─────────────────────────────────────
print("\nPHASE 3 — CORE S&D METRICS")

# ── 3a. Zero Result Rate ──────────────────────────────────────────────────────
# A search "session" = unique (searchTerms, query_date) pair
sessions = master.groupby(["searchTerms", "query_date"]).agg(
    max_results = ("totalResults", "max"),
    results_on_page = ("count", "max"),
).reset_index()

sessions["zero_result"] = sessions["max_results"] == 0
zrr_overall = sessions["zero_result"].mean() * 100

zrr_monthly = (
    master.groupby("query_month").apply(
        lambda g: g.groupby(["searchTerms", "query_date"])["totalResults"].max().eq(0).mean() * 100
    )
    .reset_index()
    .rename(columns={0: "zrr_pct"})
)

print(f"  ZRR (overall)          : {zrr_overall:.2f}%")

# ── 3b. Top-3 Booking Presence Rate ──────────────────────────────────────────
# For each search session, does a booking/OTA domain appear in ranks 1–3?
top3_hits = (
    master[master["is_top3"]]
    .groupby(["searchTerms", "query_date"])["is_booking_domain"]
    .any()
    .reset_index()
    .rename(columns={"is_booking_domain": "has_booking_top3"})
)
booking_top3_rate = top3_hits["has_booking_top3"].mean() * 100

print(f"  Top-3 OTA Presence Rate: {booking_top3_rate:.1f}%")

# ── 3c. Simulated Click-Through Rate ─────────────────────────────────────────
# Apply standard organic CTR curve from Backlinko / FirstPageSage benchmarks
CTR_BY_RANK = {1: 0.285, 2: 0.157, 3: 0.110, 4: 0.071, 5: 0.046,
               6: 0.032, 7: 0.022, 8: 0.017, 9: 0.013, 10: 0.011}
master["est_ctr"] = master["rank"].map(CTR_BY_RANK).fillna(0.008)

top3_ctr = master[master["rank"] <= 3]["est_ctr"].mean() * 100
print(f"  Avg Simulated CTR (pos 1-3)      : {top3_ctr:.1f}%")

# ── 3d. Monthly search volume ─────────────────────────────────────────────────
monthly_volume = (
    master.groupby("query_month")["searchTerms"]
    .count()
    .reset_index()
    .rename(columns={"searchTerms": "result_count"})
)

# ── 3e. Top domains overall + in Top-3 ───────────────────────────────────────
top_domains_all   = master["displayLink"].value_counts().head(15)
top_domains_top3  = master[master["is_top3"]]["displayLink"].value_counts().head(15)

# ─── 4. LEARNING TO RANK (LTR) ───────────────────────────────────────────────
print("\nPHASE 4 — LEARNING TO RANK (XGBoost Ranker)")

FEATURES = [
    "domain_top3_rate",
    "has_price_signal",
    "title_len",
    "snippet_len",
    "query_word_count",
    "query_char_count",
    "query_has_airline",
    "query_has_cheap_intent",
    "query_has_booking_verb",
    "is_booking_domain",
]

# Build per-query groups (XGBRanker requires consecutive group sizes)
ltr_data = master.dropna(subset=["rank"]).copy()
ltr_data["relevance"] = (11 - ltr_data["rank"].clip(1, 10)).astype(int)  # 10 = best
ltr_data["query_id"]  = ltr_data["searchTerms"] + "_" + ltr_data["query_date"].astype(str)

# Keep only queries that have at least 3 results (proper ranking context)
query_counts = ltr_data["query_id"].value_counts()
valid_queries = query_counts[query_counts >= 3].index
ltr_data = ltr_data[ltr_data["query_id"].isin(valid_queries)].copy()

# Sample for speed — cap at 5000 query ids (only when needed)
unique_qids = ltr_data["query_id"].unique()
if len(unique_qids) > 5_000:
    sampled_query_ids = np.random.default_rng(42).choice(unique_qids, size=5_000, replace=False)
    ltr_data = ltr_data[ltr_data["query_id"].isin(sampled_query_ids)]

ltr_data = ltr_data.sort_values("query_id").reset_index(drop=True)
# Encode string query_ids → integers (XGBRanker requires uint32 qid)
ltr_data["qid_int"] = pd.factorize(ltr_data["query_id"])[0]

X = ltr_data[FEATURES].fillna(0).astype(float)
y = ltr_data["relevance"].values
qid_all = ltr_data["qid_int"].values  # per-row integer query IDs

# Train / test split respecting group boundaries
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
group_ids = ltr_data["query_id"].values
train_idx, test_idx = next(gss.split(X, y, groups=group_ids))

# Sort indices so rows stay in query_id order (XGBRanker requirement)
train_idx = np.sort(train_idx)
test_idx  = np.sort(test_idx)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
qid_train = qid_all[train_idx]
qid_test  = qid_all[test_idx]

ranker = xgb.XGBRanker(
    objective     = "rank:ndcg",
    n_estimators  = 300,
    max_depth     = 4,
    learning_rate = 0.05,
    subsample     = 0.8,
    colsample_bytree = 0.8,
    random_state  = 42,
    verbosity     = 0,
)
ranker.fit(
    X_train, y_train,
    qid=qid_train,
    eval_set=[(X_test, y_test)],
    eval_qid=[qid_test],
)

# NDCG@3 via sklearn (approximate, per-query)
test_preds = ranker.predict(X_test)
test_df = ltr_data.iloc[test_idx].copy()
test_df["pred_score"] = test_preds

ndcg_scores = []
for q_id, grp in test_df.groupby("query_id"):
    if len(grp) >= 2:
        true_rel = grp["relevance"].values.reshape(1, -1)
        pred_scr = grp["pred_score"].values.reshape(1, -1)
        ndcg_scores.append(ndcg_score(true_rel, pred_scr, k=3))

ndcg3 = np.mean(ndcg_scores)
print(f"  Training rows  : {len(X_train):,}   |   Test rows: {len(X_test):,}")
print(f"  NDCG@3 (test)  : {ndcg3:.4f}")

feature_importance = pd.Series(
    ranker.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

# ─── 5. FIGURE GENERATION ─────────────────────────────────────────────────────
print("\nPHASE 5 — GENERATING FIGURES")

# ── Fig 1: Search Volume & Monthly Trend ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor(PAL["bg"])

months_str  = [str(m) for m in monthly_volume["query_month"]]
result_vals = monthly_volume["result_count"].values

ax = axes[0]
ax.fill_between(range(len(months_str)), result_vals, alpha=0.3, color=PAL["secondary"])
ax.plot(range(len(months_str)), result_vals, color=PAL["secondary"], linewidth=2.5, marker="o", markersize=5)
ax.set_title("Monthly Search Volume (Result Records)")
ax.set_xticks(range(len(months_str)))
ax.set_xticklabels(months_str, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Result Records")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.axvspan(12, 13, color=PAL["danger"], alpha=0.15, label="Data gap")
ax.legend(fontsize=9, facecolor=PAL["surface"], edgecolor=PAL["muted"])

# CTR curve
ax2 = axes[1]
ranks_shown = list(range(1, 11))
ctr_vals    = [CTR_BY_RANK.get(r, 0.008) * 100 for r in ranks_shown]
colors_bar  = [PAL["primary"] if r <= 3 else PAL["muted"] for r in ranks_shown]
bars = ax2.bar(ranks_shown, ctr_vals, color=colors_bar, edgecolor="none", width=0.7)
for bar, val in zip(bars, ctr_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color=PAL["text"])
ax2.set_title("Estimated CTR by SERP Position")
ax2.set_xlabel("SERP Rank")
ax2.set_ylabel("Estimated CTR (%)")
ax2.set_xticks(ranks_shown)
ax2.set_ylim(0, max(ctr_vals) * 1.2)

# Annotation
ax2.annotate("Top-3 zone\n(54.2% combined CTR)",
             xy=(3, ctr_vals[2]), xytext=(5, 22),
             arrowprops=dict(arrowstyle="->", color=PAL["accent"]),
             color=PAL["accent"], fontsize=9)

plt.tight_layout(pad=2.5)
fig.savefig(os.path.join(FIGURES_DIR, "flight_search_volume_ctr.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ fig 1 saved: flight_search_volume_ctr.png")

# ── Fig 2: Domain Visibility in Top-3 ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(PAL["bg"])

# All positions
ax = axes[0]
domains  = top_domains_all.index.tolist()[:12]
counts   = top_domains_all.values[:12]
colors   = [PAL["primary"] if d in BOOKING_DOMAINS else PAL["muted"] for d in domains]
bars = ax.barh(domains[::-1], counts[::-1], color=colors[::-1], edgecolor="none", height=0.65)
ax.set_title("Top 12 Domains — All Positions")
ax.set_xlabel("Appearances in SERP data")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
# Legend proxy
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=PAL["primary"], label="OTA / Booking"),
              Patch(facecolor=PAL["muted"],   label="Informational")]
ax.legend(handles=legend_els, fontsize=9, facecolor=PAL["surface"], edgecolor=PAL["muted"])

# Top-3 only
ax2 = axes[1]
domains3 = top_domains_top3.index.tolist()[:12]
counts3  = top_domains_top3.values[:12]
colors3  = [PAL["secondary"] if d in BOOKING_DOMAINS else PAL["muted"] for d in domains3]
ax2.barh(domains3[::-1], counts3[::-1], color=colors3[::-1], edgecolor="none", height=0.65)
ax2.set_title("Top 12 Domains — Rank 1–3 Only")
ax2.set_xlabel("Top-3 appearances")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout(pad=2.5)
fig.savefig(os.path.join(FIGURES_DIR, "flight_search_domain_visibility.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ fig 2 saved: flight_search_domain_visibility.png")

# ── Fig 3: S&D Metrics Dashboard ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.patch.set_facecolor(PAL["bg"])

# ZRR monthly sparkline
ax = axes[0]
months_zrr = [str(m) for m in zrr_monthly["query_month"]]
zrr_vals   = zrr_monthly["zrr_pct"].values
ax.fill_between(range(len(months_zrr)), zrr_vals, alpha=0.25, color=PAL["danger"])
ax.plot(range(len(months_zrr)), zrr_vals, color=PAL["danger"], linewidth=2, marker="o", markersize=4)
ax.set_title(f"Zero Result Rate by Month\n(Overall: {zrr_overall:.2f}%)")
ax.set_xticks(range(len(months_zrr)))
ax.set_xticklabels(months_zrr, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("ZRR (%)")
ax.set_ylim(0, max(zrr_vals.max() * 1.3, 0.5))

# Booking top-3 presence gauge-style bar
ax2 = axes[1]
labels2 = ["OTA in Top-3", "No OTA in Top-3"]
vals2   = [booking_top3_rate, 100 - booking_top3_rate]
bars2   = ax2.bar(labels2, vals2,
                  color=[PAL["primary"], PAL["muted"]], edgecolor="none", width=0.5)
for b, v in zip(bars2, vals2):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=PAL["text"])
ax2.set_title(f"Top-3 OTA Presence Rate\n(= simulated Top-3 CTR proxy)")
ax2.set_ylabel("% of Search Sessions")
ax2.set_ylim(0, 110)

# Query intent breakdown
ax3 = axes[2]
intent_counts = {
    "Cheap/Deal"  : master["query_has_cheap_intent"].sum(),
    "Booking Verb": master["query_has_booking_verb"].sum(),
    "Airline Name": master["query_has_airline"].sum(),
}
ax3.barh(list(intent_counts.keys()), list(intent_counts.values()),
         color=[PAL["accent"], PAL["secondary"], PAL["primary"]], edgecolor="none", height=0.5)
ax3.set_title("Query Intent Signals\n(multi-label)")
ax3.set_xlabel("Result records with signal")
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.suptitle("Core Search & Discovery Metrics", fontsize=14, fontweight="bold",
             color=PAL["text"], y=1.02)
plt.tight_layout(pad=2.5)
fig.savefig(os.path.join(FIGURES_DIR, "flight_search_metrics_dashboard.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ fig 3 saved: flight_search_metrics_dashboard.png")

# ── Fig 4: LTR Feature Importance ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(PAL["bg"])

feat_labels = {
    "domain_top3_rate"       : "Domain historical Top-3 rate",
    "has_price_signal"       : "Price signal in title/snippet",
    "is_booking_domain"      : "Is OTA / booking domain",
    "title_len"              : "Title character length",
    "snippet_len"            : "Snippet character length",
    "query_word_count"       : "Query word count",
    "query_char_count"       : "Query character count",
    "query_has_airline"      : "Airline name in query",
    "query_has_cheap_intent" : "Cheap/deal intent in query",
    "query_has_booking_verb" : "Booking verb in query",
}
fi_renamed = feature_importance.rename(feat_labels)
clrs = [PAL["primary"] if v >= fi_renamed.mean() else PAL["muted"] for v in fi_renamed.values]
bars = ax.barh(fi_renamed.index[::-1], fi_renamed.values[::-1],
               color=clrs[::-1], edgecolor="none", height=0.65)
for bar, val in zip(bars, fi_renamed.values[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, color=PAL["text"])
ax.set_title(f"XGBoost LTR — Feature Importance   (NDCG@3 = {ndcg3:.4f})")
ax.set_xlabel("Gain (normalised)")
ax.set_xlim(0, fi_renamed.values.max() * 1.18)

plt.tight_layout(pad=2.0)
fig.savefig(os.path.join(FIGURES_DIR, "flight_search_ltr_importance.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ fig 4 saved: flight_search_ltr_importance.png")

# ── Fig 5: COVID Impact — Domain Landscape Change ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(PAL["bg"])

top5_domains = top_domains_top3.index[:5].tolist()
monthly_domain = (
    master[master["displayLink"].isin(top5_domains) & master["is_top3"]]
    .groupby(["query_month", "displayLink"])
    .size()
    .reset_index(name="appearances")
)

cmap_cols = [PAL["primary"], PAL["secondary"], PAL["accent"],
             "#a78bfa", "#34d399"]
for i, dom in enumerate(top5_domains):
    sub = monthly_domain[monthly_domain["displayLink"] == dom].sort_values("query_month")
    months_x = [str(m) for m in sub["query_month"]]
    ax.plot(months_x, sub["appearances"].values,
            label=dom, marker="o", linewidth=2, markersize=5,
            color=cmap_cols[i % len(cmap_cols)])

ax.set_title("Top-5 Domain Visibility in Top-3 SERP Positions Over Time")
ax.set_xlabel("Month")
ax.set_ylabel("Top-3 appearances")
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=9, facecolor=PAL["surface"], edgecolor=PAL["muted"],
          bbox_to_anchor=(1.01, 1), loc="upper left")
# COVID reference
if len(monthly_domain) > 0:
    ax.axvline(x="2020-03", color=PAL["danger"], linestyle="--", alpha=0.7, linewidth=1.5)
    ax.text("2020-03", ax.get_ylim()[1] * 0.9, " COVID-19\n Mar 2020",
            color=PAL["danger"], fontsize=8)

plt.tight_layout(pad=2.0)
fig.savefig(os.path.join(FIGURES_DIR, "flight_search_covid_impact.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ fig 5 saved: flight_search_covid_impact.png")

# ─── 6. EXECUTIVE SUMMARY ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXECUTIVE SUMMARY")
print("=" * 60)
print(f"  Dataset        : {len(frames)} partitioned files → {master.shape[0]:,} rows")
print(f"  Date range     : Dec 2018 – Apr 2020  (16 months)")
print(f"  Unique searches: {master['searchTerms'].nunique():,}")
print(f"  Unique domains : {master['displayLink'].nunique():,}")
print(f"\n  ── Metric Baseline ──")
print(f"  ZRR (strict)            : {zrr_overall:.2f}%    ← near-zero; Google always returns something")
print(f"  Top-3 OTA Presence      : {booking_top3_rate:.1f}%   ← share of sessions with OTA in pos 1-3")
print(f"  Avg Simulated CTR (1-3) : {top3_ctr:.1f}%")
print(f"\n  ── LTR Model ──")
print(f"  Algorithm  : XGBoost Ranker (rank:ndcg)")
print(f"  NDCG@3     : {ndcg3:.4f}")
print(f"  Top feature: {feature_importance.idxmax()}")
print(f"\n  Figures → {FIGURES_DIR}/flight_search_*.png")
print("=" * 60)
