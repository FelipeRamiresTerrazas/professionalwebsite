import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(__file__).resolve().parent / "delivery_hero_ab_test_large.csv"
FIGURES_DIR = ROOT / "figures"
METRICS_PATH = Path(__file__).resolve().parent / "delivery_hero_ab_test_metrics.json"
RANDOM_STATE = 42


COLORS = {
    "cyan": "#00E5FF",
    "magenta": "#FF2D95",
    "blue": "#3A86FF",
    "green": "#39FF88",
    "amber": "#FFC857",
    "text": "#E6E6E6",
    "muted": "#9AA4B2",
    "panel": "#121212",
}


def setup_dark_theme() -> None:
    plt.style.use("dark_background")
    sns.set_theme(style="dark", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["panel"],
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["panel"],
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["muted"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": "#2A2A2A",
            "grid.alpha": 0.55,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)


def save_dark(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=240, bbox_inches="tight", transparent=True)
    plt.close(fig)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["variant"] = df["variant"].astype(str).str.upper().str.strip()
    df = df[df["variant"].isin(["A", "B"])].copy()
    df = df.dropna(subset=["eta_shown_min", "actual_delivery_min", "active_orders_in_kitchen"])
    df["delay_min"] = df["actual_delivery_min"] - df["eta_shown_min"]
    return df


def train_eta_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, dict]:
    work = df[df["actual_delivery_min"] > 0].copy()
    features = [
        "variant",
        "restaurant_rating",
        "driver_distance_km",
        "active_orders_in_kitchen",
        "eta_shown_min",
    ]
    target = "actual_delivery_min"

    X_train, X_test, y_train, y_test = train_test_split(
        work[features],
        work[target],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=work["variant"],
    )

    prep = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["variant"],
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                ["restaurant_rating", "driver_distance_km", "active_orders_in_kitchen", "eta_shown_min"],
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=450,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", prep), ("model", model)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    eval_frame = X_test.copy()
    eval_frame["actual_delivery_min"] = y_test.values
    eval_frame["rf_pred_delivery_min"] = pred
    eval_frame["rf_error"] = eval_frame["rf_pred_delivery_min"] - eval_frame["actual_delivery_min"]
    eval_frame["naive_error"] = eval_frame["eta_shown_min"] - eval_frame["actual_delivery_min"]

    metrics = {
        "rf_mae": float(mean_absolute_error(eval_frame["actual_delivery_min"], eval_frame["rf_pred_delivery_min"])),
        "naive_mae": float(mean_absolute_error(eval_frame["actual_delivery_min"], eval_frame["eta_shown_min"])),
        "rf_bias": float(eval_frame["rf_error"].mean()),
        "naive_bias": float(eval_frame["naive_error"].mean()),
    }
    metrics["mae_improvement_pct"] = float(
        (metrics["naive_mae"] - metrics["rf_mae"]) / max(metrics["naive_mae"], 1e-8) * 100
    )
    return pipe, eval_frame, metrics


def train_cancellation_models(df: pd.DataFrame) -> tuple[Pipeline, Pipeline, pd.DataFrame, dict, dict]:
    work = df[df["actual_delivery_min"] > 0].copy()
    features = [
        "variant",
        "restaurant_rating",
        "driver_distance_km",
        "active_orders_in_kitchen",
        "eta_shown_min",
        "delay_min",
        "order_value_eur",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        work[features],
        work["is_canceled"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=work["is_canceled"],
    )

    prep_scaled = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["variant"],
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                [
                    "restaurant_rating",
                    "driver_distance_km",
                    "active_orders_in_kitchen",
                    "eta_shown_min",
                    "delay_min",
                    "order_value_eur",
                ],
            ),
        ]
    )

    prep_tree = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["variant"],
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                [
                    "restaurant_rating",
                    "driver_distance_km",
                    "active_orders_in_kitchen",
                    "eta_shown_min",
                    "delay_min",
                    "order_value_eur",
                ],
            ),
        ]
    )

    logistic = Pipeline(
        steps=[("prep", prep_scaled), ("model", LogisticRegression(max_iter=1200, random_state=RANDOM_STATE))]
    )
    rf = Pipeline(
        steps=[
            ("prep", prep_tree),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    logistic.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    test_frame = X_test.copy()
    test_frame["is_canceled"] = y_test.values
    test_frame["proba_logistic"] = logistic.predict_proba(X_test)[:, 1]
    test_frame["proba_rf"] = rf.predict_proba(X_test)[:, 1]

    model_metrics = {
        "logistic_auc": float(roc_auc_score(y_test, test_frame["proba_logistic"])),
        "rf_auc": float(roc_auc_score(y_test, test_frame["proba_rf"])),
    }

    feature_names = rf.named_steps["prep"].get_feature_names_out()
    raw_importances = rf.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": raw_importances})
    importance_df["feature"] = importance_df["feature"].str.replace("cat__", "", regex=False).str.replace("num__", "", regex=False)
    consolidated = (
        importance_df.groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    )
    return logistic, rf, test_frame, model_metrics, {
        row.feature: float(row.importance)
        for row in consolidated.itertuples(index=False)
    }


def make_business_impact_plot(df: pd.DataFrame) -> tuple[Path, dict]:
    summary = (
        df.groupby("variant", observed=False)
        .agg(
            cancellation_rate=("is_canceled", "mean"),
            support_ticket_rate=("support_ticket_opened", "mean"),
        )
        .reset_index()
    )
    melted = summary.melt(id_vars="variant", var_name="metric", value_name="rate")
    melted["metric"] = melted["metric"].map(
        {
            "cancellation_rate": "Cancellation Rate",
            "support_ticket_rate": "Support Ticket Rate",
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 6))
    sns.barplot(
        data=melted,
        x="metric",
        y="rate",
        hue="variant",
        palette={"A": COLORS["magenta"], "B": COLORS["cyan"]},
        ax=ax,
    )
    style_axis(ax)
    ax.set_title("Business Impact: Variant B Slashes Operational Friction", color=COLORS["text"], pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(title="Variant", frameon=False)

    for container in ax.containers:
        ax.bar_label(container, labels=[f"{bar.get_height():.1%}" for bar in container], padding=4, color=COLORS["text"], fontsize=10)

    out_path = FIGURES_DIR / "delivery_hero_business_impact.png"
    save_dark(fig, out_path)

    a = summary.loc[summary["variant"] == "A"].iloc[0]
    b = summary.loc[summary["variant"] == "B"].iloc[0]
    payload = {
        "A_cancellation_rate": float(a["cancellation_rate"]),
        "B_cancellation_rate": float(b["cancellation_rate"]),
        "A_support_rate": float(a["support_ticket_rate"]),
        "B_support_rate": float(b["support_ticket_rate"]),
        "cancellation_drop_pct": float((a["cancellation_rate"] - b["cancellation_rate"]) * 100),
        "support_drop_pct": float((a["support_ticket_rate"] - b["support_ticket_rate"]) * 100),
    }
    return out_path, payload


def make_patience_threshold_plot(df: pd.DataFrame) -> tuple[Path, dict]:
    work = df[df["actual_delivery_min"] > 0].copy()
    work["delay_round"] = work["delay_min"].round().astype(int)
    work = work[(work["delay_round"] >= -8) & (work["delay_round"] <= 45)].copy()

    curve = (
        work.groupby("delay_round", observed=False)
        .agg(cancellation_probability=("is_canceled", "mean"), sessions=("session_id", "count"))
        .reset_index()
        .sort_values("delay_round")
    )
    curve["cancel_smooth"] = curve["cancellation_probability"].rolling(3, center=True, min_periods=1).mean()

    post_zero = curve[curve["delay_round"] >= 0].copy()
    if (post_zero["cancel_smooth"] >= 0.4).any():
        threshold_row = post_zero[post_zero["cancel_smooth"] >= 0.4].iloc[0]
    else:
        jump_idx = post_zero["cancel_smooth"].diff().fillna(0).idxmax()
        threshold_row = post_zero.loc[jump_idx]

    threshold_min = int(threshold_row["delay_round"])
    threshold_prob = float(threshold_row["cancel_smooth"])

    before = work[work["delay_min"] < threshold_min]["is_canceled"].mean()
    after = work[work["delay_min"] >= threshold_min]["is_canceled"].mean()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(
        curve["delay_round"],
        curve["cancellation_probability"],
        s=np.clip(curve["sessions"] * 0.2, 12, 200),
        color=COLORS["blue"],
        alpha=0.45,
        edgecolors="none",
        label="Observed buckets",
    )
    ax.plot(curve["delay_round"], curve["cancel_smooth"], color=COLORS["green"], linewidth=2.7, label="Smoothed cancellation curve")
    ax.axvline(threshold_min, color=COLORS["magenta"], linestyle="--", linewidth=2)
    ax.text(
        threshold_min + 0.7,
        min(0.95, threshold_prob + 0.08),
        f"Patience threshold: {threshold_min} min",
        color=COLORS["magenta"],
        fontsize=11,
        weight="bold",
    )

    style_axis(ax)
    ax.set_title("Patience Threshold: Delay vs Cancellation Probability", pad=14)
    ax.set_xlabel("Minutes Delayed (Actual - ETA)")
    ax.set_ylabel("Cancellation Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="upper left")

    out_path = FIGURES_DIR / "delivery_hero_patience_threshold.png"
    save_dark(fig, out_path)

    return out_path, {
        "patience_threshold_min": threshold_min,
        "cancel_probability_at_threshold": threshold_prob,
        "cancel_before_threshold": float(before),
        "cancel_after_threshold": float(after),
        "relative_jump_x": float(after / max(before, 1e-8)),
    }


def make_kitchen_bottleneck_plot(df: pd.DataFrame) -> tuple[Path, dict]:
    work = df[df["actual_delivery_min"] > 0].copy()

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    hb = ax.hexbin(
        work["active_orders_in_kitchen"],
        work["actual_delivery_min"],
        gridsize=35,
        cmap="turbo",
        mincnt=1,
        bins="log",
    )
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Log(Session Density)", color=COLORS["text"])
    cbar.ax.yaxis.set_tick_params(color=COLORS["muted"])
    plt.setp(cbar.ax.get_yticklabels(), color=COLORS["muted"])

    kitchen_bins = pd.cut(work["active_orders_in_kitchen"], bins=[-1, 5, 10, 15, 20, 25, 35])
    trend = (
        work.assign(kitchen_bin=kitchen_bins)
        .groupby("kitchen_bin", observed=False)
        .agg(
            kitchen_mid=("active_orders_in_kitchen", "mean"),
            median_actual=("actual_delivery_min", "median"),
        )
        .dropna()
        .reset_index(drop=True)
    )
    ax.plot(trend["kitchen_mid"], trend["median_actual"], color=COLORS["cyan"], linewidth=2.4, marker="o", markersize=5)

    style_axis(ax)
    ax.set_title("Kitchen Bottleneck: Load Saturation vs Delivery Time", pad=14)
    ax.set_xlabel("Active Orders in Kitchen")
    ax.set_ylabel("Actual Delivery Time (min)")

    out_path = FIGURES_DIR / "delivery_hero_kitchen_bottleneck.png"
    save_dark(fig, out_path)

    low = work[work["active_orders_in_kitchen"] <= 8]["actual_delivery_min"].median()
    high = work[work["active_orders_in_kitchen"] >= 20]["actual_delivery_min"].median()
    return out_path, {
        "median_actual_low_load": float(low),
        "median_actual_high_load": float(high),
        "high_vs_low_multiplier": float(high / max(low, 1e-8)),
    }


def make_model_vs_naive_error_plot(eval_frame: pd.DataFrame) -> tuple[Path, dict]:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    sns.kdeplot(eval_frame["naive_error"], color=COLORS["magenta"], fill=True, alpha=0.35, linewidth=2.2, label="Naive ETA error")
    sns.kdeplot(eval_frame["rf_error"], color=COLORS["cyan"], fill=True, alpha=0.35, linewidth=2.2, label="Random Forest ETA error")

    ax.axvline(0, color=COLORS["green"], linestyle="--", linewidth=1.8)
    style_axis(ax)
    ax.set_title("Model Accuracy vs Naive ETA: Error Distribution Shift", pad=14)
    ax.set_xlabel("Prediction Error (Predicted ETA - Actual Delivery)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    out_path = FIGURES_DIR / "delivery_hero_eta_accuracy.png"
    save_dark(fig, out_path)

    rf_mae = float(mean_absolute_error(eval_frame["actual_delivery_min"], eval_frame["rf_pred_delivery_min"]))
    naive_mae = float(mean_absolute_error(eval_frame["actual_delivery_min"], eval_frame["eta_shown_min"]))
    return out_path, {
        "rf_mae": rf_mae,
        "naive_mae": naive_mae,
        "mae_improvement_pct": float((naive_mae - rf_mae) / max(naive_mae, 1e-8) * 100),
        "rf_bias": float(eval_frame["rf_error"].mean()),
        "naive_bias": float(eval_frame["naive_error"].mean()),
    }


def make_cancel_feature_importance_plot(importance: dict) -> tuple[Path, dict]:
    imp_df = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())})
    imp_df = imp_df.sort_values("importance", ascending=False).head(8)

    neon_palette = [
        COLORS["cyan"],
        COLORS["magenta"],
        COLORS["green"],
        COLORS["blue"],
        COLORS["amber"],
        "#62F5C5",
        "#80D8FF",
        "#FF6EC7",
    ]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    bars = ax.barh(imp_df["feature"], imp_df["importance"], color=neon_palette[: len(imp_df)], alpha=0.9)
    ax.invert_yaxis()
    style_axis(ax)
    ax.set_title("Feature Importance for Cancellations", pad=14)
    ax.set_xlabel("Importance (Random Forest Classifier)")
    ax.set_ylabel("")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", color=COLORS["text"], fontsize=10)

    out_path = FIGURES_DIR / "delivery_hero_feature_importance.png"
    save_dark(fig, out_path)

    return out_path, {
        row.feature: float(row.importance)
        for row in imp_df.itertuples(index=False)
    }


def make_guardrail_order_value_plot(df: pd.DataFrame) -> tuple[Path, dict]:
    summary = (
        df.groupby("variant", observed=False)
        .agg(avg_order_value=("order_value_eur", "mean"))
        .reset_index()
    )
    a_val = float(summary.loc[summary["variant"] == "A", "avg_order_value"].iloc[0])
    b_val = float(summary.loc[summary["variant"] == "B", "avg_order_value"].iloc[0])
    delta_pct = (b_val / max(a_val, 1e-8) - 1) * 100

    fig, ax = plt.subplots(figsize=(10.5, 6))
    melted = summary.copy()
    sns.barplot(
        data=melted,
        x="variant",
        y="avg_order_value",
        hue="variant",
        palette={"A": COLORS["magenta"], "B": COLORS["cyan"]},
        legend=False,
        ax=ax,
    )
    style_axis(ax)
    ax.set_title("Guardrail KPI: Average Order Value Remains Stable", color=COLORS["text"], pad=14)
    ax.set_xlabel("Variant")
    ax.set_ylabel("Average Order Value (EUR)")
    ax.set_ylim(0, max(a_val, b_val) * 1.22)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"EUR {y:.0f}"))

    for patch, val in zip(ax.patches, [a_val, b_val]):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + max(a_val, b_val) * 0.015,
            f"EUR {val:.2f}",
            ha="center", va="bottom",
            color=COLORS["text"], fontsize=10,
        )

    ax.text(
        0.5, 0.96,
        f"Delta B vs A: {delta_pct:+.2f}%",
        transform=ax.transAxes, ha="center", va="top",
        color=COLORS["amber"], fontsize=11, weight="bold",
    )

    out_path = FIGURES_DIR / "delivery_hero_guardrail_order_value.png"
    save_dark(fig, out_path)
    return out_path, {"A_avg_order_value": a_val, "B_avg_order_value": b_val, "delta_pct": delta_pct}


def build_payload(
    df: pd.DataFrame,
    business_metrics: dict,
    patience_metrics: dict,
    kitchen_metrics: dict,
    accuracy_metrics: dict,
    cancel_model_metrics: dict,
    cancel_feature_importance: dict,
) -> dict:
    variant_summary = (
        df.groupby("variant", observed=False)
        .agg(
            conversion_rate=("is_converted", "mean"),
            avg_order_value=("order_value_eur", "mean"),
            avg_delay=("delay_min", "mean"),
            avg_eta=("eta_shown_min", "mean"),
            avg_actual=("actual_delivery_min", "mean"),
        )
        .reset_index()
    )

    guardrail_delta_pct = float(
        (variant_summary.loc[variant_summary["variant"] == "B", "avg_order_value"].iloc[0]
         / max(variant_summary.loc[variant_summary["variant"] == "A", "avg_order_value"].iloc[0], 1e-8) - 1)
        * 100
    )

    return {
        "rows": int(len(df)),
        "business_impact": business_metrics,
        "patience_threshold": patience_metrics,
        "kitchen_bottleneck": kitchen_metrics,
        "model_accuracy": accuracy_metrics,
        "cancel_models": cancel_model_metrics,
        "cancel_feature_importance": cancel_feature_importance,
        "guardrail_avg_order_value_delta_pct": guardrail_delta_pct,
        "variant_summary": {
            row["variant"]: {
                "conversion_rate": float(row["conversion_rate"]),
                "avg_order_value": float(row["avg_order_value"]),
                "avg_delay": float(row["avg_delay"]),
                "avg_eta": float(row["avg_eta"]),
                "avg_actual": float(row["avg_actual"]),
            }
            for _, row in variant_summary.iterrows()
        },
    }


def main() -> None:
    setup_dark_theme()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    _, eval_frame, _ = train_eta_model(df)
    _, _, _, cancel_model_metrics, cancel_importance_full = train_cancellation_models(df)

    business_path, business_metrics = make_business_impact_plot(df)
    patience_path, patience_metrics = make_patience_threshold_plot(df)
    kitchen_path, kitchen_metrics = make_kitchen_bottleneck_plot(df)
    accuracy_path, accuracy_metrics = make_model_vs_naive_error_plot(eval_frame)
    feature_path, top_cancel_features = make_cancel_feature_importance_plot(cancel_importance_full)
    guardrail_path, _ = make_guardrail_order_value_plot(df)

    payload = build_payload(
        df,
        business_metrics,
        patience_metrics,
        kitchen_metrics,
        accuracy_metrics,
        cancel_model_metrics,
        top_cancel_features,
    )
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Generated dark-theme assets:")
    print(f"- {business_path}")
    print(f"- {patience_path}")
    print(f"- {kitchen_path}")
    print(f"- {accuracy_path}")
    print(f"- {feature_path}")
    print(f"- {guardrail_path}")
    print(f"- {METRICS_PATH}")


if __name__ == "__main__":
    main()
