
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

print("Loading labeled features...")

df = pd.read_parquet(
    ROOT / "data" / "labeled_features.parquet"
)

original = pd.read_parquet(
    ROOT / "data" / "model_features.parquet"
)

feature_cols = [
    c for c in original.columns
    if c != "timestamp"
]

target = "failure_next_24h"

df = df.dropna(subset=[target]).copy()
df[target] = df[target].astype(int)

df = df.sort_values("timestamp")

# ----------------------------------------
# 1. Chronological partitions
# ----------------------------------------

validation_start = pd.Timestamp("2020-06-02")
test_start = pd.Timestamp("2020-07-01")

horizon = pd.Timedelta(hours=24)
history = pd.Timedelta(hours=3)

# Purge overlapping forecast horizons
# and rolling feature histories

train = df[
    df["timestamp"] < validation_start - horizon
]

validation = df[
    (df["timestamp"] >= validation_start + history)
    & (df["timestamp"] < test_start - horizon)
]

test = df[
    df["timestamp"] >= test_start + history
]

for name, partition in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    print(f"\n=== {name.upper()} ===")
    print("Observations:", len(partition))
    print("Positive labels:", partition[target].sum())
    print("Positive rate:",
          round(partition[target].mean() * 100, 2))

X_train = train[feature_cols]
y_train = train[target]

X_val = validation[feature_cols]
y_val = validation[target]

X_test = test[feature_cols]
y_test = test[target]

# ----------------------------------------
# 2. Define candidate models
# ----------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = []
thresholds = {}

# ----------------------------------------
# 3. Train and validate
# ----------------------------------------

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    scores = model.predict_proba(X_val)[:, 1]

    # Derive threshold from validation
    # reference windows only

    reference_scores = scores[
        y_val.to_numpy() == 0
    ]

    threshold = np.quantile(
        reference_scores, 0.95
    )

    thresholds[name] = threshold

    predictions = (
        scores > threshold
    ).astype(int)

    results.append({
        "model": name,
        "validation_roc_auc":
            roc_auc_score(y_val, scores),
        "validation_average_precision":
            average_precision_score(y_val, scores),
        "precision":
            precision_score(
                y_val, predictions, zero_division=0
            ),
        "recall":
            recall_score(
                y_val, predictions, zero_division=0
            ),
        "f1":
            f1_score(
                y_val, predictions, zero_division=0
            ),
        "reference_alert_pct":
            predictions[
                y_val.to_numpy() == 0
            ].mean() * 100,
        "threshold": threshold
    })

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "validation_average_precision",
    ascending=False
)

comparison.to_csv(
    REPORTS / "supervised_model_comparison.csv",
    index=False
)

print("\n=== VALIDATION MODEL COMPARISON ===")
print(comparison.round(4).to_string(index=False))

# ----------------------------------------
# 4. Select model using validation results
# ----------------------------------------

best_name = comparison.iloc[0]["model"]
best_model = models[best_name]
threshold = thresholds[best_name]

print("\nSelected model:", best_name)
print("Alert threshold:", round(threshold, 4))

# ----------------------------------------
# 5. Evaluate selected model
# ----------------------------------------

test_scores = best_model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_scores > threshold
).astype(int)

test_results = test.copy()
test_results["failure_risk_score"] = test_scores
test_results["forecast_alert"] = test_predictions

metrics = {
    "model": best_name,
    "roc_auc": roc_auc_score(y_test, test_scores),
    "average_precision":
        average_precision_score(y_test, test_scores),
    "baseline_average_precision": y_test.mean(),
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "reference_alert_pct":
        test_predictions[
            y_test.to_numpy() == 0
        ].mean() * 100
}

print("\n=== TEST PERFORMANCE ===")

for key, value in metrics.items():
    print(
        key,
        round(value, 4)
        if isinstance(value, (float, np.floating))
        else value
    )

# ----------------------------------------
# 6. Evaluate July failure warning
# ----------------------------------------

failures = pd.read_csv(
    REPORTS / "failure_events.csv",
    parse_dates=["start", "end"]
)

event = failures.loc[
    failures["event_id"] == 4
].iloc[0]

before = test_results[
    (test_results["timestamp"] >=
     event["start"] - horizon)
    & (test_results["timestamp"] < event["start"])
]

alerts = before[
    before["forecast_alert"] == 1
]

print("\n=== JULY 15 FAILURE FORECAST ===")
print("Available windows:", len(before))
print("Alert windows:", len(alerts))

lead_hours = np.nan

if not alerts.empty:

    first_alert = alerts["timestamp"].min()

    lead_hours = (
        event["start"] - first_alert
    ).total_seconds() / 3600

    print("First alert:", first_alert)
    print("Warning lead time:",
          round(lead_hours, 2), "hours")

else:
    print("No advance warning generated.")

metrics["july_available_windows"] = len(before)
metrics["july_alert_windows"] = len(alerts)
metrics["july_first_alert_lead_hours"] = lead_hours

# ----------------------------------------
# 7. Save results and model
# ----------------------------------------

pd.DataFrame([metrics]).to_csv(
    REPORTS / "supervised_test_summary.csv",
    index=False
)

test_results.to_parquet(
    ROOT / "data" / "forecast_results.parquet",
    index=False
)

joblib.dump({
    "model": best_model,
    "model_name": best_name,
    "feature_columns": feature_cols,
    "threshold": threshold,
    "prediction_horizon_hours": 24
}, MODELS / "failure_forecast.joblib")

print("\n=== FAILURE PREDICTION COMPLETE ===")
