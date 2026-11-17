
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "model_features.parquet"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
FIGURES = REPORTS / "figures"

MODELS.mkdir(exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

print("Loading engineered features...")

df = pd.read_parquet(DATA)
df = df.sort_values("timestamp").reset_index(drop=True)

# Capture feature names before adding labels
feature_cols = [
    c for c in df.columns if c != "timestamp"
]

failures = pd.read_csv(
    REPORTS / "failure_events.csv",
    parse_dates=["start", "end"]
)

# ----------------------------------------
# 1. Create failure labels
# ----------------------------------------

df["failure_active"] = 0
df["failure_next_24h"] = 0
df["event_id"] = 0
df["recovery_period"] = 0

for event in failures.itertuples():

    warning_start = event.start - pd.Timedelta(hours=24)

    before = (
        (df["timestamp"] >= warning_start)
        & (df["timestamp"] < event.start)
    )

    during = (
        (df["timestamp"] >= event.start)
        & (df["timestamp"] <= event.end)
    )

    recovery = (
        (df["timestamp"] > event.end)
        & (df["timestamp"] <=
           event.end + pd.Timedelta(hours=24))
    )

    df.loc[before, "failure_next_24h"] = 1
    df.loc[during, "failure_active"] = 1
    df.loc[before | during, "event_id"] = event.event_id
    df.loc[recovery, "recovery_period"] = 1

# Exclude active failures, recovery periods,
# and the final 24 hours from forecast labels

raw_end = pd.read_parquet(
    ROOT / "data" / "processed_sensor_data.parquet",
    columns=["timestamp"]
)["timestamp"].max()

unavailable = (
    (df["failure_active"] == 1)
    | (df["recovery_period"] == 1)
    | (df["timestamp"] > raw_end - pd.Timedelta(hours=24))
)

df["failure_next_24h"] = (
    df["failure_next_24h"].astype("Int64")
)

df.loc[unavailable, "failure_next_24h"] = pd.NA

print("\n=== FAILURE LABELS ===")
print(df["failure_next_24h"].value_counts(dropna=False))

df.to_parquet(
    ROOT / "data" / "labeled_features.parquet",
    index=False
)

# ----------------------------------------
# 2. Chronological split
# ----------------------------------------

train_end = pd.Timestamp("2020-03-15")
test_start = pd.Timestamp("2020-04-17")

train = df[df["timestamp"] < train_end]

calibration = df[
    (df["timestamp"] >= train_end)
    & (df["timestamp"] < test_start)
]

test_mask = df["timestamp"] >= test_start

print("\n=== MODEL PARTITIONS ===")
print("Training:", len(train))
print("Calibration:", len(calibration))
print("Evaluation:", test_mask.sum())

# ----------------------------------------
# 3. Train Isolation Forest
# ----------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    max_samples=2048,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(train[feature_cols])

# Higher score = more abnormal
calibration_scores = -model.score_samples(
    calibration[feature_cols]
)

# Preliminary threshold:
# 99th percentile of calibration scores
threshold = np.quantile(
    calibration_scores, 0.99
)

df["anomaly_score"] = -model.score_samples(
    df[feature_cols]
)

df["anomaly_flag"] = (
    df["anomaly_score"] >= threshold
).astype(int)

df["partition"] = "evaluation"

df.loc[
    df["timestamp"] < train_end,
    "partition"
] = "training"

df.loc[
    (df["timestamp"] >= train_end)
    & (df["timestamp"] < test_start),
    "partition"
] = "calibration"

print("\nAnomaly threshold:", round(threshold, 4))

# ----------------------------------------
# 4. Evaluate individual failure events
# ----------------------------------------

results = []

for event in failures.itertuples():

    warning_start = event.start - pd.Timedelta(hours=24)

    before = df[
        (df["timestamp"] >= warning_start)
        & (df["timestamp"] < event.start)
    ]

    during = df[
        (df["timestamp"] >= event.start)
        & (df["timestamp"] <= event.end)
    ]

    alerts = before[before["anomaly_flag"] == 1]

    if not alerts.empty:
        lead_hours = (
            event.start - alerts["timestamp"].min()
        ).total_seconds() / 3600
    else:
        lead_hours = np.nan

    results.append({
        "event_id": event.event_id,
        "pre_failure_windows": len(before),
        "pre_failure_coverage_pct":
            len(before) / 288 * 100,
        "pre_failure_alerts": len(alerts),
        "first_alert_lead_hours": lead_hours,
        "failure_windows": len(during),
        "failure_detection_pct":
            during["anomaly_flag"].mean() * 100
    })

summary = pd.DataFrame(results)

summary.to_csv(
    REPORTS / "anomaly_event_evaluation.csv",
    index=False
)

print("\n=== EVENT EVALUATION ===")
print(summary.round(2).to_string(index=False))

# Alert rate outside reported failure,
# warning and recovery periods

reference = df[
    test_mask
    & (df["failure_next_24h"] == 0)
]

print("\n=== REFERENCE ALERT RATE ===")

print("Reference windows:", len(reference))

print(
    "Alert percentage:",
    round(reference["anomaly_flag"].mean() * 100, 2)
)

# ----------------------------------------
# 5. Save the trained model
# ----------------------------------------

joblib.dump({
    "model": model,
    "feature_columns": feature_cols,
    "threshold": threshold
}, MODELS / "isolation_forest.joblib")

df.to_parquet(
    ROOT / "data" / "anomaly_results.parquet",
    index=False
)

# ----------------------------------------
# 6. Visualize anomaly scores
# ----------------------------------------

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(18, 6))

evaluation = df.loc[test_mask].set_index("timestamp")

scores = evaluation["anomaly_score"].resample(
    "5min"
).mean()

ax.plot(
    scores.index,
    scores,
    color="turquoise",
    linewidth=0.7,
    label="Anomaly score"
)

ax.axhline(
    threshold,
    color="orange",
    linestyle="--",
    label="Alert threshold"
)

for event in failures.itertuples():

    ax.axvspan(
        event.start - pd.Timedelta(hours=24),
        event.start,
        color="orange",
        alpha=0.12
    )

    ax.axvspan(
        event.start,
        event.end,
        color="red",
        alpha=0.25
    )

ax.set_title(
    "Industrial Compressor - Isolation Forest Monitoring",
    fontsize=17
)

ax.set_ylabel("Anomaly score")
ax.set_xlabel("Timestamp")
ax.legend()
ax.grid(alpha=0.15)

fig.tight_layout()

fig.savefig(
    FIGURES / "anomaly_detection.png",
    dpi=150
)

plt.close(fig)

print("\n=== ANOMALY DETECTION COMPLETE ===")
