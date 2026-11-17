
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

ROOT = Path(__file__).resolve().parents[1]

# 1. Load existing data and models

df = pd.read_parquet(
    ROOT / "data" / "anomaly_results.parquet"
).sort_values("timestamp").reset_index(drop=True)

bundle = joblib.load(
    ROOT / "models" / "isolation_forest.joblib"
)

features = bundle["feature_columns"]

failures = pd.read_csv(
    ROOT / "reports" / "failure_events.csv",
    parse_dates=["start", "end"]
)

reference = df["failure_next_24h"].eq(0).fillna(False)
active = df["failure_active"].eq(1)

# 2. Chronological partitions

train = df[
    (df["timestamp"] < "2020-06-01")
    & (reference | active)
]

calibration = df[
    (df["timestamp"] >= "2020-06-02 03:00")
    & (df["timestamp"] < "2020-06-30")
    & reference
]

test = df[
    df["timestamp"] >= "2020-07-01 03:00"
].copy().reset_index(drop=True)

# 3. Train active-failure classifiers

models = {
    "Isolation Forest": bundle["model"],

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=15,
        learning_rate=0.05,
        early_stopping=False,
        random_state=42
    ),

    "Motor Duty Cycle": None
}

X_train = train[features]
y_train = train["failure_active"]

weights = compute_sample_weight("balanced", y_train)

for name in ["Random Forest", "Gradient Boosting"]:

    print("Training:", name)

    models[name].fit(
        X_train, y_train,
        sample_weight=weights
    )

# 4. Calculate model scores

def get_scores(name, data):

    if name == "Motor Duty Cycle":
        return data["motor_duty_cycle_30min"].to_numpy()

    if name == "Isolation Forest":
        return -models[name].score_samples(data[features])

    return models[name].predict_proba(data[features])[:, 1]


# July failure event

event = failures.loc[
    failures["event_id"] == 4
].iloc[0]

during = (
    (test["timestamp"] >= event["start"])
    & (test["timestamp"] <= event["end"])
)

test_reference = (
    test["failure_next_24h"].eq(0).fillna(False)
)

eligible = during | test_reference

# Check for uninterrupted five-minute windows

consecutive = test["timestamp"].diff().eq(
    pd.Timedelta(minutes=5)
)

results = []
thresholds = {}

# 5. Evaluate each model

for name in models:

    print("Evaluating:", name)

    calibration_scores = get_scores(name, calibration)

    # Common calibration alert budget: about 5%
    threshold = np.quantile(calibration_scores, 0.95)

    thresholds[name] = threshold

    scores = pd.Series(
        get_scores(name, test),
        index=test.index
    )

    above = scores > threshold

    # Require two consecutive abnormal windows
    alarm = (
        above
        & above.shift(1, fill_value=False)
        & consecutive
    )

    # A new alarm episode begins only once
    alarm_start = alarm & (
        ~alarm.shift(1, fill_value=False)
        | ~consecutive
    )

    detected = test.loc[during & alarm, "timestamp"]

    delay = np.nan

    if not detected.empty:
        delay = (
            detected.min() - event["start"]
        ).total_seconds() / 60

    reference_hours = test_reference.sum() * 5 / 60

    reference_episodes = int(
        (test_reference & alarm_start).sum()
    )

    results.append({
        "model": name,
        "threshold": threshold,

        "roc_auc": roc_auc_score(
            test.loc[eligible, "failure_active"],
            scores[eligible]
        ),

        "average_precision": average_precision_score(
            test.loc[eligible, "failure_active"],
            scores[eligible]
        ),

        "failure_detected": not detected.empty,

        "detection_delay_minutes": delay,

        "failure_alarm_pct":
            alarm[during].mean() * 100,

        "reference_alarm_pct":
            alarm[test_reference].mean() * 100,

        "reference_alarm_episodes":
            reference_episodes,

        "episodes_per_24h_monitored":
            reference_episodes / reference_hours * 24
    })

    slug = name.lower().replace(" ", "_")

    test[f"{slug}_score"] = scores
    test[f"{slug}_alarm"] = alarm
    test[f"{slug}_alarm_start"] = alarm_start

# 6. Save evaluation results

summary = pd.DataFrame(results)

summary.to_csv(
    ROOT / "reports" / "final_detection_evaluation.csv",
    index=False
)

test.to_parquet(
    ROOT / "data" / "active_detection_results.parquet",
    index=False
)

print("\n=== FINAL MODEL EVALUATION ===")
print(summary.round(4).to_string(index=False))

print("\nJuly failure start:", event["start"])
print("July failure end:", event["end"])
print("Available failure windows:", during.sum())

# 7. Preserve our previously preferred model
# Do not select a new winner using July test results

joblib.dump({
    "model": models["Isolation Forest"],
    "model_name": "Isolation Forest",
    "feature_columns": features,
    "threshold": thresholds["Isolation Forest"],
    "consecutive_windows": 2,
    "window_minutes": 5,
    "score_direction": "negative_score_samples"
}, ROOT / "models" / "active_fault_detector.joblib")

print("\n=== FINAL EVALUATION COMPLETE ===")
