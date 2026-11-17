
from pathlib import Path
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(
    ROOT / "data" / "anomaly_results.parquet"
)

original = pd.read_parquet(
    ROOT / "data" / "model_features.parquet"
)

features = [
    c for c in original.columns if c != "timestamp"
]

failures = pd.read_csv(
    ROOT / "reports" / "failure_events.csv",
    parse_dates=["start", "end"]
)

# Create one-hour prediction labels
df["failure_next_1h"] = 0

for event in failures.itertuples():
    mask = (
        (df["timestamp"] >=
         event.start - pd.Timedelta(hours=1))
        & (df["timestamp"] < event.start)
    )
    df.loc[mask, "failure_next_1h"] = 1

# Prediction: exclude active failures
# and recovery periods
prediction = df.dropna(
    subset=["failure_next_24h"]
).copy()

# Detection: compare active failures against
# reference periods, excluding ambiguous
# pre-failure and recovery periods
reference = df["failure_next_24h"].eq(0).fillna(False)

detection = df[
    df["failure_active"].eq(1) | reference
].copy()

experiments = [
    ("1-hour prediction", prediction, "failure_next_1h"),
    ("Active detection", detection, "failure_active")
]

folds = [
    ("May", "2020-05-01", "2020-06-02"),
    ("June", "2020-06-02", "2020-07-01")
]

results = []

for task, data, target in experiments:

    for fold, start, end in folds:

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        train = data[
            data["timestamp"] <
            start - pd.Timedelta(hours=24)
        ]

        val = data[
            (data["timestamp"] >=
             start + pd.Timedelta(hours=3))
            & (data["timestamp"] <
               end - pd.Timedelta(hours=24))
        ]

        X_train = train[features]
        y_train = train[target]

        X_val = val[features]
        y_val = val[target]

        if y_train.nunique() < 2 or y_val.nunique() < 2:
            continue

        print(f"\n{task} | {fold}")

        models = {
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
            )
        }

        scores = {
            "Motor Duty Cycle":
                val["motor_duty_cycle_30min"],

            "Isolation Forest":
                val["anomaly_score"]
        }

        weights = compute_sample_weight(
            "balanced", y_train
        )

        for name, model in models.items():

            print("Training:", name)

            model.fit(
                X_train, y_train,
                sample_weight=weights
            )

            scores[name] = model.predict_proba(
                X_val
            )[:, 1]

        for name, score in scores.items():

            ap = average_precision_score(y_val, score)

            results.append({
                "task": task,
                "fold": fold,
                "model": name,
                "positive_windows": int(y_val.sum()),
                "roc_auc": roc_auc_score(y_val, score),
                "average_precision": ap,
                "ap_lift": ap / y_val.mean()
            })

results = pd.DataFrame(results)

summary = results.groupby(
    ["task", "model"], as_index=False
).agg(
    mean_auc=("roc_auc", "mean"),
    mean_ap=("average_precision", "mean"),
    mean_ap_lift=("ap_lift", "mean")
)

summary = summary.sort_values(
    ["task", "mean_ap_lift"],
    ascending=[True, False]
)

results.to_csv(
    ROOT / "reports" / "detection_prediction_details.csv",
    index=False
)

summary.to_csv(
    ROOT / "reports" / "detection_prediction_comparison.csv",
    index=False
)

print("\n=== INDIVIDUAL RESULTS ===")
print(results.round(4).to_string(index=False))

print("\n=== MODEL COMPARISON ===")
print(summary.round(4).to_string(index=False))
