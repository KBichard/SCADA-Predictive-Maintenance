
from pathlib import Path
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

feature_cols = [
    c for c in original.columns
    if c != "timestamp"
]

failures = pd.read_csv(
    ROOT / "reports" / "failure_events.csv",
    parse_dates=["start", "end"]
)

# Use the same eligible observations for both horizons
df = df.dropna(subset=["failure_next_24h"]).copy()

# Keep July out of this experiment
df = df[df["timestamp"] < "2020-07-01"].copy()

folds = [
    ("May", "2020-05-01", "2020-06-02"),
    ("June", "2020-06-02", "2020-07-01")
]

results = []

for hours in [24, 6]:

    print(f"\n=== {hours}-HOUR PREDICTION ===")

    # Create the target variable
    df["target"] = 0

    for event in failures.itertuples():

        mask = (
            (df["timestamp"] >=
             event.start - pd.Timedelta(hours=hours))
            & (df["timestamp"] < event.start)
        )

        df.loc[mask, "target"] = 1

    for fold, start, end in folds:

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        # Keep feature histories and target horizons
        # separated across chronological partitions
        train = df[
            df["timestamp"] <
            start - pd.Timedelta(hours=24)
        ]

        val = df[
            (df["timestamp"] >=
             start + pd.Timedelta(hours=3))
            & (df["timestamp"] <
               end - pd.Timedelta(hours=24))
        ]

        X_train = train[feature_cols]
        y_train = train["target"]

        X_val = val[feature_cols]
        y_val = val["target"]

        if y_train.nunique() < 2 or y_val.nunique() < 2:
            print(f"Skipping {fold}: insufficient classes")
            continue

        models = {
            "Logistic Regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000
                )
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced",
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

        baseline = y_val.mean()

        print(f"\nValidation event: {fold}")
        print("Positive training rows:", y_train.sum())
        print("Positive validation rows:", y_val.sum())

        # Existing unsupervised baseline
        scores = {
            "Isolation Forest":
                val["anomaly_score"].to_numpy()
        }

        # Train each supervised model
        for name, model in models.items():

            print("Training:", name)

            if name == "Gradient Boosting":

                weights = compute_sample_weight(
                    "balanced", y_train
                )

                model.fit(
                    X_train, y_train,
                    sample_weight=weights
                )

            else:
                model.fit(X_train, y_train)

            scores[name] = model.predict_proba(
                X_val
            )[:, 1]

        # Compare all models on identical validation data
        for name, predictions in scores.items():

            ap = average_precision_score(
                y_val, predictions
            )

            results.append({
                "horizon_hours": hours,
                "validation_event": fold,
                "model": name,
                "roc_auc": roc_auc_score(
                    y_val, predictions
                ),
                "average_precision": ap,
                "baseline_ap": baseline,
                "ap_lift": ap / baseline
            })

results = pd.DataFrame(results)

summary = results.groupby(
    ["horizon_hours", "model"],
    as_index=False
).agg(
    mean_roc_auc=("roc_auc", "mean"),
    mean_ap=("average_precision", "mean"),
    mean_ap_lift=("ap_lift", "mean"),
    folds=("validation_event", "count")
)

summary = summary.sort_values(
    ["horizon_hours", "mean_ap_lift"],
    ascending=[False, False]
)

results.to_csv(
    ROOT / "reports" / "model_comparison_details.csv",
    index=False
)

summary.to_csv(
    ROOT / "reports" / "model_comparison_summary.csv",
    index=False
)

print("\n=== INDIVIDUAL RESULTS ===")
print(results.round(4).to_string(index=False))

print("\n=== OVERALL MODEL COMPARISON ===")
print(summary.round(4).to_string(index=False))
