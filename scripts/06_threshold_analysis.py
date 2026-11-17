
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(
    ROOT / "data" / "anomaly_results.parquet"
)

failures = pd.read_csv(
    ROOT / "reports" / "failure_events.csv",
    parse_dates=["start", "end"]
)

calibration = df[df["partition"] == "calibration"]
evaluation = df[df["partition"] == "evaluation"]

reference = evaluation[
    evaluation["failure_next_24h"] == 0
]

results = []
summary = []

for percentile in [99, 97.5, 95, 90]:

    threshold = np.percentile(
        calibration["anomaly_score"],
        percentile
    )

    reference_rate = (
        reference["anomaly_score"] >= threshold
    ).mean() * 100

    events_warned = 0

    for event in failures.itertuples():

        before = evaluation[
            (evaluation["timestamp"] >=
             event.start - pd.Timedelta(hours=24))
            & (evaluation["timestamp"] < event.start)
        ]

        during = evaluation[
            (evaluation["timestamp"] >= event.start)
            & (evaluation["timestamp"] <= event.end)
        ]

        alerts = before[
            before["anomaly_score"] >= threshold
        ]

        lead = np.nan

        if not alerts.empty:
            events_warned += 1

            lead = (
                event.start - alerts["timestamp"].min()
            ).total_seconds() / 3600

        results.append({
            "percentile": percentile,
            "threshold": threshold,
            "event_id": event.event_id,
            "max_pre_failure_score":
                before["anomaly_score"].max(),
            "pre_failure_alerts": len(alerts),
            "lead_hours": lead,
            "failure_detection_pct":
                (during["anomaly_score"] >= threshold)
                .mean() * 100
        })

    summary.append({
        "percentile": percentile,
        "threshold": threshold,
        "events_warned": events_warned,
        "total_events": len(failures),
        "reference_alert_pct": reference_rate
    })

details = pd.DataFrame(results)
summary = pd.DataFrame(summary)

details.to_csv(
    ROOT / "reports" / "threshold_event_details.csv",
    index=False
)

summary.to_csv(
    ROOT / "reports" / "threshold_comparison.csv",
    index=False
)

print("\n=== THRESHOLD COMPARISON ===")
print(summary.round(4).to_string(index=False))

print("\n=== INDIVIDUAL FAILURE RESULTS ===")
print(details.round(4).to_string(index=False))

print("\nThreshold analysis complete.")
