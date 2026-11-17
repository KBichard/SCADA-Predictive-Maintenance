
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Load sensor readings
raw = pd.read_parquet(
    ROOT / "data" / "processed_sensor_data.parquet"
)

# Load our model results
results = pd.read_parquet(
    ROOT / "data" / "active_detection_results.parquet"
)

# Historical simulation period
start = pd.Timestamp("2020-07-15 10:00")
end = pd.Timestamp("2020-07-15 22:00")

sensors = [
    "motor_current",
    "oil_temperature",
    "tp2",
    "tp3",
    "h1",
    "dv_pressure",
    "reservoir_pressure"
]

timeline = pd.DataFrame({
    "timestamp": pd.date_range(
        start, end, freq="5min"
    )
})

# Match the most recent available sensor readings
raw = raw[
    (raw["timestamp"] >= start - pd.Timedelta(seconds=30))
    & (raw["timestamp"] <= end)
][["timestamp"] + sensors]

raw = raw.rename(columns={
    "timestamp": "sensor_timestamp"
})

replay = pd.merge_asof(
    timeline,
    raw.sort_values("sensor_timestamp"),
    left_on="timestamp",
    right_on="sensor_timestamp",
    direction="backward",
    tolerance=pd.Timedelta(seconds=30)
)

# Attach the Isolation Forest results
replay = replay.merge(
    results[[
        "timestamp",
        "isolation_forest_score",
        "isolation_forest_alarm"
    ]],
    on="timestamp",
    how="left"
)

replay["sensor_ready"] = (
    replay["sensor_timestamp"].notna()
)

replay["model_ready"] = (
    replay["isolation_forest_score"].notna()
    & replay["sensor_ready"]
)

replay["urgent_alarm"] = (
    replay["isolation_forest_alarm"].eq(True)
    & replay["model_ready"]
)

replay["alarm_status"] = "NORMAL"

replay.loc[
    replay["urgent_alarm"],
    "alarm_status"
] = "URGENT"

replay.loc[
    ~replay["model_ready"],
    "alarm_status"
] = "DATA UNAVAILABLE"

output = ROOT / "data" / "scada_replay.json"

replay.to_json(
    output,
    orient="records",
    date_format="iso",
    indent=2
)

print("\n=== SCADA REPLAY READY ===")
print("Simulation windows:", len(replay))
print("Available model windows:", replay["model_ready"].sum())
print("Urgent alarm windows:", replay["urgent_alarm"].sum())

print("\n=== ALARM STATUS COUNTS ===")
print(replay["alarm_status"].value_counts())

print("\nSaved:", output)
