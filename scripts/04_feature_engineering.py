
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed_sensor_data.parquet"
OUTPUT = ROOT / "data" / "model_features.parquet"

print("Loading sensor data...")
df = pd.read_parquet(DATA)
df = df.set_index("timestamp").sort_index()

# Preliminary motor activity threshold
# Subject to validation against equipment behavior
df["motor_active"] = (
    df["motor_current"] > 1.0
).astype(float)

# Identify collection interruptions
df["gap_flag"] = (
    df.index.to_series().diff().dt.total_seconds() > 30
)

print("Aggregating sensor readings...")

# Each window contains only readings available
# at or before its timestamp
windows = df.resample(
    "5min", label="right", closed="right"
).agg(
    current_mean=("motor_current", "mean"),
    current_std=("motor_current", "std"),
    temperature_mean=("oil_temperature", "mean"),
    pressure_mean=("reservoir_pressure", "mean"),
    pressure_min=("reservoir_pressure", "min"),
    pressure_std=("reservoir_pressure", "std"),
    tp2_mean=("tp2", "mean"),
    h1_mean=("h1", "mean"),
    dv_pressure_mean=("dv_pressure", "mean"),
    motor_duty_cycle=("motor_active", "mean"),
    observations=("motor_current", "count"),
    gap_flag=("gap_flag", "max")
)

# Require at least 80% of the expected
# 30 readings per five-minute window
valid = (
    (windows["observations"] >= 24)
    & windows["gap_flag"].eq(False)
)

features = windows.drop(
    columns=["observations", "gap_flag"]
)

# Do not interpolate interrupted readings
features.loc[~valid, :] = float("nan")

print("Engineering historical features...")

rolling_columns = [
    "current_mean",
    "pressure_mean",
    "temperature_mean",
    "tp2_mean",
    "h1_mean",
    "dv_pressure_mean",
    "motor_duty_cycle"
]

# Trailing 30-minute, 1-hour and 3-hour windows
for name, periods in [
    ("30min", 6),
    ("1h", 12),
    ("3h", 36)
]:
    for col in rolling_columns:

        rolling = features[col].rolling(
            periods,
            min_periods=periods
        )

        features[f"{col}_{name}"] = rolling.mean()

# Changes over the previous hour
for col in [
    "pressure_mean",
    "temperature_mean",
    "current_mean",
    "h1_mean"
]:
    complete = (
        features[col].rolling(13).count() == 13
    )

    change = features[col] - features[col].shift(12)

    features[f"{col}_change_1h"] = change.where(
        complete
    )

# Preserve only complete feature windows
total_windows = len(features)

features = features.dropna().reset_index()

features.to_parquet(OUTPUT, index=False)

print("\n=== FEATURE ENGINEERING COMPLETE ===")
print(f"Total five-minute windows: {total_windows:,}")
print(f"Usable observations: {len(features):,}")
print(f"Excluded windows: {total_windows-len(features):,}")
print(f"Predictive features: {len(features.columns)-1}")
print("Missing values:", features.isna().sum().sum())

print("\n=== FEATURE NAMES ===")
for col in features.columns:
    print(col)

print("\nSaved:", OUTPUT)
