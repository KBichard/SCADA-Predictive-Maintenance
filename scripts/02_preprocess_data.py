
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "MetroPT3(AirCompressor).csv"
OUTPUT = ROOT / "data" / "processed_sensor_data.parquet"
REPORTS = ROOT / "reports"

REPORTS.mkdir(exist_ok=True)

print("Loading raw sensor data...")
df = pd.read_csv(INPUT)

# Remove exported row index
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

df = df.rename(columns={
    "reservoirs": "reservoir_pressure",
    "dv_eletric": "dv_electric"
})

# Convert timestamps
df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="raise"
)

# Sort observations chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

# Inspect temporal integrity
duplicate_times = df["timestamp"].duplicated().sum()

intervals = df["timestamp"].diff().dt.total_seconds()

print("\n=== SAMPLING ANALYSIS ===")
print("Duplicate timestamps:", duplicate_times)
print("Median interval:", intervals.median(), "seconds")
print("Mean interval:", round(intervals.mean(), 2), "seconds")
print("Maximum interval:", intervals.max(), "seconds")

# Investigate interruptions in sensor collection
# 30 seconds is a preliminary data-quality threshold
gaps = pd.DataFrame({
    "previous_timestamp": df["timestamp"].shift(1),
    "next_timestamp": df["timestamp"],
    "gap_seconds": intervals
})

gaps = gaps.loc[gaps["gap_seconds"] > 30]

print("Gaps exceeding 30 seconds:", len(gaps))

gaps.to_csv(
    REPORTS / "sensor_data_gaps.csv", index=False
)

# Generate sensor summary
summary = df.drop(columns=["timestamp"]).describe().T

summary.to_csv(
    REPORTS / "sensor_summary.csv"
)

# Save processed dataset
df.to_parquet(OUTPUT, index=False)

print("\n=== PREPROCESSING COMPLETE ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print("Missing values:", df.isna().sum().sum())
print("Saved:", OUTPUT.name)

print("\n=== PROCESSED COLUMNS ===")
print(df.columns.tolist())
