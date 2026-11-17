
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MetroPT3(AirCompressor).csv"

print("Loading industrial sensor data...")
df = pd.read_csv(DATA)

print("\n=== DATASET DIMENSIONS ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\n=== COLUMN NAMES ===")
for col in df.columns:
    print(col)

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== FIRST FIVE OBSERVATIONS ===")
print(df.head().to_string())

print("\n=== MISSING VALUES ===")
print(df.isna().sum())

print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== TIMESTAMP ANALYSIS ===")

timestamp_col = next(
    (c for c in df.columns if c.strip().lower() == "timestamp"),
    None
)

if timestamp_col:
    timestamps = pd.to_datetime(
        df[timestamp_col], errors="coerce"
    )

    print("Start:", timestamps.min())
    print("End:", timestamps.max())
    print("Invalid timestamps:", timestamps.isna().sum())
    print("Chronological:", timestamps.is_monotonic_increasing)

print("\n=== SENSOR STATISTICS ===")
print(df.describe().round(3).T.to_string())

print("\nInspection complete.")
