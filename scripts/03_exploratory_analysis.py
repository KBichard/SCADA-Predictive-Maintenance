
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_sensor_data.parquet"
REPORTS = ROOT / "reports"
PLOTS = REPORTS / "figures"

PLOTS.mkdir(parents=True, exist_ok=True)

print("Loading processed sensor data...")
df = pd.read_parquet(DATA)

df = df.set_index("timestamp").sort_index()

# Documented equipment failure events
failures = pd.DataFrame([
    (1, "2020-04-18 00:00", "2020-04-18 23:59"),
    (2, "2020-05-29 23:30", "2020-05-30 06:00"),
    (3, "2020-06-05 10:00", "2020-06-07 14:30"),
    (4, "2020-07-15 14:30", "2020-07-15 19:00")
], columns=["event_id", "start", "end"])

failures["start"] = pd.to_datetime(failures["start"])
failures["end"] = pd.to_datetime(failures["end"])
failures["failure_type"] = "Air leak"

failures.to_csv(
    REPORTS / "failure_events.csv", index=False
)

sensors = [
    "tp2",
    "tp3",
    "h1",
    "dv_pressure",
    "reservoir_pressure",
    "oil_temperature",
    "motor_current"
]

plt.style.use("dark_background")

# 1. Long-term sensor trends
print("Generating historical sensor trends...")

hourly = df[sensors].resample("1h").mean()

fig, axes = plt.subplots(
    len(sensors), 1,
    figsize=(16, 15),
    sharex=True
)

for ax, sensor in zip(axes, sensors):
    ax.plot(
        hourly.index,
        hourly[sensor],
        linewidth=0.7
    )

    for event in failures.itertuples():
        ax.axvspan(
            event.start,
            event.end,
            color="red",
            alpha=0.25
        )

    ax.set_ylabel(sensor)
    ax.grid(alpha=0.15)

fig.suptitle(
    "Industrial Compressor - Historical Sensor Trends",
    fontsize=18
)

fig.tight_layout()
fig.savefig(
    PLOTS / "historical_sensor_trends.png",
    dpi=150
)
plt.close(fig)

# 2. Sensor correlation matrix
print("Generating correlation matrix...")

corr = df[sensors].corr()

fig, ax = plt.subplots(figsize=(10, 8))

img = ax.imshow(
    corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(sensors)))
ax.set_yticks(range(len(sensors)))
ax.set_xticklabels(sensors, rotation=45, ha="right")
ax.set_yticklabels(sensors)

for i in range(len(sensors)):
    for j in range(len(sensors)):
        ax.text(
            j, i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black"
        )

fig.colorbar(img)
ax.set_title("Industrial Sensor Correlations")
fig.tight_layout()
fig.savefig(PLOTS / "sensor_correlations.png", dpi=150)
plt.close(fig)

# 3. Analyze individual failure events
print("Investigating documented failure events...")

event_summary = []

for event in failures.itertuples():

    start = event.start - pd.Timedelta(hours=24)
    end = event.end + pd.Timedelta(hours=12)

    window = df.loc[start:end, sensors]
    sampled = window.resample("5min").mean()

    fig, axes = plt.subplots(
        len(sensors), 1,
        figsize=(16, 13),
        sharex=True
    )

    for ax, sensor in zip(axes, sensors):

        ax.plot(
            sampled.index,
            sampled[sensor],
            linewidth=1
        )

        ax.axvspan(
            event.start,
            event.end,
            color="red",
            alpha=0.25
        )

        ax.axvline(
            event.start,
            color="orange",
            linestyle="--"
        )

        ax.set_ylabel(sensor)
        ax.grid(alpha=0.15)

    fig.suptitle(
        f"Failure Event {event.event_id} - Air Leak",
        fontsize=18
    )

    fig.tight_layout()

    fig.savefig(
        PLOTS / f"failure_event_{event.event_id}.png",
        dpi=150
    )

    plt.close(fig)

    before = df.loc[
        (df.index >= start) &
        (df.index < event.start)
    ]

    during = df.loc[event.start:event.end]

    event_summary.append({
        "event_id": event.event_id,
        "pre_failure_records": len(before),
        "failure_records": len(during),
        "pre_failure_avg_pressure":
            before["reservoir_pressure"].mean(),
        "failure_avg_pressure":
            during["reservoir_pressure"].mean(),
        "pre_failure_avg_current":
            before["motor_current"].mean(),
        "failure_avg_current":
            during["motor_current"].mean()
    })

summary = pd.DataFrame(event_summary)

summary.to_csv(
    REPORTS / "failure_event_summary.csv",
    index=False
)

print("\n=== FAILURE EVENT SUMMARY ===")
print(summary.round(3).to_string(index=False))

print("\n=== ANALYSIS COMPLETE ===")
print("Figures saved to:", PLOTS)
