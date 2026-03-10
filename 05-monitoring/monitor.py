"""
Generate an Evidently HTML report comparing past vs recent predictions.

Usage:
    python monitor.py
This will:
- Load the logged predictions from data/predictions.csv
- Split them into reference (older) vs current (newer) data
- Generate a data + performance drift report
"""

import pandas as pd
from pathlib import Path
from evidently import Dataset, DataDefinition, Report, Regression
from evidently.presets import DataDriftPreset, RegressionPreset


LOG_PATH = Path("data/predictions.csv")
REPORT_PATH = Path("monitoring_report.html")


def main():
    print("\n📊 Starting monitoring report...\n")

    if not LOG_PATH.exists():
        raise FileNotFoundError("❌ No logged predictions found. Run simulate.py first!")

    df = pd.read_csv(LOG_PATH, parse_dates=["ts"])
    df = df.dropna(subset=["prediction", "duration"])
    print(f"✓ Loaded {len(df)} logged predictions")

    # Sort by timestamp and split into reference (older) vs current (recent)
    df = df.sort_values("ts")
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()

    print(f"Reference: {len(reference)}  |  Current: {len(current)}")

    # Evidently 0.7+: DataDefinition for column mapping
    data_def = DataDefinition(
        regression=[Regression(target="duration", prediction="prediction")],
        numerical_columns=["trip_distance"],
        categorical_columns=["PU_DO"],
    )
    ref_dataset = Dataset.from_pandas(reference, data_definition=data_def)
    cur_dataset = Dataset.from_pandas(current, data_definition=data_def)

    # Build report (current first, reference second)
    print("\n🧮 Generating Evidently drift report...")
    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    snapshot = report.run(cur_dataset, ref_dataset)

    snapshot.save_html(str(REPORT_PATH))
    print(f"✅ Report saved: {REPORT_PATH.resolve()}")
    print("Open it in your browser to explore drift metrics.\n")


if __name__ == "__main__":
    main()
