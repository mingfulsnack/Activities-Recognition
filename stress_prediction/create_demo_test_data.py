"""
Create a new demo dataset for stress prediction inference.

This script preserves the existing data-generation logic:
1. Generate the full health dataset with the current 44-field generator.
2. Reduce it to the same 23-field schema used by the project.
3. Reduce it again to the final 13-feature schema used by the LSTM model.

The output filenames are demo-specific, so the original training datasets are
not overwritten.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GENERATOR_DIR = BASE_DIR / "generate_and_verify_data" / "Data generator"

DEFAULT_FULL_OUTPUT = DATA_DIR / "demo_health_data_44features.csv"
DEFAULT_23_OUTPUT = DATA_DIR / "demo_health_data_23features.csv"
DEFAULT_13_OUTPUT = DATA_DIR / "demo_stress_test_13features.csv"

PROTECTED_OUTPUTS = {
    DATA_DIR / "optimized_health_data_13features.csv",
    DATA_DIR / "optimized_health_data_23features.csv",
    GENERATOR_DIR / "data" / "optimized_health_data_23features.csv",
    GENERATOR_DIR / "data" / "quota_balanced_health_data_30days_v2.csv",
}

SELECTED_23_FEATURES = [
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Timestamp",
    "Activity",
    "Location",
    "Heart_Rate",
    "Sleep_Duration",
    "Sleep_Quality",
    "Energy_Level",
    "Mood_Score",
    "Stress_Level",
    "Screen_Usage_Current",
    "Screen_Usage_15min_Avg",
    "Screen_Usage_Trend",
    "Phone_Usage_Intensity",
    "Phone_Event_Frequency",
    "Social_Current_Level",
    "Social_1hour_Avg",
    "Ambient_Light",
    "Noise_Level",
    "Weather_Condition",
    "Exercise_Minutes",
]

SELECTED_13_FEATURES = [
    "Hour",
    "Day_of_Week",
    "Activity",
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Heart_Rate",
    "Location",
    "Screen_Usage_Current",
    "Phone_Event_Frequency",
    "Mood_Score",
    "Energy_Level",
    "Sleep_Duration",
    "Stress_Level",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a new demo dataset without overwriting training data."
    )
    parser.add_argument("--start-date", default="2024-03-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days to generate. Default is 3 for a quick defense demo.",
    )
    parser.add_argument("--age", type=int, default=22, help="Demo user age.")
    parser.add_argument("--gender", default="Male", help="Demo user gender.")
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL_OUTPUT)
    parser.add_argument("--features23-output", type=Path, default=DEFAULT_23_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_13_OUTPUT)
    return parser.parse_args()


def resolve_output(path: Path) -> Path:
    return path if path.is_absolute() else BASE_DIR / path


def guard_output_path(path: Path) -> None:
    resolved = path.resolve()
    protected = {p.resolve() for p in PROTECTED_OUTPUTS}
    if resolved in protected:
        raise ValueError(
            f"Refusing to overwrite protected dataset: {resolved}. "
            "Use a demo-specific filename instead."
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)


# Generates a fresh full 44-field demo dataset using the existing generator.
def generate_full_dataset(start_date: str, days: int, age: int, gender: str, output_path: Path) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("--days must be greater than 0.")

    if str(GENERATOR_DIR) not in sys.path:
        sys.path.insert(0, str(GENERATOR_DIR))

    from refactored_health_data_generator import RefactoredHealthDataGenerator

    print("=" * 80)
    print("[1/3] Generating full 44-field demo dataset")
    print("=" * 80)
    print(f"Start date: {start_date}")
    print(f"Days:       {days}")
    print(f"Output:     {output_path}")

    generator = RefactoredHealthDataGenerator(age=age, gender=gender)
    return generator.generate_enhanced_dataset(start_date, days, filename=str(output_path))


# Reduces generated full data to the same 23-field schema used by the project.
def create_23feature_dataset(full_path: Path, output_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("[2/3] Reducing full dataset to 23 fields")
    print("=" * 80)

    df_full = pd.read_csv(full_path)
    missing = [col for col in SELECTED_23_FEATURES if col not in df_full.columns]
    if missing:
        raise ValueError(f"Missing columns for 23-feature dataset: {missing}")

    df_23 = df_full[SELECTED_23_FEATURES].copy()
    df_23.to_csv(output_path, index=False)

    print(f"Input shape:  {df_full.shape}")
    print(f"Output shape: {df_23.shape}")
    print(f"Saved:        {output_path}")
    return df_23


# DA: DEMO_DATA_44_TO_23_TO_13
# Converts demo 23-field data into the final 13-feature LSTM schema.
def create_13feature_dataset(features23_path: Path, output_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("[3/3] Reducing 23 fields to final 13-feature LSTM schema")
    print("=" * 80)

    df = pd.read_csv(features23_path)
    if "Timestamp" in df.columns:
        timestamp = pd.to_datetime(df["Timestamp"], errors="raise")
        df["Hour"] = timestamp.dt.hour
        df["Day_of_Week"] = timestamp.dt.dayofweek

    missing = [col for col in SELECTED_13_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for 13-feature dataset: {missing}")

    df_13 = df[SELECTED_13_FEATURES].copy()

    numeric_df = df_13.select_dtypes(include=[np.number])
    nan_count = int(df_13.isnull().sum().sum())
    inf_count = int(np.isinf(numeric_df).sum().sum()) if not numeric_df.empty else 0
    if nan_count or inf_count:
        raise ValueError(f"Demo dataset has invalid values: NaN={nan_count}, Inf={inf_count}")

    df_13.to_csv(output_path, index=False)

    print(f"Input shape:  {df.shape}")
    print(f"Output shape: {df_13.shape}")
    print(f"Saved:        {output_path}")
    print("\nFinal columns:")
    for idx, col in enumerate(df_13.columns, start=1):
        role = "target" if col == "Stress_Level" else "feature"
        print(f"  {idx:2d}. {col} ({role})")

    return df_13


def main() -> None:
    args = parse_args()

    full_output = resolve_output(args.full_output)
    features23_output = resolve_output(args.features23_output)
    final_output = resolve_output(args.output)

    for path in [full_output, features23_output, final_output]:
        guard_output_path(path)

    generate_full_dataset(args.start_date, args.days, args.age, args.gender, full_output)
    create_23feature_dataset(full_output, features23_output)
    df_13 = create_13feature_dataset(features23_output, final_output)

    print("\n" + "=" * 80)
    print("[DONE] Demo dataset is ready")
    print("=" * 80)
    print(f"Rows:       {len(df_13):,}")
    print(f"13-feature: {final_output}")
    print("\nNext command:")
    print(f"python -m stress_prediction.demo_predict_stress --input \"{final_output}\" --model tuned")


if __name__ == "__main__":
    main()
