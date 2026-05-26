"""
Run stress prediction demo on a new CSV file.

The script loads an already-trained 13-feature LSTM model together with the
matching scaler and label encoders. It does not retrain the model.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "data" / "demo_stress_test_13features.csv"
DEFAULT_OUTPUT = BASE_DIR / "results" / "demo_predictions.csv"

FEATURE_COLUMNS = [
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
]

TARGET_COLUMN = "Stress_Level"
MODEL_ARTIFACTS = {
    "tuned": {
        "model": BASE_DIR / "models" / "lstm_13features_tuned.keras",
        "scaler": BASE_DIR / "models" / "scaler_13features_tuned.pkl",
        "activity_encoder": BASE_DIR / "models" / "label_encoder_13features_tuned_Activity.pkl",
        "location_encoder": BASE_DIR / "models" / "label_encoder_13features_tuned_Location.pkl",
    },
    "baseline": {
        "model": BASE_DIR / "models" / "lstm_13features_best.keras",
        "scaler": BASE_DIR / "models" / "scaler_13features.pkl",
        "activity_encoder": BASE_DIR / "models" / "label_encoder_13features_Activity.pkl",
        "location_encoder": BASE_DIR / "models" / "label_encoder_13features_Location.pkl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict stress with an already-trained LSTM model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV file to predict.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save predictions.")
    parser.add_argument("--model", choices=["tuned", "baseline"], default="tuned")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--sample-rows", type=int, default=10, help="Rows to print in the terminal summary.")
    parser.add_argument(
        "--show-case-study",
        action="store_true",
        help="Print a representative high-stress case after the aggregate metrics.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else BASE_DIR / path


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifacts(model_name: str):
    artifacts = MODEL_ARTIFACTS[model_name]
    for label, path in artifacts.items():
        require_file(path, label)

    print("=" * 80)
    print("[1/3] Loading trained model and preprocessing artifacts")
    print("=" * 80)
    print(f"Model type: {model_name}")
    print(f"Model:      {artifacts['model']}")
    print(f"Scaler:     {artifacts['scaler']}")

    model = keras.models.load_model(artifacts["model"])
    scaler = load_pickle(artifacts["scaler"])
    activity_encoder = load_pickle(artifacts["activity_encoder"])
    location_encoder = load_pickle(artifacts["location_encoder"])
    return model, scaler, activity_encoder, location_encoder


def add_time_features_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if ("Hour" not in df.columns or "Day_of_Week" not in df.columns) and "Timestamp" in df.columns:
        timestamp = pd.to_datetime(df["Timestamp"], errors="raise")
        df["Hour"] = timestamp.dt.hour
        df["Day_of_Week"] = timestamp.dt.dayofweek
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required feature columns: {missing}")


def encode_category(series: pd.Series, encoder, column_name: str) -> np.ndarray:
    values = series.astype(str)
    allowed = set(encoder.classes_)
    unknown = sorted(set(values.unique()) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown {column_name} values: {unknown}. "
            f"Allowed values from training: {list(encoder.classes_)}"
        )
    return encoder.transform(values)


def preprocess(df: pd.DataFrame, scaler, activity_encoder, location_encoder) -> np.ndarray:
    print("\n" + "=" * 80)
    print("[2/3] Preprocessing input CSV with saved train-set artifacts")
    print("=" * 80)

    df = add_time_features_if_needed(df)
    validate_columns(df)

    X = df[FEATURE_COLUMNS].copy()
    X["Activity"] = encode_category(X["Activity"], activity_encoder, "Activity")
    X["Location"] = encode_category(X["Location"], location_encoder, "Location")

    scaled = scaler.transform(X.values)
    print(f"Rows:     {len(X):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print("Pipeline: encode categorical values -> scale with saved StandardScaler")
    return scaled


def create_sequences(features: np.ndarray, df: pd.DataFrame, sequence_length: int):
    if sequence_length <= 0:
        raise ValueError("--sequence-length must be greater than 0.")
    if len(features) <= sequence_length:
        raise ValueError(
            f"Need more than {sequence_length} rows to create LSTM sequences. "
            f"Current rows: {len(features)}"
        )

    X_seq = []
    for i in range(len(features) - sequence_length):
        X_seq.append(features[i : i + sequence_length])

    # Each prediction uses rows [i, i + sequence_length - 1] as input
    # and predicts the next row [i + sequence_length].
    target_df = df.iloc[sequence_length:].reset_index(drop=True)
    last_input_df = df.iloc[sequence_length - 1 : -1].reset_index(drop=True)
    return np.asarray(X_seq), target_df, last_input_df


def build_output(target_df: pd.DataFrame, last_input_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    keep_cols = [
        col
        for col in [
            "Timestamp",
            "Hour",
            "Day_of_Week",
            "Activity",
            "Location",
            "Heart_Rate",
            "Mood_Score",
            "Energy_Level",
            "Sleep_Duration",
            "Screen_Usage_Current",
            "Phone_Event_Frequency",
            TARGET_COLUMN,
        ]
        if col in target_df.columns
    ]
    output = target_df[keep_cols].copy()
    output.insert(0, "Prediction_Index", np.arange(len(output)))

    input_context_cols = [
        "Hour",
        "Day_of_Week",
        "Activity",
        "Location",
        "Heart_Rate",
        "Mood_Score",
        "Energy_Level",
        "Sleep_Duration",
        "Screen_Usage_Current",
        "Phone_Event_Frequency",
    ]
    for col in input_context_cols:
        if col in last_input_df.columns:
            output[f"Input_Last_{col}"] = last_input_df[col].values[: len(output)]

    output["Predicted_Stress"] = predictions
    output["Predicted_Stress_Clipped"] = np.clip(predictions, 1, 9)

    if TARGET_COLUMN in output.columns:
        output["Absolute_Error"] = np.abs(output[TARGET_COLUMN] - output["Predicted_Stress"])
        output["Absolute_Error_Clipped"] = np.abs(
            output[TARGET_COLUMN] - output["Predicted_Stress_Clipped"]
        )

    return output


def print_metrics_if_available(output: pd.DataFrame) -> None:
    if TARGET_COLUMN not in output.columns:
        print("\nTarget column not found. Metrics are skipped; predictions are still saved.")
        return

    y_true = output[TARGET_COLUMN].values
    y_pred = output["Predicted_Stress"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    print("\nDemo metrics on this CSV:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")


def find_work_stress_case(output: pd.DataFrame):
    if TARGET_COLUMN not in output.columns:
        return None, "No ground-truth Stress_Level column, so case-study selection is skipped."

    required_cols = [
        "Input_Last_Activity",
        "Input_Last_Location",
        "Input_Last_Heart_Rate",
        TARGET_COLUMN,
        "Predicted_Stress_Clipped",
        "Absolute_Error_Clipped",
    ]
    missing = [col for col in required_cols if col not in output.columns]
    if missing:
        return None, f"Missing columns for case-study selection: {missing}"

    candidates = output[
        (output["Input_Last_Activity"] == "Sitting")
        & (output["Input_Last_Location"] == "work")
        & (output["Input_Last_Heart_Rate"].between(88, 100))
        & (output[TARGET_COLUMN].between(7.5, 9.0))
        & (output["Predicted_Stress_Clipped"].between(7.0, 9.0))
    ].copy()

    reason = (
        "Preferred case: last observed input is Sitting at work, heart rate near 95 bpm, "
        "actual next-step stress is high, and prediction is also high."
    )

    if candidates.empty:
        candidates = output[
            (output[TARGET_COLUMN].between(7.5, 9.0))
            & (output["Predicted_Stress_Clipped"].between(7.0, 9.0))
        ].copy()
        reason = (
            "Fallback case: no Sitting/work/HR-near-95 case found in this CSV, "
            "so the script selected a high-actual/high-predicted stress case instead."
        )

    if candidates.empty:
        return None, "No high-stress case found for this CSV."

    candidates["HR_Diff_95"] = (candidates.get("Input_Last_Heart_Rate", 95) - 95).abs()
    candidates["Raw_Overflow"] = np.maximum(candidates["Predicted_Stress"] - 9, 0)
    candidates["Case_Score"] = (
        candidates["Absolute_Error_Clipped"]
        + candidates["HR_Diff_95"] * 0.05
        + candidates["Raw_Overflow"] * 0.1
    )
    best = candidates.sort_values(["Case_Score", "Absolute_Error"]).iloc[0]
    return best, reason


def find_jogging_outdoor_case(output: pd.DataFrame):
    if TARGET_COLUMN not in output.columns:
        return None, "No ground-truth Stress_Level column, so case-study selection is skipped."

    required_cols = [
        "Input_Last_Activity",
        "Input_Last_Location",
        TARGET_COLUMN,
        "Predicted_Stress_Clipped",
        "Absolute_Error_Clipped",
    ]
    missing = [col for col in required_cols if col not in output.columns]
    if missing:
        return None, f"Missing columns for jogging/outdoor case selection: {missing}"

    candidates = output[
        (output["Input_Last_Activity"] == "Jogging")
        & (output["Input_Last_Location"] == "outdoor")
        & (output[TARGET_COLUMN].between(3.0, 6.0))
        & (output["Predicted_Stress_Clipped"].between(3.0, 6.5))
    ].copy()

    reason = (
        "Preferred case: last observed input is Jogging outdoors, and both actual "
        "and predicted next-step stress are in the low-to-medium range."
    )

    if candidates.empty:
        candidates = output[
            (output["Input_Last_Activity"].isin(["Jogging", "Walking"]))
            & (output["Input_Last_Location"].isin(["outdoor", "gym"]))
            & (output[TARGET_COLUMN].between(2.0, 6.5))
            & (output["Predicted_Stress_Clipped"].between(2.0, 7.0))
        ].copy()
        reason = (
            "Fallback case: no exact Jogging/outdoor low-to-medium case found, "
            "so the script selected a similar active-context case."
        )

    if candidates.empty:
        return None, "No suitable active low-to-medium stress case found for this CSV."

    candidates["Distance_To_Mid_Stress"] = (candidates[TARGET_COLUMN] - 4.5).abs()
    candidates["Case_Score"] = (
        candidates["Absolute_Error_Clipped"] + candidates["Distance_To_Mid_Stress"] * 0.05
    )
    best = candidates.sort_values(["Case_Score", "Absolute_Error_Clipped"]).iloc[0]
    return best, reason


def print_single_case(case: pd.Series, title: str, reason: str, sequence_length: int) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)
    print(reason)

    if case is None:
        return

    print("\nHow to read this case:")
    print(f"  The LSTM uses a {sequence_length}-step time window as input.")
    print("  The context below is the LAST observed step inside that input window.")
    print("  The actual and predicted stress values are for the NEXT time step.")

    print("\nLast observed input context:")
    print(f"  Activity:              {case.get('Input_Last_Activity', 'N/A')}")
    print(f"  Location:              {case.get('Input_Last_Location', 'N/A')}")
    print(f"  Hour / Day of week:    {case.get('Input_Last_Hour', 'N/A')} / {case.get('Input_Last_Day_of_Week', 'N/A')}")
    print(f"  Heart rate:            {case.get('Input_Last_Heart_Rate', 'N/A')} bpm")
    print(f"  Mood score:            {case.get('Input_Last_Mood_Score', 'N/A')} / 10")
    print(f"  Energy level:          {case.get('Input_Last_Energy_Level', 'N/A')}")
    print(f"  Sleep duration:        {case.get('Input_Last_Sleep_Duration', 'N/A')} hours")
    print(f"  Screen usage current:  {case.get('Input_Last_Screen_Usage_Current', 'N/A')}")
    print(f"  Phone event frequency: {case.get('Input_Last_Phone_Event_Frequency', 'N/A')}")

    print("\nNext-step stress prediction:")
    print(f"  Actual stress:         {case[TARGET_COLUMN]:.2f} / 9")
    print(f"  Predicted stress raw:  {case['Predicted_Stress']:.2f}")
    print(f"  Predicted stress:      {case['Predicted_Stress_Clipped']:.2f} / 9 after clipping")
    print(f"  Absolute error:        {case['Absolute_Error_Clipped']:.2f} after clipping")
    print(f"  Prediction index:      {int(case['Prediction_Index'])}")


def print_case_studies(output: pd.DataFrame, sequence_length: int) -> None:
    print("\n" + "=" * 80)
    print("CASE STUDIES: Interpretable stress predictions")
    print("=" * 80)

    work_case, work_reason = find_work_stress_case(output)
    print_single_case(
        work_case,
        "CASE 1 - High work-stress context",
        work_reason,
        sequence_length,
    )

    jogging_case, jogging_reason = find_jogging_outdoor_case(output)
    print_single_case(
        jogging_case,
        "CASE 2 - Outdoor jogging low-to-medium stress context",
        jogging_reason,
        sequence_length,
    )


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    require_file(input_path, "input CSV")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, scaler, activity_encoder, location_encoder = load_artifacts(args.model)

    raw_df = pd.read_csv(input_path)
    raw_df = add_time_features_if_needed(raw_df)
    scaled = preprocess(raw_df, scaler, activity_encoder, location_encoder)

    print("\n" + "=" * 80)
    print("[3/3] Creating LSTM windows and predicting stress")
    print("=" * 80)

    X_seq, target_df, last_input_df = create_sequences(scaled, raw_df, args.sequence_length)
    predictions = model.predict(X_seq, verbose=0).flatten()
    output = build_output(target_df, last_input_df, predictions)
    output.to_csv(output_path, index=False)

    print(f"Input shape:     {scaled.shape}")
    print(f"LSTM sequences:  {X_seq.shape}")
    print(f"Predictions:     {len(predictions):,}")
    print(f"Saved results:   {output_path}")

    print_metrics_if_available(output)

    print("\nSample predictions:")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(output.head(args.sample_rows).to_string(index=False))

    if args.show_case_study:
        print_case_studies(output, args.sequence_length)


if __name__ == "__main__":
    main()
