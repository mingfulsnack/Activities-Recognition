"""
Validate the usefulness of the Context-Stress Modifier.

This script is designed for thesis defense evidence, not for model training.
It produces three complementary checks:
1. Rule-level counterfactuals: same base stress, different context.
2. Dataset-level slices: stress differs by activity/location, even under similar HR.
3. Model-level ablation: break context features and observe metric degradation.

Outputs are written to results/context_modifier_validation/.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "context_modifier_validation"
DATA_PATH = PROJECT_ROOT / "data" / "optimized_health_data_13features.csv"
MODEL_DIR = PROJECT_ROOT / "models"

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
CATEGORICAL_FEATURES = ["Activity", "Location"]
SEQ_LENGTH = 60
RANDOM_SEED = 42

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from core.context_stress_modifier import ContextStressModifier


def ensure_inputs() -> None:
    required_files = [
        DATA_PATH,
        MODEL_DIR / "lstm_13features_tuned.keras",
        MODEL_DIR / "scaler_13features_tuned.pkl",
        MODEL_DIR / "label_encoder_13features_tuned_Activity.pkl",
        MODEL_DIR / "label_encoder_13features_tuned_Location.pkl",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def clip_stress(value: float) -> float:
    return max(1.0, min(9.0, value))


def run_rule_level_counterfactuals(n_runs: int = 300) -> pd.DataFrame:
    print("\n[1/4] Rule-level counterfactual validation")

    scenarios = [
        {
            "Scenario": "Sitting at work under high workload",
            "Activity": "Sitting",
            "Location": "work",
            "Hour": 14,
            "Work_Intensity": "high",
            "Is_Weekend": False,
            "Sleep_Duration": 6.5,
            "Noise": "crowded_space",
            "Social": "conflict",
            "Base_Stress": 5.0,
        },
        {
            "Scenario": "Sitting at home in relaxed evening",
            "Activity": "Sitting",
            "Location": "home",
            "Hour": 21,
            "Work_Intensity": "low",
            "Is_Weekend": False,
            "Sleep_Duration": 8.0,
            "Noise": "quiet_environment",
            "Social": "supportive",
            "Base_Stress": 5.0,
        },
        {
            "Scenario": "Walking during commute rush",
            "Activity": "Walking",
            "Location": "commute",
            "Hour": 8,
            "Work_Intensity": "normal",
            "Is_Weekend": False,
            "Sleep_Duration": 7.0,
            "Noise": "heavy_traffic",
            "Social": None,
            "Base_Stress": 5.0,
        },
        {
            "Scenario": "Walking outdoors in evening",
            "Activity": "Walking",
            "Location": "outdoor",
            "Hour": 18,
            "Work_Intensity": "low",
            "Is_Weekend": False,
            "Sleep_Duration": 8.0,
            "Noise": "nature_sounds",
            "Social": None,
            "Base_Stress": 5.0,
        },
        {
            "Scenario": "Jogging outdoors after work",
            "Activity": "Jogging",
            "Location": "outdoor",
            "Hour": 18,
            "Work_Intensity": "normal",
            "Is_Weekend": False,
            "Sleep_Duration": 8.0,
            "Noise": "nature_sounds",
            "Social": None,
            "Base_Stress": 5.0,
        },
    ]

    rows = []
    random.seed(RANDOM_SEED)
    for scenario in scenarios:
        modifiers = []
        final_stresses = []
        for _ in range(n_runs):
            modifier = ContextStressModifier.calculate_context_stress_modifier(
                activity=scenario["Activity"],
                location=scenario["Location"],
                hour=scenario["Hour"],
                work_intensity=scenario["Work_Intensity"],
                is_weekend=scenario["Is_Weekend"],
                sleep_duration=scenario["Sleep_Duration"],
                noise_environment=scenario["Noise"],
                social_context=scenario["Social"],
            )
            modifiers.append(modifier)
            final_stresses.append(clip_stress(scenario["Base_Stress"] + modifier))

        rows.append(
            {
                **scenario,
                "Runs": n_runs,
                "Modifier_Mean": np.mean(modifiers),
                "Modifier_Std": np.std(modifiers),
                "Modifier_Min": np.min(modifiers),
                "Modifier_Max": np.max(modifiers),
                "Final_Stress_Mean": np.mean(final_stresses),
                "Final_Stress_Std": np.std(final_stresses),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "scenario_counterfactuals.csv", index=False)
    return result


def run_dataset_context_slices(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/4] Dataset-level context slice validation")

    all_context = (
        df.groupby(["Activity", "Location"])["Stress_Level"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    all_context.insert(0, "Slice", "all_rows")

    hr_controlled = df[df["Heart_Rate"].between(85, 100)].copy()
    hr_context = (
        hr_controlled.groupby(["Activity", "Location"])["Stress_Level"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    hr_context.insert(0, "Slice", "heart_rate_85_100")

    result = pd.concat([all_context, hr_context], ignore_index=True)
    result = result.rename(
        columns={
            "mean": "Stress_Mean",
            "std": "Stress_Std",
            "count": "Count",
        }
    )
    result = result.sort_values(["Slice", "Stress_Mean"], ascending=[True, False])
    result.to_csv(RESULTS_DIR / "dataset_context_slices.csv", index=False)
    return result


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def split_test_data(df: pd.DataFrame):
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].values
    n = len(df)
    val_end = int(n * 0.85)
    return x.iloc[val_end:].reset_index(drop=True), y[val_end:]


def encode_scale_sequence(x_raw: pd.DataFrame, y_raw: np.ndarray, scaler, encoders):
    x = x_raw.copy()
    for col in CATEGORICAL_FEATURES:
        encoder = encoders[col]
        x[col] = encoder.transform(x[col].astype(str))
    x_scaled = scaler.transform(x.values)

    x_seq = []
    y_seq = []
    for idx in range(len(x_scaled) - SEQ_LENGTH):
        x_seq.append(x_scaled[idx : idx + SEQ_LENGTH])
        y_seq.append(y_raw[idx + SEQ_LENGTH])
    return np.asarray(x_seq), np.asarray(y_seq)


def evaluate_model(model, x_seq: np.ndarray, y_seq: np.ndarray) -> dict:
    y_pred = model.predict(x_seq, verbose=0).flatten()
    return {
        "MAE": mean_absolute_error(y_seq, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_seq, y_pred))),
        "R2": r2_score(y_seq, y_pred),
        "Num_Sequences": len(y_seq),
    }


def run_model_context_ablation(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/4] Model-level context ablation validation")

    from tensorflow import keras

    model = keras.models.load_model(MODEL_DIR / "lstm_13features_tuned.keras")
    scaler = load_pickle(MODEL_DIR / "scaler_13features_tuned.pkl")
    encoders = {
        "Activity": load_pickle(MODEL_DIR / "label_encoder_13features_tuned_Activity.pkl"),
        "Location": load_pickle(MODEL_DIR / "label_encoder_13features_tuned_Location.pkl"),
    }

    x_test_raw, y_test_raw = split_test_data(df)
    rng = np.random.default_rng(RANDOM_SEED)

    experiments = [
        ("baseline", []),
        ("permute_location", ["Location"]),
        ("permute_activity", ["Activity"]),
        ("permute_time", ["Hour", "Day_of_Week"]),
        ("permute_full_context", ["Location", "Activity", "Hour", "Day_of_Week"]),
    ]

    rows = []
    baseline_metrics = None
    for experiment_name, columns in experiments:
        x_ablation = x_test_raw.copy()
        for col in columns:
            x_ablation[col] = rng.permutation(x_ablation[col].values)

        x_seq, y_seq = encode_scale_sequence(x_ablation, y_test_raw, scaler, encoders)
        metrics = evaluate_model(model, x_seq, y_seq)

        if experiment_name == "baseline":
            baseline_metrics = metrics

        rows.append(
            {
                "Experiment": experiment_name,
                "Permuted_Features": ", ".join(columns) if columns else "None",
                **metrics,
            }
        )

    result = pd.DataFrame(rows)
    if baseline_metrics:
        result["Delta_MAE_vs_Baseline"] = result["MAE"] - baseline_metrics["MAE"]
        result["Delta_RMSE_vs_Baseline"] = result["RMSE"] - baseline_metrics["RMSE"]
        result["Delta_R2_vs_Baseline"] = result["R2"] - baseline_metrics["R2"]

    result.to_csv(RESULTS_DIR / "model_context_ablation.csv", index=False)
    return result


def lookup_slice(
    dataset_slices: pd.DataFrame, slice_name: str, activity: str, location: str
) -> dict | None:
    rows = dataset_slices[
        (dataset_slices["Slice"] == slice_name)
        & (dataset_slices["Activity"] == activity)
        & (dataset_slices["Location"] == location)
    ]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def fmt_num(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def create_markdown_report(
    scenario_df: pd.DataFrame,
    dataset_slices: pd.DataFrame,
    model_ablation: pd.DataFrame,
) -> None:
    print("[4/4] Writing markdown defense report")

    sitting_work = lookup_slice(dataset_slices, "all_rows", "Sitting", "work")
    sitting_home = lookup_slice(dataset_slices, "all_rows", "Sitting", "home")
    walking_work = lookup_slice(dataset_slices, "all_rows", "Walking", "work")
    walking_outdoor = lookup_slice(dataset_slices, "all_rows", "Walking", "outdoor")
    sitting_work_hr = lookup_slice(dataset_slices, "heart_rate_85_100", "Sitting", "work")
    walking_outdoor_hr = lookup_slice(dataset_slices, "heart_rate_85_100", "Walking", "outdoor")

    baseline = model_ablation[model_ablation["Experiment"] == "baseline"].iloc[0]
    full_context = model_ablation[model_ablation["Experiment"] == "permute_full_context"].iloc[0]
    location_perm = model_ablation[model_ablation["Experiment"] == "permute_location"].iloc[0]

    report = f"""# Context-Stress Modifier Validation

## Kết luận ngắn

Context-Stress Modifier không nên được trình bày như một công thức y sinh chính xác. Nó là một cơ chế heuristic có kiểm soát để mã hóa ý tưởng: cùng một hoạt động hoặc cùng mức nhịp tim có thể mang ý nghĩa stress khác nhau tùy bối cảnh.

## 1. Rule-level counterfactual

Script giữ `base_stress = 5.0` và chỉ thay đổi tổ hợp `Activity x Location x Time x Workload x Environment/Social`. Kết quả nằm trong `scenario_counterfactuals.csv`.

| Scenario | Modifier mean | Final stress mean |
|---|---:|---:|
"""

    for _, row in scenario_df.iterrows():
        report += (
            f"| {row['Scenario']} | {fmt_num(row['Modifier_Mean'])} | "
            f"{fmt_num(row['Final_Stress_Mean'])} |\n"
        )

    report += f"""
Ý nghĩa: cùng stress nền, bối cảnh `work/high workload/conflict` làm stress tăng, còn `home/quiet/supportive` hoặc `outdoor/nature` làm stress giảm. Đây là bằng chứng cơ chế modifier có tác động đúng chiều, không chỉ là nhiễu ngẫu nhiên.

## 2. Dataset-level evidence

Các thống kê từ `data/optimized_health_data_13features.csv` cho thấy context tạo ra khác biệt rõ trong phân phối stress:

- `Sitting/work`: mean stress `{fmt_num(sitting_work['Stress_Mean'] if sitting_work else None)}`, trong khi `Sitting/home`: `{fmt_num(sitting_home['Stress_Mean'] if sitting_home else None)}`.
- `Walking/work`: mean stress `{fmt_num(walking_work['Stress_Mean'] if walking_work else None)}`, trong khi `Walking/outdoor`: `{fmt_num(walking_outdoor['Stress_Mean'] if walking_outdoor else None)}`.
- Khi giữ HR trong khoảng `85-100 bpm`, `Sitting/work` vẫn có mean stress `{fmt_num(sitting_work_hr['Stress_Mean'] if sitting_work_hr else None)}`, còn `Walking/outdoor` là `{fmt_num(walking_outdoor_hr['Stress_Mean'] if walking_outdoor_hr else None)}`.

Ý nghĩa: modifier giúp tránh việc diễn giải máy móc rằng “HR cao luôn là stress cao”. HR cao khi vận động ngoài trời có thể khác với HR cao khi ngồi làm việc ở môi trường áp lực.

## 3. Model-level evidence

Model tuned được đánh giá lại trên test split, sau đó hoán vị các feature context để phá quan hệ ngữ cảnh mà không train lại model.

| Experiment | MAE | RMSE | R2 | Delta MAE | Delta R2 |
|---|---:|---:|---:|---:|---:|
"""

    for _, row in model_ablation.iterrows():
        report += (
            f"| {row['Experiment']} | {fmt_num(row['MAE'], 4)} | "
            f"{fmt_num(row['RMSE'], 4)} | {fmt_num(row['R2'], 4)} | "
            f"{fmt_num(row['Delta_MAE_vs_Baseline'], 4)} | "
            f"{fmt_num(row['Delta_R2_vs_Baseline'], 4)} |\n"
        )

    report += f"""
Baseline của model: MAE `{fmt_num(baseline['MAE'], 4)}`, R2 `{fmt_num(baseline['R2'], 4)}`. Khi phá toàn bộ nhóm context (`Location`, `Activity`, `Hour`, `Day_of_Week`), MAE đổi `{fmt_num(full_context['Delta_MAE_vs_Baseline'], 4)}` và R2 đổi `{fmt_num(full_context['Delta_R2_vs_Baseline'], 4)}`. Khi chỉ phá `Location`, MAE đổi `{fmt_num(location_perm['Delta_MAE_vs_Baseline'], 4)}`.

## Câu trả lời khi hội đồng hỏi

Em không khẳng định modifier là công thức lâm sàng. Đây là cơ chế mô phỏng có kiểm soát, dựa trên nguyên lý stress phụ thuộc bối cảnh. Tác dụng của nó được kiểm tra theo ba lớp: rule-level cho thấy cùng stress nền nhưng context khác tạo delta khác; dataset-level cho thấy cùng activity/HR nhưng location khác có stress khác; model-level cho thấy khi phá context features thì kết quả dự đoán thay đổi. Vì vậy modifier có vai trò làm dữ liệu nhạy với ngữ cảnh, thay vì chỉ cộng một nhiễu heuristic vô nghĩa.
"""

    (RESULTS_DIR / "context_modifier_validation.md").write_text(report, encoding="utf-8")


def save_run_summary(
    scenario_df: pd.DataFrame,
    dataset_slices: pd.DataFrame,
    model_ablation: pd.DataFrame,
) -> None:
    summary = {
        "data_path": str(DATA_PATH),
        "num_scenarios": int(len(scenario_df)),
        "num_dataset_slices": int(len(dataset_slices)),
        "model_ablation_experiments": model_ablation["Experiment"].tolist(),
        "baseline_metrics": model_ablation[model_ablation["Experiment"] == "baseline"]
        .iloc[0][["MAE", "RMSE", "R2"]]
        .to_dict(),
    }
    (RESULTS_DIR / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    print("=" * 80)
    print("CONTEXT-STRESS MODIFIER VALIDATION")
    print("=" * 80)
    print("No model retraining. No source dataset modification.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_inputs()

    df = pd.read_csv(DATA_PATH)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    scenario_df = run_rule_level_counterfactuals()
    dataset_slices = run_dataset_context_slices(df)
    model_ablation = run_model_context_ablation(df)
    create_markdown_report(scenario_df, dataset_slices, model_ablation)
    save_run_summary(scenario_df, dataset_slices, model_ablation)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Results directory: {RESULTS_DIR}")
    print("Generated files:")
    for filename in [
        "scenario_counterfactuals.csv",
        "dataset_context_slices.csv",
        "model_context_ablation.csv",
        "context_modifier_validation.md",
        "validation_summary.json",
    ]:
        print(f"  - {RESULTS_DIR / filename}")


if __name__ == "__main__":
    main()
