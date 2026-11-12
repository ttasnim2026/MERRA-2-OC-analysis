# -*- coding: utf-8 -*-
"""Post-selection robustness and ablation analysis for Bangladesh OCSMASS.

This script intentionally excludes every 2025 row before any analysis. It does
not modify or replace the locked primary model and does not revise the reported
2025 final-test result. All analyses below use 2019-2024 development data only.
"""

from pathlib import Path
import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# =============================================================================
# 1. PATHS AND SETTINGS
# =============================================================================

PROJECT_FOLDER = Path(
    os.environ.get(
        "OA_PROJECT_FOLDER",
        str(Path.cwd()),
    )
)

INPUT_FILE = (
    PROJECT_FOLDER
    / "EDA_Results_2019_2025"
    / "Model_Data"
    / "Bangladesh_OA_Model_Features_2019_2025.csv"
)

OUTPUT_FOLDER = PROJECT_FOLDER / "Robustness_Results_2019_2024"
TABLE_FOLDER = OUTPUT_FOLDER / "Tables"
FIGURE_FOLDER = OUTPUT_FOLDER / "Figures"

TARGET_RAW = "OCSMASS_ug_m3"
TARGET_LOG = "log1p_OCSMASS"
RANDOM_SEED = 42

# This environment option is only for a short technical test. Leave it unset
# for the full analysis.
QUICK_TEST = os.environ.get("OA_QUICK_TEST", "0") == "1"
RF_TREES = 20 if QUICK_TEST else 250
HGB_ITERATIONS = 30 if QUICK_TEST else 300

CALENDAR_LOCATION_FEATURES = [
    "latitude",
    "longitude",
    "day_of_year_sin",
    "day_of_year_cos",
]

METEOROLOGY_FEATURES = [
    "temperature_C",
    "relative_humidity_percent",
    "boundary_layer_height_m",
    "wind_speed_m_s",
    "wind_direction_sin",
    "wind_direction_cos",
    "total_precipitation_mm",
    "solar_radiation_MJ_m2_day",
]

SATELLITE_FEATURES = [
    "HCHO",
    "NO2",
    "HCHO_missing_flag",
    "NO2_missing_flag",
    "HCHO_negative_flag",
    "NO2_negative_flag",
]

FULL_FEATURES = (
    CALENDAR_LOCATION_FEATURES
    + METEOROLOGY_FEATURES
    + SATELLITE_FEATURES
)

FEATURE_GROUPS = {
    "Calendar + location": CALENDAR_LOCATION_FEATURES,
    "Calendar/location + meteorology": (
        CALENDAR_LOCATION_FEATURES + METEOROLOGY_FEATURES
    ),
    "Calendar/location + satellite": (
        CALENDAR_LOCATION_FEATURES + SATELLITE_FEATURES
    ),
    "Full model": FULL_FEATURES,
}

EXPANDING_FOLDS = [
    {
        "fold": "Train 2019-2021, validate 2022",
        "train_start": 2019,
        "train_end": 2021,
        "validation_year": 2022,
    },
    {
        "fold": "Train 2019-2022, validate 2023",
        "train_start": 2019,
        "train_end": 2022,
        "validation_year": 2023,
    },
    {
        "fold": "Train 2019-2023, validate 2024",
        "train_start": 2019,
        "train_end": 2023,
        "validation_year": 2024,
    },
]


# =============================================================================
# 2. HELPERS
# =============================================================================

def ensure_output_folders() -> None:
    for folder in [OUTPUT_FOLDER, TABLE_FOLDER, FIGURE_FOLDER]:
        folder.mkdir(parents=True, exist_ok=True)


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_FOLDER / filename, dpi=300, bbox_inches="tight")
    plt.close()


def elapsed_text(start_time: float) -> str:
    elapsed = time.perf_counter() - start_time
    if elapsed < 60:
        return f"{elapsed:.1f} seconds"
    return f"{elapsed / 60:.1f} minutes"


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "date",
        "year",
        "grid_id",
        TARGET_RAW,
        TARGET_LOG,
        *FULL_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    duplicates = int(df.duplicated(["date", "grid_id"]).sum())
    if duplicates:
        raise ValueError(f"Found {duplicates:,} duplicate date-grid rows.")

    if df[TARGET_RAW].isna().any() or df[TARGET_LOG].isna().any():
        raise ValueError("The target contains missing values.")

    if (df[TARGET_RAW] < 0).any():
        raise ValueError("The OCSMASS target contains negative values.")

    expected_log = np.log1p(df[TARGET_RAW].to_numpy())
    if not np.allclose(expected_log, df[TARGET_LOG], rtol=1e-7, atol=1e-8):
        raise ValueError("The log target is inconsistent with raw OCSMASS.")


def prepare_data(
    source: pd.DataFrame,
    negative_treatment: str,
) -> pd.DataFrame:
    """Create an in-memory modeling copy with specified satellite treatment."""
    data = source.copy()

    for variable in ["HCHO", "NO2"]:
        data[f"{variable}_negative_flag"] = (
            data[variable] < 0
        ).astype("int8")
        data[f"{variable}_missing_flag"] = (
            data[variable].isna()
        ).astype("int8")

        if negative_treatment == "set_negative_to_missing":
            data.loc[data[variable] < 0, variable] = np.nan
        elif negative_treatment == "retain_negative":
            pass
        else:
            raise ValueError(
                "negative_treatment must be 'set_negative_to_missing' "
                "or 'retain_negative'."
            )

    return data


def build_hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=HGB_ITERATIONS,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=0.5,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def build_random_forest() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=RF_TREES,
                    max_depth=22,
                    min_samples_leaf=5,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def metrics_from_log_prediction(
    y_true_raw: np.ndarray,
    predicted_log: np.ndarray,
) -> dict:
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_true_log = np.log1p(y_true_raw)
    predicted_log = np.asarray(predicted_log, dtype=float)
    predicted_raw = np.maximum(np.expm1(predicted_log), 0.0)
    return calculate_metrics(y_true_raw, y_true_log, predicted_raw, predicted_log)


def metrics_from_raw_prediction(
    y_true_raw: np.ndarray,
    predicted_raw: np.ndarray,
) -> dict:
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    predicted_raw = np.maximum(np.asarray(predicted_raw, dtype=float), 0.0)
    y_true_log = np.log1p(y_true_raw)
    predicted_log = np.log1p(predicted_raw)
    return calculate_metrics(y_true_raw, y_true_log, predicted_raw, predicted_log)


def calculate_metrics(
    y_true_raw: np.ndarray,
    y_true_log: np.ndarray,
    predicted_raw: np.ndarray,
    predicted_log: np.ndarray,
) -> dict:
    if np.std(predicted_raw) == 0 or np.std(y_true_raw) == 0:
        pearson_r = np.nan
    else:
        pearson_r = float(np.corrcoef(y_true_raw, predicted_raw)[0, 1])

    return {
        "n_rows": len(y_true_raw),
        "MAE_ug_m3": mean_absolute_error(y_true_raw, predicted_raw),
        "RMSE_ug_m3": np.sqrt(mean_squared_error(y_true_raw, predicted_raw)),
        "R2_raw": r2_score(y_true_raw, predicted_raw),
        "mean_bias_ug_m3": float(np.mean(predicted_raw - y_true_raw)),
        "Pearson_r_raw": pearson_r,
        "MAE_log": mean_absolute_error(y_true_log, predicted_log),
        "RMSE_log": np.sqrt(mean_squared_error(y_true_log, predicted_log)),
        "R2_log": r2_score(y_true_log, predicted_log),
    }


def fit_and_evaluate(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target_scale: str = "log1p",
) -> dict:
    if model_name == "Histogram Gradient Boosting":
        model = build_hgb()
    elif model_name == "Random Forest":
        model = build_random_forest()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if target_scale == "log1p":
        model.fit(train[features], train[TARGET_LOG])
        prediction = model.predict(validation[features])
        metrics = metrics_from_log_prediction(
            validation[TARGET_RAW].to_numpy(), prediction
        )
    elif target_scale == "raw":
        model.fit(train[features], train[TARGET_RAW])
        prediction = model.predict(validation[features])
        metrics = metrics_from_raw_prediction(
            validation[TARGET_RAW].to_numpy(), prediction
        )
    else:
        raise ValueError("target_scale must be 'log1p' or 'raw'.")

    return metrics


# =============================================================================
# 3. EXPANDING-YEAR MODEL COMPARISON
# =============================================================================

def run_expanding_year_comparison(
    primary_data: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for fold in EXPANDING_FOLDS:
        train = primary_data[
            primary_data["year"].between(
                fold["train_start"], fold["train_end"]
            )
        ]
        validation = primary_data[
            primary_data["year"] == fold["validation_year"]
        ]

        for model_name in ["Histogram Gradient Boosting", "Random Forest"]:
            print(f"  {fold['fold']} — {model_name}")
            start = time.perf_counter()
            metrics = fit_and_evaluate(
                model_name,
                train,
                validation,
                FULL_FEATURES,
                target_scale="log1p",
            )
            records.append(
                {
                    "analysis": "expanding_year_model_comparison",
                    "fold": fold["fold"],
                    "train_start_year": fold["train_start"],
                    "train_end_year": fold["train_end"],
                    "validation_year": fold["validation_year"],
                    "model": model_name,
                    "feature_group": "Full model",
                    "target_scale": "log1p",
                    "negative_treatment": "set_negative_to_missing",
                    **metrics,
                }
            )
            print(f"    completed in {elapsed_text(start)}")

    return pd.DataFrame(records)


# =============================================================================
# 4. FEATURE-GROUP ABLATION ACROSS ALL FOLDS
# =============================================================================

def run_feature_ablation(primary_data: pd.DataFrame) -> pd.DataFrame:
    records = []

    for fold in EXPANDING_FOLDS:
        train = primary_data[
            primary_data["year"].between(
                fold["train_start"], fold["train_end"]
            )
        ]
        validation = primary_data[
            primary_data["year"] == fold["validation_year"]
        ]

        for group_name, features in FEATURE_GROUPS.items():
            print(f"  {fold['fold']} — {group_name}")
            start = time.perf_counter()
            metrics = fit_and_evaluate(
                "Histogram Gradient Boosting",
                train,
                validation,
                features,
                target_scale="log1p",
            )
            records.append(
                {
                    "analysis": "feature_group_ablation",
                    "fold": fold["fold"],
                    "train_start_year": fold["train_start"],
                    "train_end_year": fold["train_end"],
                    "validation_year": fold["validation_year"],
                    "model": "Histogram Gradient Boosting",
                    "feature_group": group_name,
                    "n_features": len(features),
                    "target_scale": "log1p",
                    "negative_treatment": "set_negative_to_missing",
                    **metrics,
                }
            )
            print(f"    completed in {elapsed_text(start)}")

    return pd.DataFrame(records)


# =============================================================================
# 5. 2024 SENSITIVITY ANALYSES
# =============================================================================

def run_sensitivity_2024(
    source_development: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    specifications = [
        {
            "specification": "Primary: HGB, log target, negatives missing",
            "model": "Histogram Gradient Boosting",
            "target_scale": "log1p",
            "negative_treatment": "set_negative_to_missing",
        },
        {
            "specification": "Retain negative satellite retrievals",
            "model": "Histogram Gradient Boosting",
            "target_scale": "log1p",
            "negative_treatment": "retain_negative",
        },
        {
            "specification": "Model raw OCSMASS target",
            "model": "Histogram Gradient Boosting",
            "target_scale": "raw",
            "negative_treatment": "set_negative_to_missing",
        },
        {
            "specification": "Random Forest, log target",
            "model": "Random Forest",
            "target_scale": "log1p",
            "negative_treatment": "set_negative_to_missing",
        },
    ]

    for specification in specifications:
        print(f"  {specification['specification']}")
        data = prepare_data(
            source_development,
            specification["negative_treatment"],
        )
        train = data[data["year"].between(2019, 2023)]
        validation = data[data["year"] == 2024]

        start = time.perf_counter()
        metrics = fit_and_evaluate(
            specification["model"],
            train,
            validation,
            FULL_FEATURES,
            target_scale=specification["target_scale"],
        )
        records.append(
            {
                "analysis": "sensitivity_2024",
                **specification,
                **metrics,
            }
        )
        print(f"    completed in {elapsed_text(start)}")

    return pd.DataFrame(records)


# =============================================================================
# 6. SUMMARY TABLES AND FIGURES
# =============================================================================

def summarize_expanding_results(expanding: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "MAE_ug_m3",
        "RMSE_ug_m3",
        "R2_raw",
        "mean_bias_ug_m3",
        "Pearson_r_raw",
        "MAE_log",
        "RMSE_log",
        "R2_log",
    ]
    summary = (
        expanding.groupby("model", observed=True)[metric_columns]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def summarize_ablation(ablation: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "MAE_ug_m3",
        "RMSE_ug_m3",
        "R2_raw",
        "Pearson_r_raw",
        "RMSE_log",
        "R2_log",
    ]
    summary = (
        ablation.groupby("feature_group", observed=True)[metric_columns]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    calendar_rmse = float(
        summary.loc[
            summary["feature_group"] == "Calendar + location",
            "RMSE_log_mean",
        ].iloc[0]
    )
    summary["log_RMSE_reduction_vs_calendar_location_percent"] = (
        (calendar_rmse - summary["RMSE_log_mean"]) / calendar_rmse * 100
    )
    return summary.sort_values("RMSE_log_mean")


def make_figures(
    expanding: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # Expanding-year log-RMSE by model.
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in expanding.groupby("model", observed=True):
        group = group.sort_values("validation_year")
        ax.plot(
            group["validation_year"],
            group["RMSE_log"],
            marker="o",
            linewidth=2,
            label=model_name,
        )
    ax.set_xticks([2022, 2023, 2024])
    ax.set_xlabel("Validation year")
    ax.set_ylabel("Log-scale RMSE")
    ax.set_title("Expanding-year validation performance")
    ax.legend(frameon=False)
    save_figure("01_expanding_year_model_comparison.png")

    # Feature-group ablation averaged over validation years.
    ordered = ablation_summary.sort_values("RMSE_log_mean", ascending=False)
    plt.figure(figsize=(10, 5.5))
    plt.barh(
        ordered["feature_group"],
        ordered["RMSE_log_mean"],
        xerr=ordered["RMSE_log_std"].fillna(0),
        color="#477f92",
        alpha=0.9,
    )
    plt.xlabel("Mean log-scale RMSE across 2022–2024")
    plt.title("Feature-group ablation: Histogram Gradient Boosting")
    save_figure("02_feature_group_ablation.png")

    # Sensitivity of 2024 validation performance.
    sensitivity_plot = sensitivity.sort_values("RMSE_log", ascending=False)
    plt.figure(figsize=(10, 5.5))
    plt.barh(
        sensitivity_plot["specification"],
        sensitivity_plot["RMSE_log"],
        color="#d2833e",
        alpha=0.9,
    )
    plt.xlabel("2024 validation log-scale RMSE")
    plt.title("Preprocessing and model sensitivity")
    save_figure("03_preprocessing_model_sensitivity.png")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.random.seed(RANDOM_SEED)
    ensure_output_folders()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature table was not found:\n{INPUT_FILE}\n\n"
            "Run the EDA feature-preparation script first or correct "
            "PROJECT_FOLDER at the beginning of this script."
        )

    print("Reading the feature table...")
    source = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
    source["date"] = source["date"].dt.normalize()
    source["grid_id"] = source["grid_id"].astype(str).str.strip()
    validate_input(source)

    # Critical leakage guard: remove every 2025 row before analysis.
    development_source = source[source["year"].between(2019, 2024)].copy()
    del source

    if len(development_source) != 149_056:
        raise ValueError(
            f"Expected 149,056 development rows, found "
            f"{len(development_source):,}."
        )
    if development_source["year"].max() != 2024:
        raise RuntimeError("The leakage guard failed: post-2024 rows remain.")

    primary_data = prepare_data(
        development_source,
        negative_treatment="set_negative_to_missing",
    )

    print("\nA. Expanding-year HGB and Random Forest comparison")
    expanding = run_expanding_year_comparison(primary_data)
    expanding.to_csv(
        TABLE_FOLDER / "01_expanding_year_model_metrics.csv", index=False
    )
    expanding_summary = summarize_expanding_results(expanding)
    expanding_summary.to_csv(
        TABLE_FOLDER / "02_expanding_year_model_summary.csv", index=False
    )

    print("\nB. Feature-group ablation across 2022-2024")
    ablation = run_feature_ablation(primary_data)
    ablation.to_csv(
        TABLE_FOLDER / "03_feature_group_ablation_metrics.csv", index=False
    )
    ablation_summary = summarize_ablation(ablation)
    ablation_summary.to_csv(
        TABLE_FOLDER / "04_feature_group_ablation_summary.csv", index=False
    )

    print("\nC. Preprocessing and target sensitivity using 2024")
    sensitivity = run_sensitivity_2024(development_source)
    sensitivity.to_csv(
        TABLE_FOLDER / "05_preprocessing_model_sensitivity_2024.csv",
        index=False,
    )

    feature_group_rows = []
    for group_name, features in FEATURE_GROUPS.items():
        for feature in features:
            feature_group_rows.append(
                {
                    "feature_group": group_name,
                    "feature": feature,
                }
            )
    pd.DataFrame(feature_group_rows).to_csv(
        TABLE_FOLDER / "06_feature_group_definitions.csv", index=False
    )

    analysis_record = {
        "purpose": "Post-selection robustness and ablation analysis",
        "years_used": [2019, 2020, 2021, 2022, 2023, 2024],
        "year_2025_used": False,
        "locked_primary_model_changed": False,
        "primary_model": "Histogram Gradient Boosting",
        "primary_target": "log1p_OCSMASS",
        "negative_satellite_primary_treatment": "set negative values to missing",
        "note": (
            "Results are robustness evidence only. They must not be used to "
            "retune the already locked model based on the known 2025 result."
        ),
    }
    (TABLE_FOLDER / "07_analysis_design_record.json").write_text(
        json.dumps(analysis_record, indent=2), encoding="utf-8"
    )

    make_figures(expanding, ablation_summary, sensitivity)

    best_ablation = ablation_summary.iloc[0]
    hgb_summary = expanding_summary[
        expanding_summary["model"] == "Histogram Gradient Boosting"
    ].iloc[0]
    rf_summary = expanding_summary[
        expanding_summary["model"] == "Random Forest"
    ].iloc[0]

    report_lines = [
        "Bangladesh OCSMASS robustness and feature-ablation report",
        "",
        "Leakage protection:",
        "Only 2019-2024 data were used.",
        "All 2025 records were removed before model fitting or evaluation.",
        "The locked primary model and reported 2025 result were not changed.",
        "",
        "Expanding-year validation (mean across 2022, 2023, and 2024):",
        (
            "Histogram Gradient Boosting mean log-RMSE: "
            f"{hgb_summary['RMSE_log_mean']:.6f}"
        ),
        (
            "Random Forest mean log-RMSE: "
            f"{rf_summary['RMSE_log_mean']:.6f}"
        ),
        (
            "Histogram Gradient Boosting mean raw R²: "
            f"{hgb_summary['R2_raw_mean']:.6f}"
        ),
        (
            "Random Forest mean raw R²: "
            f"{rf_summary['R2_raw_mean']:.6f}"
        ),
        "",
        "Feature-group ablation:",
        f"Lowest mean log-RMSE feature group: {best_ablation['feature_group']}",
        f"Mean log-RMSE: {best_ablation['RMSE_log_mean']:.6f}",
        (
            "Reduction relative to calendar/location-only: "
            f"{best_ablation['log_RMSE_reduction_vs_calendar_location_percent']:.2f}%"
        ),
        "",
        "Interpretation warning:",
        "Predictive improvement does not establish causality.",
        "The target is MERRA-2 surface organic carbon concentration, not direct SOA.",
    ]
    (OUTPUT_FOLDER / "Robustness_run_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 78)
    print("ROBUSTNESS AND ABLATION ANALYSIS COMPLETED")
    print("=" * 78)
    print("Only 2019-2024 data were used; 2025 was excluded.")
    print(f"Tables saved in: {TABLE_FOLDER}")
    print(f"Figures saved in: {FIGURE_FOLDER}")
    print("The locked model and original feature table were not modified.")


if __name__ == "__main__":
    main()
