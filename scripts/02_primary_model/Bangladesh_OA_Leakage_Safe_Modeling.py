# -*- coding: utf-8 -*-
"""Benchmark modeling for Bangladesh MERRA-2 OCSMASS.

Design
------
* Train:      2019-2023
* Validation: 2024 (used for model and Ridge-alpha selection)
* Final test: 2025 (evaluated once after model selection)
* Target:     log1p(OCSMASS_ug_m3)
"""

from pathlib import Path
import json
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. FILE LOCATIONS AND SETTINGS
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

OUTPUT_FOLDER = PROJECT_FOLDER / "Modeling_Results_2019_2025"
TABLE_FOLDER = OUTPUT_FOLDER / "Tables"
FIGURE_FOLDER = OUTPUT_FOLDER / "Figures"
MODEL_FOLDER = OUTPUT_FOLDER / "Saved_Model"

RANDOM_SEED = 42
TARGET_RAW = "OCSMASS_ug_m3"
TARGET_LOG = "log1p_OCSMASS"
LOCKED_MODEL = "Histogram Gradient Boosting"

# The environment setting is used only for a short technical test. Leave it
# unset for the full analysis.
QUICK_TEST = os.environ.get("OA_QUICK_TEST", "0") == "1"

RF_TREES = 20 if QUICK_TEST else 250
HGB_ITERATIONS = 30 if QUICK_TEST else 300
PERMUTATION_SAMPLE = 1_000 if QUICK_TEST else 5_000
PERMUTATION_REPEATS = 2 if QUICK_TEST else 7

# Primary predictor set. Deliberate exclusions:
# - dewpoint_C: mathematically related to temperature and relative humidity
# - raw wind direction: circular and incorrectly treated as linear
# - u10/v10: excluded because speed + direction sine/cosine are used
# - raw HCHO/NO2 ratio: unstable when NO2 approaches zero
# - year: avoids learning a simple time index that must extrapolate into 2025
FEATURES = [
    "HCHO",
    "NO2",
    "temperature_C",
    "relative_humidity_percent",
    "boundary_layer_height_m",
    "wind_speed_m_s",
    "wind_direction_sin",
    "wind_direction_cos",
    "total_precipitation_mm",
    "solar_radiation_MJ_m2_day",
    "latitude",
    "longitude",
    "day_of_year_sin",
    "day_of_year_cos",
    "HCHO_missing_flag",
    "NO2_missing_flag",
    "HCHO_negative_flag",
    "NO2_negative_flag",
]

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def ensure_output_folders() -> None:
    for folder in [OUTPUT_FOLDER, TABLE_FOLDER, FIGURE_FOLDER, MODEL_FOLDER]:
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
        "month",
        "grid_id",
        TARGET_RAW,
        TARGET_LOG,
        *FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"The feature table is missing columns: {missing}")

    if len(df) != 173_876:
        raise ValueError(f"Expected 173,876 rows, found {len(df):,}.")

    if df["date"].nunique() != 2_557:
        raise ValueError(f"Expected 2,557 dates, found {df['date'].nunique():,}.")

    if df["grid_id"].nunique() != 68:
        raise ValueError(f"Expected 68 grid cells, found {df['grid_id'].nunique()}.")

    duplicates = int(df.duplicated(["date", "grid_id"]).sum())
    if duplicates:
        raise ValueError(f"Found {duplicates:,} duplicate date-grid records.")

    if df[TARGET_RAW].isna().any() or df[TARGET_LOG].isna().any():
        raise ValueError("The target contains missing values.")

    if (df[TARGET_RAW] < 0).any():
        raise ValueError("The OCSMASS target contains negative values.")

    log_check = np.log1p(df[TARGET_RAW].to_numpy())
    if not np.allclose(log_check, df[TARGET_LOG], rtol=1e-7, atol=1e-8):
        raise ValueError("log1p_OCSMASS is inconsistent with OCSMASS_ug_m3.")


def prepare_satellite_retrievals(df: pd.DataFrame) -> pd.DataFrame:
    """Treat physically negative satellite retrievals as missing.

    Raw input values remain unchanged on disk. Flags are recomputed before
    replacing negative values with NaN in this in-memory modeling copy.
    """
    model_df = df.copy()

    for variable in ["HCHO", "NO2"]:
        model_df[f"{variable}_negative_flag"] = (
            model_df[variable] < 0
        ).astype("int8")
        model_df[f"{variable}_missing_flag"] = (
            model_df[variable].isna()
        ).astype("int8")
        model_df.loc[model_df[variable] < 0, variable] = np.nan

    return model_df


# =============================================================================
# 3. METRICS
# =============================================================================

def calculate_metrics(
    model_name: str,
    period: str,
    y_true_raw: np.ndarray,
    predicted_log: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Calculate metrics on both log and original concentration scales."""
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_true_log = np.log1p(y_true_raw)
    predicted_log = np.asarray(predicted_log, dtype=float)

    # log1p modeling guarantees nonnegative inverse predictions after clipping.
    predicted_raw = np.maximum(np.expm1(predicted_log), 0.0)

    metrics = {
        "model": model_name,
        "period": period,
        "n_rows": len(y_true_raw),
        "MAE_ug_m3": mean_absolute_error(y_true_raw, predicted_raw),
        "RMSE_ug_m3": np.sqrt(mean_squared_error(y_true_raw, predicted_raw)),
        "R2_raw": r2_score(y_true_raw, predicted_raw),
        "mean_bias_ug_m3": float(np.mean(predicted_raw - y_true_raw)),
        "MAE_log": mean_absolute_error(y_true_log, predicted_log),
        "RMSE_log": np.sqrt(mean_squared_error(y_true_log, predicted_log)),
        "R2_log": r2_score(y_true_log, predicted_log),
    }
    return metrics, predicted_raw


# =============================================================================
# 4. MONTH-GRID CLIMATOLOGY BASELINE
# =============================================================================

def fit_climatology(training: pd.DataFrame) -> dict:
    """Fit a hierarchical log-target climatology without future information."""
    grid_month = (
        training.groupby(["grid_id", "month"], observed=True)[TARGET_LOG]
        .mean()
        .rename("prediction")
    )
    month = (
        training.groupby("month", observed=True)[TARGET_LOG]
        .mean()
        .rename("prediction")
    )
    global_mean = float(training[TARGET_LOG].mean())

    return {
        "grid_month": grid_month,
        "month": month,
        "global_mean": global_mean,
    }


def predict_climatology(model: dict, data: pd.DataFrame) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(data[["grid_id", "month"]])
    prediction = model["grid_month"].reindex(keys).to_numpy(dtype=float)

    missing = np.isnan(prediction)
    if missing.any():
        month_prediction = (
            model["month"].reindex(data.loc[missing, "month"]).to_numpy(dtype=float)
        )
        prediction[missing] = month_prediction

    prediction = np.where(
        np.isnan(prediction), model["global_mean"], prediction
    )
    return prediction


# =============================================================================
# 5. SCIKIT-LEARN MODELS
# =============================================================================

def build_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
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


def build_hist_gradient_boosting() -> HistGradientBoostingRegressor:
    # This estimator natively handles NaN values in HCHO and NO2.
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


# =============================================================================
# 6. PLOTS
# =============================================================================

def plot_validation_metrics(metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("RMSE_log")
    x = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(x, ordered["RMSE_log"], color="#3b82a0")
    axes[0].set_xticks(x, ordered["model"], rotation=25, ha="right")
    axes[0].set_ylabel("RMSE on log1p scale")
    axes[0].set_title("2024 validation: log-scale error")

    axes[1].bar(x, ordered["MAE_ug_m3"], color="#d8873f")
    axes[1].set_xticks(x, ordered["model"], rotation=25, ha="right")
    axes[1].set_ylabel("MAE (µg/m³)")
    axes[1].set_title("2024 validation: original-scale error")
    save_figure("01_validation_model_comparison.png")


def plot_observed_predicted(
    observed_raw: np.ndarray,
    predicted_raw: np.ndarray,
    title: str,
    filename: str,
) -> None:
    observed_log = np.log1p(observed_raw)
    predicted_log = np.log1p(predicted_raw)
    low = float(min(observed_log.min(), predicted_log.min()))
    high = float(max(observed_log.max(), predicted_log.max()))

    plt.figure(figsize=(7, 6))
    plot = plt.hexbin(
        observed_log,
        predicted_log,
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    plt.plot([low, high], [low, high], color="red", linestyle="--", linewidth=1.2)
    plt.colorbar(plot, label="log10 observation count")
    plt.xlabel("Observed log(1 + OCSMASS)")
    plt.ylabel("Predicted log(1 + OCSMASS)")
    plt.title(title)
    save_figure(filename)


def plot_residuals(
    observed_raw: np.ndarray,
    predicted_raw: np.ndarray,
    title: str,
    filename: str,
) -> None:
    residual = predicted_raw - observed_raw
    display_limit = np.quantile(np.abs(residual), 0.995)
    clipped = residual[np.abs(residual) <= display_limit]

    plt.figure(figsize=(8, 5))
    plt.hist(clipped, bins=80, color="#557a95", alpha=0.9)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Prediction error: predicted − observed (µg/m³)")
    plt.ylabel("Observations")
    plt.title(title + "\n(central 99.5% shown)")
    save_figure(filename)


def plot_permutation_importance(importance: pd.DataFrame) -> None:
    shown = importance.sort_values("importance_mean", ascending=True).tail(15)
    plt.figure(figsize=(9, 7))
    plt.barh(
        shown["feature"],
        shown["importance_mean"],
        xerr=shown["importance_std"],
        color="#3f7f93",
        alpha=0.9,
    )
    plt.xlabel("Increase in validation log-RMSE after permutation")
    plt.title("2024 permutation importance of selected model")
    save_figure("05_selected_model_permutation_importance.png")


# =============================================================================
# 7. MAIN WORKFLOW
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.random.seed(RANDOM_SEED)
    ensure_output_folders()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature table was not found:\n{INPUT_FILE}\n\n"
            "Run the EDA/feature-preparation script first, or correct "
            "PROJECT_FOLDER at the beginning of this script."
        )

    print("Reading the feature-prepared dataset...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
    df["date"] = df["date"].dt.normalize()
    df["grid_id"] = df["grid_id"].astype(str).str.strip()
    validate_input(df)
    df = prepare_satellite_retrievals(df)

    train = df[df["year"].between(2019, 2023)].copy()
    validation = df[df["year"] == 2024].copy()
    development = df[df["year"].between(2019, 2024)].copy()
    final_test = df[df["year"] == 2025].copy()

    expected = {
        "train_2019_2023": 124_168,
        "validation_2024": 24_888,
        "development_2019_2024": 149_056,
        "final_test_2025": 24_820,
    }
    actual = {
        "train_2019_2023": len(train),
        "validation_2024": len(validation),
        "development_2019_2024": len(development),
        "final_test_2025": len(final_test),
    }
    if actual != expected:
        raise ValueError(f"Unexpected temporal split sizes: {actual}")

    X_train = train[FEATURES]
    y_train_log = train[TARGET_LOG]
    X_validation = validation[FEATURES]
    y_validation_raw = validation[TARGET_RAW].to_numpy()

    print()
    print("Temporal split confirmed:")
    for name, count in actual.items():
        print(f"  {name}: {count:,} rows")

    validation_metrics = []
    validation_predictions = {}
    trained_models = {}
    model_parameters = []

    # -------------------------------------------------------------------------
    # Model 1: month-grid climatology baseline
    # -------------------------------------------------------------------------
    print("\n[1/4] Fitting month-grid climatology baseline...")
    start = time.perf_counter()
    climatology = fit_climatology(train)
    prediction_log = predict_climatology(climatology, validation)
    metrics, prediction_raw = calculate_metrics(
        "Month-grid climatology", "Validation 2024", y_validation_raw, prediction_log
    )
    validation_metrics.append(metrics)
    validation_predictions["Month-grid climatology"] = (
        prediction_log,
        prediction_raw,
    )
    trained_models["Month-grid climatology"] = climatology
    model_parameters.append(
        {
            "model": "Month-grid climatology",
            "settings": "Mean log1p target by training grid_id and month",
        }
    )
    print(f"Completed in {elapsed_text(start)}")

    # -------------------------------------------------------------------------
    # Model 2: Ridge regression, with alpha selected using 2024
    # -------------------------------------------------------------------------
    print("\n[2/4] Fitting Ridge regression candidates...")
    start = time.perf_counter()
    ridge_candidates = []
    for alpha in RIDGE_ALPHAS:
        ridge = build_ridge(alpha)
        ridge.fit(X_train, y_train_log)
        candidate_log = ridge.predict(X_validation)
        candidate_metrics, candidate_raw = calculate_metrics(
            f"Ridge alpha={alpha:g}",
            "Validation 2024",
            y_validation_raw,
            candidate_log,
        )
        ridge_candidates.append(
            (candidate_metrics["RMSE_log"], alpha, ridge, candidate_log, candidate_raw)
        )

    ridge_candidates.sort(key=lambda item: item[0])
    _, best_alpha, best_ridge, prediction_log, prediction_raw = ridge_candidates[0]
    metrics, prediction_raw = calculate_metrics(
        "Ridge regression", "Validation 2024", y_validation_raw, prediction_log
    )
    validation_metrics.append(metrics)
    validation_predictions["Ridge regression"] = (prediction_log, prediction_raw)
    trained_models["Ridge regression"] = best_ridge
    model_parameters.append(
        {
            "model": "Ridge regression",
            "settings": f"Median imputation; scaling; selected alpha={best_alpha:g}",
        }
    )
    print(f"Selected alpha: {best_alpha:g}")
    print(f"Completed in {elapsed_text(start)}")

    # -------------------------------------------------------------------------
    # Model 3: Random forest
    # -------------------------------------------------------------------------
    print("\n[3/4] Fitting Random Forest...")
    start = time.perf_counter()
    random_forest = build_random_forest()
    random_forest.fit(X_train, y_train_log)
    prediction_log = random_forest.predict(X_validation)
    metrics, prediction_raw = calculate_metrics(
        "Random Forest", "Validation 2024", y_validation_raw, prediction_log
    )
    validation_metrics.append(metrics)
    validation_predictions["Random Forest"] = (prediction_log, prediction_raw)
    trained_models["Random Forest"] = random_forest
    model_parameters.append(
        {
            "model": "Random Forest",
            "settings": (
                f"trees={RF_TREES}; max_depth=22; min_samples_leaf=5; "
                "max_features=0.8; median imputation"
            ),
        }
    )
    print(f"Completed in {elapsed_text(start)}")

    # -------------------------------------------------------------------------
    # Model 4: Histogram gradient boosting
    # -------------------------------------------------------------------------
    print("\n[4/4] Fitting Histogram Gradient Boosting...")
    start = time.perf_counter()
    gradient_boosting = build_hist_gradient_boosting()
    gradient_boosting.fit(X_train, y_train_log)
    prediction_log = gradient_boosting.predict(X_validation)
    metrics, prediction_raw = calculate_metrics(
        "Histogram Gradient Boosting",
        "Validation 2024",
        y_validation_raw,
        prediction_log,
    )
    validation_metrics.append(metrics)
    validation_predictions["Histogram Gradient Boosting"] = (
        prediction_log,
        prediction_raw,
    )
    trained_models["Histogram Gradient Boosting"] = gradient_boosting
    model_parameters.append(
        {
            "model": "Histogram Gradient Boosting",
            "settings": (
                f"iterations={HGB_ITERATIONS}; learning_rate=0.05; "
                "max_leaf_nodes=31; min_samples_leaf=30; l2=0.5; native NaN"
            ),
        }
    )
    print(f"Completed in {elapsed_text(start)}")

    # -------------------------------------------------------------------------
    # Save the 2024 ranking and reproduce the model locked in the original run.
    # -------------------------------------------------------------------------
    validation_metrics_df = pd.DataFrame(validation_metrics).sort_values(
        ["RMSE_log", "MAE_log"]
    )
    validation_metrics_df.insert(
        0, "validation_rank", range(1, len(validation_metrics_df) + 1)
    )
    validation_metrics_df.to_csv(
        TABLE_FOLDER / "01_validation_metrics_2024.csv", index=False
    )
    pd.DataFrame(model_parameters).to_csv(
        TABLE_FOLDER / "02_model_settings.csv", index=False
    )

    validation_winner = str(validation_metrics_df.iloc[0]["model"])
    selected_name = LOCKED_MODEL
    selected_validation_log, selected_validation_raw = validation_predictions[
        selected_name
    ]

    print("\n2024 validation results:")
    print(
        validation_metrics_df[
            [
                "validation_rank",
                "model",
                "MAE_ug_m3",
                "RMSE_ug_m3",
                "R2_raw",
                "RMSE_log",
                "R2_log",
            ]
        ].to_string(index=False)
    )
    print(f"\nLocked model from the paper: {selected_name}")
    if validation_winner != selected_name and not QUICK_TEST:
        warnings.warn(
            f"{validation_winner} ranked first in this run. The original ranking "
            "used scikit-learn 1.7.2. The 2025 test still uses the paper's locked model."
        )

    # Save validation predictions for the selected model.
    validation_output = validation[
        ["date", "grid_id", "latitude", "longitude", TARGET_RAW, TARGET_LOG]
    ].copy()
    validation_output["predicted_log1p_OCSMASS"] = selected_validation_log
    validation_output["predicted_OCSMASS_ug_m3"] = selected_validation_raw
    validation_output["residual_ug_m3"] = (
        selected_validation_raw - y_validation_raw
    )
    validation_output.to_csv(
        TABLE_FOLDER / "03_selected_model_validation_predictions_2024.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        float_format="%.10g",
    )

    # -------------------------------------------------------------------------
    # Permutation importance uses validation data only, before final test use.
    # -------------------------------------------------------------------------
    importance_df = None
    if selected_name != "Month-grid climatology":
        print("\nCalculating 2024 permutation importance...")
        rng = np.random.default_rng(RANDOM_SEED)
        sample_size = min(PERMUTATION_SAMPLE, len(validation))
        sample_positions = rng.choice(
            len(validation), size=sample_size, replace=False
        )
        selected_model = trained_models[selected_name]
        result = permutation_importance(
            selected_model,
            X_validation.iloc[sample_positions],
            validation[TARGET_LOG].iloc[sample_positions],
            scoring="neg_root_mean_squared_error",
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        importance_df = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        importance_df.to_csv(
            TABLE_FOLDER / "04_selected_model_permutation_importance_2024.csv",
            index=False,
        )

    pd.DataFrame(
        {
            "feature": FEATURES,
            "role": "primary predictor",
        }
    ).to_csv(TABLE_FOLDER / "05_primary_feature_list.csv", index=False)

    # -------------------------------------------------------------------------
    # Final refit on 2019-2024, then one-time 2025 evaluation
    # -------------------------------------------------------------------------
    print("\nRefitting the selected model on all 2019-2024 development data...")
    start = time.perf_counter()

    if selected_name == "Month-grid climatology":
        final_model = fit_climatology(development)
        test_prediction_log = predict_climatology(final_model, final_test)
    else:
        final_model = clone(trained_models[selected_name])
        final_model.fit(development[FEATURES], development[TARGET_LOG])
        test_prediction_log = final_model.predict(final_test[FEATURES])

    print(f"Final refit completed in {elapsed_text(start)}")
    print("Evaluating the locked model once on the 2025 final test period...")

    test_metrics, test_prediction_raw = calculate_metrics(
        selected_name,
        "Final test 2025",
        final_test[TARGET_RAW].to_numpy(),
        test_prediction_log,
    )
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv(
        TABLE_FOLDER / "06_final_test_metrics_2025.csv", index=False
    )

    test_output = final_test[
        [
            "date",
            "year",
            "month",
            "day_of_year",
            "season",
            "grid_id",
            "latitude",
            "longitude",
            TARGET_RAW,
            TARGET_LOG,
            "HCHO",
            "NO2",
        ]
    ].copy()
    test_output["predicted_log1p_OCSMASS"] = test_prediction_log
    test_output["predicted_OCSMASS_ug_m3"] = test_prediction_raw
    test_output["residual_ug_m3"] = (
        test_prediction_raw - final_test[TARGET_RAW].to_numpy()
    )
    test_output.to_csv(
        TABLE_FOLDER / "07_final_test_predictions_2025.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        float_format="%.10g",
    )

    # Save model and preprocessing together.
    model_bundle = {
        "selected_model_name": selected_name,
        "model": final_model,
        "features": FEATURES,
        "target": TARGET_RAW,
        "target_transformation": "log1p",
        "inverse_transformation": "expm1 followed by clipping at zero",
        "training_period": "2019-01-01 through 2024-12-31",
        "final_test_period": "2025-01-01 through 2025-12-31",
        "negative_satellite_treatment": "Set negative HCHO/NO2 to NaN; preserve flags",
        "random_seed": RANDOM_SEED,
    }
    model_path = MODEL_FOLDER / "Bangladesh_OCSMASS_Selected_Model.joblib"
    joblib.dump(model_bundle, model_path, compress=3)

    # Machine-readable selection record.
    selection_record = {
        "selection_criterion": "Lowest 2024 validation log-RMSE in the original run",
        "original_scikit_learn_version": "1.7.2",
        "validation_winner_this_run": validation_winner,
        "selected_model": selected_name,
        "development_period": "2019-2024",
        "final_test_period": "2025",
        "test_was_used_for_selection": False,
    }
    (TABLE_FOLDER / "08_model_selection_record.json").write_text(
        json.dumps(selection_record, indent=2), encoding="utf-8"
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_validation_metrics(validation_metrics_df)
    plot_observed_predicted(
        y_validation_raw,
        selected_validation_raw,
        f"{selected_name}: 2024 validation",
        "02_selected_model_validation_observed_vs_predicted.png",
    )
    plot_observed_predicted(
        final_test[TARGET_RAW].to_numpy(),
        test_prediction_raw,
        f"{selected_name}: 2025 final test",
        "03_selected_model_test_observed_vs_predicted.png",
    )
    plot_residuals(
        final_test[TARGET_RAW].to_numpy(),
        test_prediction_raw,
        f"{selected_name}: 2025 final-test residuals",
        "04_selected_model_test_residuals.png",
    )
    if importance_df is not None:
        plot_permutation_importance(importance_df)

    # -------------------------------------------------------------------------
    # Human-readable report
    # -------------------------------------------------------------------------
    selected_metrics = validation_metrics_df.loc[
        validation_metrics_df["model"].eq(selected_name)
    ].iloc[0]
    report_lines = [
        "Bangladesh OCSMASS leakage-safe model report",
        "",
        "Important interpretation:",
        "The target is MERRA-2 surface organic carbon mass concentration.",
        "It is not a direct measurement of secondary organic aerosol.",
        "",
        "Temporal design:",
        "Training: 2019-2023",
        "Validation and model selection: 2024",
        "Final locked-model test: 2025",
        "",
        "Primary preprocessing:",
        "Negative HCHO and NO2 retrievals were set to missing in memory.",
        "Negative and missing retrieval flags were retained.",
        "Linear and random-forest models used training-only median imputation.",
        "Histogram gradient boosting used native missing-value handling.",
        "The unstable raw HCHO/NO2 ratio was excluded.",
        "The target was modeled as log1p(OCSMASS).",
        "",
        f"Selected model: {selected_name}",
        "Selection criterion in the original run: lowest 2024 validation log-RMSE",
        f"Lowest 2024 validation log-RMSE in this run: {validation_winner}",
        f"2024 validation log-RMSE: {selected_metrics['RMSE_log']:.6f}",
        f"2024 validation raw MAE: {selected_metrics['MAE_ug_m3']:.6f} µg/m³",
        f"2024 validation raw RMSE: {selected_metrics['RMSE_ug_m3']:.6f} µg/m³",
        f"2024 validation raw R²: {selected_metrics['R2_raw']:.6f}",
        "",
        "Final 2025 test performance:",
        f"MAE: {test_metrics['MAE_ug_m3']:.6f} µg/m³",
        f"RMSE: {test_metrics['RMSE_ug_m3']:.6f} µg/m³",
        f"R²: {test_metrics['R2_raw']:.6f}",
        f"Log-RMSE: {test_metrics['RMSE_log']:.6f}",
        f"Log-R²: {test_metrics['R2_log']:.6f}",
        f"Mean bias: {test_metrics['mean_bias_ug_m3']:.6f} µg/m³",
    ]
    (OUTPUT_FOLDER / "Modeling_run_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 76)
    print("MODELING COMPLETED SUCCESSFULLY")
    print("=" * 76)
    print(f"Selected model: {selected_name}")
    print("Final 2025 metrics:")
    print(
        test_metrics_df[
            ["MAE_ug_m3", "RMSE_ug_m3", "R2_raw", "RMSE_log", "R2_log"]
        ].to_string(index=False)
    )
    print(f"\nTables: {TABLE_FOLDER}")
    print(f"Figures: {FIGURE_FOLDER}")
    print(f"Saved model: {model_path}")
    print("\nThe source feature CSV was not modified.")


if __name__ == "__main__":
    main()
