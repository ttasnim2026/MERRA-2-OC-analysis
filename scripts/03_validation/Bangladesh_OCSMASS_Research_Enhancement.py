# -*- coding: utf-8 -*-
"""Enhancements for the Bangladesh OCSMASS model.

This is a post-selection robustness analysis. It does NOT change the locked
primary model, model choice, hyperparameters, or the final 2025 test result.

Analyses
--------
1. Predictor-group ablation under the existing five spatial blocks (2019-2024).
2. Seasonal, spatial-block, grid-cell, and upper-tail error diagnostics.
3. Combined space-time test with conformal prediction intervals:
      train = 2019-2022, calibrate = 2023, test = withheld region in 2024.

Spyder use
----------
Edit PROJECT_FOLDER if needed, save the file, and click Run.

Terminal use (optional)
-----------------------
python Bangladesh_OCSMASS_Research_Enhancement.py \
  --features ".../Bangladesh_OA_Model_Features_2019_2025.csv" \
  --blocks ".../01_spatial_block_assignments.csv" \
  --output ".../Research_Enhancement_Results_2019_2024"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

PROJECT_FOLDER = Path(
    os.environ.get("OA_PROJECT_FOLDER", str(Path.cwd()))
)

DEFAULT_FEATURE_FILE = (
    PROJECT_FOLDER
    / "EDA_Results_2019_2025"
    / "Model_Data"
    / "Bangladesh_OA_Model_Features_2019_2025.csv"
)

DEFAULT_BLOCK_FILE = (
    PROJECT_FOLDER
    / "Spatial_Validation_Results_2019_2024"
    / "Tables"
    / "01_spatial_block_assignments.csv"
)

DEFAULT_OUTPUT_FOLDER = PROJECT_FOLDER / "Research_Enhancement_Results_2019_2024"

RANDOM_SEED = 42
TARGET_RAW = "OCSMASS_ug_m3"
TARGET_LOG = "log1p_OCSMASS"


CALENDAR_FEATURES = [
    "day_of_year_sin",
    "day_of_year_cos",
]

COORDINATE_FEATURES = [
    "latitude",
    "longitude",
]

SATELLITE_FEATURES = [
    "HCHO",
    "NO2",
    "HCHO_missing_flag",
    "NO2_missing_flag",
    "HCHO_negative_flag",
    "NO2_negative_flag",
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

# The order is used in tables and figures.
ABLATION_SPECIFICATIONS = {
    "Calendar only": CALENDAR_FEATURES,
    "Spatial-seasonal baseline": COORDINATE_FEATURES + CALENDAR_FEATURES,
    "Meteorology + calendar (no coordinates)": METEOROLOGY_FEATURES + CALENDAR_FEATURES,
    "Satellite + calendar (no coordinates)": SATELLITE_FEATURES + CALENDAR_FEATURES,
    "Atmospheric predictors (no coordinates)": (
        SATELLITE_FEATURES + METEOROLOGY_FEATURES + CALENDAR_FEATURES
    ),
    "Full model without satellite": (
        METEOROLOGY_FEATURES + COORDINATE_FEATURES + CALENDAR_FEATURES
    ),
    "Full model without meteorology": (
        SATELLITE_FEATURES + COORDINATE_FEATURES + CALENDAR_FEATURES
    ),
    "Full model": (
        SATELLITE_FEATURES
        + METEOROLOGY_FEATURES
        + COORDINATE_FEATURES
        + CALENDAR_FEATURES
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--blocks", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Technical test only: 40 boosting iterations instead of 300.",
    )
    return parser.parse_args()


def locate_file(explicit: Path | None, preferred: Path, filename: str) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"File not found: {explicit}")
        return explicit
    if preferred.exists():
        return preferred
    matches = sorted(PROJECT_FOLDER.rglob(filename)) if PROJECT_FOLDER.exists() else []
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename}. Edit PROJECT_FOLDER or pass the file path."
        )
    return matches[0]


def make_model(iterations: int) -> HistGradientBoostingRegressor:
    """Use the locked primary-model hyperparameters."""
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=iterations,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=0.5,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def inverse_log(predicted_log: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(np.asarray(predicted_log, dtype=float)), 0.0)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(stats.pearsonr(x, y).statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def metrics(
    observed_raw: np.ndarray,
    predicted_raw: np.ndarray,
    observed_log: np.ndarray | None = None,
    predicted_log: np.ndarray | None = None,
) -> dict:
    observed_raw = np.asarray(observed_raw, dtype=float)
    predicted_raw = np.asarray(predicted_raw, dtype=float)
    if observed_log is None:
        observed_log = np.log1p(observed_raw)
    if predicted_log is None:
        predicted_log = np.log1p(predicted_raw)
    observed_log = np.asarray(observed_log, dtype=float)
    predicted_log = np.asarray(predicted_log, dtype=float)

    return {
        "n_rows": int(len(observed_raw)),
        "MAE_ug_m3": float(mean_absolute_error(observed_raw, predicted_raw)),
        "RMSE_ug_m3": float(np.sqrt(mean_squared_error(observed_raw, predicted_raw))),
        "R2_raw": float(r2_score(observed_raw, predicted_raw)),
        "mean_bias_ug_m3": float(np.mean(predicted_raw - observed_raw)),
        "Pearson_r_raw": safe_pearson(observed_raw, predicted_raw),
        "Spearman_rho_raw": safe_spearman(observed_raw, predicted_raw),
        "MAE_log": float(mean_absolute_error(observed_log, predicted_log)),
        "RMSE_log": float(np.sqrt(mean_squared_error(observed_log, predicted_log))),
        "R2_log": float(r2_score(observed_log, predicted_log)),
    }


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data["grid_id"] = data["grid_id"].astype(str).str.strip()

    for variable in ["HCHO", "NO2"]:
        data[f"{variable}_negative_flag"] = (data[variable] < 0).astype("int8")
        data[f"{variable}_missing_flag"] = data[variable].isna().astype("int8")
        data.loc[data[variable] < 0, variable] = np.nan

    required = {
        "date", "year", "grid_id", "latitude", "longitude", TARGET_RAW,
        TARGET_LOG, *ABLATION_SPECIFICATIONS["Full model"],
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    if data.duplicated(["date", "grid_id"]).any():
        raise ValueError("The feature table has duplicate date-grid rows.")
    if data["grid_id"].nunique() != 68:
        raise ValueError(f"Expected 68 grids; found {data['grid_id'].nunique()}.")
    if not np.allclose(np.log1p(data[TARGET_RAW]), data[TARGET_LOG]):
        raise ValueError("The raw and log1p targets are inconsistent.")
    return data


def attach_blocks(data: pd.DataFrame, block_table: pd.DataFrame) -> pd.DataFrame:
    blocks = block_table[["grid_id", "spatial_block"]].copy()
    blocks["grid_id"] = blocks["grid_id"].astype(str).str.strip()
    if len(blocks) != 68 or blocks["grid_id"].nunique() != 68:
        raise ValueError("Block assignment table must contain exactly 68 unique grids.")
    if sorted(blocks["spatial_block"].unique()) != [1, 2, 3, 4, 5]:
        raise ValueError("Expected spatial blocks numbered 1 through 5.")
    merged = data.merge(blocks, on="grid_id", how="left", validate="many_to_one")
    if merged["spatial_block"].isna().any():
        missing = merged.loc[merged["spatial_block"].isna(), "grid_id"].unique()
        raise ValueError(f"Missing block assignments for: {missing.tolist()}")
    merged["spatial_block"] = merged["spatial_block"].astype(int)
    return merged


def run_ablation(
    development: pd.DataFrame,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    pooled_rows = []
    full_predictions = []

    for spec_number, (specification, features) in enumerate(
        ABLATION_SPECIFICATIONS.items(), start=1
    ):
        print(
            f"Ablation {spec_number}/{len(ABLATION_SPECIFICATIONS)}: "
            f"{specification} ({len(features)} predictors)"
        )
        specification_predictions = []
        for block in range(1, 6):
            train = development[development["spatial_block"] != block]
            test = development[development["spatial_block"] == block]

            model = make_model(iterations)
            model.fit(train[features], train[TARGET_LOG])
            predicted_log = model.predict(test[features])
            predicted_raw = inverse_log(predicted_log)

            fold_metric = metrics(
                test[TARGET_RAW].to_numpy(),
                predicted_raw,
                test[TARGET_LOG].to_numpy(),
                predicted_log,
            )
            fold_metric.update(
                {
                    "specification": specification,
                    "spatial_block": block,
                    "n_features": len(features),
                    "n_train_grid_cells": int(train["grid_id"].nunique()),
                    "n_validation_grid_cells": int(test["grid_id"].nunique()),
                    "n_train_rows": int(len(train)),
                }
            )
            fold_rows.append(fold_metric)

            prediction = test[
                ["date", "year", "grid_id", "latitude", "longitude", "spatial_block", TARGET_RAW, TARGET_LOG]
            ].copy()
            prediction["predicted_log1p_OCSMASS"] = predicted_log
            prediction["predicted_OCSMASS_ug_m3"] = predicted_raw
            prediction["residual_ug_m3"] = predicted_raw - test[TARGET_RAW].to_numpy()
            specification_predictions.append(prediction)

        pooled = pd.concat(specification_predictions, ignore_index=True)
        pooled_metric = metrics(
            pooled[TARGET_RAW].to_numpy(),
            pooled["predicted_OCSMASS_ug_m3"].to_numpy(),
            pooled[TARGET_LOG].to_numpy(),
            pooled["predicted_log1p_OCSMASS"].to_numpy(),
        )
        pooled_metric.update(
            {
                "specification": specification,
                "n_features": len(features),
                "n_unique_grid_cells": int(pooled["grid_id"].nunique()),
            }
        )
        pooled_rows.append(pooled_metric)

        if specification == "Full model":
            full_predictions.append(pooled)

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(pooled_rows),
        pd.concat(full_predictions, ignore_index=True),
    )


def group_metrics(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group_value, group in data.groupby(group_column, observed=True):
        row = metrics(
            group[TARGET_RAW].to_numpy(),
            group["predicted_OCSMASS_ug_m3"].to_numpy(),
            group[TARGET_LOG].to_numpy(),
            group["predicted_log1p_OCSMASS"].to_numpy(),
        )
        row[group_column] = group_value
        rows.append(row)
    return pd.DataFrame(rows)


def add_diagnostic_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    month = out["date"].dt.month
    out["season"] = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8, 9]),
            month.isin([10, 11]),
        ],
        ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"],
        default="Unknown",
    )
    p90 = float(out[TARGET_RAW].quantile(0.90))
    p95 = float(out[TARGET_RAW].quantile(0.95))
    out["observed_concentration_group"] = pd.cut(
        out[TARGET_RAW],
        bins=[-np.inf, p90, p95, np.inf],
        labels=["Below 90th percentile", "90th-95th percentile", "Highest 5%"],
        include_lowest=True,
    )
    out.attrs["p90"] = p90
    out.attrs["p95"] = p95
    return out


def finite_sample_quantile(residuals: np.ndarray, coverage: float) -> float:
    residuals = np.sort(np.asarray(residuals, dtype=float))
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        raise ValueError("No finite calibration residuals.")
    rank = math.ceil((len(residuals) + 1) * coverage)
    rank = min(max(rank, 1), len(residuals))
    return float(residuals[rank - 1])


def run_spatiotemporal_intervals(
    development: pd.DataFrame,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = ABLATION_SPECIFICATIONS["Full model"]
    predictions = []
    fold_rows = []

    for block in range(1, 6):
        print(f"Space-time uncertainty fold {block}/5")
        training = development[
            development["year"].between(2019, 2022)
            & development["spatial_block"].ne(block)
        ]
        calibration = development[
            development["year"].eq(2023)
            & development["spatial_block"].ne(block)
        ]
        test = development[
            development["year"].eq(2024)
            & development["spatial_block"].eq(block)
        ]

        model = make_model(iterations)
        model.fit(training[features], training[TARGET_LOG])

        calibration_prediction = model.predict(calibration[features])
        calibration_residual = np.abs(
            calibration[TARGET_LOG].to_numpy() - calibration_prediction
        )
        q80 = finite_sample_quantile(calibration_residual, 0.80)
        q95 = finite_sample_quantile(calibration_residual, 0.95)

        predicted_log = model.predict(test[features])
        predicted_raw = inverse_log(predicted_log)

        output = test[
            ["date", "year", "grid_id", "latitude", "longitude", "spatial_block", TARGET_RAW, TARGET_LOG]
        ].copy()
        output["predicted_log1p_OCSMASS"] = predicted_log
        output["predicted_OCSMASS_ug_m3"] = predicted_raw
        output["residual_ug_m3"] = predicted_raw - output[TARGET_RAW]

        for coverage, quantile in [(80, q80), (95, q95)]:
            lower_log = predicted_log - quantile
            upper_log = predicted_log + quantile
            lower_raw = inverse_log(lower_log)
            upper_raw = inverse_log(upper_log)
            output[f"lower_{coverage}_ug_m3"] = lower_raw
            output[f"upper_{coverage}_ug_m3"] = upper_raw
            output[f"covered_{coverage}"] = (
                output[TARGET_RAW].between(lower_raw, upper_raw)
            )

        row = metrics(
            output[TARGET_RAW].to_numpy(),
            output["predicted_OCSMASS_ug_m3"].to_numpy(),
            output[TARGET_LOG].to_numpy(),
            output["predicted_log1p_OCSMASS"].to_numpy(),
        )
        row.update(
            {
                "spatial_block": block,
                "training_period": "2019-2022",
                "calibration_period": "2023",
                "test_period": "2024",
                "n_training_rows": int(len(training)),
                "n_calibration_rows": int(len(calibration)),
                "n_test_rows": int(len(test)),
                "n_training_grid_cells": int(training["grid_id"].nunique()),
                "n_test_grid_cells": int(test["grid_id"].nunique()),
                "nominal_80_coverage": 0.80,
                "empirical_80_coverage": float(output["covered_80"].mean()),
                "median_80_width_ug_m3": float(
                    (output["upper_80_ug_m3"] - output["lower_80_ug_m3"]).median()
                ),
                "nominal_95_coverage": 0.95,
                "empirical_95_coverage": float(output["covered_95"].mean()),
                "median_95_width_ug_m3": float(
                    (output["upper_95_ug_m3"] - output["lower_95_ug_m3"]).median()
                ),
                "calibration_q80_log": q80,
                "calibration_q95_log": q95,
            }
        )
        fold_rows.append(row)
        predictions.append(output)

    pooled = pd.concat(predictions, ignore_index=True)
    pooled_row = metrics(
        pooled[TARGET_RAW].to_numpy(),
        pooled["predicted_OCSMASS_ug_m3"].to_numpy(),
        pooled[TARGET_LOG].to_numpy(),
        pooled["predicted_log1p_OCSMASS"].to_numpy(),
    )
    pooled_row.update(
        {
            "spatial_block": "Pooled",
            "training_period": "2019-2022",
            "calibration_period": "2023",
            "test_period": "2024",
            "n_training_rows": np.nan,
            "n_calibration_rows": np.nan,
            "n_test_rows": int(len(pooled)),
            "n_training_grid_cells": np.nan,
            "n_test_grid_cells": int(pooled["grid_id"].nunique()),
            "nominal_80_coverage": 0.80,
            "empirical_80_coverage": float(pooled["covered_80"].mean()),
            "median_80_width_ug_m3": float(
                (pooled["upper_80_ug_m3"] - pooled["lower_80_ug_m3"]).median()
            ),
            "nominal_95_coverage": 0.95,
            "empirical_95_coverage": float(pooled["covered_95"].mean()),
            "median_95_width_ug_m3": float(
                (pooled["upper_95_ug_m3"] - pooled["lower_95_ug_m3"]).median()
            ),
            "calibration_q80_log": np.nan,
            "calibration_q95_log": np.nan,
        }
    )
    fold_rows.append(pooled_row)
    return pd.DataFrame(fold_rows), pooled


def save_plot(destination: Path) -> None:
    plt.tight_layout()
    plt.savefig(destination, dpi=350, bbox_inches="tight")
    plt.close()


def plot_ablation(pooled: pd.DataFrame, destination: Path) -> None:
    order = list(ABLATION_SPECIFICATIONS)
    shown = pooled.set_index("specification").loc[order].reset_index()
    colors = ["#8da0cb"] * len(shown)
    colors[order.index("Spatial-seasonal baseline")] = "#fc8d62"
    colors[order.index("Full model")] = "#2c7fb8"
    plt.figure(figsize=(11, 6.5))
    y = np.arange(len(shown))
    plt.barh(y, shown["RMSE_log"], color=colors)
    plt.yticks(y, shown["specification"])
    plt.gca().invert_yaxis()
    plt.xlabel("Pooled spatial out-of-fold RMSE on log1p scale")
    plt.title("Predictor-group ablation under identical spatial folds")
    for index, value in enumerate(shown["RMSE_log"]):
        plt.text(value + 0.004, index, f"{value:.3f}", va="center")
    plt.grid(axis="x", alpha=0.2)
    save_plot(destination)


def plot_seasonal_error(seasonal: pd.DataFrame, destination: Path) -> None:
    order = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
    shown = seasonal.set_index("season").loc[order].reset_index()
    plt.figure(figsize=(8.5, 5.5))
    bars = plt.bar(shown["season"], shown["RMSE_ug_m3"], color="#4c78a8")
    plt.ylabel("Spatial out-of-fold RMSE (µg m⁻³)")
    plt.title("Full-model error varies strongly by season")
    plt.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, shown["RMSE_ug_m3"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}",
                 ha="center", va="bottom")
    save_plot(destination)


def plot_extreme_bias(extreme: pd.DataFrame, destination: Path) -> None:
    order = ["Below 90th percentile", "90th-95th percentile", "Highest 5%"]
    shown = extreme.set_index("observed_concentration_group").loc[order].reset_index()
    plt.figure(figsize=(9, 5.5))
    colors = ["#72b7b2" if x >= 0 else "#e45756" for x in shown["mean_bias_ug_m3"]]
    bars = plt.bar(shown["observed_concentration_group"], shown["mean_bias_ug_m3"], color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Mean prediction bias (µg m⁻³)")
    plt.title("Full-model bias by observed OCSMASS concentration")
    plt.xticks(rotation=12, ha="right")
    for bar, value in zip(bars, shown["mean_bias_ug_m3"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value,
                 f"{value:.2f}", ha="center",
                 va="bottom" if value >= 0 else "top")
    save_plot(destination)


def plot_interval_coverage(interval_metrics: pd.DataFrame, destination: Path) -> None:
    shown = interval_metrics[interval_metrics["spatial_block"].astype(str) != "Pooled"].copy()
    shown["spatial_block"] = shown["spatial_block"].astype(int)
    shown = shown.sort_values("spatial_block")
    x = np.arange(len(shown))
    width = 0.36
    plt.figure(figsize=(8.5, 5.5))
    plt.bar(x - width / 2, shown["empirical_80_coverage"], width, label="80% interval")
    plt.bar(x + width / 2, shown["empirical_95_coverage"], width, label="95% interval")
    plt.axhline(0.80, color="#1f77b4", linestyle="--", linewidth=1)
    plt.axhline(0.95, color="#ff7f0e", linestyle="--", linewidth=1)
    plt.xticks(x, [f"Block {b}" for b in shown["spatial_block"]])
    plt.ylim(0, 1.05)
    plt.ylabel("Empirical interval coverage")
    plt.title("Combined space-time prediction-interval coverage in 2024")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    save_plot(destination)


def write_manuscript_text(
    destination: Path,
    pooled: pd.DataFrame,
    seasonal: pd.DataFrame,
    extreme: pd.DataFrame,
    interval_metrics: pd.DataFrame,
) -> None:
    full = pooled.loc[pooled["specification"].eq("Full model")].iloc[0]
    baseline = pooled.loc[
        pooled["specification"].eq("Spatial-seasonal baseline")
    ].iloc[0]
    no_coordinates = pooled.loc[
        pooled["specification"].eq("Atmospheric predictors (no coordinates)")
    ].iloc[0]
    no_satellite = pooled.loc[
        pooled["specification"].eq("Full model without satellite")
    ].iloc[0]
    premonsoon = seasonal.loc[seasonal["season"].eq("Pre-monsoon")].iloc[0]
    monsoon = seasonal.loc[seasonal["season"].eq("Monsoon")].iloc[0]
    upper = extreme.loc[
        extreme["observed_concentration_group"].astype(str).eq("Highest 5%")
    ].iloc[0]
    interval = interval_metrics.loc[
        interval_metrics["spatial_block"].astype(str).eq("Pooled")
    ].iloc[0]

    full_vs_baseline = 100 * (baseline["RMSE_log"] - full["RMSE_log"]) / baseline["RMSE_log"]
    satellite_gain = 100 * (no_satellite["RMSE_log"] - full["RMSE_log"]) / no_satellite["RMSE_log"]

    text = f"""MANUSCRIPT-READY ENHANCEMENT TEXT

Methods — predictor-group ablation
To distinguish atmospheric predictive information from stable geographic and
seasonal structure, we repeated the five-fold spatial-block validation using
eight prespecified predictor groups. All dates from a grid cell remained in a
single fold, the Histogram Gradient Boosting configuration was held constant,
and only 2019–2024 records were used. The specifications ranged from a calendar-
only model and a spatial-seasonal baseline to meteorology-only, satellite-only,
coordinate-free atmospheric, and complete models. The locked model selection
and independent 2025 test result were not altered.

Results — incremental predictor contribution
The full model obtained pooled spatial out-of-fold log-RMSE={full['RMSE_log']:.3f},
raw-scale RMSE={full['RMSE_ug_m3']:.2f} µg m-3, and R2={full['R2_raw']:.3f}.
Relative to the spatial-seasonal baseline (log-RMSE={baseline['RMSE_log']:.3f}),
the full model changed log-RMSE by {full_vs_baseline:+.1f}%. The atmospheric
predictor model without coordinates obtained log-RMSE={no_coordinates['RMSE_log']:.3f}.
Removing the satellite predictor group changed log-RMSE from
{full['RMSE_log']:.3f} to {no_satellite['RMSE_log']:.3f}, corresponding to a
{satellite_gain:+.1f}% full-model improvement relative to the no-satellite model.

Results — heterogeneous error
Performance differed substantially by season. Raw-scale RMSE was
{premonsoon['RMSE_ug_m3']:.2f} µg m-3 during the pre-monsoon season and
{monsoon['RMSE_ug_m3']:.2f} µg m-3 during the monsoon. For observations in the
highest 5% of OCSMASS, mean prediction bias was {upper['mean_bias_ug_m3']:.2f}
µg m-3 and RMSE was {upper['RMSE_ug_m3']:.2f} µg m-3. These results show that
pooled performance metrics conceal lower skill during high-concentration events.

Methods and results — combined space-time uncertainty test
For a stricter post-selection test, each geographic block was withheld in turn.
The full model was trained on 2019–2022 observations from the remaining blocks,
absolute log-scale residuals from 2023 in those blocks were used for split-
conformal calibration, and predictions were evaluated in the withheld block in
2024. Across all 68 grid cells, the pooled 2024 raw-scale RMSE was
{interval['RMSE_ug_m3']:.2f} µg m-3. The nominal 80% and 95% intervals achieved
empirical coverages of {100*interval['empirical_80_coverage']:.1f}% and
{100*interval['empirical_95_coverage']:.1f}%, respectively. This analysis
evaluates uncertainty under simultaneous temporal and geographic transfer and
does not use the locked 2025 test set.
"""
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    start = time.perf_counter()
    iterations = 40 if args.quick else 300

    feature_file = locate_file(
        args.features, DEFAULT_FEATURE_FILE, "Bangladesh_OA_Model_Features_2019_2025.csv"
    )
    block_file = locate_file(
        args.blocks, DEFAULT_BLOCK_FILE, "01_spatial_block_assignments.csv"
    )
    output = args.output
    table_folder = output / "Tables"
    figure_folder = output / "Figures"
    table_folder.mkdir(parents=True, exist_ok=True)
    figure_folder.mkdir(parents=True, exist_ok=True)

    print("Reading model feature table...")
    data = pd.read_csv(feature_file, parse_dates=["date"], low_memory=False)
    data = prepare_data(data)
    blocks = pd.read_csv(block_file)
    data = attach_blocks(data, blocks)

    development = data[data["year"].between(2019, 2024)].copy()
    excluded_2025_rows = int(data["year"].eq(2025).sum())
    if len(development) != 149_056:
        raise ValueError(f"Expected 149,056 development rows; found {len(development):,}.")
    if development["year"].max() >= 2025:
        raise RuntimeError("2025 leakage detected in development data.")

    print("\nRunning predictor-group spatial ablation...")
    fold_metrics, pooled_metrics, full_predictions = run_ablation(
        development, iterations
    )

    baseline_rmse = float(
        pooled_metrics.loc[
            pooled_metrics["specification"].eq("Spatial-seasonal baseline"),
            "RMSE_log",
        ].iloc[0]
    )
    pooled_metrics["log_RMSE_improvement_vs_spatial_seasonal_percent"] = (
        100 * (baseline_rmse - pooled_metrics["RMSE_log"]) / baseline_rmse
    )
    pooled_metrics["feature_list"] = pooled_metrics["specification"].map(
        {k: " | ".join(v) for k, v in ABLATION_SPECIFICATIONS.items()}
    )

    fold_metrics.to_csv(table_folder / "01_ablation_fold_metrics.csv", index=False)
    pooled_metrics.to_csv(table_folder / "02_ablation_pooled_metrics.csv", index=False)

    print("\nCalculating seasonal, extreme-event, and grid diagnostics...")
    diagnostic_predictions = add_diagnostic_groups(full_predictions)
    seasonal = group_metrics(diagnostic_predictions, "season")
    by_block = group_metrics(diagnostic_predictions, "spatial_block")
    extreme = group_metrics(diagnostic_predictions, "observed_concentration_group")
    by_grid = group_metrics(diagnostic_predictions, "grid_id")
    coordinates = diagnostic_predictions.groupby("grid_id", as_index=False)[
        ["latitude", "longitude", "spatial_block"]
    ].first()
    by_grid = by_grid.merge(coordinates, on="grid_id", how="left", validate="one_to_one")

    seasonal.to_csv(table_folder / "03_full_model_seasonal_metrics.csv", index=False)
    by_block.to_csv(table_folder / "04_full_model_spatial_block_metrics.csv", index=False)
    extreme.to_csv(table_folder / "05_full_model_extreme_metrics.csv", index=False)
    by_grid.to_csv(table_folder / "06_full_model_grid_metrics.csv", index=False)
    diagnostic_predictions.to_csv(
        table_folder / "07_full_model_spatial_oof_predictions.csv.gz",
        index=False,
        compression="gzip",
        date_format="%Y-%m-%d",
    )

    print("\nRunning combined space-time uncertainty validation...")
    interval_metrics, interval_predictions = run_spatiotemporal_intervals(
        development, iterations
    )
    interval_metrics.to_csv(
        table_folder / "08_spatiotemporal_interval_metrics.csv", index=False
    )
    interval_predictions.to_csv(
        table_folder / "09_spatiotemporal_interval_predictions_2024.csv.gz",
        index=False,
        compression="gzip",
        date_format="%Y-%m-%d",
    )

    print("\nCreating figures...")
    plot_ablation(pooled_metrics, figure_folder / "01_predictor_group_ablation.png")
    plot_seasonal_error(seasonal, figure_folder / "02_seasonal_spatial_error.png")
    plot_extreme_bias(extreme, figure_folder / "03_extreme_concentration_bias.png")
    plot_interval_coverage(
        interval_metrics, figure_folder / "04_spatiotemporal_interval_coverage.png"
    )

    design = {
        "purpose": "Post-selection research enhancement",
        "primary_model_changed": False,
        "locked_2025_test_used": False,
        "excluded_2025_rows": excluded_2025_rows,
        "development_years": [2019, 2020, 2021, 2022, 2023, 2024],
        "ablation_block_method": "Existing five geographic blocks; each grid remains in one fold",
        "ablation_specifications": ABLATION_SPECIFICATIONS,
        "uncertainty_training_years": [2019, 2020, 2021, 2022],
        "uncertainty_calibration_year": 2023,
        "uncertainty_test_year": 2024,
        "uncertainty_test_design": "Withhold one complete spatial block at a time",
        "interval_method": "Absolute log-residual split conformal with finite-sample quantile",
        "boosting_iterations": iterations,
        "random_seed": RANDOM_SEED,
        "quick_test": bool(args.quick),
    }
    (table_folder / "10_analysis_design_record.json").write_text(
        json.dumps(design, indent=2), encoding="utf-8"
    )

    write_manuscript_text(
        output / "Manuscript_ready_enhancement_text.txt",
        pooled_metrics,
        seasonal,
        extreme,
        interval_metrics,
    )

    elapsed_minutes = (time.perf_counter() - start) / 60
    report = f"""Bangladesh OCSMASS research-enhancement run report

Feature table: {feature_file}
Spatial blocks: {block_file}
Rows used: {len(development):,} (2019-2024 only)
Rows from 2025 excluded before all fitting: {excluded_2025_rows:,}
Grid cells: {development['grid_id'].nunique()}
Spatial blocks: {development['spatial_block'].nunique()}
Boosting iterations: {iterations}
Quick technical test: {args.quick}
Runtime: {elapsed_minutes:.1f} minutes

The primary model, model selection, and locked 2025 final test were not changed.
"""
    (output / "Research_enhancement_run_report.txt").write_text(
        report, encoding="utf-8"
    )

    script_path = Path(__file__).resolve()
    if script_path.parent != output.resolve():
        shutil.copy2(script_path, output / script_path.name)

    archive_path = shutil.make_archive(
        str(output), "zip", root_dir=output.parent, base_dir=output.name
    )

    print("\nCompleted successfully.")
    print(f"Results folder: {output}")
    print(f"ZIP archive: {archive_path}")
    print("\nAblation pooled results:")
    print(
        pooled_metrics[
            [
                "specification", "n_features", "RMSE_log", "RMSE_ug_m3",
                "R2_raw", "log_RMSE_improvement_vs_spatial_seasonal_percent",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
