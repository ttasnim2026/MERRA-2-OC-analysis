# -*- coding: utf-8 -*-
"""Spatial-block validation for the Bangladesh OCSMASS model.

This is a post-selection robustness analysis. It intentionally excludes every
2025 record before creating folds or fitting models. It does not modify the
locked primary model and must not be used to retune that model based on the
already observed 2025 test result.

The 68 MERRA-2 grid cells are grouped into five geographic blocks with K-means
clustering of latitude and longitude. In each fold, one complete block is held
out, so every date belonging to its grid cells remains outside training.

Two fixed Histogram Gradient Boosting specifications are compared:
    1. Full model.
    2. Full model without latitude and longitude.
"""

from pathlib import Path
import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. PATHS AND FIXED SETTINGS
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

OUTPUT_FOLDER = PROJECT_FOLDER / "Spatial_Validation_Results_2019_2024"
TABLE_FOLDER = OUTPUT_FOLDER / "Tables"
FIGURE_FOLDER = OUTPUT_FOLDER / "Figures"

TARGET_RAW = "OCSMASS_ug_m3"
TARGET_LOG = "log1p_OCSMASS"
RANDOM_SEED = 42
N_SPATIAL_BLOCKS = 5
EXPECTED_GRID_CELLS = 68
EXPECTED_DEVELOPMENT_ROWS = 149_056

# This option is only for a short technical test. Leave it unset during the
# actual analysis.
QUICK_TEST = os.environ.get("OA_QUICK_TEST", "0") == "1"
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

FULL_WITHOUT_COORDINATES = [
    feature
    for feature in FULL_FEATURES
    if feature not in {"latitude", "longitude"}
]

MODEL_SPECIFICATIONS = {
    "Full model": FULL_FEATURES,
    "Full model without coordinates": FULL_WITHOUT_COORDINATES,
}


# =============================================================================
# 2. GENERAL HELPERS
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


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def calculate_metrics(
    y_true_raw: np.ndarray,
    predicted_log: np.ndarray,
) -> dict:
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_true_log = np.log1p(y_true_raw)
    predicted_log = np.asarray(predicted_log, dtype=float)
    predicted_raw = np.maximum(np.expm1(predicted_log), 0.0)

    return {
        "n_rows": int(len(y_true_raw)),
        "MAE_ug_m3": float(mean_absolute_error(y_true_raw, predicted_raw)),
        "RMSE_ug_m3": float(
            mean_squared_error(y_true_raw, predicted_raw) ** 0.5
        ),
        "R2_raw": float(r2_score(y_true_raw, predicted_raw)),
        "mean_bias_ug_m3": float(np.mean(predicted_raw - y_true_raw)),
        "Pearson_r_raw": safe_pearson(y_true_raw, predicted_raw),
        "MAE_log": float(mean_absolute_error(y_true_log, predicted_log)),
        "RMSE_log": float(
            mean_squared_error(y_true_log, predicted_log) ** 0.5
        ),
        "R2_log": float(r2_score(y_true_log, predicted_log)),
    }


def build_hgb() -> HistGradientBoostingRegressor:
    """Return the fixed primary HGB specification; no tuning occurs here."""
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
# 3. DATA VALIDATION AND PREPARATION
# =============================================================================

def validate_input(df: pd.DataFrame) -> None:
    required = {
        "date",
        "year",
        "grid_id",
        "latitude",
        "longitude",
        TARGET_RAW,
        TARGET_LOG,
        *FULL_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    duplicate_count = int(df.duplicated(["date", "grid_id"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate date-grid records."
        )

    if df[TARGET_RAW].isna().any() or df[TARGET_LOG].isna().any():
        raise ValueError("The target contains missing values.")

    if (df[TARGET_RAW] < 0).any():
        raise ValueError("The OCSMASS target contains negative values.")

    expected_log = np.log1p(df[TARGET_RAW].to_numpy(dtype=float))
    if not np.allclose(
        expected_log,
        df[TARGET_LOG].to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-8,
    ):
        raise ValueError("The stored log target is inconsistent with OCSMASS.")

    coordinate_counts = df.groupby("grid_id")[["latitude", "longitude"]].nunique()
    conflicting = coordinate_counts.gt(1).any(axis=1)
    if conflicting.any():
        bad_ids = coordinate_counts.index[conflicting].tolist()[:10]
        raise ValueError(
            "Some grid IDs have conflicting coordinates. Examples: "
            f"{bad_ids}"
        )


def prepare_development_data(source: pd.DataFrame) -> pd.DataFrame:
    """Remove 2025 first, then apply the locked satellite treatment."""
    data = source.loc[source["year"].between(2019, 2024)].copy()

    # Recalculate flags from the unmodified satellite columns, then set
    # negative retrievals to missing, matching the locked primary analysis.
    for variable in ["HCHO", "NO2"]:
        data[f"{variable}_negative_flag"] = (
            data[variable] < 0
        ).astype("int8")
        data[f"{variable}_missing_flag"] = (
            data[variable].isna()
        ).astype("int8")
        data.loc[data[variable] < 0, variable] = np.nan

    if int(data["year"].max()) != 2024:
        raise RuntimeError("Development data do not end in 2024.")
    if (data["year"] == 2025).any():
        raise RuntimeError("Leakage guard failed: 2025 remains in the data.")
    if len(data) != EXPECTED_DEVELOPMENT_ROWS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_DEVELOPMENT_ROWS:,} rows for 2019-2024, "
            f"but found {len(data):,}."
        )
    if data["grid_id"].nunique() != EXPECTED_GRID_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_GRID_CELLS} grid cells, but found "
            f"{data['grid_id'].nunique()}."
        )

    return data


def create_spatial_blocks(data: pd.DataFrame) -> pd.DataFrame:
    """Create five geographically compact, reproducible validation blocks."""
    grids = (
        data[["grid_id", "latitude", "longitude"]]
        .drop_duplicates()
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    coordinates = grids[["latitude", "longitude"]].to_numpy(dtype=float)
    scaled_coordinates = StandardScaler().fit_transform(coordinates)

    kmeans = KMeans(
        n_clusters=N_SPATIAL_BLOCKS,
        random_state=RANDOM_SEED,
        n_init=50,
    )
    raw_labels = kmeans.fit_predict(scaled_coordinates)

    # Relabel clusters deterministically from west to east, then south to north.
    grids["raw_block"] = raw_labels
    centers = (
        grids.groupby("raw_block", as_index=False)
        .agg(center_longitude=("longitude", "mean"),
             center_latitude=("latitude", "mean"))
        .sort_values(["center_longitude", "center_latitude"])
        .reset_index(drop=True)
    )
    block_map = {
        int(raw): int(index + 1)
        for index, raw in enumerate(centers["raw_block"])
    }
    grids["spatial_block"] = grids["raw_block"].map(block_map).astype(int)
    grids = grids.drop(columns="raw_block")

    counts = grids.groupby("spatial_block")["grid_id"].nunique()
    if len(counts) != N_SPATIAL_BLOCKS or (counts == 0).any():
        raise RuntimeError("Spatial-block creation failed.")

    return grids


# =============================================================================
# 4. FIVE-FOLD SPATIAL-BLOCK VALIDATION
# =============================================================================

def run_spatial_validation(
    data: pd.DataFrame,
    block_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_lookup = block_assignments.set_index("grid_id")["spatial_block"]
    data = data.copy()
    data["spatial_block"] = data["grid_id"].map(block_lookup)

    if data["spatial_block"].isna().any():
        raise RuntimeError("At least one row did not receive a spatial block.")

    metric_rows = []
    prediction_tables = []

    for specification, features in MODEL_SPECIFICATIONS.items():
        print(f"\n{specification}")

        for block in sorted(data["spatial_block"].unique()):
            start_time = time.perf_counter()
            train_mask = data["spatial_block"] != block
            validation_mask = data["spatial_block"] == block

            train_grid_ids = set(data.loc[train_mask, "grid_id"].unique())
            validation_grid_ids = set(
                data.loc[validation_mask, "grid_id"].unique()
            )
            overlap = train_grid_ids & validation_grid_ids
            if overlap:
                raise RuntimeError(
                    f"Grid leakage detected in block {block}: {sorted(overlap)}"
                )

            X_train = data.loc[train_mask, features]
            y_train = data.loc[train_mask, TARGET_LOG]
            X_validation = data.loc[validation_mask, features]
            y_validation_raw = data.loc[validation_mask, TARGET_RAW]

            model = build_hgb()
            model.fit(X_train, y_train)
            predicted_log = model.predict(X_validation)
            predicted_raw = np.maximum(np.expm1(predicted_log), 0.0)

            metrics = calculate_metrics(
                y_validation_raw.to_numpy(dtype=float),
                predicted_log,
            )
            metric_rows.append(
                {
                    "analysis": "five_fold_spatial_block_validation",
                    "specification": specification,
                    "fold": f"Spatial block {int(block)}",
                    "spatial_block": int(block),
                    "n_features": len(features),
                    "n_train_grid_cells": len(train_grid_ids),
                    "n_validation_grid_cells": len(validation_grid_ids),
                    "n_train_rows": int(train_mask.sum()),
                    **metrics,
                }
            )

            fold_predictions = data.loc[
                validation_mask,
                [
                    "date",
                    "year",
                    "grid_id",
                    "latitude",
                    "longitude",
                    "spatial_block",
                    TARGET_RAW,
                ],
            ].copy()
            fold_predictions["specification"] = specification
            fold_predictions["predicted_OCSMASS_ug_m3"] = predicted_raw
            fold_predictions["observed_log1p_OCSMASS"] = np.log1p(
                fold_predictions[TARGET_RAW].to_numpy(dtype=float)
            )
            fold_predictions["predicted_log1p_OCSMASS"] = predicted_log
            fold_predictions["residual_ug_m3"] = (
                fold_predictions["predicted_OCSMASS_ug_m3"]
                - fold_predictions[TARGET_RAW]
            )
            prediction_tables.append(fold_predictions)

            print(
                f"  Block {int(block)}: "
                f"{len(validation_grid_ids)} held-out grids, "
                f"log-RMSE {metrics['RMSE_log']:.4f}, "
                f"completed in {elapsed_text(start_time)}"
            )

    metrics_table = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    return metrics_table, predictions


# =============================================================================
# 5. SUMMARIES AND GRID-LEVEL DIAGNOSTICS
# =============================================================================

METRIC_COLUMNS = [
    "MAE_ug_m3",
    "RMSE_ug_m3",
    "R2_raw",
    "mean_bias_ug_m3",
    "Pearson_r_raw",
    "MAE_log",
    "RMSE_log",
    "R2_log",
]


def summarize_fold_metrics(metrics_table: pd.DataFrame) -> pd.DataFrame:
    summary = metrics_table.groupby("specification")[METRIC_COLUMNS].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]
    return summary.reset_index()


def calculate_pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for specification, group in predictions.groupby("specification"):
        metrics = calculate_metrics(
            group[TARGET_RAW].to_numpy(dtype=float),
            group["predicted_log1p_OCSMASS"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "specification": specification,
                "validation_type": "pooled spatial out-of-fold predictions",
                "n_unique_grid_cells": int(group["grid_id"].nunique()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def calculate_grid_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "specification",
        "grid_id",
        "latitude",
        "longitude",
        "spatial_block",
    ]
    for keys, group in predictions.groupby(group_columns, sort=True):
        metrics = calculate_metrics(
            group[TARGET_RAW].to_numpy(dtype=float),
            group["predicted_log1p_OCSMASS"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "specification": keys[0],
                "grid_id": keys[1],
                "latitude": keys[2],
                "longitude": keys[3],
                "spatial_block": int(keys[4]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 6. FIGURES
# =============================================================================

def plot_spatial_blocks(block_assignments: pd.DataFrame) -> None:
    plt.figure(figsize=(8.2, 7.0))
    scatter = plt.scatter(
        block_assignments["longitude"],
        block_assignments["latitude"],
        c=block_assignments["spatial_block"],
        cmap="tab10",
        s=95,
        edgecolor="black",
        linewidth=0.6,
    )
    colorbar = plt.colorbar(scatter, ticks=range(1, N_SPATIAL_BLOCKS + 1))
    colorbar.set_label("Spatial block")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Five geographic validation blocks")
    plt.grid(alpha=0.25)
    save_figure("01_spatial_block_assignments.png")


def plot_fold_performance(metrics_table: pd.DataFrame) -> None:
    pivot = metrics_table.pivot(
        index="spatial_block",
        columns="specification",
        values="RMSE_log",
    ).sort_index()

    ax = pivot.plot(kind="bar", figsize=(10.5, 6.2), width=0.78)
    ax.set_xlabel("Held-out spatial block")
    ax.set_ylabel("Log-scale RMSE")
    ax.set_title("Spatial-block validation performance")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Model specification")
    save_figure("02_spatial_block_fold_performance.png")


def plot_pooled_comparison(pooled_metrics: pd.DataFrame) -> None:
    plot_data = pooled_metrics.sort_values("RMSE_log")
    plt.figure(figsize=(9.2, 4.8))
    bars = plt.barh(
        plot_data["specification"],
        plot_data["RMSE_log"],
        color=["#2b7bba", "#d98f3d"],
    )
    for bar, value in zip(bars, plot_data["RMSE_log"]):
        plt.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.4f}",
            va="center",
        )
    plt.xlabel("Pooled out-of-fold log-scale RMSE")
    plt.ylabel("")
    plt.title("Effect of removing latitude and longitude")
    plt.grid(axis="x", alpha=0.3)
    save_figure("03_pooled_spatial_performance.png")


def plot_observed_vs_predicted(predictions: pd.DataFrame) -> None:
    full = predictions.loc[
        predictions["specification"] == "Full model"
    ].copy()

    observed = full[TARGET_RAW].to_numpy(dtype=float)
    predicted = full["predicted_OCSMASS_ug_m3"].to_numpy(dtype=float)
    upper = float(np.nanpercentile(np.concatenate([observed, predicted]), 99.5))

    plt.figure(figsize=(7.2, 6.5))
    plt.hexbin(
        observed,
        predicted,
        gridsize=65,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    plt.plot([0, upper], [0, upper], linestyle="--", color="red", linewidth=1.5)
    plt.xlim(0, upper)
    plt.ylim(0, upper)
    plt.xlabel("Observed OCSMASS (µg m⁻³)")
    plt.ylabel("Spatial out-of-fold prediction (µg m⁻³)")
    plt.title("Observed versus spatial out-of-fold predictions")
    colorbar = plt.colorbar()
    colorbar.set_label("Log₁₀ observation count")
    save_figure("04_observed_vs_spatial_oof_prediction.png")


def plot_grid_rmse(grid_metrics: pd.DataFrame) -> None:
    full = grid_metrics.loc[
        grid_metrics["specification"] == "Full model"
    ].copy()
    plt.figure(figsize=(8.2, 7.0))
    scatter = plt.scatter(
        full["longitude"],
        full["latitude"],
        c=full["RMSE_log"],
        cmap="magma_r",
        s=105,
        edgecolor="black",
        linewidth=0.5,
    )
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Grid-level log-scale RMSE")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial distribution of out-of-fold error")
    plt.grid(alpha=0.25)
    save_figure("05_grid_level_spatial_error.png")


# =============================================================================
# 7. SAVE REPRODUCIBILITY RECORDS
# =============================================================================

def save_design_record(
    block_assignments: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
) -> None:
    full_rmse = float(
        pooled_metrics.loc[
            pooled_metrics["specification"] == "Full model",
            "RMSE_log",
        ].iloc[0]
    )
    no_coordinate_rmse = float(
        pooled_metrics.loc[
            pooled_metrics["specification"]
            == "Full model without coordinates",
            "RMSE_log",
        ].iloc[0]
    )

    coordinate_effect_percent = (
        (no_coordinate_rmse - full_rmse) / full_rmse * 100
    )

    design = {
        "purpose": "Post-selection spatial-block robustness analysis",
        "years_used": [2019, 2020, 2021, 2022, 2023, 2024],
        "year_2025_used": False,
        "locked_primary_model_changed": False,
        "number_of_grid_cells": int(block_assignments["grid_id"].nunique()),
        "number_of_spatial_blocks": N_SPATIAL_BLOCKS,
        "block_method": (
            "K-means clustering of standardized grid-cell latitude and "
            "longitude; all dates from each grid cell remain in one fold"
        ),
        "primary_model": "Histogram Gradient Boosting",
        "primary_target": "log1p_OCSMASS",
        "negative_satellite_treatment": "set negative values to missing",
        "full_model_pooled_log_RMSE": full_rmse,
        "no_coordinate_model_pooled_log_RMSE": no_coordinate_rmse,
        "log_RMSE_change_without_coordinates_percent": (
            coordinate_effect_percent
        ),
        "note": (
            "These results provide robustness evidence only and must not be "
            "used to retune the locked model based on the known 2025 result."
        ),
    }

    with open(
        TABLE_FOLDER / "07_analysis_design_record.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(design, file, indent=2)


def write_run_report(
    metrics_table: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    block_assignments: pd.DataFrame,
) -> None:
    full = pooled_metrics.loc[
        pooled_metrics["specification"] == "Full model"
    ].iloc[0]
    no_coordinates = pooled_metrics.loc[
        pooled_metrics["specification"]
        == "Full model without coordinates"
    ].iloc[0]

    change_percent = (
        (no_coordinates["RMSE_log"] - full["RMSE_log"])
        / full["RMSE_log"]
        * 100
    )

    block_counts = (
        block_assignments.groupby("spatial_block")["grid_id"]
        .nunique()
        .sort_index()
    )
    count_text = ", ".join(
        f"block {block}: {count} grids"
        for block, count in block_counts.items()
    )

    report_lines = [
        "Bangladesh OCSMASS spatial-block validation report",
        "",
        "Leakage protection:",
        "Only 2019-2024 records were used.",
        "All 2025 records were removed before block creation and fitting.",
        "Every grid cell and all its dates appeared in one validation fold.",
        "The locked primary model and 2025 final-test result were not changed.",
        "",
        "Spatial design:",
        f"{N_SPATIAL_BLOCKS} geographic blocks were formed from 68 grid cells.",
        count_text,
        "",
        "Pooled spatial out-of-fold results:",
        f"Full model log-RMSE: {full['RMSE_log']:.6f}",
        f"Full model raw RMSE: {full['RMSE_ug_m3']:.6f} µg m^-3",
        f"Full model raw R²: {full['R2_raw']:.6f}",
        f"Full model Pearson r: {full['Pearson_r_raw']:.6f}",
        (
            "Without-coordinate model log-RMSE: "
            f"{no_coordinates['RMSE_log']:.6f}"
        ),
        (
            "Log-RMSE change after removing coordinates: "
            f"{change_percent:+.2f}%"
        ),
        "",
        "Interpretation warning:",
        "Spatial interpolation performance does not establish causality.",
        "The target is MERRA-2 surface organic carbon, not direct SOA.",
        "This test evaluates withheld grid cells within the Bangladesh domain;",
        "it does not establish transferability outside that domain.",
    ]

    with open(
        OUTPUT_FOLDER / "Spatial_validation_run_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(report_lines))


# =============================================================================
# 8. MAIN PROGRAM
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ensure_output_folders()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "The prepared modeling table was not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run the EDA/feature-preparation script first or correct "
            "PROJECT_FOLDER."
        )

    print("Reading the prepared feature table...")
    source = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    validate_input(source)

    data = prepare_development_data(source)
    del source

    print(
        f"Confirmed {len(data):,} development rows, "
        f"{data['grid_id'].nunique()} grid cells, and no 2025 records."
    )

    print("Creating five geographic validation blocks...")
    block_assignments = create_spatial_blocks(data)
    block_assignments.to_csv(
        TABLE_FOLDER / "01_spatial_block_assignments.csv",
        index=False,
    )

    print("Running spatial-block validation...")
    metrics_table, predictions = run_spatial_validation(
        data,
        block_assignments,
    )

    expected_prediction_rows = len(data) * len(MODEL_SPECIFICATIONS)
    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(
            "Out-of-fold predictions are incomplete: expected "
            f"{expected_prediction_rows:,}, found {len(predictions):,}."
        )

    duplicate_predictions = int(
        predictions.duplicated(
            ["specification", "date", "grid_id"]
        ).sum()
    )
    if duplicate_predictions:
        raise RuntimeError(
            f"Found {duplicate_predictions:,} duplicate out-of-fold predictions."
        )

    fold_summary = summarize_fold_metrics(metrics_table)
    pooled_metrics = calculate_pooled_metrics(predictions)
    grid_metrics = calculate_grid_metrics(predictions)

    metrics_table.to_csv(
        TABLE_FOLDER / "02_spatial_block_fold_metrics.csv",
        index=False,
    )
    fold_summary.to_csv(
        TABLE_FOLDER / "03_spatial_block_metric_summary.csv",
        index=False,
    )
    pooled_metrics.to_csv(
        TABLE_FOLDER / "04_spatial_oof_pooled_metrics.csv",
        index=False,
    )
    grid_metrics.to_csv(
        TABLE_FOLDER / "05_grid_level_spatial_metrics.csv",
        index=False,
    )

    prediction_columns = [
        "date",
        "year",
        "grid_id",
        "latitude",
        "longitude",
        "spatial_block",
        "specification",
        TARGET_RAW,
        "predicted_OCSMASS_ug_m3",
        "observed_log1p_OCSMASS",
        "predicted_log1p_OCSMASS",
        "residual_ug_m3",
    ]
    predictions[prediction_columns].to_csv(
        TABLE_FOLDER / "06_spatial_oof_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    save_design_record(block_assignments, pooled_metrics)
    write_run_report(metrics_table, pooled_metrics, block_assignments)

    print("Creating figures...")
    plot_spatial_blocks(block_assignments)
    plot_fold_performance(metrics_table)
    plot_pooled_comparison(pooled_metrics)
    plot_observed_vs_predicted(predictions)
    plot_grid_rmse(grid_metrics)

    print("\n" + "=" * 78)
    print("SPATIAL-BLOCK VALIDATION COMPLETED")
    print("=" * 78)
    print("Only 2019-2024 data were used; 2025 was excluded.")
    print(f"Tables saved in: {TABLE_FOLDER}")
    print(f"Figures saved in: {FIGURE_FOLDER}")
    print("The locked primary model and original feature table were not modified.")


if __name__ == "__main__":
    main()
