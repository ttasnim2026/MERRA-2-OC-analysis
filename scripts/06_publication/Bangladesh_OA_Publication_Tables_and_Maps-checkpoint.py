# -*- coding: utf-8 -*-
"""Create manuscript-ready tables and figures for the Bangladesh OCSMASS study.

This script performs no model fitting or tuning. It consolidates outputs from
the completed EDA, locked-model, robustness, and spatial-validation analyses.
The 2025 final test is reported exactly as previously evaluated.

Expected project folders
------------------------
PROJECT_FOLDER/
    EDA_Results_2019_2025/
    Modeling_Results_2019_2025/
    Robustness_Results_2019_2024/
    Spatial_Validation_Results_2019_2024/

Created output
--------------
PROJECT_FOLDER/Publication_Outputs/
    Tables/
    Figures/
    Publication_outputs_run_report.txt
"""

from pathlib import Path
import json
import os
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_FOLDER = Path(
    os.environ.get(
        "OA_PROJECT_FOLDER",
        str(Path.cwd()),
    )
)

EDA_FOLDER = PROJECT_FOLDER / "EDA_Results_2019_2025"
MODEL_FOLDER = PROJECT_FOLDER / "Modeling_Results_2019_2025"
ROBUSTNESS_FOLDER = PROJECT_FOLDER / "Robustness_Results_2019_2024"
SPATIAL_FOLDER = PROJECT_FOLDER / "Spatial_Validation_Results_2019_2024"

OUTPUT_FOLDER = PROJECT_FOLDER / "Publication_Outputs"
TABLE_FOLDER = OUTPUT_FOLDER / "Tables"
FIGURE_FOLDER = OUTPUT_FOLDER / "Figures"

FEATURE_DATA_FILE = (
    EDA_FOLDER
    / "Model_Data"
    / "Bangladesh_OA_Model_Features_2019_2025.csv"
)

VALIDATION_METRICS_FILE = (
    MODEL_FOLDER / "Tables" / "01_validation_metrics_2024.csv"
)
VALIDATION_PREDICTIONS_FILE = (
    MODEL_FOLDER
    / "Tables"
    / "03_selected_model_validation_predictions_2024.csv"
)
FEATURE_IMPORTANCE_FILE = (
    MODEL_FOLDER
    / "Tables"
    / "04_selected_model_permutation_importance_2024.csv"
)
TEST_METRICS_FILE = (
    MODEL_FOLDER / "Tables" / "06_final_test_metrics_2025.csv"
)
TEST_PREDICTIONS_FILE = (
    MODEL_FOLDER / "Tables" / "07_final_test_predictions_2025.csv"
)
MODEL_SELECTION_RECORD_FILE = (
    MODEL_FOLDER / "Tables" / "08_model_selection_record.json"
)

EXPANDING_METRICS_FILE = (
    ROBUSTNESS_FOLDER
    / "Tables"
    / "01_expanding_year_model_metrics.csv"
)
EXPANDING_SUMMARY_FILE = (
    ROBUSTNESS_FOLDER
    / "Tables"
    / "02_expanding_year_model_summary.csv"
)
ABLATION_SUMMARY_FILE = (
    ROBUSTNESS_FOLDER
    / "Tables"
    / "04_feature_group_ablation_summary.csv"
)
SENSITIVITY_FILE = (
    ROBUSTNESS_FOLDER
    / "Tables"
    / "05_preprocessing_model_sensitivity_2024.csv"
)
ROBUSTNESS_DESIGN_FILE = (
    ROBUSTNESS_FOLDER
    / "Tables"
    / "07_analysis_design_record.json"
)

SPATIAL_FOLD_METRICS_FILE = (
    SPATIAL_FOLDER
    / "Tables"
    / "02_spatial_block_fold_metrics.csv"
)
SPATIAL_POOLED_METRICS_FILE = (
    SPATIAL_FOLDER
    / "Tables"
    / "04_spatial_oof_pooled_metrics.csv"
)
SPATIAL_GRID_METRICS_FILE = (
    SPATIAL_FOLDER
    / "Tables"
    / "05_grid_level_spatial_metrics.csv"
)
SPATIAL_DESIGN_FILE = (
    SPATIAL_FOLDER
    / "Tables"
    / "07_analysis_design_record.json"
)

TARGET = "OCSMASS_ug_m3"
PREDICTED = "predicted_OCSMASS_ug_m3"
LOG_TARGET = "log1p_OCSMASS"
LOG_PREDICTED = "predicted_log1p_OCSMASS"
SELECTED_MODEL = "Histogram Gradient Boosting"


# =============================================================================
# 2. STYLE AND GENERAL HELPERS
# =============================================================================

COLORS = {
    "blue": "#2F6B9A",
    "orange": "#D97732",
    "green": "#3E8E6D",
    "red": "#B84A4A",
    "gray": "#6E7781",
    "light_gray": "#D9DEE3",
}

SEASON_ORDER = ["winter", "pre-monsoon", "monsoon", "post-monsoon"]
SEASON_LABELS = {
    "winter": "Winter",
    "pre-monsoon": "Pre-monsoon",
    "monsoon": "Monsoon",
    "post-monsoon": "Post-monsoon",
}
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_output_folders() -> None:
    TABLE_FOLDER.mkdir(parents=True, exist_ok=True)
    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)


def save_figure(filename: str) -> None:
    plt.savefig(
        FIGURE_FOLDER / filename,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def safe_pearson(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if len(observed) < 2:
        return np.nan
    if np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan
    return float(np.corrcoef(observed, predicted)[0, 1])


def calculate_metrics(observed: pd.Series, predicted: pd.Series) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = np.maximum(predicted[valid], 0.0)
    observed_log = np.log1p(observed)
    predicted_log = np.log1p(predicted)

    return {
        "n_rows": int(len(observed)),
        "MAE_ug_m3": float(mean_absolute_error(observed, predicted)),
        "RMSE_ug_m3": float(mean_squared_error(observed, predicted) ** 0.5),
        "R2_raw": float(r2_score(observed, predicted)),
        "mean_bias_ug_m3": float(np.mean(predicted - observed)),
        "Pearson_r_raw": safe_pearson(observed, predicted),
        "MAE_log": float(mean_absolute_error(observed_log, predicted_log)),
        "RMSE_log": float(mean_squared_error(observed_log, predicted_log) ** 0.5),
        "R2_log": float(r2_score(observed_log, predicted_log)),
    }


def require_files() -> None:
    required = [
        FEATURE_DATA_FILE,
        VALIDATION_METRICS_FILE,
        VALIDATION_PREDICTIONS_FILE,
        FEATURE_IMPORTANCE_FILE,
        TEST_METRICS_FILE,
        TEST_PREDICTIONS_FILE,
        MODEL_SELECTION_RECORD_FILE,
        EXPANDING_METRICS_FILE,
        EXPANDING_SUMMARY_FILE,
        ABLATION_SUMMARY_FILE,
        SENSITIVITY_FILE,
        ROBUSTNESS_DESIGN_FILE,
        SPATIAL_FOLD_METRICS_FILE,
        SPATIAL_POOLED_METRICS_FILE,
        SPATIAL_GRID_METRICS_FILE,
        SPATIAL_DESIGN_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required result files are missing. Extract all four result "
            f"folders inside PROJECT_FOLDER. Missing files:\n{formatted}"
        )


# =============================================================================
# 3. LOAD AND VERIFY COMPLETED ANALYSES
# =============================================================================

def load_inputs() -> dict:
    print("Reading completed analysis outputs...")
    data = {
        "features": pd.read_csv(FEATURE_DATA_FILE, parse_dates=["date"]),
        "validation_metrics": pd.read_csv(VALIDATION_METRICS_FILE),
        "validation_predictions": pd.read_csv(
            VALIDATION_PREDICTIONS_FILE,
            encoding="utf-8-sig",
            parse_dates=["date"],
        ),
        "importance": pd.read_csv(FEATURE_IMPORTANCE_FILE),
        "test_metrics": pd.read_csv(TEST_METRICS_FILE),
        "test_predictions": pd.read_csv(
            TEST_PREDICTIONS_FILE,
            encoding="utf-8-sig",
            parse_dates=["date"],
        ),
        "expanding_metrics": pd.read_csv(EXPANDING_METRICS_FILE),
        "expanding_summary": pd.read_csv(EXPANDING_SUMMARY_FILE),
        "ablation": pd.read_csv(ABLATION_SUMMARY_FILE),
        "sensitivity": pd.read_csv(SENSITIVITY_FILE),
        "spatial_fold_metrics": pd.read_csv(SPATIAL_FOLD_METRICS_FILE),
        "spatial_pooled": pd.read_csv(SPATIAL_POOLED_METRICS_FILE),
        "spatial_grid": pd.read_csv(SPATIAL_GRID_METRICS_FILE),
    }

    with open(MODEL_SELECTION_RECORD_FILE, encoding="utf-8") as file:
        data["model_record"] = json.load(file)
    with open(ROBUSTNESS_DESIGN_FILE, encoding="utf-8") as file:
        data["robustness_record"] = json.load(file)
    with open(SPATIAL_DESIGN_FILE, encoding="utf-8") as file:
        data["spatial_record"] = json.load(file)

    # Use one spelling for season names across saved outputs. Earlier scripts
    # stored these labels with underscores (for example, "pre_monsoon"), while
    # publication labels use hyphens.
    data["test_predictions"]["season"] = (
        data["test_predictions"]["season"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", "-", regex=False)
        .str.replace(" ", "-", regex=False)
        .replace(
            {
                "premonsoon": "pre-monsoon",
                "postmonsoon": "post-monsoon",
            }
        )
    )

    return data


def validate_inputs(data: dict) -> None:
    features = data["features"]
    validation_predictions = data["validation_predictions"]
    test_predictions = data["test_predictions"]

    required_feature_columns = {
        "date", "year", "grid_id", "latitude", "longitude", TARGET,
    }
    missing = required_feature_columns - set(features.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")

    if features.duplicated(["date", "grid_id"]).any():
        raise ValueError("Feature table contains duplicate date-grid rows.")
    if features["grid_id"].nunique() != 68:
        raise ValueError("Expected 68 Bangladesh grid cells.")

    for name, table, expected_year, expected_rows in [
        ("validation", validation_predictions, 2024, 24_888),
        ("test", test_predictions, 2025, 24_820),
    ]:
        required = {
            "date", "grid_id", "latitude", "longitude", TARGET, PREDICTED,
        }
        missing_columns = required - set(table.columns)
        if missing_columns:
            raise ValueError(
                f"{name.title()} predictions are missing: "
                f"{sorted(missing_columns)}"
            )
        if len(table) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows:,} {name} predictions, "
                f"found {len(table):,}."
            )
        if table["date"].dt.year.nunique() != 1:
            raise ValueError(f"{name.title()} predictions span multiple years.")
        if int(table["date"].dt.year.iloc[0]) != expected_year:
            raise ValueError(
                f"{name.title()} prediction year is not {expected_year}."
            )
        if table.duplicated(["date", "grid_id"]).any():
            raise ValueError(f"{name.title()} predictions contain duplicates.")

    if data["robustness_record"].get("year_2025_used") is not False:
        raise RuntimeError("Robustness record does not confirm 2025 exclusion.")
    if data["spatial_record"].get("year_2025_used") is not False:
        raise RuntimeError("Spatial record does not confirm 2025 exclusion.")

    if data["spatial_record"].get("locked_primary_model_changed") is not False:
        raise RuntimeError("Spatial analysis indicates the locked model changed.")


def reconcile_saved_metrics(data: dict) -> None:
    """Independently recalculate the two consequential prediction tables."""
    checks = [
        (
            "2024 validation",
            data["validation_predictions"],
            data["validation_metrics"].loc[
                data["validation_metrics"]["model"] == SELECTED_MODEL
            ].iloc[0],
        ),
        (
            "2025 final test",
            data["test_predictions"],
            data["test_metrics"].iloc[0],
        ),
    ]

    for label, predictions, saved in checks:
        recalculated = calculate_metrics(predictions[TARGET], predictions[PREDICTED])
        for metric in [
            "MAE_ug_m3", "RMSE_ug_m3", "R2_raw", "mean_bias_ug_m3",
            "MAE_log", "RMSE_log", "R2_log",
        ]:
            if not np.isclose(
                recalculated[metric],
                float(saved[metric]),
                rtol=1e-7,
                atol=1e-8,
            ):
                raise RuntimeError(
                    f"{label} reconciliation failed for {metric}: "
                    f"recalculated={recalculated[metric]}, saved={saved[metric]}"
                )
        print(f"  Reconciled {label} metrics.")


# =============================================================================
# 4. MANUSCRIPT TABLES
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


def create_consolidated_performance_table(data: dict) -> pd.DataFrame:
    validation_metrics = data["validation_metrics"].loc[
        data["validation_metrics"]["model"] == SELECTED_MODEL
    ].iloc[0]
    test_metrics = data["test_metrics"].iloc[0]
    expanding = data["expanding_summary"].loc[
        data["expanding_summary"]["model"] == SELECTED_MODEL
    ].iloc[0]
    spatial = data["spatial_pooled"].loc[
        data["spatial_pooled"]["specification"] == "Full model"
    ].iloc[0]

    validation_pearson = safe_pearson(
        data["validation_predictions"][TARGET],
        data["validation_predictions"][PREDICTED],
    )
    test_pearson = safe_pearson(
        data["test_predictions"][TARGET],
        data["test_predictions"][PREDICTED],
    )

    rows = [
        {
            "evaluation": "Temporal validation (2024)",
            "role": "Model selection validation",
            "years": "Train 2019-2023; validate 2024",
            "aggregation": "All 2024 date-grid records",
            "n_rows": int(validation_metrics["n_rows"]),
            **{
                metric: (
                    validation_pearson
                    if metric == "Pearson_r_raw"
                    else float(validation_metrics[metric])
                )
                for metric in METRIC_COLUMNS
            },
        },
        {
            "evaluation": "Expanding-year temporal robustness",
            "role": "Post-selection robustness",
            "years": "Validate 2022, 2023, and 2024",
            "aggregation": "Unweighted mean across three validation years",
            "n_rows": int(
                data["expanding_metrics"].loc[
                    data["expanding_metrics"]["model"] == SELECTED_MODEL,
                    "n_rows",
                ].sum()
            ),
            **{
                metric: float(expanding[f"{metric}_mean"])
                for metric in METRIC_COLUMNS
            },
        },
        {
            "evaluation": "Five-fold spatial-block validation",
            "role": "Post-selection spatial robustness",
            "years": "2019-2024",
            "aggregation": "Pooled out-of-fold predictions from 68 grids",
            "n_rows": int(spatial["n_rows"]),
            **{metric: float(spatial[metric]) for metric in METRIC_COLUMNS},
        },
        {
            "evaluation": "Independent final test (2025)",
            "role": "Locked final test",
            "years": "Refit 2019-2024; test 2025 once",
            "aggregation": "All 2025 date-grid records",
            "n_rows": int(test_metrics["n_rows"]),
            **{
                metric: (
                    test_pearson
                    if metric == "Pearson_r_raw"
                    else float(test_metrics[metric])
                )
                for metric in METRIC_COLUMNS
            },
        },
    ]
    return pd.DataFrame(rows)


def create_temporal_table(data: dict) -> pd.DataFrame:
    columns = [
        "validation_year", "fold", "n_rows", *METRIC_COLUMNS,
    ]
    table = data["expanding_metrics"].loc[
        data["expanding_metrics"]["model"] == SELECTED_MODEL,
        columns,
    ].copy()
    return table.sort_values("validation_year").reset_index(drop=True)


def create_ablation_table(data: dict) -> pd.DataFrame:
    columns = [
        "feature_group",
        "MAE_ug_m3_mean",
        "RMSE_ug_m3_mean",
        "R2_raw_mean",
        "Pearson_r_raw_mean",
        "RMSE_log_mean",
        "R2_log_mean",
        "log_RMSE_reduction_vs_calendar_location_percent",
    ]
    return (
        data["ablation"][columns]
        .sort_values("RMSE_log_mean")
        .reset_index(drop=True)
    )


def create_spatial_table(data: dict) -> pd.DataFrame:
    columns = [
        "specification",
        "spatial_block",
        "n_validation_grid_cells",
        "n_rows",
        *METRIC_COLUMNS,
    ]
    return (
        data["spatial_fold_metrics"][columns]
        .sort_values(["specification", "spatial_block"])
        .reset_index(drop=True)
    )


def create_grid_statistics(test_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["grid_id", "latitude", "longitude"]
    for keys, group in test_predictions.groupby(group_columns, sort=True):
        metrics = calculate_metrics(group[TARGET], group[PREDICTED])
        rows.append(
            {
                "grid_id": keys[0],
                "latitude": float(keys[1]),
                "longitude": float(keys[2]),
                "observed_mean_ug_m3": float(group[TARGET].mean()),
                "predicted_mean_ug_m3": float(group[PREDICTED].mean()),
                "mean_residual_ug_m3": float(
                    (group[PREDICTED] - group[TARGET]).mean()
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def create_period_statistics(
    test_predictions: pd.DataFrame,
    period_column: str,
) -> pd.DataFrame:
    rows = []
    for period, group in test_predictions.groupby(period_column, sort=False):
        metrics = calculate_metrics(group[TARGET], group[PREDICTED])
        rows.append(
            {
                period_column: period,
                "observed_mean_ug_m3": float(group[TARGET].mean()),
                "predicted_mean_ug_m3": float(group[PREDICTED].mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def save_tables(data: dict) -> dict:
    print("Creating manuscript tables...")
    consolidated = create_consolidated_performance_table(data)
    temporal = create_temporal_table(data)
    ablation = create_ablation_table(data)
    spatial = create_spatial_table(data)
    grid = create_grid_statistics(data["test_predictions"])
    monthly = create_period_statistics(data["test_predictions"], "month")
    monthly = monthly.sort_values("month").reset_index(drop=True)
    seasonal = create_period_statistics(data["test_predictions"], "season")
    seasonal["season"] = pd.Categorical(
        seasonal["season"], categories=SEASON_ORDER, ordered=True
    )
    seasonal = seasonal.sort_values("season").reset_index(drop=True)
    seasonal["season"] = seasonal["season"].astype(str)

    importance = data["importance"].copy()
    sensitivity = data["sensitivity"].copy()

    outputs = {
        "01_consolidated_model_performance.csv": consolidated,
        "02_temporal_robustness_by_year.csv": temporal,
        "03_feature_group_ablation.csv": ablation,
        "04_spatial_block_performance.csv": spatial,
        "05_permutation_importance_2024.csv": importance,
        "06_final_test_grid_statistics_2025.csv": grid,
        "07_final_test_monthly_statistics_2025.csv": monthly,
        "08_final_test_seasonal_statistics_2025.csv": seasonal,
        "09_preprocessing_model_sensitivity_2024.csv": sensitivity,
    }

    for filename, table in outputs.items():
        table.to_csv(TABLE_FOLDER / filename, index=False)

    return {
        "consolidated": consolidated,
        "temporal": temporal,
        "ablation": ablation,
        "spatial": spatial,
        "grid": grid,
        "monthly": monthly,
        "seasonal": seasonal,
        "importance": importance,
        "sensitivity": sensitivity,
    }


# =============================================================================
# 5. PUBLICATION FIGURES
# =============================================================================

def draw_grid_map(
    axis,
    table: pd.DataFrame,
    value_column: str,
    title: str,
    cmap,
    vmin=None,
    vmax=None,
    norm=None,
):
    scatter = axis.scatter(
        table["longitude"],
        table["latitude"],
        c=table[value_column],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        marker="s",
        s=155,
        edgecolors="white",
        linewidths=0.35,
    )
    axis.set_title(title)
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color=COLORS["light_gray"], linewidth=0.5, alpha=0.7)
    return scatter


def figure_study_grid_long_term_mean(features: pd.DataFrame) -> None:
    grid = (
        features.groupby(["grid_id", "latitude", "longitude"], as_index=False)
        .agg(
            mean_OCSMASS_ug_m3=(TARGET, "mean"),
            sd_OCSMASS_ug_m3=(TARGET, "std"),
        )
    )

    figure, axis = plt.subplots(figsize=(7.0, 7.3))
    scatter = draw_grid_map(
        axis,
        grid,
        "mean_OCSMASS_ug_m3",
        "Mean MERRA-2 surface organic carbon, 2019–2025",
        "viridis",
    )
    colorbar = figure.colorbar(scatter, ax=axis, shrink=0.82, pad=0.03)
    colorbar.set_label("Mean OCSMASS (µg m⁻³)")
    figure.text(
        0.5,
        0.015,
        "Squares represent MERRA-2 grid cells intersecting the Bangladesh study domain.",
        ha="center",
        fontsize=8.5,
        color=COLORS["gray"],
    )
    figure.tight_layout(rect=[0, 0.035, 1, 1])
    save_figure("01_study_grid_mean_OCSMASS_2019_2025.png")


def figure_test_map_comparison(grid: pd.DataFrame) -> None:
    concentration_min = float(
        min(grid["observed_mean_ug_m3"].min(), grid["predicted_mean_ug_m3"].min())
    )
    concentration_max = float(
        max(grid["observed_mean_ug_m3"].max(), grid["predicted_mean_ug_m3"].max())
    )
    residual_limit = float(np.max(np.abs(grid["mean_residual_ug_m3"])))
    if residual_limit == 0:
        residual_limit = 1.0
    residual_norm = TwoSlopeNorm(
        vmin=-residual_limit,
        vcenter=0.0,
        vmax=residual_limit,
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.5), sharex=True, sharey=True)
    observed_map = draw_grid_map(
        axes[0], grid, "observed_mean_ug_m3", "(a) Observed", "viridis",
        vmin=concentration_min, vmax=concentration_max,
    )
    draw_grid_map(
        axes[1], grid, "predicted_mean_ug_m3", "(b) Predicted", "viridis",
        vmin=concentration_min, vmax=concentration_max,
    )
    residual_map = draw_grid_map(
        axes[2], grid, "mean_residual_ug_m3", "(c) Residual", "RdBu_r",
        norm=residual_norm,
    )

    concentration_bar = figure.colorbar(
        observed_map, ax=axes[:2], shrink=0.78, pad=0.02
    )
    concentration_bar.set_label("Annual mean OCSMASS (µg m⁻³)")
    residual_bar = figure.colorbar(
        residual_map, ax=axes[2], shrink=0.78, pad=0.02
    )
    residual_bar.set_label("Mean residual (predicted − observed; µg m⁻³)")
    figure.suptitle("Independent 2025 final-test spatial comparison", y=0.995)
    figure.text(
        0.5,
        0.01,
        "Observed values are MERRA-2 OCSMASS; predictions are from the locked HGB model.",
        ha="center",
        fontsize=8.5,
        color=COLORS["gray"],
    )
    figure.subplots_adjust(left=0.06, right=0.95, bottom=0.14, top=0.90, wspace=0.20)
    save_figure("02_final_test_spatial_comparison_2025.png")


def figure_monthly_test_series(
    test_predictions: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    daily = (
        test_predictions.groupby("date", as_index=False)
        .agg(
            observed=(TARGET, "mean"),
            predicted=(PREDICTED, "mean"),
        )
    )

    figure, axes = plt.subplots(
        2, 1, figsize=(11.0, 7.2), gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    axes[0].plot(
        daily["date"], daily["observed"],
        color=COLORS["blue"], linewidth=1.25, label="Observed MERRA-2",
    )
    axes[0].plot(
        daily["date"], daily["predicted"],
        color=COLORS["orange"], linewidth=1.25, label="HGB prediction",
    )
    axes[0].set_ylabel("Daily national mean OCSMASS (µg m⁻³)")
    axes[0].set_title("(a) Daily national mean during the independent 2025 test")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(axis="y", alpha=0.25)

    months = monthly["month"].astype(int).to_numpy()
    axes[1].plot(
        months, monthly["observed_mean_ug_m3"],
        marker="o", color=COLORS["blue"], linewidth=1.8,
        label="Observed MERRA-2",
    )
    axes[1].plot(
        months, monthly["predicted_mean_ug_m3"],
        marker="s", color=COLORS["orange"], linewidth=1.8,
        label="HGB prediction",
    )
    axes[1].set_xticks(range(1, 13), MONTH_LABELS)
    axes[1].set_ylabel("Monthly mean (µg m⁻³)")
    axes[1].set_title("(b) Monthly mean comparison")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure("03_final_test_temporal_comparison_2025.png")


def figure_seasonal_test_performance(seasonal: pd.DataFrame) -> None:
    labels = [SEASON_LABELS.get(value, value) for value in seasonal["season"]]
    x = np.arange(len(labels))
    width = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    axes[0].bar(
        x - width / 2,
        seasonal["observed_mean_ug_m3"],
        width,
        label="Observed MERRA-2",
        color=COLORS["blue"],
    )
    axes[0].bar(
        x + width / 2,
        seasonal["predicted_mean_ug_m3"],
        width,
        label="HGB prediction",
        color=COLORS["orange"],
    )
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Mean OCSMASS (µg m⁻³)")
    axes[0].set_title("(a) Seasonal mean")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, seasonal["RMSE_ug_m3"], color=COLORS["green"])
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("RMSE (µg m⁻³)")
    axes[1].set_title("(b) Seasonal prediction error")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Seasonal performance during the independent 2025 test")
    figure.tight_layout()
    save_figure("04_final_test_seasonal_performance_2025.png")


def figure_performance_summary(consolidated: pd.DataFrame) -> None:
    labels = [
        "2024 temporal\nvalidation",
        "2022–2024\nexpanding-year mean",
        "2019–2024\nspatial OOF",
        "2025 final\ntest",
    ]
    x = np.arange(len(consolidated))
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    axes[0].bar(x, consolidated["RMSE_log"], color=COLORS["blue"])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Log-scale RMSE")
    axes[0].set_title("(a) Error across evaluation designs")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, consolidated["R2_log"], color=COLORS["green"])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Log-scale R²")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("(b) Explained variation")
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle("Locked HGB model performance summary")
    figure.tight_layout()
    save_figure("05_model_performance_summary.png")


def figure_feature_importance(importance: pd.DataFrame) -> None:
    top = importance.nlargest(12, "importance_mean").sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=(8.6, 5.8))
    axis.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color=COLORS["blue"],
        alpha=0.9,
        ecolor=COLORS["gray"],
        capsize=2,
    )
    axis.set_xlabel("Decrease in 2024 validation log-scale R² after permutation")
    axis.set_ylabel("")
    axis.set_title("Permutation importance of the selected HGB model")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    save_figure("06_permutation_importance_2024.png")


def figure_robustness_summary(
    temporal: pd.DataFrame,
    spatial: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))

    axes[0].plot(
        temporal["validation_year"],
        temporal["RMSE_log"],
        color=COLORS["blue"],
        marker="o",
        linewidth=2.0,
    )
    axes[0].set_xticks(temporal["validation_year"])
    axes[0].set_xlabel("Validation year")
    axes[0].set_ylabel("Log-scale RMSE")
    axes[0].set_title("(a) Expanding-year temporal validation")
    axes[0].grid(alpha=0.25)

    pivot = spatial.pivot(
        index="spatial_block",
        columns="specification",
        values="RMSE_log",
    ).sort_index()
    pivot.plot(
        kind="bar",
        ax=axes[1],
        color=[COLORS["blue"], COLORS["orange"]],
        width=0.75,
    )
    axes[1].set_xlabel("Held-out spatial block")
    axes[1].set_ylabel("Log-scale RMSE")
    axes[1].set_title("(b) Five-fold spatial-block validation")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="", frameon=False, fontsize=8.4)
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    save_figure("07_temporal_and_spatial_robustness.png")


def save_figure_index() -> None:
    index = pd.DataFrame(
        [
            {
                "figure": 1,
                "filename": "01_study_grid_mean_OCSMASS_2019_2025.png",
                "purpose": "Study grid and long-term descriptive OCSMASS pattern",
            },
            {
                "figure": 2,
                "filename": "02_final_test_spatial_comparison_2025.png",
                "purpose": "Observed, predicted, and residual 2025 spatial patterns",
            },
            {
                "figure": 3,
                "filename": "03_final_test_temporal_comparison_2025.png",
                "purpose": "Daily and monthly 2025 temporal comparison",
            },
            {
                "figure": 4,
                "filename": "04_final_test_seasonal_performance_2025.png",
                "purpose": "Seasonal means and errors in the final test",
            },
            {
                "figure": 5,
                "filename": "05_model_performance_summary.png",
                "purpose": "Comparison across temporal, spatial, and final tests",
            },
            {
                "figure": 6,
                "filename": "06_permutation_importance_2024.png",
                "purpose": "Permutation importance from untouched 2024 validation",
            },
            {
                "figure": 7,
                "filename": "07_temporal_and_spatial_robustness.png",
                "purpose": "Temporal and spatial robustness evidence",
            },
        ]
    )
    index.to_csv(TABLE_FOLDER / "10_publication_figure_index.csv", index=False)


# =============================================================================
# 6. REPORT
# =============================================================================

def write_run_report(tables: dict) -> None:
    consolidated = tables["consolidated"]
    test = consolidated.loc[
        consolidated["evaluation"] == "Independent final test (2025)"
    ].iloc[0]
    spatial = consolidated.loc[
        consolidated["evaluation"] == "Five-fold spatial-block validation"
    ].iloc[0]

    report = [
        "Bangladesh OCSMASS publication-output report",
        "",
        "Integrity checks:",
        "No model was trained, tuned, or replaced by this script.",
        "Saved 2024 validation and 2025 final-test metrics were independently reconciled.",
        "The robustness and spatial records confirm that 2025 was excluded.",
        "",
        "Headline locked-model results:",
        f"2025 final-test RMSE: {test['RMSE_ug_m3']:.4f} µg m^-3",
        f"2025 final-test raw R²: {test['R2_raw']:.4f}",
        f"2025 final-test log-RMSE: {test['RMSE_log']:.4f}",
        f"2025 final-test log R²: {test['R2_log']:.4f}",
        f"2025 final-test Pearson r: {test['Pearson_r_raw']:.4f}",
        f"Spatial OOF log-RMSE: {spatial['RMSE_log']:.4f}",
        f"Spatial OOF log R²: {spatial['R2_log']:.4f}",
        "",
        "Output:",
        "10 manuscript tables/index files were created.",
        "7 high-resolution PNG figures were created at 400 dpi.",
        "",
        "Interpretation limits:",
        "The target is MERRA-2 surface organic carbon concentration, not direct SOA.",
        "The model supports prediction and association, not causal inference.",
        "Spatial validation applies within the Bangladesh study domain.",
        "Coordinate dependence must be discussed when describing transferability.",
    ]
    with open(
        OUTPUT_FOLDER / "Publication_outputs_run_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(report))


# =============================================================================
# 7. MAIN
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    apply_publication_style()
    ensure_output_folders()
    require_files()

    data = load_inputs()
    validate_inputs(data)
    print("Checking saved metrics against prediction records...")
    reconcile_saved_metrics(data)

    tables = save_tables(data)

    print("Creating publication figures...")
    figure_study_grid_long_term_mean(data["features"])
    figure_test_map_comparison(tables["grid"])
    figure_monthly_test_series(data["test_predictions"], tables["monthly"])
    figure_seasonal_test_performance(tables["seasonal"])
    figure_performance_summary(tables["consolidated"])
    figure_feature_importance(tables["importance"])
    figure_robustness_summary(tables["temporal"], tables["spatial"])
    save_figure_index()
    write_run_report(tables)

    print("\n" + "=" * 78)
    print("PUBLICATION TABLES AND FIGURES COMPLETED")
    print("=" * 78)
    print(f"Tables saved in: {TABLE_FOLDER}")
    print(f"Figures saved in: {FIGURE_FOLDER}")
    print("No model fitting or tuning was performed.")


if __name__ == "__main__":
    main()
