"""
Integrated TROPOMI CO and VIIRS fire sensitivity analysis for Bangladesh OA.

Primary question
----------------
Do TROPOMI total-column CO and local VIIRS thermal-anomaly variables improve
out-of-year prediction of MERRA-2 surface organic carbon (OCSMASS) beyond
HCHO, NO2, meteorology, season, and location?

Four models are evaluated on exactly the same rows:
    1. Baseline
    2. Baseline + CO
    3. Baseline + VIIRS fire
    4. Baseline + CO + VIIRS fire

Design safeguards
-----------------
* Analysis period: 2019-2024.
* Rows without valid CO are excluded from all four models. This makes the
  comparison fair and prevents CO missingness from acting as a predictor.
* Entire calendar years are held out using leave-one-year-out validation.
* Imputation is fitted separately inside every training fold.
* Satellite HCHO/NO2 values below zero are treated as unavailable in the
  in-memory modeling copy; their availability indicators are retained.
* The target is modeled on the log1p scale and evaluated on the original
  concentration scale.
* Paired uncertainty intervals use bootstrap resampling of whole dates rather
  than individual grid rows.

Required packages
-----------------
pandas, numpy, matplotlib, scikit-learn
"""

from pathlib import Path
import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# =============================================================================
# 1. FILE LOCATIONS — NORMALLY CHANGE ONLY PROJECT_FOLDER
# =============================================================================

PROJECT_FOLDER = Path(os.environ.get(
    "OA_PROJECT_FOLDER",
    str(Path.cwd()),
))

FIRE_MASTER_FILE = Path(os.environ.get(
    "OA_FIRE_MASTER_FILE",
    str(
        PROJECT_FOLDER
        / "Fire_Sensitivity_Results_2019_2024"
        / "Bangladesh_OA_HCHO_NO2_ERA5_VIIRS_Master_2019_2024.csv"
    ),
))

CO_FILE = Path(os.environ.get(
    "OA_CO_FILE",
    str(
        PROJECT_FOLDER
        / "Bangladesh_TROPOMI_CO_Daily_MERRAgrid_2019_2024_FINAL.csv"
    ),
))

OUTPUT_FOLDER = Path(os.environ.get(
    "OA_CO_FIRE_OUTPUT_FOLDER",
    str(PROJECT_FOLDER / "CO_Fire_Integrated_Model_Results_2019_2024"),
))

TABLE_FOLDER = OUTPUT_FOLDER / "Tables"
FIGURE_FOLDER = OUTPUT_FOLDER / "Figures"


# =============================================================================
# 2. ANALYSIS SETTINGS
# =============================================================================

START_DATE = pd.Timestamp("2019-01-01")
END_DATE = pd.Timestamp("2024-12-31")
EXPECTED_YEARS = list(range(2019, 2025))
EXPECTED_GRID_CELLS = 68

TARGET = "OCSMASS_ug_m3"
CO_VARIABLE = "CO_mol_m2"
RANDOM_SEED = 42

# Set OA_QUICK_TEST=1 only for a short technical test.
QUICK_TEST = os.environ.get("OA_QUICK_TEST", "0") == "1"
MODEL_ITERATIONS = 35 if QUICK_TEST else 180
BOOTSTRAP_ITERATIONS = 100 if QUICK_TEST else 1000

# High-OA threshold is fitted on each training fold only.
EXTREME_QUANTILE = 0.90
MIN_GRIDS_FOR_REGIONAL_CO_EVENT = 50

BASE_FEATURES = [
    "HCHO",
    "NO2",
    "HCHO_available",
    "NO2_available",
    "temperature_C",
    "relative_humidity_percent",
    "boundary_layer_height_m",
    "wind_speed_m_s",
    "total_precipitation_mm",
    "solar_radiation_MJ_m2_day",
    "latitude",
    "longitude",
    "month_sin",
    "month_cos",
]

CO_FEATURES = [CO_VARIABLE]

FIRE_FEATURES = [
    "fire_count_50km_3day_log1p",
    "fire_frp_MW_50km_3day_log1p",
]

MODEL_FEATURES = {
    "Baseline": BASE_FEATURES,
    "Baseline_plus_CO": BASE_FEATURES + CO_FEATURES,
    "Baseline_plus_fire": BASE_FEATURES + FIRE_FEATURES,
    "Baseline_plus_CO_and_fire": BASE_FEATURES + CO_FEATURES + FIRE_FEATURES,
}

# Candidate episodes identified from the CO diagnostics. Additional top-CO
# dates are selected automatically later.
PRESELECTED_EVENT_DATES = pd.to_datetime([
    "2019-03-15",
    "2020-03-28",
    "2021-03-24",
    "2023-04-11",
])


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def elapsed_text(start_time: float) -> str:
    elapsed = time.perf_counter() - start_time
    return f"{elapsed:.1f} seconds" if elapsed < 60 else f"{elapsed / 60:.1f} minutes"


def ensure_folders() -> None:
    for folder in [OUTPUT_FOLDER, TABLE_FOLDER, FIGURE_FOLDER]:
        folder.mkdir(parents=True, exist_ok=True)


def require_columns(frame: pd.DataFrame, columns, table_name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def metric_values(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "MAE_ug_m3": mean_absolute_error(y_true, y_pred),
        "RMSE_ug_m3": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mean_bias_ug_m3": float(np.mean(y_pred - y_true)),
    }


def build_model() -> Pipeline:
    """Imputation is learned only from the training rows of each year fold."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        (
            "model",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.06,
                max_iter=MODEL_ITERATIONS,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=20,
                random_state=RANDOM_SEED,
            ),
        ),
    ])


def safe_spearman(frame: pd.DataFrame, first: str, second: str) -> float:
    valid = frame[[first, second]].dropna()
    if len(valid) < 3 or valid[first].nunique() < 2 or valid[second].nunique() < 2:
        return np.nan
    return valid[first].corr(valid[second], method="spearman")


def save_figure(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_FOLDER / filename, dpi=350, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 4. READ, VALIDATE, AND MERGE CO
# =============================================================================

start_time = time.perf_counter()
ensure_folders()

if not FIRE_MASTER_FILE.exists():
    alternative = PROJECT_FOLDER / FIRE_MASTER_FILE.name
    if alternative.exists():
        FIRE_MASTER_FILE = alternative
    else:
        raise FileNotFoundError(
            "The VIIRS-enhanced master CSV was not found. Expected:\n"
            f"{FIRE_MASTER_FILE}\n\n"
            "Copy Bangladesh_OA_HCHO_NO2_ERA5_VIIRS_Master_2019_2024.csv "
            "to that folder or change FIRE_MASTER_FILE."
        )

if not CO_FILE.exists():
    raise FileNotFoundError(f"The final TROPOMI CO file was not found:\n{CO_FILE}")

master = pd.read_csv(FIRE_MASTER_FILE, low_memory=False)
co = pd.read_csv(CO_FILE, low_memory=False)

master.columns = master.columns.astype(str).str.strip()
co.columns = co.columns.astype(str).str.strip()

require_columns(
    master,
    [
        "date", "grid_id", "latitude", "longitude", "season", TARGET,
        *BASE_FEATURES[:-4], *FIRE_FEATURES,
        "fire_count_50km_3day", "fire_frp_MW_50km_3day",
        "fire_any_50km_3day", "fire_exposure_class",
    ],
    "VIIRS-enhanced master",
)
require_columns(co, ["date", "grid_id", CO_VARIABLE], "TROPOMI CO table")

master["date"] = pd.to_datetime(master["date"], errors="coerce").dt.normalize()
co["date"] = pd.to_datetime(co["date"], errors="coerce").dt.normalize()
master["grid_id"] = master["grid_id"].astype(str).str.strip()
co["grid_id"] = co["grid_id"].astype(str).str.strip()
co[CO_VARIABLE] = pd.to_numeric(co[CO_VARIABLE], errors="coerce")

master = master.loc[master["date"].between(START_DATE, END_DATE)].copy()
co = co.loc[co["date"].between(START_DATE, END_DATE)].copy()

if master[["date", "grid_id"]].isna().any().any():
    raise ValueError("The VIIRS-enhanced master contains invalid dates or grid IDs.")
if co[["date", "grid_id"]].isna().any().any():
    raise ValueError("The CO table contains invalid dates or grid IDs.")
if master.duplicated(["date", "grid_id"]).any():
    raise ValueError("The VIIRS-enhanced master has duplicate date-grid rows.")
if co.duplicated(["date", "grid_id"]).any():
    raise ValueError("The CO table has duplicate date-grid rows.")

# Remove an earlier CO column if an old attempt placed one in the fire master.
old_co_columns = [
    column for column in master.columns
    if column.lower() in {"co_mol_m2", "co_column_number_density"}
]
if old_co_columns:
    master = master.drop(columns=old_co_columns)

co_for_merge = co[["date", "grid_id", CO_VARIABLE]].copy()
combined = master.merge(
    co_for_merge,
    on=["date", "grid_id"],
    how="left",
    validate="one_to_one",
)

if combined["grid_id"].nunique() != EXPECTED_GRID_CELLS:
    warnings.warn(
        f"Expected {EXPECTED_GRID_CELLS} grids; found {combined['grid_id'].nunique()}."
    )

combined["year"] = combined["date"].dt.year
combined["month"] = combined["date"].dt.month
combined["month_sin"] = np.sin(2 * np.pi * combined["month"] / 12.0)
combined["month_cos"] = np.cos(2 * np.pi * combined["month"] / 12.0)

# Treat physically negative trace-gas retrievals as unavailable in memory.
for gas in ["HCHO", "NO2"]:
    combined[gas] = pd.to_numeric(combined[gas], errors="coerce")
    combined.loc[combined[gas] < 0, gas] = np.nan
    combined[f"{gas}_available"] = combined[gas].notna().astype("int8")

for column in set(sum(MODEL_FEATURES.values(), [])) | {TARGET}:
    combined[column] = pd.to_numeric(combined[column], errors="coerce")

if combined[TARGET].isna().any():
    raise ValueError("The OA target contains missing values.")
if (combined[TARGET] < 0).any():
    raise ValueError("The OA target contains negative values.")

matched_co = int(combined[CO_VARIABLE].notna().sum())
if matched_co == 0:
    raise RuntimeError(
        "Zero CO values matched. Check that both files use the same date and grid_id."
    )

# Primary analysis uses complete CO rows for a like-for-like comparison.
model_data = combined.loc[
    combined[CO_VARIABLE].notna()
    & combined["year"].isin(EXPECTED_YEARS)
].copy()
model_data = model_data.sort_values(["date", "grid_id"]).reset_index(drop=True)
model_data["target_log1p"] = np.log1p(model_data[TARGET])

observed_years = sorted(model_data["year"].unique().tolist())
if observed_years != EXPECTED_YEARS:
    raise ValueError(
        f"Expected valid CO in every year {EXPECTED_YEARS}; found {observed_years}."
    )


# =============================================================================
# 5. LEAVE-ONE-YEAR-OUT FOUR-MODEL COMPARISON
# =============================================================================

prediction_parts = []
fold_metric_rows = []

for test_year in EXPECTED_YEARS:
    train = model_data.loc[model_data["year"].ne(test_year)].copy()
    test = model_data.loc[model_data["year"].eq(test_year)].copy()

    if train.empty or test.empty:
        raise ValueError(f"Empty train or test set for held-out year {test_year}.")

    extreme_threshold = float(train[TARGET].quantile(EXTREME_QUANTILE))
    prediction_table = test[
        ["date", "grid_id", "year", "season", TARGET, CO_VARIABLE,
         "fire_any_50km_3day", "fire_exposure_class"]
    ].copy()
    prediction_table["held_out_year"] = test_year
    prediction_table["training_extreme_threshold_ug_m3"] = extreme_threshold
    prediction_table["extreme_OA"] = test[TARGET].ge(extreme_threshold).to_numpy()

    print(f"\nHeld-out year {test_year}: train={len(train):,}, test={len(test):,}")

    for model_name, features in MODEL_FEATURES.items():
        model = build_model()
        model.fit(train[features], train["target_log1p"])
        predicted_log = model.predict(test[features])
        predicted_raw = np.maximum(np.expm1(predicted_log), 0.0)
        prediction_table[f"prediction_{model_name}"] = predicted_raw

        metrics = metric_values(test[TARGET], predicted_raw)
        metrics.update({
            "held_out_year": test_year,
            "model": model_name,
            "extreme_threshold_ug_m3": extreme_threshold,
            "number_features": len(features),
        })
        fold_metric_rows.append(metrics)
        print(
            f"  {model_name:28s} "
            f"R2={metrics['R2']:.4f}, MAE={metrics['MAE_ug_m3']:.4f}, "
            f"RMSE={metrics['RMSE_ug_m3']:.4f}"
        )

    prediction_parts.append(prediction_table)

predictions = pd.concat(prediction_parts, ignore_index=True)
fold_metrics = pd.DataFrame(fold_metric_rows)


# =============================================================================
# 6. POOLED, SEASONAL, AND EXTREME-EVENT METRICS
# =============================================================================

pooled_rows = []
seasonal_rows = []
extreme_rows = []

for model_name in MODEL_FEATURES:
    prediction_column = f"prediction_{model_name}"

    pooled = metric_values(predictions[TARGET], predictions[prediction_column])
    pooled.update({"model": model_name})
    pooled_rows.append(pooled)

    for season, group in predictions.groupby("season", observed=True):
        metrics = metric_values(group[TARGET], group[prediction_column])
        metrics.update({"model": model_name, "season": season})
        seasonal_rows.append(metrics)

    for extreme_status, group in predictions.groupby("extreme_OA", observed=True):
        metrics = metric_values(group[TARGET], group[prediction_column])
        metrics.update({
            "model": model_name,
            "subset": "Fold-defined extreme OA" if extreme_status else "Non-extreme OA",
        })
        extreme_rows.append(metrics)

pooled_metrics = pd.DataFrame(pooled_rows)
seasonal_metrics = pd.DataFrame(seasonal_rows)
extreme_metrics = pd.DataFrame(extreme_rows)


# =============================================================================
# 7. PAIRED DATE-BLOCK BOOTSTRAP OF INCREMENTAL MODEL VALUE
# =============================================================================

def date_level_error_components(prediction_frame: pd.DataFrame, model_name: str):
    prediction_column = f"prediction_{model_name}"
    work = prediction_frame[["date", TARGET, prediction_column]].copy()
    error = work[prediction_column] - work[TARGET]
    work["absolute_error"] = error.abs()
    work["squared_error"] = error ** 2
    work["target_sum"] = work[TARGET]
    work["target_squared_sum"] = work[TARGET] ** 2
    return (
        work.groupby("date", sort=True)
        .agg(
            n=(TARGET, "size"),
            absolute_error_sum=("absolute_error", "sum"),
            squared_error_sum=("squared_error", "sum"),
            target_sum=("target_sum", "sum"),
            target_squared_sum=("target_squared_sum", "sum"),
        )
    )


date_components = {
    model_name: date_level_error_components(predictions, model_name)
    for model_name in MODEL_FEATURES
}

common_dates = date_components["Baseline"].index
for model_name in MODEL_FEATURES:
    if not date_components[model_name].index.equals(common_dates):
        raise RuntimeError("Model predictions do not have identical dates.")

rng = np.random.default_rng(RANDOM_SEED)
number_dates = len(common_dates)
bootstrap_records = []

component_arrays = {
    model_name: frame.to_numpy(dtype=float)
    for model_name, frame in date_components.items()
}

# Column positions in component_arrays:
# n, absolute_error_sum, squared_error_sum, target_sum, target_squared_sum
for iteration in range(BOOTSTRAP_ITERATIONS):
    sampled_positions = rng.integers(0, number_dates, size=number_dates)
    iteration_metrics = {}

    for model_name, values in component_arrays.items():
        sampled = values[sampled_positions]
        n = sampled[:, 0].sum()
        absolute_error = sampled[:, 1].sum()
        squared_error = sampled[:, 2].sum()
        target_sum = sampled[:, 3].sum()
        target_squared_sum = sampled[:, 4].sum()
        total_sum_squares = target_squared_sum - (target_sum ** 2 / n)

        iteration_metrics[model_name] = {
            "MAE": absolute_error / n,
            "RMSE": np.sqrt(squared_error / n),
            "R2": 1.0 - squared_error / total_sum_squares,
        }

    baseline = iteration_metrics["Baseline"]
    for model_name in MODEL_FEATURES:
        if model_name == "Baseline":
            continue
        current = iteration_metrics[model_name]
        bootstrap_records.append({
            "bootstrap_iteration": iteration,
            "model": model_name,
            "delta_R2_vs_baseline": current["R2"] - baseline["R2"],
            "delta_MAE_vs_baseline_ug_m3": current["MAE"] - baseline["MAE"],
            "delta_RMSE_vs_baseline_ug_m3": current["RMSE"] - baseline["RMSE"],
        })

bootstrap_samples = pd.DataFrame(bootstrap_records)

incremental_rows = []
baseline_point = pooled_metrics.set_index("model").loc["Baseline"]
for model_name in MODEL_FEATURES:
    if model_name == "Baseline":
        continue

    current_point = pooled_metrics.set_index("model").loc[model_name]
    samples = bootstrap_samples.loc[bootstrap_samples["model"].eq(model_name)]

    row = {
        "model": model_name,
        "delta_R2_vs_baseline": current_point["R2"] - baseline_point["R2"],
        "delta_MAE_vs_baseline_ug_m3": (
            current_point["MAE_ug_m3"] - baseline_point["MAE_ug_m3"]
        ),
        "delta_RMSE_vs_baseline_ug_m3": (
            current_point["RMSE_ug_m3"] - baseline_point["RMSE_ug_m3"]
        ),
    }

    for metric in [
        "delta_R2_vs_baseline",
        "delta_MAE_vs_baseline_ug_m3",
        "delta_RMSE_vs_baseline_ug_m3",
    ]:
        row[f"{metric}_CI_low"] = samples[metric].quantile(0.025)
        row[f"{metric}_CI_high"] = samples[metric].quantile(0.975)

    row["bootstrap_probability_R2_improved"] = (
        samples["delta_R2_vs_baseline"] > 0
    ).mean()
    row["bootstrap_probability_MAE_improved"] = (
        samples["delta_MAE_vs_baseline_ug_m3"] < 0
    ).mean()
    row["bootstrap_probability_RMSE_improved"] = (
        samples["delta_RMSE_vs_baseline_ug_m3"] < 0
    ).mean()
    incremental_rows.append(row)

incremental_value = pd.DataFrame(incremental_rows)


# =============================================================================
# 8. CO/FIRE EVENT TABLES AND DESCRIPTIVE RELATIONSHIPS
# =============================================================================

# Daily Bangladesh summaries are used to find regional CO episodes.
daily_summary = (
    model_data.groupby("date")
    .agg(
        valid_grid_cells=(CO_VARIABLE, "count"),
        OA_mean_ug_m3=(TARGET, "mean"),
        OA_max_ug_m3=(TARGET, "max"),
        CO_mean_mol_m2=(CO_VARIABLE, "mean"),
        CO_max_mol_m2=(CO_VARIABLE, "max"),
        HCHO_mean=("HCHO", "mean"),
        NO2_mean=("NO2", "mean"),
        fire_count_50km_3day_mean=("fire_count_50km_3day", "mean"),
        fire_FRP_MW_50km_3day_mean=("fire_frp_MW_50km_3day", "mean"),
        temperature_C_mean=("temperature_C", "mean"),
        RH_percent_mean=("relative_humidity_percent", "mean"),
        BLH_m_mean=("boundary_layer_height_m", "mean"),
        wind_speed_m_s_mean=("wind_speed_m_s", "mean"),
        precipitation_mm_mean=("total_precipitation_mm", "mean"),
    )
    .reset_index()
)

regional_event_candidates = daily_summary.loc[
    daily_summary["valid_grid_cells"] >= MIN_GRIDS_FOR_REGIONAL_CO_EVENT
]
top_regional_dates = regional_event_candidates.nlargest(
    10, "CO_mean_mol_m2"
)["date"]
top_local_dates = daily_summary.nlargest(10, "CO_max_mol_m2")["date"]
event_dates = pd.DatetimeIndex(
    sorted(set(PRESELECTED_EVENT_DATES) | set(top_regional_dates) | set(top_local_dates))
)

event_daily = daily_summary.loc[daily_summary["date"].isin(event_dates)].copy()
event_daily["preselected_CO_episode"] = event_daily["date"].isin(
    PRESELECTED_EVENT_DATES
)
event_daily["top10_regional_daily_CO"] = event_daily["date"].isin(
    top_regional_dates
)
event_daily["top10_local_grid_CO"] = event_daily["date"].isin(top_local_dates)

event_grid = model_data.loc[
    model_data["date"].isin(event_dates),
    [
        "date", "grid_id", "latitude", "longitude", TARGET,
        "HCHO", "NO2", CO_VARIABLE,
        "fire_count_50km_3day", "fire_frp_MW_50km_3day",
        "fire_count_50km_3day_log1p", "fire_frp_MW_50km_3day_log1p",
        "temperature_C", "relative_humidity_percent",
        "boundary_layer_height_m", "wind_speed_m_s",
        "total_precipitation_mm", "solar_radiation_MJ_m2_day",
    ],
].copy()

# Add interpretable within-day CO spatial anomaly for the event table only.
event_grid["CO_daily_BD_mean_mol_m2"] = event_grid.groupby("date")[
    CO_VARIABLE
].transform("mean")
event_grid["CO_spatial_anomaly_mol_m2"] = (
    event_grid[CO_VARIABLE] - event_grid["CO_daily_BD_mean_mol_m2"]
)

relationship_rows = []
relationship_groups = [
    ("All", model_data),
    ("No local fire", model_data.loc[model_data["fire_any_50km_3day"].eq(0)]),
    ("Any local fire", model_data.loc[model_data["fire_any_50km_3day"].eq(1)]),
]
relationship_groups.extend(
    (f"Season: {season}", group)
    for season, group in model_data.groupby("season", observed=True)
)

for group_name, group in relationship_groups:
    relationship_rows.append({
        "group": group_name,
        "n": len(group),
        "OA_CO_spearman_rho": safe_spearman(group, TARGET, CO_VARIABLE),
        "OA_HCHO_spearman_rho": safe_spearman(group, TARGET, "HCHO"),
        "OA_NO2_spearman_rho": safe_spearman(group, TARGET, "NO2"),
        "mean_OA_ug_m3": group[TARGET].mean(),
        "mean_CO_mol_m2": group[CO_VARIABLE].mean(),
    })

relationship_summary = pd.DataFrame(relationship_rows)


# =============================================================================
# 9. SAVE ALL ANALYSIS TABLES
# =============================================================================

combined.to_csv(
    TABLE_FOLDER / "01_OA_HCHO_NO2_ERA5_VIIRS_CO_merged_2019_2024.csv",
    index=False,
)
predictions.to_csv(
    TABLE_FOLDER / "02_leave_one_year_out_predictions.csv.gz",
    index=False,
    compression="gzip",
)
fold_metrics.to_csv(TABLE_FOLDER / "03_model_metrics_by_held_out_year.csv", index=False)
pooled_metrics.to_csv(TABLE_FOLDER / "04_pooled_model_metrics.csv", index=False)
incremental_value.to_csv(
    TABLE_FOLDER / "05_incremental_value_with_date_block_CI.csv", index=False
)
seasonal_metrics.to_csv(TABLE_FOLDER / "06_seasonal_model_metrics.csv", index=False)
extreme_metrics.to_csv(TABLE_FOLDER / "07_extreme_OA_model_metrics.csv", index=False)
event_daily.to_csv(TABLE_FOLDER / "08_CO_event_daily_summary.csv", index=False)
event_grid.to_csv(TABLE_FOLDER / "09_CO_event_grid_details.csv", index=False)
relationship_summary.to_csv(
    TABLE_FOLDER / "10_descriptive_OA_trace_gas_relationships.csv", index=False
)
bootstrap_samples.to_csv(
    TABLE_FOLDER / "11_date_block_bootstrap_samples.csv.gz",
    index=False,
    compression="gzip",
)


# =============================================================================
# 10. FIGURE 1 — POOLED PERFORMANCE AND INCREMENTAL VALUE
# =============================================================================

display_names = {
    "Baseline": "Baseline",
    "Baseline_plus_CO": "+ CO",
    "Baseline_plus_fire": "+ fire",
    "Baseline_plus_CO_and_fire": "+ CO + fire",
}
model_order = list(MODEL_FEATURES)
colors = ["#777777", "#2C7FB8", "#D95F0E", "#6A51A3"]

figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
metric_definitions = [
    ("R2", "Pooled R²", None),
    ("MAE_ug_m3", r"Pooled MAE (µg m$^{-3}$)", None),
    ("RMSE_ug_m3", r"Pooled RMSE (µg m$^{-3}$)", None),
]

pooled_index = pooled_metrics.set_index("model")
for axis, (metric, ylabel, _) in zip(axes, metric_definitions):
    values = [pooled_index.loc[name, metric] for name in model_order]
    axis.bar(range(len(model_order)), values, color=colors)
    axis.set_xticks(range(len(model_order)))
    axis.set_xticklabels([display_names[name] for name in model_order], rotation=25, ha="right")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    for position, value in enumerate(values):
        axis.text(position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

figure.suptitle("Out-of-year OA prediction with CO and VIIRS fire information", y=1.02)
save_figure(figure, "Figure_1_Four_Model_Pooled_Performance.png")


# =============================================================================
# 11. FIGURE 2 — PAIRED DELTAS WITH DATE-BLOCK CONFIDENCE INTERVALS
# =============================================================================

comparison_order = [
    "Baseline_plus_CO",
    "Baseline_plus_fire",
    "Baseline_plus_CO_and_fire",
]
incremental_index = incremental_value.set_index("model")

figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

for axis, metric, label, better_text in [
    (axes[0], "delta_R2_vs_baseline", "ΔR² versus baseline", "Improvement > 0"),
    (
        axes[1],
        "delta_MAE_vs_baseline_ug_m3",
        r"ΔMAE versus baseline (µg m$^{-3}$)",
        "Improvement < 0",
    ),
]:
    point = np.array([incremental_index.loc[name, metric] for name in comparison_order])
    low = np.array([
        incremental_index.loc[name, f"{metric}_CI_low"] for name in comparison_order
    ])
    high = np.array([
        incremental_index.loc[name, f"{metric}_CI_high"] for name in comparison_order
    ])
    errors = np.vstack([point - low, high - point])
    axis.errorbar(
        range(len(comparison_order)), point, yerr=errors,
        fmt="o", capsize=4, color="#252525", ecolor="#636363",
    )
    axis.axhline(0, color="#B2182B", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(comparison_order)))
    axis.set_xticklabels(
        [display_names[name] for name in comparison_order], rotation=20, ha="right"
    )
    axis.set_ylabel(label)
    axis.set_title(better_text)
    axis.grid(axis="y", alpha=0.25)

figure.suptitle("Incremental predictive value with paired date-block 95% intervals", y=1.02)
save_figure(figure, "Figure_2_Incremental_Value_Date_Block_CI.png")


# =============================================================================
# 12. FIGURE 3 — PERFORMANCE CHANGES BY HELD-OUT YEAR
# =============================================================================

fold_wide = fold_metrics.pivot(index="held_out_year", columns="model", values=["R2", "MAE_ug_m3"])
figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))

for model_name, color in zip(comparison_order, colors[1:]):
    delta_r2 = fold_wide[("R2", model_name)] - fold_wide[("R2", "Baseline")]
    delta_mae = fold_wide[("MAE_ug_m3", model_name)] - fold_wide[("MAE_ug_m3", "Baseline")]
    axes[0].plot(delta_r2.index, delta_r2.values, marker="o", label=display_names[model_name], color=color)
    axes[1].plot(delta_mae.index, delta_mae.values, marker="o", label=display_names[model_name], color=color)

axes[0].axhline(0, color="#444444", linewidth=0.8, linestyle="--")
axes[1].axhline(0, color="#444444", linewidth=0.8, linestyle="--")
axes[0].set_ylabel("ΔR² versus baseline")
axes[1].set_ylabel(r"ΔMAE versus baseline (µg m$^{-3}$)")
for axis in axes:
    axis.set_xlabel("Held-out year")
    axis.set_xticks(EXPECTED_YEARS)
    axis.grid(alpha=0.25)
axes[0].set_title("Positive values indicate improvement")
axes[1].set_title("Negative values indicate improvement")
axes[1].legend(frameon=False, fontsize=8)
save_figure(figure, "Figure_3_Incremental_Value_by_Held_Out_Year.png")


# =============================================================================
# 13. FIGURE 4 — NATIONAL OA AND CO CONTEXT
# =============================================================================

daily_plot = daily_summary.sort_values("date").copy()
daily_plot["OA_30day"] = daily_plot["OA_mean_ug_m3"].rolling(30, center=True, min_periods=10).mean()
daily_plot["CO_30day"] = daily_plot["CO_mean_mol_m2"].rolling(30, center=True, min_periods=10).mean()

figure, axis_left = plt.subplots(figsize=(11, 4.5))
axis_right = axis_left.twinx()
axis_left.plot(daily_plot["date"], daily_plot["OA_30day"], color="#7A0177", linewidth=1.4, label="OA 30-day mean")
axis_right.plot(daily_plot["date"], daily_plot["CO_30day"], color="#238B45", linewidth=1.4, label="CO 30-day mean")
axis_left.set_ylabel(r"MERRA-2 OCSMASS (µg m$^{-3}$)", color="#7A0177")
axis_right.set_ylabel(r"TROPOMI CO column (mol m$^{-2}$)", color="#238B45")
axis_left.set_xlabel("Date")
axis_left.grid(alpha=0.22)
lines = axis_left.get_lines() + axis_right.get_lines()
axis_left.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")
axis_left.set_title("Bangladesh-mean OA and total-column CO context")
save_figure(figure, "Figure_4_National_OA_CO_Time_Series.png")


# =============================================================================
# 14. QUALITY REPORT AND INTERPRETATION TEMPLATE
# =============================================================================

quality_report = {
    "fire_master_file": str(FIRE_MASTER_FILE),
    "CO_file": str(CO_FILE),
    "input_fire_master_rows": int(len(master)),
    "merged_rows": int(len(combined)),
    "valid_CO_rows": matched_co,
    "valid_CO_coverage_percent": float(100 * matched_co / len(combined)),
    "model_rows_complete_CO": int(len(model_data)),
    "unique_dates_in_model": int(model_data["date"].nunique()),
    "unique_grid_cells_in_model": int(model_data["grid_id"].nunique()),
    "duplicate_date_grid_keys": int(combined.duplicated(["date", "grid_id"]).sum()),
    "held_out_years": EXPECTED_YEARS,
    "model_iterations": MODEL_ITERATIONS,
    "date_block_bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    "target": TARGET,
    "target_transformation": "log1p",
    "primary_comparison_rows": "Only rows with valid TROPOMI CO; identical rows for all four models",
}

with open(TABLE_FOLDER / "12_analysis_quality_report.json", "w", encoding="utf-8") as file:
    json.dump(quality_report, file, indent=2)

best_row = pooled_metrics.sort_values("MAE_ug_m3").iloc[0]
combined_increment = incremental_value.loc[
    incremental_value["model"].eq("Baseline_plus_CO_and_fire")
].iloc[0]

interpretation_lines = [
    "CO AND FIRE INTEGRATED ANALYSIS — INTERPRETATION TEMPLATE",
    "=" * 68,
    "",
    f"Rows used in every model: {len(model_data):,}",
    f"Valid CO coverage before complete-case restriction: {quality_report['valid_CO_coverage_percent']:.2f}%",
    f"Best pooled-MAE model: {best_row['model']}",
    f"Best pooled MAE: {best_row['MAE_ug_m3']:.4f} ug/m3",
    "",
    "Combined CO + fire versus baseline:",
    f"  Delta R2: {combined_increment['delta_R2_vs_baseline']:.6f}",
    f"  Delta MAE: {combined_increment['delta_MAE_vs_baseline_ug_m3']:.6f} ug/m3",
    f"  Delta RMSE: {combined_increment['delta_RMSE_vs_baseline_ug_m3']:.6f} ug/m3",
    "",
    "Interpretation rules:",
    "* Delta R2 > 0 and Delta MAE/RMSE < 0 indicate improvement.",
    "* Improvement is more convincing when the paired date-block 95% interval",
    "  does not cross zero and the direction is reasonably stable by year.",
    "* CO is a broad total-column combustion/transport proxy, not a direct",
    "  measurement of surface CO, OA, or SOA.",
    "* VIIRS thermal anomalies provide fire context but do not capture every",
    "  combustion source, especially traffic, industry, and small fires.",
    "* Associations cannot by themselves prove secondary organic aerosol formation.",
]

with open(OUTPUT_FOLDER / "READ_ME_Integrated_CO_Fire_Results.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(interpretation_lines))

print("\n" + "=" * 72)
print("INTEGRATED CO + FIRE ANALYSIS COMPLETED")
print("=" * 72)
print(f"Valid CO values merged: {matched_co:,} of {len(combined):,}")
print(f"Rows used in all four models: {len(model_data):,}")
print(f"Dates used: {model_data['date'].nunique():,}")
print(f"Grid cells used: {model_data['grid_id'].nunique():,}")
print("\nPooled model performance:")
print(pooled_metrics[["model", "R2", "MAE_ug_m3", "RMSE_ug_m3", "mean_bias_ug_m3"]].to_string(index=False))
print("\nIncremental value versus baseline:")
print(incremental_value[[
    "model", "delta_R2_vs_baseline", "delta_MAE_vs_baseline_ug_m3",
    "delta_RMSE_vs_baseline_ug_m3",
]].to_string(index=False))
print(f"\nRuntime: {elapsed_text(start_time)}")
print("All outputs were saved to:")
print(OUTPUT_FOLDER)
