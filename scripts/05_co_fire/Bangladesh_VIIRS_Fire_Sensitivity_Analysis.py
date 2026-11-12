"""
Bangladesh OA project: VIIRS fire-context and sensitivity analysis (2019-2024).

What this script does
---------------------
1. Reads the six NASA FIRMS VIIRS S-NPP Collection 2 ZIP/CSV files.
2. Retains vegetation-fire detections (type 0) with nominal or high confidence.
3. Creates same-day and recent 3-day fire-exposure indicators for each of the
   68 MERRA-2 grid-cell centres using 25 km and 50 km radii.
4. Merges the indicators with the OA-HCHO-NO2-ERA5 master dataset.
5. Produces descriptive fire summaries, OA/trace-gas relationship summaries,
   figures, and a leave-one-year-out model comparison with and without fire.
6. Optionally merges TROPOMI CO if its daily grid CSV is present.

Scientific scope
----------------
The FIRMS files cover Bangladesh (and any buffer included in the request).
Therefore, these variables measure local/in-domain thermal-anomaly exposure.
They do not represent all transboundary smoke reaching Bangladesh and should
not be described as a regional 300-km transport analysis or as direct SOA
measurements. The results are a combustion/fire sensitivity check.

Run in Spyder after changing ROOT below if your project folder is different.
The script reads ZIP files directly; extraction is not required.
"""

from __future__ import annotations

import math
import os
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# =============================================================================
# 1. PATHS AND SETTINGS
# =============================================================================
ROOT = Path(os.environ.get("OA_PROJECT_FOLDER", str(Path.cwd())))

MASTER_FILE = ROOT / "Bangladesh_OA_HCHO_NO2_ERA5_Master_2019_2025.csv"
FIRMS_FOLDER = Path(
    os.environ.get("FIRMS_FOLDER", str(ROOT / "FIRMS_VIIRS_2019_2024"))
)
CO_FILE = ROOT / "Bangladesh_TROPOMI_CO_Daily_MERRAgrid_2019_2024.csv"
OUTPUT_DIR = Path(
    os.environ.get(
        "FIRE_OUTPUT_DIR", str(ROOT / "Fire_Sensitivity_Results_2019_2024")
    )
)

START_DATE = pd.Timestamp("2019-01-01")
END_DATE = pd.Timestamp("2024-12-31")
EXPECTED_YEARS = set(range(2019, 2025))
EXPECTED_GRIDS = 68
EXPECTED_ROWS = 149_056  # 2,192 days x 68 cells
EARTH_RADIUS_KM = 6371.0088
RANDOM_STATE = 42


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================
def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def find_column(frame: pd.DataFrame, candidates, required=True):
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise KeyError(
            f"None of {list(candidates)} was found. Available columns:\n"
            f"{list(frame.columns)}"
        )
    return None


def haversine_matrix_km(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    fire_lat: np.ndarray,
    fire_lon: np.ndarray,
) -> np.ndarray:
    """Return a grid-by-fire matrix of great-circle distances in kilometres."""
    lat1 = np.radians(grid_lat)[:, None]
    lon1 = np.radians(grid_lon)[:, None]
    lat2 = np.radians(fire_lat)[None, :]
    lon2 = np.radians(fire_lon)[None, :]

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def read_firms_archives(folder: Path) -> pd.DataFrame:
    """Read all FIRMS ZIPs and loose CSVs without double-counting detections."""
    if not folder.exists():
        raise FileNotFoundError(f"FIRMS folder not found:\n{folder}")

    tables = []
    zip_paths = sorted(folder.glob("DL_FIRE_SV-C2_*.zip"))
    csv_paths = sorted(folder.glob("fire_archive_SV-C2_*.csv"))

    for archive_path in zip_paths:
        with zipfile.ZipFile(archive_path) as archive:
            csv_names = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]
            if not csv_names:
                warnings.warn(f"No CSV found in {archive_path.name}; skipped.")
                continue
            for csv_name in csv_names:
                with archive.open(csv_name) as stream:
                    table = pd.read_csv(stream, low_memory=False)
                table["source_file"] = archive_path.name
                tables.append(table)

    # Support already-extracted downloads. Duplicate removal below prevents
    # double counting if both ZIP and CSV copies are present.
    for csv_path in csv_paths:
        table = pd.read_csv(csv_path, low_memory=False)
        table["source_file"] = csv_path.name
        tables.append(table)

    if not tables:
        raise FileNotFoundError(
            f"No FIRMS .zip or .csv files were found in:\n{folder}"
        )

    raw = normalise_columns(pd.concat(tables, ignore_index=True))
    required = ["latitude", "longitude", "acq_date", "confidence", "frp", "type"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise KeyError(f"FIRMS data are missing required columns: {missing}")

    fires = pd.DataFrame(
        {
            "fire_date": pd.to_datetime(raw["acq_date"], errors="coerce").dt.normalize(),
            "fire_latitude": pd.to_numeric(raw["latitude"], errors="coerce"),
            "fire_longitude": pd.to_numeric(raw["longitude"], errors="coerce"),
            "fire_frp_MW": pd.to_numeric(raw["frp"], errors="coerce"),
            "confidence": raw["confidence"].astype(str).str.strip().str.lower(),
            "fire_type": pd.to_numeric(raw["type"], errors="coerce"),
            "source_file": raw["source_file"],
        }
    )

    fires = fires.dropna(
        subset=["fire_date", "fire_latitude", "fire_longitude", "fire_frp_MW"]
    )
    fires = fires.loc[fires["fire_date"].between(START_DATE, END_DATE)].copy()

    # VIIRS C2: l=low, n=nominal, h=high. Primary analysis uses nominal+high.
    fires = fires.loc[fires["confidence"].isin(["n", "h"])].copy()

    # Type 0 is presumed vegetation fire. Types 2 and 3 represent static land
    # sources and offshore detections, which are excluded from the fire check.
    fires = fires.loc[fires["fire_type"].eq(0)].copy()
    fires["fire_frp_MW"] = fires["fire_frp_MW"].clip(lower=0)

    duplicate_key = [
        "fire_date", "fire_latitude", "fire_longitude", "fire_frp_MW",
        "confidence", "fire_type",
    ]
    fires = fires.drop_duplicates(duplicate_key).sort_values("fire_date")
    fires = fires.reset_index(drop=True)
    return fires


def add_optional_co(master: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Merge TROPOMI CO when the expected CSV is available."""
    if not CO_FILE.exists():
        print("TROPOMI CO file not found; continuing without CO.")
        return master, None

    co = normalise_columns(pd.read_csv(CO_FILE, low_memory=False))
    date_column = find_column(co, ["date"])
    grid_column = find_column(co, ["grid_id"])
    value_column = find_column(
        co,
        ["CO_mol_m2", "co_mol_m2", "CO", "carbon_monoxide_column_number_density"],
    )
    count_column = find_column(co, ["CO_pixel_count", "co_pixel_count"], required=False)

    co[date_column] = pd.to_datetime(co[date_column], errors="coerce").dt.normalize()
    co[grid_column] = co[grid_column].astype(str).str.strip()
    co[value_column] = pd.to_numeric(co[value_column], errors="coerce")

    source_nonmissing = int(co[value_column].notna().sum())
    if source_nonmissing == 0:
        print(
            "TROPOMI CO file was found, but its CO value column is entirely "
            "empty. Continuing without CO."
        )
        return master, None

    keep = [date_column, grid_column, value_column]
    rename = {
        date_column: "date",
        grid_column: "grid_id",
        value_column: "CO_mol_m2",
    }
    if count_column is not None:
        co[count_column] = pd.to_numeric(co[count_column], errors="coerce")
        keep.append(count_column)
        rename[count_column] = "CO_pixel_count"

    co = co[keep].rename(columns=rename)
    if co.duplicated(["date", "grid_id"]).any():
        raise ValueError("The TROPOMI CO file contains duplicate date-grid rows.")

    merged = master.merge(co, on=["date", "grid_id"], how="left", validate="one_to_one")
    merged_nonmissing = int(merged["CO_mol_m2"].notna().sum())
    if merged_nonmissing == 0:
        print(
            "TROPOMI CO contains values, but no date-grid keys matched the "
            "master dataset. Continuing without CO."
        )
        return master, None

    print(f"TROPOMI CO values merged: {merged_nonmissing:,}")
    return merged, "CO_mol_m2"


# =============================================================================
# 3. READ AND VALIDATE THE OA MASTER DATASET
# =============================================================================
if not MASTER_FILE.exists():
    raise FileNotFoundError(f"Master dataset not found:\n{MASTER_FILE}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

master = normalise_columns(pd.read_csv(MASTER_FILE, low_memory=False))
required_master_columns = [
    "date", "grid_id", "latitude", "longitude", "OCSMASS_ug_m3", "HCHO", "NO2",
    "temperature_C", "relative_humidity_percent", "boundary_layer_height_m",
    "wind_speed_m_s", "total_precipitation_mm", "solar_radiation_MJ_m2_day",
]
missing_master = [column for column in required_master_columns if column not in master]
if missing_master:
    raise KeyError(f"Master dataset is missing columns: {missing_master}")

master["date"] = pd.to_datetime(master["date"], errors="coerce").dt.normalize()
master["grid_id"] = master["grid_id"].astype(str).str.strip()
master = master.loc[master["date"].between(START_DATE, END_DATE)].copy()
master = master.sort_values(["date", "grid_id"]).reset_index(drop=True)

if master[["date", "grid_id", "latitude", "longitude"]].isna().any().any():
    raise ValueError("Invalid master dates, grid IDs, or coordinates were found.")
if master.duplicated(["date", "grid_id"]).any():
    raise ValueError("Master dataset contains duplicate date-grid rows.")

grid_table = (
    master[["grid_id", "latitude", "longitude"]]
    .drop_duplicates("grid_id")
    .sort_values("grid_id")
    .reset_index(drop=True)
)
if len(grid_table) != EXPECTED_GRIDS:
    raise ValueError(f"Expected {EXPECTED_GRIDS} grid cells; found {len(grid_table)}.")
if len(master) != EXPECTED_ROWS:
    warnings.warn(f"Expected {EXPECTED_ROWS:,} master rows; found {len(master):,}.")

master, co_column = add_optional_co(master)


# =============================================================================
# 4. READ, FILTER, AND SUMMARISE THE VIIRS DETECTIONS
# =============================================================================
fires = read_firms_archives(FIRMS_FOLDER)
observed_years = set(fires["fire_date"].dt.year.unique())
if observed_years != EXPECTED_YEARS:
    raise ValueError(
        f"Expected FIRMS coverage for {sorted(EXPECTED_YEARS)} but found "
        f"{sorted(observed_years)}."
    )

fires.to_csv(OUTPUT_DIR / "VIIRS_filtered_nominal_high_type0_2019_2024.csv", index=False)

all_dates = pd.DataFrame({"date": pd.date_range(START_DATE, END_DATE, freq="D")})
national_daily = (
    fires.groupby("fire_date")
    .agg(fire_count=("fire_date", "size"), fire_frp_MW=("fire_frp_MW", "sum"))
    .rename_axis("date")
    .reset_index()
)
national_daily = all_dates.merge(national_daily, on="date", how="left")
national_daily[["fire_count", "fire_frp_MW"]] = national_daily[
    ["fire_count", "fire_frp_MW"]
].fillna(0)
national_daily["year"] = national_daily["date"].dt.year
national_daily["month"] = national_daily["date"].dt.month
national_daily.to_csv(
    OUTPUT_DIR / "Bangladesh_daily_VIIRS_fire_summary_2019_2024.csv", index=False
)

annual_summary = (
    national_daily.groupby("year")
    .agg(
        retained_fire_detections=("fire_count", "sum"),
        total_FRP_MW=("fire_frp_MW", "sum"),
        days_with_detection=("fire_count", lambda values: int((values > 0).sum())),
        mean_daily_fire_count=("fire_count", "mean"),
    )
    .reset_index()
)
annual_summary.to_csv(OUTPUT_DIR / "VIIRS_annual_summary.csv", index=False)


# =============================================================================
# 5. CALCULATE GRID-LEVEL LOCAL FIRE EXPOSURE
# =============================================================================
dates = pd.date_range(START_DATE, END_DATE, freq="D")
number_dates = len(dates)
number_grids = len(grid_table)

count_25 = np.zeros((number_dates, number_grids), dtype=np.int32)
count_50 = np.zeros((number_dates, number_grids), dtype=np.int32)
frp_25 = np.zeros((number_dates, number_grids), dtype=np.float64)
frp_50 = np.zeros((number_dates, number_grids), dtype=np.float64)

date_position = {date: index for index, date in enumerate(dates)}
grid_lat = grid_table["latitude"].to_numpy(dtype=float)
grid_lon = grid_table["longitude"].to_numpy(dtype=float)

for fire_date, daily_fires in fires.groupby("fire_date", sort=True):
    position = date_position.get(fire_date)
    if position is None:
        continue
    distances = haversine_matrix_km(
        grid_lat,
        grid_lon,
        daily_fires["fire_latitude"].to_numpy(dtype=float),
        daily_fires["fire_longitude"].to_numpy(dtype=float),
    )
    daily_frp = daily_fires["fire_frp_MW"].to_numpy(dtype=float)
    mask_25 = distances <= 25.0
    mask_50 = distances <= 50.0
    count_25[position, :] = mask_25.sum(axis=1)
    count_50[position, :] = mask_50.sum(axis=1)
    frp_25[position, :] = mask_25 @ daily_frp
    frp_50[position, :] = mask_50 @ daily_frp

exposure = pd.MultiIndex.from_product(
    [dates, grid_table["grid_id"]], names=["date", "grid_id"]
).to_frame(index=False)
exposure["fire_count_25km"] = count_25.reshape(-1)
exposure["fire_frp_MW_25km"] = frp_25.reshape(-1)
exposure["fire_count_50km"] = count_50.reshape(-1)
exposure["fire_frp_MW_50km"] = frp_50.reshape(-1)
exposure = exposure.sort_values(["grid_id", "date"]).reset_index(drop=True)

for base_column in ["fire_count_50km", "fire_frp_MW_50km"]:
    exposure[f"{base_column}_lag1d"] = (
        exposure.groupby("grid_id")[base_column].shift(1).fillna(0)
    )
    exposure[f"{base_column}_3day"] = (
        exposure.groupby("grid_id")[base_column]
        .transform(lambda values: values.rolling(3, min_periods=1).sum())
    )

exposure["fire_any_50km"] = (exposure["fire_count_50km"] > 0).astype(int)
exposure["fire_any_50km_3day"] = (
    exposure["fire_count_50km_3day"] > 0
).astype(int)

exposure.to_csv(
    OUTPUT_DIR / "Bangladesh_grid_daily_VIIRS_exposure_2019_2024.csv", index=False
)

combined = master.merge(
    exposure, on=["date", "grid_id"], how="left", validate="one_to_one"
)
fire_columns = [column for column in exposure.columns if column not in ["date", "grid_id"]]
combined[fire_columns] = combined[fire_columns].fillna(0)

positive_frp = combined.loc[
    combined["fire_frp_MW_50km_3day"] > 0, "fire_frp_MW_50km_3day"
]
high_threshold = float(positive_frp.quantile(0.75)) if len(positive_frp) else math.inf
combined["fire_exposure_class"] = "No local fire"
combined.loc[
    combined["fire_frp_MW_50km_3day"] > 0, "fire_exposure_class"
] = "Lower local fire"
combined.loc[
    combined["fire_frp_MW_50km_3day"] >= high_threshold,
    "fire_exposure_class",
] = "High local fire"

combined["fire_count_50km_3day_log1p"] = np.log1p(
    combined["fire_count_50km_3day"]
)
combined["fire_frp_MW_50km_3day_log1p"] = np.log1p(
    combined["fire_frp_MW_50km_3day"]
)
combined = combined.sort_values(["date", "grid_id"]).reset_index(drop=True)
combined.to_csv(
    OUTPUT_DIR / "Bangladesh_OA_HCHO_NO2_ERA5_VIIRS_Master_2019_2024.csv",
    index=False,
)


# =============================================================================
# 6. DESCRIPTIVE OA-TRACE-GAS RELATIONSHIPS BY FIRE EXPOSURE
# =============================================================================
relationship_variables = ["HCHO", "NO2"]
if co_column is not None:
    relationship_variables.append(co_column)

relationship_rows = []
class_order = ["No local fire", "Lower local fire", "High local fire"]
for exposure_class in class_order:
    subset = combined.loc[combined["fire_exposure_class"].eq(exposure_class)]
    for predictor in relationship_variables:
        valid = subset[["OCSMASS_ug_m3", predictor]].dropna()
        relationship_rows.append(
            {
                "fire_exposure_class": exposure_class,
                "predictor": predictor,
                "n": len(valid),
                "OA_mean_ug_m3": valid["OCSMASS_ug_m3"].mean(),
                "OA_median_ug_m3": valid["OCSMASS_ug_m3"].median(),
                "pearson_r": valid["OCSMASS_ug_m3"].corr(valid[predictor], method="pearson"),
                "spearman_rho": valid["OCSMASS_ug_m3"].corr(valid[predictor], method="spearman"),
            }
        )

relationship_summary = pd.DataFrame(relationship_rows)
relationship_summary.to_csv(
    OUTPUT_DIR / "OA_trace_gas_relationships_by_fire_exposure.csv", index=False
)


# =============================================================================
# 7. BLOCK-AWARE MODEL SENSITIVITY: LEAVE ONE YEAR OUT
# =============================================================================
combined["month_sin"] = np.sin(2 * np.pi * combined["date"].dt.month / 12.0)
combined["month_cos"] = np.cos(2 * np.pi * combined["date"].dt.month / 12.0)

base_features = [
    "HCHO", "NO2", "temperature_C", "relative_humidity_percent",
    "boundary_layer_height_m", "wind_speed_m_s", "total_precipitation_mm",
    "solar_radiation_MJ_m2_day", "latitude", "longitude", "month_sin", "month_cos",
]
if co_column is not None:
    base_features.append(co_column)

fire_features = [
    "fire_count_50km_3day_log1p",
    "fire_frp_MW_50km_3day_log1p",
]
target = "OCSMASS_ug_m3"

model_data = combined.dropna(subset=[target]).copy()
model_data["year"] = model_data["date"].dt.year


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.07,
                    max_iter=120,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


performance_rows = []
for test_year in sorted(EXPECTED_YEARS):
    train = model_data.loc[model_data["year"].ne(test_year)]
    test = model_data.loc[model_data["year"].eq(test_year)]

    if train.empty or test.empty:
        continue

    predictions = {}
    for model_name, features in [
        ("Base_without_fire", base_features),
        ("Base_plus_local_fire", base_features + fire_features),
    ]:
        model = make_model()
        model.fit(train[features], train[target])
        predicted = model.predict(test[features])
        predictions[model_name] = predicted
        performance_rows.append(
            {
                "test_year": test_year,
                "model": model_name,
                "n_test": len(test),
                "R2": r2_score(test[target], predicted),
                "MAE_ug_m3": mean_absolute_error(test[target], predicted),
                "RMSE_ug_m3": np.sqrt(mean_squared_error(test[target], predicted)),
            }
        )

performance = pd.DataFrame(performance_rows)
performance.to_csv(
    OUTPUT_DIR / "leave_one_year_out_fire_model_performance.csv", index=False
)

performance_wide = performance.pivot(
    index="test_year", columns="model", values=["R2", "MAE_ug_m3", "RMSE_ug_m3"]
)
delta_rows = []
for year in performance_wide.index:
    delta_rows.append(
        {
            "test_year": year,
            "delta_R2_fire_minus_base": (
                performance_wide.loc[year, ("R2", "Base_plus_local_fire")]
                - performance_wide.loc[year, ("R2", "Base_without_fire")]
            ),
            "delta_MAE_fire_minus_base_ug_m3": (
                performance_wide.loc[year, ("MAE_ug_m3", "Base_plus_local_fire")]
                - performance_wide.loc[year, ("MAE_ug_m3", "Base_without_fire")]
            ),
            "delta_RMSE_fire_minus_base_ug_m3": (
                performance_wide.loc[year, ("RMSE_ug_m3", "Base_plus_local_fire")]
                - performance_wide.loc[year, ("RMSE_ug_m3", "Base_without_fire")]
            ),
        }
    )

model_delta = pd.DataFrame(delta_rows)
model_delta.loc[len(model_delta)] = {
    "test_year": "mean",
    "delta_R2_fire_minus_base": model_delta["delta_R2_fire_minus_base"].mean(),
    "delta_MAE_fire_minus_base_ug_m3": model_delta[
        "delta_MAE_fire_minus_base_ug_m3"
    ].mean(),
    "delta_RMSE_fire_minus_base_ug_m3": model_delta[
        "delta_RMSE_fire_minus_base_ug_m3"
    ].mean(),
}
model_delta.to_csv(OUTPUT_DIR / "fire_incremental_model_value.csv", index=False)


# =============================================================================
# 8. COMPACT FIGURES FOR THE APPENDIX
# =============================================================================
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})

figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].bar(annual_summary["year"], annual_summary["retained_fire_detections"], color="#d95f02")
axes[0].set_title("Retained VIIRS fire detections")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Detection count")
axes[0].grid(axis="y", alpha=0.25)

monthly = (
    national_daily.groupby("month")
    .agg(mean_daily_count=("fire_count", "mean"), mean_daily_FRP=("fire_frp_MW", "mean"))
    .reset_index()
)
axes[1].plot(monthly["month"], monthly["mean_daily_count"], marker="o", color="#1b9e77")
axes[1].set_title("Monthly fire-detection climatology")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Mean daily detections")
axes[1].set_xticks(range(1, 13))
axes[1].grid(alpha=0.25)

figure.tight_layout()
figure.savefig(OUTPUT_DIR / "Appendix_Figure_VIIRS_fire_context.png", dpi=300, bbox_inches="tight")
plt.close(figure)


def binned_relationship(data, predictor, exposure_class, bins=10):
    subset = data.loc[
        data["fire_exposure_class"].eq(exposure_class),
        [predictor, "OCSMASS_ug_m3"],
    ].dropna()
    if len(subset) < bins * 10:
        return pd.DataFrame()
    subset = subset.copy()
    subset["bin"] = pd.qcut(subset[predictor], q=bins, duplicates="drop")
    return (
        subset.groupby("bin", observed=True)
        .agg(
            predictor_mean=(predictor, "mean"),
            OA_mean=("OCSMASS_ug_m3", "mean"),
            OA_sem=("OCSMASS_ug_m3", "sem"),
            n=("OCSMASS_ug_m3", "size"),
        )
        .reset_index(drop=True)
    )


figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
plot_definitions = [
    ("HCHO", "HCHO column (mol m$^{-2}$)"),
    ("NO2", "NO$_2$ column (mol m$^{-2}$)"),
]
colors = {"No local fire": "#377eb8", "High local fire": "#e41a1c"}

for axis, (predictor, xlabel) in zip(axes, plot_definitions):
    for exposure_class in ["No local fire", "High local fire"]:
        binned = binned_relationship(combined, predictor, exposure_class)
        if binned.empty:
            continue
        axis.errorbar(
            binned["predictor_mean"],
            binned["OA_mean"],
            yerr=binned["OA_sem"],
            marker="o",
            linewidth=1.4,
            capsize=2,
            color=colors[exposure_class],
            label=exposure_class,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"Mean OCSMASS ($\mu$g m$^{-3}$)")
    axis.set_title(f"OA relationship with {predictor}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

figure.tight_layout()
figure.savefig(
    OUTPUT_DIR / "Appendix_Figure_OA_trace_gases_by_fire_exposure.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)


# =============================================================================
# 9. QUALITY REPORT AND INTERPRETATION GUARDRAILS
# =============================================================================
quality_rows = [
    ("master_rows_2019_2024", len(master)),
    ("unique_grid_cells", master["grid_id"].nunique()),
    ("FIRMS_detections_after_filters", len(fires)),
    ("FIRMS_high_confidence", int(fires["confidence"].eq("h").sum())),
    ("FIRMS_nominal_confidence", int(fires["confidence"].eq("n").sum())),
    ("grid_days_with_same_day_fire_within_50km", int((combined["fire_count_50km"] > 0).sum())),
    ("grid_days_with_3day_fire_within_50km", int((combined["fire_count_50km_3day"] > 0).sum())),
    ("enhanced_master_rows", len(combined)),
]
if co_column is not None:
    quality_rows.append(("TROPOMI_CO_nonmissing_rows", int(combined[co_column].notna().sum())))

quality_report = pd.DataFrame(quality_rows, columns=["check", "value"])
quality_report.to_csv(OUTPUT_DIR / "fire_analysis_quality_report.csv", index=False)

mean_delta_r2 = float(model_delta.loc[model_delta["test_year"].eq("mean"), "delta_R2_fire_minus_base"].iloc[0])
mean_delta_mae = float(model_delta.loc[model_delta["test_year"].eq("mean"), "delta_MAE_fire_minus_base_ug_m3"].iloc[0])

notes = f"""VIIRS FIRE SENSITIVITY ANALYSIS, 2019-2024

Retained detections: {len(fires):,}
Grid cells: {len(grid_table):,}
Enhanced master rows: {len(combined):,}
High-fire threshold (trailing 3-day FRP within 50 km): {high_threshold:.3f} MW
Mean leave-one-year-out delta R2 after adding local fire: {mean_delta_r2:.6f}
Mean leave-one-year-out delta MAE after adding local fire: {mean_delta_mae:.6f} ug/m3

Interpretation:
- Positive delta R2 and negative delta MAE indicate that the local fire proxy
  adds out-of-year predictive information beyond trace gases, meteorology,
  seasonality, and location.
- Near-zero or inconsistent changes indicate that explicit local fire exposure
  adds little incremental predictive value to this reanalysis-based OA model.
- These are associations, not causal source apportionment.
- FIRMS detects thermal anomalies, not smoke concentration or SOA.
- Because the requested domain is Bangladesh, fires outside the download area
  are not represented. Do not interpret this as a complete transboundary-smoke
  analysis.
- MERRA-2 OCSMASS is reanalysis surface organic carbon mass, not a direct
  measurement of ambient OA or SOA.
"""
(OUTPUT_DIR / "READ_ME_fire_results.txt").write_text(notes, encoding="utf-8")

print("\nAnalysis completed successfully.")
print(f"Retained nominal/high-confidence vegetation-fire detections: {len(fires):,}")
print(f"Enhanced master rows: {len(combined):,}")
print(f"Mean delta R2 from local fire variables: {mean_delta_r2:.6f}")
print(f"Mean delta MAE from local fire variables: {mean_delta_mae:.6f} ug/m3")
print(f"\nAll outputs were saved to:\n{OUTPUT_DIR}")
