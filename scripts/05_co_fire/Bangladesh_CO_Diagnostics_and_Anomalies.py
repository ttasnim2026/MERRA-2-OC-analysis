"""
TROPOMI CO diagnostics and anomaly analysis for Bangladesh.

This script:
1. checks whether daily grid-cell CO values are genuinely duplicated;
2. calculates daily spatial statistics;
3. calculates within-day spatial and grid/month temporal anomalies;
4. saves analysis-ready CSV tables; and
5. creates publication-quality figures.

Required packages:
    pandas, numpy, matplotlib
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


# ============================================================
# 1. FILE LOCATIONS — CHANGE ONLY THESE PATHS IF NEEDED
# ============================================================

PROJECT_FOLDER = Path(os.environ.get("OA_PROJECT_FOLDER", str(Path.cwd())))

INPUT_FILE = Path(os.environ.get(
    "CO_INPUT_FILE",
    str(PROJECT_FOLDER / "Bangladesh_TROPOMI_CO_Daily_MERRAgrid_2019_2024_FINAL.csv"),
))

OUTPUT_FOLDER = Path(os.environ.get(
    "CO_OUTPUT_FOLDER",
    str(PROJECT_FOLDER / "CO_Diagnostics_2019_2024"),
))

# Native MERRA-2 grid spacing used in the CSV.
GRID_WIDTH_DEGREES = 0.625
GRID_HEIGHT_DEGREES = 0.500


# ============================================================
# 2. SETTINGS
# ============================================================

CO_COLUMN = "CO_mol_m2"
KEY_COLUMNS = ["date", "grid_id"]
REQUIRED_COLUMNS = {
    "date", "grid_id", "latitude", "longitude", CO_COLUMN
}

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 400,
})


# ============================================================
# 3. READ AND VALIDATE THE CO TABLE
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "CO input file was not found. Check INPUT_FILE at the top of the script:\n"
        f"{INPUT_FILE}"
    )

co = pd.read_csv(INPUT_FILE)

missing_columns = REQUIRED_COLUMNS.difference(co.columns)
if missing_columns:
    raise ValueError(
        "The input CSV is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

co["date"] = pd.to_datetime(co["date"], errors="coerce")
for column in ["latitude", "longitude", CO_COLUMN]:
    co[column] = pd.to_numeric(co[column], errors="coerce")

if co["date"].isna().any():
    raise ValueError("At least one date could not be parsed.")

duplicate_keys = int(co.duplicated(KEY_COLUMNS).sum())
if duplicate_keys:
    raise ValueError(
        f"Found {duplicate_keys:,} duplicate date-grid_id rows. "
        "Resolve these before continuing."
    )

valid = co.dropna(subset=[CO_COLUMN]).copy()
if valid.empty:
    raise ValueError(f"{CO_COLUMN} contains no valid numerical values.")


# ============================================================
# 4. DAILY SPATIAL DIAGNOSTICS
# ============================================================

daily_stats = (
    valid.groupby("date")[CO_COLUMN]
    .agg(
        valid_grid_cells="count",
        unique_exact_values="nunique",
        mean_CO_mol_m2="mean",
        median_CO_mol_m2="median",
        std_CO_mol_m2="std",
        minimum_CO_mol_m2="min",
        maximum_CO_mol_m2="max",
    )
    .reset_index()
)

daily_stats["range_CO_mol_m2"] = (
    daily_stats["maximum_CO_mol_m2"]
    - daily_stats["minimum_CO_mol_m2"]
)
daily_stats["spatial_CV_percent"] = (
    100.0
    * daily_stats["std_CO_mol_m2"]
    / daily_stats["mean_CO_mol_m2"]
)
daily_stats["all_valid_grids_identical"] = (
    daily_stats["unique_exact_values"] == 1
)

# Count visually distinct values at several display precisions.
for decimals in [2, 3, 4, 5, 6]:
    rounded_counts = (
        valid.assign(_rounded=valid[CO_COLUMN].round(decimals))
        .groupby("date")["_rounded"]
        .nunique()
    )
    daily_stats[f"unique_values_at_{decimals}_decimals"] = (
        daily_stats["date"].map(rounded_counts)
    )


# ============================================================
# 5. ANOMALY VARIABLES
# ============================================================

# A. Spatial anomaly: difference from the Bangladesh mean on the same date.
# This highlights which grid cells were relatively high or low that day.
co["CO_daily_BD_mean_mol_m2"] = (
    co.groupby("date")[CO_COLUMN].transform("mean")
)
co["CO_spatial_anomaly_mol_m2"] = (
    co[CO_COLUMN] - co["CO_daily_BD_mean_mol_m2"]
)

# B. Grid/month climatology: the expected value for each grid cell and month.
# Subtracting it removes the normal seasonal cycle and persistent grid contrast.
co["calendar_month"] = co["date"].dt.month
climatology_group = co.groupby(["grid_id", "calendar_month"])[CO_COLUMN]
co["CO_grid_month_climatology_mol_m2"] = climatology_group.transform("mean")
co["CO_grid_month_std_mol_m2"] = climatology_group.transform("std")
co["CO_deseasonalized_anomaly_mol_m2"] = (
    co[CO_COLUMN] - co["CO_grid_month_climatology_mol_m2"]
)

safe_std = co["CO_grid_month_std_mol_m2"].replace(0, np.nan)
co["CO_deseasonalized_zscore"] = (
    co["CO_deseasonalized_anomaly_mol_m2"] / safe_std
)


# ============================================================
# 6. MONTHLY AND GRID SUMMARIES
# ============================================================

daily_national = (
    valid.groupby("date")[CO_COLUMN]
    .mean()
    .rename("national_daily_mean_CO_mol_m2")
    .reset_index()
)
daily_national["rolling_30day_mean_CO_mol_m2"] = (
    daily_national["national_daily_mean_CO_mol_m2"]
    .rolling(30, center=True, min_periods=10)
    .mean()
)
daily_national["year"] = daily_national["date"].dt.year
daily_national["month"] = daily_national["date"].dt.month

monthly_national = (
    daily_national.groupby(["year", "month"])
    ["national_daily_mean_CO_mol_m2"]
    .agg(monthly_mean_CO_mol_m2="mean", monthly_std_CO_mol_m2="std", days="count")
    .reset_index()
)

grid_summary = (
    valid.groupby(["grid_id", "latitude", "longitude"])[CO_COLUMN]
    .agg(
        mean_CO_mol_m2="mean",
        std_CO_mol_m2="std",
        minimum_CO_mol_m2="min",
        maximum_CO_mol_m2="max",
        valid_days="count",
    )
    .reset_index()
)


# ============================================================
# 7. SAVE TABLES
# ============================================================

co.to_csv(
    OUTPUT_FOLDER / "Bangladesh_TROPOMI_CO_with_Anomalies_2019_2024.csv",
    index=False,
    float_format="%.8f",
)
daily_stats.to_csv(
    OUTPUT_FOLDER / "CO_Daily_Spatial_Diagnostics_2019_2024.csv",
    index=False,
    float_format="%.8f",
)
daily_national.to_csv(
    OUTPUT_FOLDER / "CO_National_Daily_Mean_2019_2024.csv",
    index=False,
    float_format="%.8f",
)
monthly_national.to_csv(
    OUTPUT_FOLDER / "CO_National_Monthly_Mean_2019_2024.csv",
    index=False,
    float_format="%.8f",
)
grid_summary.to_csv(
    OUTPUT_FOLDER / "CO_Grid_Summary_2019_2024.csv",
    index=False,
    float_format="%.8f",
)


# ============================================================
# 8. FIGURE 1 — NATIONAL DAILY CO TIME SERIES
# ============================================================

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(
    daily_national["date"],
    daily_national["national_daily_mean_CO_mol_m2"],
    color="#8CBBD9",
    linewidth=0.55,
    alpha=0.70,
    label="Daily Bangladesh mean",
)
ax.plot(
    daily_national["date"],
    daily_national["rolling_30day_mean_CO_mol_m2"],
    color="#B2182B",
    linewidth=1.5,
    label="30-day moving mean",
)
ax.set_title("TROPOMI total-column CO over Bangladesh, 2019–2024")
ax.set_ylabel(r"CO column (mol m$^{-2}$)")
ax.set_xlabel("Date")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "Figure_CO_National_Daily_TimeSeries.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# 9. FIGURE 2 — MONTHLY DISTRIBUTIONS
# ============================================================

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_values = [
    daily_national.loc[
        daily_national["month"] == month,
        "national_daily_mean_CO_mol_m2",
    ].dropna().to_numpy()
    for month in range(1, 13)
]

fig, ax = plt.subplots(figsize=(9.5, 4.8))
box = ax.boxplot(
    monthly_values,
    patch_artist=True,
    showfliers=False,
    widths=0.65,
)
for patch in box["boxes"]:
    patch.set_facecolor("#5AB4AC")
    patch.set_alpha(0.75)
for median in box["medians"]:
    median.set_color("#8C2D04")
    median.set_linewidth(1.5)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_title("Seasonal distribution of Bangladesh-mean TROPOMI CO")
ax.set_ylabel(r"CO column (mol m$^{-2}$)")
ax.set_xlabel("Calendar month")
ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "Figure_CO_Monthly_Distribution.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# 10. FIGURE 3 — RAW CO VERSUS SPATIAL ANOMALY
# ============================================================

# Select the earliest date having the maximum number of valid grid cells.
maximum_coverage = int(daily_stats["valid_grid_cells"].max())
example_date = daily_stats.loc[
    daily_stats["valid_grid_cells"] == maximum_coverage, "date"
].min()
example = co.loc[
    (co["date"] == example_date) & co[CO_COLUMN].notna()
].copy()


def draw_grid_map(ax, table, value_column, cmap, norm, title, colorbar_label):
    """Draw complete MERRA-2 grid rectangles centered on lon/lat coordinates."""
    for row in table.itertuples(index=False):
        value = getattr(row, value_column)
        rectangle = Rectangle(
            (
                row.longitude - GRID_WIDTH_DEGREES / 2,
                row.latitude - GRID_HEIGHT_DEGREES / 2,
            ),
            GRID_WIDTH_DEGREES,
            GRID_HEIGHT_DEGREES,
            facecolor=cmap(norm(value)),
            edgecolor="white",
            linewidth=0.35,
        )
        ax.add_patch(rectangle)

    ax.set_xlim(
        table["longitude"].min() - GRID_WIDTH_DEGREES / 2,
        table["longitude"].max() + GRID_WIDTH_DEGREES / 2,
    )
    ax.set_ylim(
        table["latitude"].min() - GRID_HEIGHT_DEGREES / 2,
        table["latitude"].max() + GRID_HEIGHT_DEGREES / 2,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = plt.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


raw_cmap = plt.get_cmap("viridis")
raw_norm = plt.Normalize(
    vmin=example[CO_COLUMN].min(),
    vmax=example[CO_COLUMN].max(),
)

anomaly_limit = float(example["CO_spatial_anomaly_mol_m2"].abs().max())
anomaly_cmap = plt.get_cmap("RdBu_r")
anomaly_norm = TwoSlopeNorm(
    vmin=-anomaly_limit,
    vcenter=0,
    vmax=anomaly_limit,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
date_label = pd.Timestamp(example_date).strftime("%Y-%m-%d")

draw_grid_map(
    axes[0], example, CO_COLUMN, raw_cmap, raw_norm,
    f"Raw CO column: {date_label}",
    r"CO (mol m$^{-2}$)",
)
draw_grid_map(
    axes[1], example, "CO_spatial_anomaly_mol_m2",
    anomaly_cmap, anomaly_norm,
    f"Difference from Bangladesh daily mean: {date_label}",
    r"CO anomaly (mol m$^{-2}$)",
)

fig.savefig(
    OUTPUT_FOLDER / "Figure_CO_Raw_and_Spatial_Anomaly_Example.png",
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 11. PRINT AUDIT RESULTS
# ============================================================

dates_all_identical = int(daily_stats["all_valid_grids_identical"].sum())
median_spatial_cv = float(daily_stats["spatial_CV_percent"].median())
median_spatial_range = float(daily_stats["range_CO_mol_m2"].median())

first_date = daily_stats["date"].min()
first = valid.loc[valid["date"] == first_date, CO_COLUMN]

print("\nCO ANALYSIS COMPLETED")
print("=" * 60)
print(f"Input rows: {len(co):,}")
print(f"Valid CO rows: {len(valid):,}")
print(f"Unique dates with valid CO: {valid['date'].nunique():,}")
print(f"Unique grid cells: {co['grid_id'].nunique():,}")
print(f"Duplicate date-grid keys: {duplicate_keys:,}")
print(f"Dates with all valid grid values exactly identical: {dates_all_identical:,}")
print(f"Median daily spatial CV: {median_spatial_cv:.2f}%")
print(f"Median daily spatial range: {median_spatial_range:.6f} mol/m2")
print("\nFirst valid date check:")
print(f"Date: {first_date.date()}")
print(f"Valid grids: {first.count():,}")
print(f"Exact unique values: {first.nunique():,}")
print(f"Minimum: {first.min():.6f}")
print(f"Maximum: {first.max():.6f}")
print(f"Unique values when shown to 2 decimals: {first.round(2).nunique():,}")
print(f"Unique values when shown to 6 decimals: {first.round(6).nunique():,}")
print("\nUse at least 5–6 decimal places when viewing CO in Excel.")
print("All outputs were saved to:")
print(OUTPUT_FOLDER)
