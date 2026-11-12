# -*- coding: utf-8 -*-
"""Exploratory analysis and feature preparation for the Bangladesh OA dataset.
"""

from pathlib import Path
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_FOLDER = Path(
    os.environ.get(
        "OA_PROJECT_FOLDER",
        str(Path.cwd()),
    )
)

INPUT_FILE = PROJECT_FOLDER / (
    "Bangladesh_OA_HCHO_NO2_ERA5_Master_2019_2025.csv"
)

OUTPUT_FOLDER = PROJECT_FOLDER / "EDA_Results_2019_2025"
TABLE_FOLDER = OUTPUT_FOLDER / "Tables"
FIGURE_FOLDER = OUTPUT_FOLDER / "Figures"
MODEL_FOLDER = OUTPUT_FOLDER / "Model_Data"

DEVELOPMENT_START = pd.Timestamp("2019-01-01")
DEVELOPMENT_END = pd.Timestamp("2024-12-31")
FINAL_TEST_START = pd.Timestamp("2025-01-01")
FINAL_TEST_END = pd.Timestamp("2025-12-31")

EXPECTED_GRID_CELLS = 68
EXPECTED_DATES = 2557
EXPECTED_ROWS = 173_876

TARGET = "OCSMASS_ug_m3"

METEOROLOGY_COLUMNS = ["temperature_C", "dewpoint_C", "relative_humidity_percent", "boundary_layer_height_m", "u10_m_s", "v10_m_s", "wind_speed_m_s", "wind_direction_degrees", "total_precipitation_mm", "solar_radiation_MJ_m2_day",]

ANALYSIS_COLUMNS = [
    TARGET,
    "HCHO",
    "NO2",
] + METEOROLOGY_COLUMNS

REQUIRED_COLUMNS = {
    "date",
    "year",
    "month",
    "day_of_year",
    "season",
    "grid_id",
    "latitude",
    "longitude",
    TARGET,
    "HCHO",
    "NO2",
    "hcho_cloud_fraction",
    "no2_cloud_fraction",
    "HCHO_available",
    "NO2_available",
    "temperature_K",
    "dewpoint_K",
    "boundary_layer_height_m",
    "u10_m_s",
    "v10_m_s",
    "total_precipitation_m",
    "solar_radiation_J_m2",
    "temperature_C",
    "dewpoint_C",
    "relative_humidity_percent",
    "wind_speed_m_s",
    "wind_direction_degrees",
    "total_precipitation_mm",
    "solar_radiation_MJ_m2_day",
    "mean_hour_count",
    "sum_hour_count",
}


# -----------------------------------------------------------------------------
# 2. Helpers
# -----------------------------------------------------------------------------

def save_figure(filename: str) -> None:
    """Save the current Matplotlib figure and close it."""
    plt.tight_layout()
    plt.savefig(FIGURE_FOLDER / filename, dpi=300, bbox_inches="tight")
    plt.close()


def write_text_report(lines: list[str], path: Path) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_source(df: pd.DataFrame) -> None:
    """Stop immediately if the master table is structurally incomplete."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, but found {len(df):,}."
        )

    if df["date"].nunique() != EXPECTED_DATES:
        raise ValueError(
            f"Expected {EXPECTED_DATES:,} dates, but found "
            f"{df['date'].nunique():,}."
        )

    if df["grid_id"].nunique() != EXPECTED_GRID_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_GRID_CELLS} grid cells, but found "
            f"{df['grid_id'].nunique()}."
        )

    duplicates = int(df.duplicated(["date", "grid_id"]).sum())
    if duplicates:
        raise ValueError(f"Found {duplicates:,} duplicate date-grid records.")

    rows_per_date = df.groupby("date", observed=True).size()
    if rows_per_date.min() != 68 or rows_per_date.max() != 68:
        raise ValueError("Every date must contain exactly 68 grid-cell rows.")

    if not (df["mean_hour_count"] == 24).all():
        raise ValueError("Some ERA5 daily means do not contain 24 hours.")

    if not (df["sum_hour_count"] == 24).all():
        raise ValueError("Some ERA5 daily sums do not contain 24 hours.")

    if (df["total_precipitation_mm"] < 0).any():
        raise ValueError("Negative rainfall remains in the master dataset.")


def make_overall_summary(df: pd.DataFrame, development: pd.DataFrame) -> None:
    summary_rows = [
        ("Total rows", len(df)),
        ("Total columns", df.shape[1]),
        ("First date", df["date"].min().date().isoformat()),
        ("Last date", df["date"].max().date().isoformat()),
        ("Unique dates", df["date"].nunique()),
        ("Unique grid cells", df["grid_id"].nunique()),
        ("Duplicate date-grid rows", df.duplicated(["date", "grid_id"]).sum()),
        ("Development rows (2019-2024)", len(development)),
        ("Final-test rows (2025)", int((df["year"] == 2025).sum())),
        ("Missing HCHO", df["HCHO"].isna().sum()),
        ("Missing NO2", df["NO2"].isna().sum()),
        ("Negative HCHO retrievals", (df["HCHO"] < 0).sum()),
        ("Negative NO2 retrievals", (df["NO2"] < 0).sum()),
        ("Negative rainfall values", (df["total_precipitation_mm"] < 0).sum()),
    ]
    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(
        TABLE_FOLDER / "01_dataset_overview.csv", index=False
    )


def make_summary_statistics(development: pd.DataFrame) -> None:
    stats = development[ANALYSIS_COLUMNS].describe(
        percentiles=[0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    ).T
    stats.insert(0, "variable", stats.index)
    stats.reset_index(drop=True).to_csv(
        TABLE_FOLDER / "02_development_summary_statistics.csv", index=False
    )


def make_missingness_tables(df: pd.DataFrame) -> None:
    missing_columns = [
        "HCHO",
        "NO2",
        "hcho_cloud_fraction",
        "no2_cloud_fraction",
    ]

    overall = pd.DataFrame(
        {
            "variable": missing_columns,
            "missing_count": [df[c].isna().sum() for c in missing_columns],
            "missing_percent": [df[c].isna().mean() * 100 for c in missing_columns],
        }
    )
    overall.to_csv(TABLE_FOLDER / "03_missingness_overall.csv", index=False)

    by_year = (
        df.groupby("year", observed=True)
        .agg(
            total_rows=("grid_id", "size"),
            HCHO_missing=("HCHO", lambda x: x.isna().sum()),
            NO2_missing=("NO2", lambda x: x.isna().sum()),
        )
        .reset_index()
    )
    by_year["HCHO_missing_percent"] = (
        by_year["HCHO_missing"] / by_year["total_rows"] * 100
    )
    by_year["NO2_missing_percent"] = (
        by_year["NO2_missing"] / by_year["total_rows"] * 100
    )
    by_year.to_csv(TABLE_FOLDER / "04_missingness_by_year.csv", index=False)

    by_month = (
        df.groupby("month", observed=True)
        .agg(
            total_rows=("grid_id", "size"),
            HCHO_missing=("HCHO", lambda x: x.isna().sum()),
            NO2_missing=("NO2", lambda x: x.isna().sum()),
        )
        .reset_index()
    )
    by_month["HCHO_missing_percent"] = (
        by_month["HCHO_missing"] / by_month["total_rows"] * 100
    )
    by_month["NO2_missing_percent"] = (
        by_month["NO2_missing"] / by_month["total_rows"] * 100
    )
    by_month.to_csv(TABLE_FOLDER / "05_missingness_by_month.csv", index=False)

    by_grid = (
        df.groupby(["grid_id", "latitude", "longitude"], observed=True)
        .agg(
            total_rows=("date", "size"),
            HCHO_missing=("HCHO", lambda x: x.isna().sum()),
            NO2_missing=("NO2", lambda x: x.isna().sum()),
        )
        .reset_index()
    )
    by_grid["HCHO_missing_percent"] = (
        by_grid["HCHO_missing"] / by_grid["total_rows"] * 100
    )
    by_grid["NO2_missing_percent"] = (
        by_grid["NO2_missing"] / by_grid["total_rows"] * 100
    )
    by_grid.to_csv(TABLE_FOLDER / "06_missingness_by_grid.csv", index=False)


def make_negative_retrieval_tables(df: pd.DataFrame) -> None:
    records = []
    for variable in ["HCHO", "NO2"]:
        available = int(df[variable].notna().sum())
        negative = int((df[variable] < 0).sum())
        records.append(
            {
                "variable": variable,
                "available_count": available,
                "negative_count": negative,
                "negative_percent_of_available": (
                    negative / available * 100 if available else np.nan
                ),
                "minimum": df[variable].min(),
                "p01": df[variable].quantile(0.01),
                "median": df[variable].median(),
                "p99": df[variable].quantile(0.99),
                "maximum": df[variable].max(),
            }
        )
    pd.DataFrame(records).to_csv(
        TABLE_FOLDER / "07_negative_satellite_retrievals.csv", index=False
    )

    by_year = (
        df.groupby("year", observed=True)
        .agg(
            HCHO_available=("HCHO", lambda x: x.notna().sum()),
            HCHO_negative=("HCHO", lambda x: (x < 0).sum()),
            NO2_available=("NO2", lambda x: x.notna().sum()),
            NO2_negative=("NO2", lambda x: (x < 0).sum()),
        )
        .reset_index()
    )
    by_year["HCHO_negative_percent"] = (
        by_year["HCHO_negative"] / by_year["HCHO_available"] * 100
    )
    by_year["NO2_negative_percent"] = (
        by_year["NO2_negative"] / by_year["NO2_available"] * 100
    )
    by_year.to_csv(
        TABLE_FOLDER / "08_negative_retrievals_by_year.csv", index=False
    )


def make_temporal_and_spatial_tables(development: pd.DataFrame) -> None:
    annual = (
        development.groupby("year", observed=True)
        .agg(
            rows=("grid_id", "size"),
            OCSMASS_mean=(TARGET, "mean"),
            OCSMASS_median=(TARGET, "median"),
            OCSMASS_std=(TARGET, "std"),
            HCHO_mean=("HCHO", "mean"),
            NO2_mean=("NO2", "mean"),
            temperature_C_mean=("temperature_C", "mean"),
            relative_humidity_mean=("relative_humidity_percent", "mean"),
            precipitation_mm_mean=("total_precipitation_mm", "mean"),
        )
        .reset_index()
    )
    annual.to_csv(TABLE_FOLDER / "09_development_annual_summary.csv", index=False)

    monthly = (
        development.groupby("month", observed=True)
        .agg(
            rows=("grid_id", "size"),
            OCSMASS_mean=(TARGET, "mean"),
            OCSMASS_median=(TARGET, "median"),
            HCHO_mean=("HCHO", "mean"),
            NO2_mean=("NO2", "mean"),
            temperature_C_mean=("temperature_C", "mean"),
            relative_humidity_mean=("relative_humidity_percent", "mean"),
            precipitation_mm_mean=("total_precipitation_mm", "mean"),
        )
        .reset_index()
    )
    monthly.to_csv(
        TABLE_FOLDER / "10_development_monthly_climatology.csv", index=False
    )

    seasonal = (
        development.groupby("season", observed=True)
        .agg(
            rows=("grid_id", "size"),
            OCSMASS_mean=(TARGET, "mean"),
            OCSMASS_median=(TARGET, "median"),
            HCHO_mean=("HCHO", "mean"),
            NO2_mean=("NO2", "mean"),
        )
        .reset_index()
    )
    seasonal.to_csv(
        TABLE_FOLDER / "11_development_seasonal_summary.csv", index=False
    )

    grid = (
        development.groupby(
            ["grid_id", "latitude", "longitude"], observed=True
        )
        .agg(
            rows=("date", "size"),
            OCSMASS_mean=(TARGET, "mean"),
            OCSMASS_median=(TARGET, "median"),
            OCSMASS_p95=(TARGET, lambda x: x.quantile(0.95)),
            HCHO_mean=("HCHO", "mean"),
            NO2_mean=("NO2", "mean"),
        )
        .reset_index()
    )
    grid.to_csv(TABLE_FOLDER / "12_development_grid_summary.csv", index=False)


def make_correlation_table(development: pd.DataFrame) -> pd.DataFrame:
    correlation = development[ANALYSIS_COLUMNS].corr(method="spearman")
    correlation.to_csv(
        TABLE_FOLDER / "13_development_spearman_correlations.csv"
    )
    return correlation


def make_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    model = df.copy()

    # Preserve the raw satellite retrieval values. Add explicit quality flags.
    model["HCHO_missing_flag"] = model["HCHO"].isna().astype("int8")
    model["NO2_missing_flag"] = model["NO2"].isna().astype("int8")
    model["HCHO_negative_flag"] = (model["HCHO"] < 0).astype("int8")
    model["NO2_negative_flag"] = (model["NO2"] < 0).astype("int8")

    # Right-skewed target transformation. Raw OCSMASS remains unchanged.
    model["log1p_OCSMASS"] = np.log1p(model[TARGET])

    # Cyclical calendar terms.
    days_in_year = np.where(model["date"].dt.is_leap_year, 366.0, 365.0)
    day_angle = 2 * np.pi * (model["day_of_year"] - 1) / days_in_year
    model["day_of_year_sin"] = np.sin(day_angle)
    model["day_of_year_cos"] = np.cos(day_angle)

    month_angle = 2 * np.pi * (model["month"] - 1) / 12.0
    model["month_sin"] = np.sin(month_angle)
    model["month_cos"] = np.cos(month_angle)

    # Circular wind direction. These are preferable to using degrees directly.
    wind_radians = np.deg2rad(model["wind_direction_degrees"])
    model["wind_direction_sin"] = np.sin(wind_radians)
    model["wind_direction_cos"] = np.cos(wind_radians)

    # Ratio is defined only for positive, available satellite retrievals.
    valid_ratio = (
        model["HCHO"].notna()
        & model["NO2"].notna()
        & (model["HCHO"] > 0)
        & (model["NO2"] > 0)
    )
    model["HCHO_NO2_ratio"] = np.nan
    model.loc[valid_ratio, "HCHO_NO2_ratio"] = (
        model.loc[valid_ratio, "HCHO"] / model.loc[valid_ratio, "NO2"]
    )
    model["HCHO_NO2_ratio_available"] = valid_ratio.astype("int8")

    # 2025 is labeled but not included in outcome-based EDA or development.
    model["dataset_partition"] = np.where(
        model["year"] <= 2024,
        "development_2019_2024",
        "final_test_2025",
    )

    return model


def make_figures(
    df: pd.DataFrame,
    development: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Raw and log-transformed target distributions (development only).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(development[TARGET], bins=100, color="#32688e", alpha=0.9)
    axes[0].set_title("OCSMASS distribution, 2019–2024")
    axes[0].set_xlabel("OCSMASS (µg/m³)")
    axes[0].set_ylabel("Observations")
    axes[1].hist(
        np.log1p(development[TARGET]), bins=100, color="#d17c34", alpha=0.9
    )
    axes[1].set_title("log(1 + OCSMASS), 2019–2024")
    axes[1].set_xlabel("log(1 + OCSMASS)")
    axes[1].set_ylabel("Observations")
    save_figure("01_OCSMASS_raw_and_log_distribution.png")

    # 2. Bangladesh-wide daily mean and 30-day rolling mean.
    daily = development.groupby("date", observed=True)[TARGET].mean().sort_index()
    rolling = daily.rolling(30, min_periods=15).mean()
    plt.figure(figsize=(13, 5))
    plt.plot(daily.index, daily, color="#9bbbd1", linewidth=0.7, label="Daily mean")
    plt.plot(
        rolling.index,
        rolling,
        color="#154c79",
        linewidth=1.6,
        label="30-day rolling mean",
    )
    plt.title("Bangladesh daily mean OCSMASS, 2019–2024")
    plt.xlabel("Date")
    plt.ylabel("OCSMASS (µg/m³)")
    plt.legend(frameon=False)
    save_figure("02_OCSMASS_daily_time_series.png")

    # 3. Monthly climatology.
    monthly = development.groupby("month", observed=True)[TARGET].agg(
        ["mean", "median"]
    )
    plt.figure(figsize=(9, 5))
    plt.plot(monthly.index, monthly["mean"], marker="o", label="Mean")
    plt.plot(monthly.index, monthly["median"], marker="s", label="Median")
    plt.xticks(range(1, 13))
    plt.title("Monthly OCSMASS climatology, 2019–2024")
    plt.xlabel("Month")
    plt.ylabel("OCSMASS (µg/m³)")
    plt.legend(frameon=False)
    save_figure("03_OCSMASS_monthly_climatology.png")

    # 4. Seasonal target boxplot. Extreme points are hidden only visually.
    preferred_order = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
    present = [s for s in preferred_order if s in set(development["season"])]
    if not present:
        present = sorted(development["season"].dropna().unique())
    arrays = [
        development.loc[development["season"] == season, TARGET].dropna().to_numpy()
        for season in present
    ]
    plt.figure(figsize=(9, 5))
    plt.boxplot(arrays, tick_labels=present, showfliers=False)
    plt.title("Seasonal OCSMASS distribution, 2019–2024")
    plt.xlabel("Season")
    plt.ylabel("OCSMASS (µg/m³)")
    save_figure("04_OCSMASS_by_season.png")

    # 5. Satellite missingness by year. This evaluates predictors, not 2025 OA.
    missing_year = (
        df.groupby("year", observed=True)
        .agg(
            HCHO_missing=("HCHO", lambda x: x.isna().mean() * 100),
            NO2_missing=("NO2", lambda x: x.isna().mean() * 100),
        )
        .reset_index()
    )
    x = np.arange(len(missing_year))
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(
        x - width / 2,
        missing_year["HCHO_missing"],
        width,
        label="HCHO",
        color="#3b82a0",
    )
    plt.bar(
        x + width / 2,
        missing_year["NO2_missing"],
        width,
        label="NO₂",
        color="#dc8b38",
    )
    plt.xticks(x, missing_year["year"])
    plt.title("Satellite retrieval missingness by year")
    plt.xlabel("Year")
    plt.ylabel("Missing observations (%)")
    plt.legend(frameon=False)
    save_figure("05_satellite_missingness_by_year.png")

    # 6. Spearman correlation heatmap (development only).
    matrix = correlation.to_numpy()
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(correlation.columns)))
    ax.set_yticks(np.arange(len(correlation.index)))
    ax.set_xticklabels(correlation.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(correlation.index, fontsize=8)
    ax.set_title("Spearman correlations, development period 2019–2024")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman correlation")
    save_figure("06_development_spearman_correlation_heatmap.png")

    # 7. Spatial pattern across the 68 MERRA grid cells.
    grid = (
        development.groupby(
            ["grid_id", "latitude", "longitude"], observed=True
        )[TARGET]
        .mean()
        .reset_index(name="OCSMASS_mean")
    )
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        grid["longitude"],
        grid["latitude"],
        c=grid["OCSMASS_mean"],
        s=120,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.35,
    )
    plt.colorbar(scatter, label="Mean OCSMASS (µg/m³)")
    plt.title("Mean OCSMASS by MERRA grid cell, 2019–2024")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    save_figure("07_spatial_mean_OCSMASS.png")

    # 8. HCHO and NO2 retrieval distributions, including negative tails.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(development["HCHO"].dropna(), bins=100, color="#4477aa")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("HCHO retrieval distribution, 2019–2024")
    axes[0].set_xlabel("HCHO column number density")
    axes[0].set_ylabel("Observations")
    axes[1].hist(development["NO2"].dropna(), bins=100, color="#cc6677")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("NO₂ retrieval distribution, 2019–2024")
    axes[1].set_xlabel("NO₂ column number density")
    axes[1].set_ylabel("Observations")
    save_figure("08_HCHO_NO2_retrieval_distributions.png")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n{INPUT_FILE}\n\n"
            "Place the final master CSV in the project folder or correct "
            "PROJECT_FOLDER at the beginning of the script."
        )

    for folder in [OUTPUT_FOLDER, TABLE_FOLDER, FIGURE_FOLDER, MODEL_FOLDER]:
        folder.mkdir(parents=True, exist_ok=True)

    print("Reading the final master dataset...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"], low_memory=False)
    df["date"] = df["date"].dt.normalize()
    df["grid_id"] = df["grid_id"].astype(str).str.strip()

    print("Validating structure and completeness...")
    validate_source(df)

    development = df.loc[
        df["date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
    ].copy()
    final_test = df.loc[
        df["date"].between(FINAL_TEST_START, FINAL_TEST_END)
    ].copy()

    if len(development) != 149_056:
        raise ValueError(
            f"Expected 149,056 development rows, found {len(development):,}."
        )
    if len(final_test) != 24_820:
        raise ValueError(
            f"Expected 24,820 final-test rows, found {len(final_test):,}."
        )

    print("Creating quality-control and EDA tables...")
    make_overall_summary(df, development)
    make_summary_statistics(development)
    make_missingness_tables(df)
    make_negative_retrieval_tables(df)
    make_temporal_and_spatial_tables(development)
    correlation = make_correlation_table(development)

    print("Creating the feature-prepared table...")
    model = make_feature_table(df)
    model_file = MODEL_FOLDER / (
        "Bangladesh_OA_Model_Features_2019_2025.csv"
    )
    model.to_csv(
        model_file,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        float_format="%.10g",
    )

    feature_dictionary = pd.DataFrame(
        [
            ("log1p_OCSMASS", "Natural log of 1 + OCSMASS; raw target retained"),
            ("day_of_year_sin", "Cyclical sine encoding of day of year"),
            ("day_of_year_cos", "Cyclical cosine encoding of day of year"),
            ("month_sin", "Cyclical sine encoding of month"),
            ("month_cos", "Cyclical cosine encoding of month"),
            ("wind_direction_sin", "Sine encoding of meteorological wind direction"),
            ("wind_direction_cos", "Cosine encoding of meteorological wind direction"),
            ("HCHO_missing_flag", "1 when HCHO is missing; otherwise 0"),
            ("NO2_missing_flag", "1 when NO2 is missing; otherwise 0"),
            ("HCHO_negative_flag", "1 when available HCHO retrieval is negative"),
            ("NO2_negative_flag", "1 when available NO2 retrieval is negative"),
            (
                "HCHO_NO2_ratio",
                "HCHO divided by NO2 only when both retrievals are positive",
            ),
            (
                "HCHO_NO2_ratio_available",
                "1 when the positive HCHO/NO2 ratio is defined",
            ),
            (
                "dataset_partition",
                "2019-2024 development or untouched 2025 final test",
            ),
        ],
        columns=["feature", "definition"],
    )
    feature_dictionary.to_csv(
        TABLE_FOLDER / "14_derived_feature_dictionary.csv", index=False
    )

    print("Creating figures...")
    make_figures(df, development, correlation)

    # Report only design and validation facts. It does not summarize 2025 OA.
    report_lines = [
        "Bangladesh OA EDA and feature-preparation report",
        "",
        f"Input: {INPUT_FILE.name}",
        f"Rows: {len(df):,}",
        f"Dates: {df['date'].nunique():,}",
        f"Grid cells: {df['grid_id'].nunique()}",
        f"Development rows (2019-2024): {len(development):,}",
        f"Untouched final-test rows (2025): {len(final_test):,}",
        f"Missing HCHO: {df['HCHO'].isna().sum():,}",
        f"Missing NO2: {df['NO2'].isna().sum():,}",
        f"Negative HCHO retrievals: {(df['HCHO'] < 0).sum():,}",
        f"Negative NO2 retrievals: {(df['NO2'] < 0).sum():,}",
        f"Negative rainfall values: {(df['total_precipitation_mm'] < 0).sum():,}",
        "",
        "Important processing decisions:",
        "- The source master CSV was not modified.",
        "- Negative HCHO and NO2 retrievals were preserved and flagged.",
        "- No HCHO or NO2 values were imputed.",
        "- The HCHO/NO2 ratio was calculated only when both values were positive.",
        "- Outcome-based EDA used 2019-2024 only.",
        "- The 2025 target was not used in plots, summaries, or correlations.",
    ]
    write_text_report(report_lines, OUTPUT_FOLDER / "EDA_run_report.txt")

    print()
    print("=" * 72)
    print("EDA AND FEATURE PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print(f"Development rows: {len(development):,}")
    print(f"Final-test rows preserved: {len(final_test):,}")
    print(f"Tables saved in: {TABLE_FOLDER}")
    print(f"Figures saved in: {FIGURE_FOLDER}")
    print(f"Feature table saved as: {model_file}")
    print()
    print("The original master CSV was not changed.")


if __name__ == "__main__":
    main()
