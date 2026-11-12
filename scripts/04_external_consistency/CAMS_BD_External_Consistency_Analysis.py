"""
CAMS-BD external-consistency analysis for the Bangladesh OCSMASS study.

What this script does
---------------------
1. Reads the nested CAMS-BD ZIP archive and combines the 2012-2021, 2022,
   2023, and 2024 workbooks.
2. Standardizes columns, dates, station names, and missing values.
3. Keeps 2019-2024 and applies configurable, conservative physical QC limits.
4. Aggregates hourly station observations to daily values.
5. Matches every station to the nearest of the 68 MERRA-2 grid cells.
6. Joins station observations to the existing OCSMASS/HCHO/NO2/ERA5 master CSV.
7. Evaluates:
      - OCSMASS versus station PM2.5 (external consistency only)
      - TROPOMI NO2 versus station NO2 (temporal consistency only)
      - ERA5 meteorology versus station meteorology
8. Saves clean data, diagnostics, statistics, and publication-ready figures.

IMPORTANT SCIENTIFIC LIMITATION
-------------------------------
PM2.5 is not organic carbon. Therefore, OCSMASS-versus-PM2.5 results must be
described as an "external consistency assessment," never as direct validation
of organic carbon. The satellite column and surface NO2 also have different
units/vertical support; their comparison is likewise based on association,
not bias or RMSE.

Required packages
-----------------
pandas, numpy, openpyxl, scipy, matplotlib

Install the repository requirements from a terminal before running the script.
"""

from __future__ import annotations

import io
import math
import os
import re
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

PROJECT_FOLDER = Path(os.environ.get("OA_PROJECT_FOLDER", str(Path.cwd())))

CAMS_ZIP = PROJECT_FOLDER / "CAMS-BD Hourly Air Quality and Meteorological Meas.zip"
MASTER_CSV = PROJECT_FOLDER / "Bangladesh_OA_HCHO_NO2_ERA5_Master_2019_2025.csv"
OUTPUT_FOLDER = PROJECT_FOLDER / "CAMS_External_Consistency_Results_2019_2024"

START_DATE = pd.Timestamp("2019-01-01")
END_DATE = pd.Timestamp("2024-12-31")

# Minimum number of valid hourly values needed for a reported daily mean/sum.
# Eighteen hours is a conservative 75% completeness requirement.
MIN_HOURS_PER_DAY = 18

# The CAMS Rain field has unresolved accumulation/reset behavior and severe
# outliers in some years. Keep it in the clean output, but do not treat it as a
# validated daily rainfall total unless its semantics are confirmed from the
# instrument metadata.
VALIDATE_STATION_RAINFALL = False

# Conservative physical/data-error screening limits. Values outside these
# ranges are converted to missing; the number removed is reported.
QC_LIMITS = {
    "SO2_ppb": (0, 1000),
    "NO_ppb": (0, 1000),
    "NO2_ppb": (0, 500),
    "NOX_ppb": (0, 1500),
    "CO_ppm": (0, 100),
    "O3_ppb": (0, 500),
    "PM25_ug_m3": (0, 2000),
    "PM10_ug_m3": (0, 3000),
    "wind_speed_m_s": (0, 60),
    "wind_direction_degrees": (0, 360),
    "temperature_C": (0, 50),
    "relative_humidity_percent": (0, 100),
    "solar_radiation_W_m2": (0, 1500),
    "pressure_hPa": (850, 1100),
    "rain_mm": (0, 300),
    "vertical_wind_speed_m_s": (-20, 20),
}


# Authoritative CAMS coordinates transcribed from Table 3 of the Department of
# Environment's Air Quality Monthly Report (April 2025). The report prints the
# decimal digits using degree/minute/second-like punctuation (for example,
# 23°.77’73.94”N); this is interpreted as decimal latitude 23.777394 because
# values such as 77 minutes and 73.94 seconds cannot be valid DMS coordinates.
STATION_COORDINATES = {
    "Dhaka_DoE": (23.777394, 90.372603),          # CAMS-1
    "Dhaka_BARC": (23.759410, 90.388679),         # CAMS-2, Farmgate
    "Dhaka_Darussalam": (23.780775, 90.355410),   # CAMS-3
    "Gazipur": (23.994128, 90.422315),            # CAMS-4
    "Narayanganj": (23.626079, 90.507200),        # CAMS-5
    "Chattogram_TV": (22.360487, 91.800454),      # CAMS-6, Khulshi
    "Chattogram_Agrabad": (22.323020, 91.802336), # CAMS-7
    "Khulna": (22.835775, 89.529056),             # CAMS-8, Boyra
    "Rajshahi": (24.383320, 88.608007),           # CAMS-9, Sapura
    "Sylhet": (24.888334, 91.867347),             # CAMS-10
    "Barishal": (22.710287, 90.362598),           # CAMS-11
    "Mymensingh": (24.762458, 90.402102),         # CAMS-12
    "Rangpur": (25.747371, 89.228931),            # CAMS-13
    "Savar": (23.953704, 90.279794),              # CAMS-14
    "Narsingdi": (23.932456, 90.716598),          # CAMS-15
    "Cumilla": (23.472988, 91.180671),            # CAMS-16
}


# =============================================================================
# 2. COLUMN AND STATION STANDARDIZATION
# =============================================================================

COLUMN_MAP = {
    "SO2": "SO2_ppb",
    "NO": "NO_ppb",
    "NO2": "NO2_ppb",
    "NOX": "NOX_ppb",
    "CO": "CO_ppm",
    "O3": "O3_ppb",
    "PM2.5": "PM25_ug_m3",
    "PM10": "PM10_ug_m3",
    "Wind Speed": "wind_speed_m_s",
    "WS": "wind_speed_m_s",
    "Wind Dir": "wind_direction_degrees",
    "WD": "wind_direction_degrees",
    "Temperature": "temperature_C",
    "Temp": "temperature_C",
    "RH": "relative_humidity_percent",
    "Solar Rad": "solar_radiation_W_m2",
    "SR": "solar_radiation_W_m2",
    "BP": "pressure_hPa",
    "Rain": "rain_mm",
    "V Wind Speed": "vertical_wind_speed_m_s",
    "VWS": "vertical_wind_speed_m_s",
}

MEAN_DAILY_VARIABLES = [
    "SO2_ppb", "NO_ppb", "NO2_ppb", "NOX_ppb", "CO_ppm", "O3_ppb",
    "PM25_ug_m3", "PM10_ug_m3", "wind_speed_m_s",
    "wind_direction_degrees", "temperature_C", "relative_humidity_percent",
    "solar_radiation_W_m2", "pressure_hPa", "vertical_wind_speed_m_s",
]


def normalized_text(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_station(value: object) -> str:
    """Map inconsistent workbook station labels to stable station names."""
    s = normalized_text(value)
    if not s or s == "nan":
        return ""
    if "barc" in s:
        return "Dhaka_BARC"
    if "darus" in s or "daruss" in s:
        return "Dhaka_Darussalam"
    if s == "doe" or "department of environment" in s or "agargaon" in s:
        return "Dhaka_DoE"
    if "agrabad" in s or s == "cda" or "chittagong cda" in s:
        return "Chattogram_Agrabad"
    if ("tv" in s and ("center" in s or "centre" in s or "station" in s or "sation" in s)):
        return "Chattogram_TV"
    if "chittagong" in s or "chattogram" in s:
        return "Chattogram_Agrabad"
    if "barisal" in s or "barishal" in s:
        return "Barishal"
    if "comilla" in s or "cumilla" in s:
        return "Cumilla"
    if "gazipur" in s:
        return "Gazipur"
    if "khulna" in s:
        return "Khulna"
    if "mymen" in s:
        return "Mymensingh"
    if "narayanganj" in s or "narayangonj" in s or "narayonganj" in s:
        return "Narayanganj"
    if "narsingdi" in s:
        return "Narsingdi"
    if "rajshahi" in s:
        return "Rajshahi"
    if "rangpur" in s:
        return "Rangpur"
    if "savar" in s:
        return "Savar"
    if "sylhet" in s:
        return "Sylhet"
    return "UNMATCHED__" + re.sub(r"\s+", "_", s)


def parse_old_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """Parse the separate Excel-serial Date and Time fields in 2012-2021."""
    date_num = pd.to_numeric(date_series, errors="coerce")
    serial_date = pd.Timestamp("1899-12-30") + pd.to_timedelta(date_num, unit="D")

    # Some cells can already be datetime-like; other cells are HH:MM strings
    # or fractions of an Excel day.
    def time_to_delta(value: object) -> pd.Timedelta:
        if pd.isna(value):
            return pd.NaT
        if hasattr(value, "hour"):
            return pd.Timedelta(
                hours=value.hour,
                minutes=getattr(value, "minute", 0),
                seconds=getattr(value, "second", 0),
            )
        if isinstance(value, (int, float, np.integer, np.floating)):
            return pd.to_timedelta(float(value), unit="D")
        text = str(value).strip()
        parsed = pd.to_timedelta(text, errors="coerce")
        if pd.isna(parsed):
            parsed_dt = pd.to_datetime(text, errors="coerce")
            if pd.notna(parsed_dt):
                return pd.Timedelta(
                    hours=parsed_dt.hour,
                    minutes=parsed_dt.minute,
                    seconds=parsed_dt.second,
                )
        return parsed

    time_delta = time_series.map(time_to_delta)
    return serial_date + time_delta


# =============================================================================
# 3. ZIP AND WORKBOOK READING
# =============================================================================

def extract_workbooks(zip_path: Path, destination: Path) -> list[Path]:
    """Extract .xlsx files whether the supplied archive is nested or direct."""
    workbook_paths: list[Path] = []
    nested_archives: list[Path] = []

    with zipfile.ZipFile(zip_path) as outer:
        for member in outer.namelist():
            lower = member.lower()
            if lower.endswith(".xlsx"):
                outer.extract(member, destination)
                workbook_paths.append(destination / member)
            elif lower.endswith(".zip"):
                outer.extract(member, destination)
                nested_archives.append(destination / member)

    for nested_path in nested_archives:
        with zipfile.ZipFile(nested_path) as nested:
            for member in nested.namelist():
                if member.lower().endswith(".xlsx"):
                    nested.extract(member, destination)
                    workbook_paths.append(destination / member)

    workbook_paths = sorted({p.resolve() for p in workbook_paths})
    if not workbook_paths:
        raise FileNotFoundError("No .xlsx workbooks were found inside the CAMS ZIP.")
    return workbook_paths


def read_one_workbook(path: Path) -> pd.DataFrame:
    print(f"Reading {path.name} ...")
    raw = pd.read_excel(
        path,
        engine="openpyxl",
        na_values=["", "-", "--", "---", "N/A", "NA", "null", "None"],
    )
    raw.columns = [str(c).strip() for c in raw.columns]

    if "Location" not in raw.columns:
        raise ValueError(f"Location column not found in {path.name}")

    raw["station_original"] = raw["Location"].astype(str).str.strip()
    raw["station"] = raw["Location"].map(canonical_station)
    raw["source_file"] = path.name

    if "Date & Time" in raw.columns:
        raw["datetime_local"] = pd.to_datetime(raw["Date & Time"], errors="coerce")
    elif "Date" in raw.columns and "Time" in raw.columns:
        raw["datetime_local"] = parse_old_datetime(raw["Date"], raw["Time"])
    else:
        raise ValueError(f"No recognized date/time columns in {path.name}")

    # Rename measurement columns and coerce unit/header rows to missing.
    raw = raw.rename(columns=COLUMN_MAP)
    measurement_columns = [c for c in QC_LIMITS if c in raw.columns]
    for column in measurement_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    # Dataset documentation confirms a Solar Radiation / Barometric Pressure
    # column swap for Savar in the 2012-2021 workbook.
    is_old = "2012-2021" in path.name
    is_savar = raw["station"].eq("Savar")
    if is_old and {"solar_radiation_W_m2", "pressure_hPa"}.issubset(raw.columns):
        old_sr = raw.loc[is_savar, "solar_radiation_W_m2"].copy()
        raw.loc[is_savar, "solar_radiation_W_m2"] = raw.loc[is_savar, "pressure_hPa"]
        raw.loc[is_savar, "pressure_hPa"] = old_sr

    keep = [
        "station", "station_original", "datetime_local", "source_file",
        *measurement_columns,
    ]
    return raw[keep].copy()


def read_all_cams(zip_path: Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="cams_bd_") as temp_dir:
        workbooks = extract_workbooks(zip_path, Path(temp_dir))
        tables = [read_one_workbook(p) for p in workbooks]
    return pd.concat(tables, ignore_index=True, sort=False)


# =============================================================================
# 4. QUALITY CONTROL AND DAILY AGGREGATION
# =============================================================================

def apply_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = df.copy()
    reports: list[dict] = []

    clean = clean.loc[clean["datetime_local"].notna()].copy()
    clean = clean.loc[clean["station"].ne("")].copy()
    clean = clean.loc[
        clean["datetime_local"].between(START_DATE, END_DATE + pd.Timedelta(days=1), inclusive="left")
    ].copy()

    # Remove exact duplicate station timestamps, preferring the last record.
    duplicate_mask = clean.duplicated(["station", "datetime_local"], keep="last")
    reports.append({
        "check": "duplicate_station_timestamps_removed",
        "variable": "all",
        "count": int(duplicate_mask.sum()),
    })
    clean = clean.loc[~duplicate_mask].copy()

    for variable, (lower, upper) in QC_LIMITS.items():
        if variable not in clean.columns:
            continue
        original_nonmissing = int(clean[variable].notna().sum())
        invalid = clean[variable].notna() & ~clean[variable].between(lower, upper)
        reports.append({
            "check": "outside_configured_QC_range_set_to_missing",
            "variable": variable,
            "lower_limit": lower,
            "upper_limit": upper,
            "nonmissing_before": original_nonmissing,
            "count": int(invalid.sum()),
        })
        clean.loc[invalid, variable] = np.nan

    clean["date"] = clean["datetime_local"].dt.floor("D")
    return clean, pd.DataFrame(reports)


def daily_aggregate(hourly: pd.DataFrame) -> pd.DataFrame:
    # Direction is circular (359° and 1° average to 0°, not 180°), so it is
    # handled separately below.
    available_means = [
        c for c in MEAN_DAILY_VARIABLES
        if c in hourly.columns and c != "wind_direction_degrees"
    ]
    grouped = hourly.groupby(["station", "date"], observed=True, sort=True)

    daily_mean = grouped[available_means].mean()
    daily_count = grouped[available_means].count().add_suffix("_hour_count")
    daily = daily_mean.join(daily_count).reset_index()

    if "wind_direction_degrees" in hourly.columns:
        direction = hourly[["station", "date", "wind_direction_degrees"]].copy()
        radians = np.deg2rad(direction["wind_direction_degrees"])
        direction["_sin"] = np.sin(radians)
        direction["_cos"] = np.cos(radians)
        circular = (
            direction.groupby(["station", "date"], observed=True)
            .agg(
                _sin=("_sin", "mean"),
                _cos=("_cos", "mean"),
                wind_direction_degrees_hour_count=("wind_direction_degrees", "count"),
            )
            .reset_index()
        )
        circular["wind_direction_degrees"] = (
            np.degrees(np.arctan2(circular["_sin"], circular["_cos"])) + 360
        ) % 360
        circular.loc[
            circular["wind_direction_degrees_hour_count"] < MIN_HOURS_PER_DAY,
            "wind_direction_degrees",
        ] = np.nan
        daily = daily.merge(
            circular.drop(columns=["_sin", "_cos"]),
            on=["station", "date"], how="left",
        )

    # Rainfall is an accumulation, so replace the mean with a sum.
    if "rain_mm" in hourly.columns:
        rain_sum = grouped["rain_mm"].sum(min_count=1).rename("rain_mm")
        rain_count = grouped["rain_mm"].count().rename("rain_mm_hour_count")
        daily = daily.drop(columns=["rain_mm", "rain_mm_hour_count"], errors="ignore")
        daily = daily.merge(
            pd.concat([rain_sum, rain_count], axis=1).reset_index(),
            on=["station", "date"], how="left",
        )

    # Apply daily completeness independently to every variable.
    for variable in [*available_means, "rain_mm"]:
        count_col = f"{variable}_hour_count"
        if variable in daily.columns and count_col in daily.columns:
            daily.loc[daily[count_col] < MIN_HOURS_PER_DAY, variable] = np.nan

    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    return daily


# =============================================================================
# 5. STATION-TO-GRID MATCHING
# =============================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def build_station_grid_crosswalk(
    station_names: list[str], grid: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict] = []
    missing = [s for s in station_names if s not in STATION_COORDINATES]
    if missing:
        raise ValueError(
            "Coordinates are missing for these standardized stations:\n  "
            + "\n  ".join(missing)
            + "\nAdd them to STATION_COORDINATES and run again."
        )
    for station in sorted(station_names):
        station_lat, station_lon = STATION_COORDINATES[station]
        candidates = grid.copy()
        candidates["distance_km"] = candidates.apply(
            lambda r: haversine_km(
                station_lat, station_lon, float(r["latitude"]), float(r["longitude"])
            ),
            axis=1,
        )
        nearest = candidates.loc[candidates["distance_km"].idxmin()]
        rows.append({
            "station": station,
            "station_latitude": station_lat,
            "station_longitude": station_lon,
            "coordinate_status": "DoE_Air_Quality_Monthly_Report_April_2025_Table_3",
            "grid_id": nearest["grid_id"],
            "grid_latitude": nearest["latitude"],
            "grid_longitude": nearest["longitude"],
            "station_to_grid_distance_km": nearest["distance_km"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6. STATISTICS
# =============================================================================

def pairwise_frame(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    return df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()


def association_statistics(
    df: pd.DataFrame, x: str, y: str, comparison: str, station: str
) -> dict:
    pair = pairwise_frame(df, x, y)
    result = {
        "comparison": comparison,
        "station": station,
        "x_variable": x,
        "y_variable": y,
        "n": len(pair),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "linear_R2": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "MAE": np.nan,
        "RMSE": np.nan,
        "mean_bias": np.nan,
    }
    if len(pair) < 10 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return result
    pearson = stats.pearsonr(pair[x], pair[y])
    spearman = stats.spearmanr(pair[x], pair[y], nan_policy="omit")
    linear = stats.linregress(pair[x], pair[y])
    result.update({
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
        "linear_R2": linear.rvalue ** 2,
        "slope": linear.slope,
        "intercept": linear.intercept,
    })
    return result


def agreement_statistics(
    df: pd.DataFrame, model: str, observed: str, comparison: str, station: str
) -> dict:
    result = association_statistics(df, model, observed, comparison, station)
    pair = pairwise_frame(df, model, observed)
    if len(pair) >= 10:
        error = pair[model] - pair[observed]
        result.update({
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "mean_bias": float(np.mean(error)),
        })
    return result


def station_month_anomaly(df: pd.DataFrame, column: str) -> pd.Series:
    # Derive month from date instead of relying on a merged `month` column;
    # both input tables contain month, so pandas suffixes those fields.
    month_key = pd.to_datetime(df["date"], errors="coerce").dt.month
    climatology = df[column].groupby([df["station"], month_key]).transform("mean")
    return df[column] - climatology


def calculate_statistics(matched: pd.DataFrame) -> pd.DataFrame:
    matched = matched.copy()
    anomaly_pairs = []
    if {"OCSMASS_ug_m3", "PM25_ug_m3"}.issubset(matched.columns):
        matched["OCSMASS_month_anomaly"] = station_month_anomaly(matched, "OCSMASS_ug_m3")
        matched["PM25_month_anomaly"] = station_month_anomaly(matched, "PM25_ug_m3")
        anomaly_pairs.append((
            "OCSMASS_month_anomaly", "PM25_month_anomaly",
            "OCSMASS_vs_station_PM2.5_within_station_month_anomalies",
        ))
    if {"NO2", "NO2_ppb"}.issubset(matched.columns):
        matched["TROPOMI_NO2_month_anomaly"] = station_month_anomaly(matched, "NO2")
        matched["station_NO2_month_anomaly"] = station_month_anomaly(matched, "NO2_ppb")
        anomaly_pairs.append((
            "TROPOMI_NO2_month_anomaly", "station_NO2_month_anomaly",
            "TROPOMI_vs_station_NO2_within_station_month_anomalies",
        ))

    output: list[dict] = []
    groups = [("ALL_STATIONS_POOLED", matched)] + list(matched.groupby("station"))

    for station, group in groups:
        # Different physical quantities/units: association only.
        if {"OCSMASS_ug_m3", "PM25_ug_m3"}.issubset(group.columns):
            output.append(association_statistics(
                group, "OCSMASS_ug_m3", "PM25_ug_m3",
                "OCSMASS_vs_station_PM2.5_external_consistency", station,
            ))
        if {"NO2", "NO2_ppb"}.issubset(group.columns):
            output.append(association_statistics(
                group, "NO2", "NO2_ppb",
                "TROPOMI_column_NO2_vs_station_surface_NO2_temporal_consistency", station,
            ))
        for x_anomaly, y_anomaly, label in anomaly_pairs:
            output.append(association_statistics(
                group, x_anomaly, y_anomaly, label, station,
            ))

        # Comparable units: agreement statistics are meaningful.
        comparisons = [
            ("temperature_C", "temperature_C_station", "ERA5_vs_station_temperature_C"),
            ("relative_humidity_percent", "relative_humidity_percent_station", "ERA5_vs_station_RH_percent"),
            ("wind_speed_m_s", "wind_speed_m_s_station", "ERA5_vs_station_wind_speed_m_s"),
        ]
        if VALIDATE_STATION_RAINFALL:
            comparisons.append(
                ("total_precipitation_mm", "rain_mm", "ERA5_vs_station_daily_rain_mm")
            )
        for model, observed, label in comparisons:
            if {model, observed}.issubset(group.columns):
                output.append(agreement_statistics(group, model, observed, label, station))

    return pd.DataFrame(output)


# =============================================================================
# 7. FIGURES
# =============================================================================

def zscore(series: pd.Series) -> pd.Series:
    sd = series.std()
    return (series - series.mean()) / sd if pd.notna(sd) and sd > 0 else np.nan


def save_scatter(
    data: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str,
    title: str, output_path: Path, one_to_one: bool = False,
) -> None:
    pair = pairwise_frame(data, x, y)
    if len(pair) < 10:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(pair[x], pair[y], s=10, alpha=0.20, color="#176B87", edgecolors="none")
    if pair[x].nunique() > 1:
        fit = stats.linregress(pair[x], pair[y])
        xx = np.linspace(pair[x].quantile(0.01), pair[x].quantile(0.99), 100)
        ax.plot(xx, fit.intercept + fit.slope * xx, color="#C0392B", lw=2,
                label=f"Linear fit: $R^2$={fit.rvalue**2:.2f}, n={len(pair):,}")
    if one_to_one:
        lo = min(pair[x].min(), pair[y].min())
        hi = max(pair[x].max(), pair[y].max())
        ax.plot([lo, hi], [lo, hi], "--", color="0.35", lw=1.2, label="1:1 line")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def make_figures(matched: pd.DataFrame, output_folder: Path) -> None:
    plot_df = matched.copy()

    if {"OCSMASS_ug_m3", "PM25_ug_m3"}.issubset(plot_df.columns):
        plot_df["OCSMASS_anomaly"] = station_month_anomaly(plot_df, "OCSMASS_ug_m3")
        plot_df["PM25_anomaly"] = station_month_anomaly(plot_df, "PM25_ug_m3")
        save_scatter(
            plot_df, "OCSMASS_anomaly", "PM25_anomaly",
            "OCSMASS anomaly (µg m⁻³)", "Station PM₂.₅ anomaly (µg m⁻³)",
            "Daily within-station seasonal anomalies: OCSMASS vs PM₂.₅",
            output_folder / "Figure_1_OCSMASS_PM25_anomalies.png",
        )

    if {"NO2", "NO2_ppb"}.issubset(plot_df.columns):
        plot_df["TROPOMI_NO2_anomaly"] = station_month_anomaly(plot_df, "NO2")
        plot_df["station_NO2_anomaly"] = station_month_anomaly(plot_df, "NO2_ppb")
        save_scatter(
            plot_df, "TROPOMI_NO2_anomaly", "station_NO2_anomaly",
            "TROPOMI NO₂ column anomaly", "Station surface NO₂ anomaly (ppb)",
            "Daily within-station seasonal anomalies: TROPOMI vs surface NO₂",
            output_folder / "Figure_2_TROPOMI_station_NO2_anomalies.png",
        )

    save_scatter(
        plot_df, "temperature_C", "temperature_C_station",
        "ERA5 temperature (°C)", "Station temperature (°C)",
        "ERA5 versus station daily temperature",
        output_folder / "Figure_3_ERA5_station_temperature.png", one_to_one=True,
    ) if {"temperature_C", "temperature_C_station"}.issubset(plot_df.columns) else None

    save_scatter(
        plot_df, "relative_humidity_percent", "relative_humidity_percent_station",
        "ERA5 relative humidity (%)", "Station relative humidity (%)",
        "ERA5 versus station daily relative humidity",
        output_folder / "Figure_4_ERA5_station_RH.png", one_to_one=True,
    ) if {"relative_humidity_percent", "relative_humidity_percent_station"}.issubset(plot_df.columns) else None

    # Monthly standardized time series: shape agreement without equating units.
    needed = {"date", "OCSMASS_ug_m3", "PM25_ug_m3"}
    if needed.issubset(plot_df.columns):
        monthly = (
            plot_df.set_index("date")[["OCSMASS_ug_m3", "PM25_ug_m3"]]
            .resample("MS").mean().dropna()
        )
        if len(monthly) >= 6:
            monthly = monthly.apply(zscore)
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(monthly.index, monthly["OCSMASS_ug_m3"], lw=2, label="OCSMASS")
            ax.plot(monthly.index, monthly["PM25_ug_m3"], lw=2, label="Station PM₂.₅")
            ax.axhline(0, color="0.4", lw=0.8)
            ax.set(
                ylabel="Standardized monthly mean (z score)",
                title="Monthly temporal consistency of OCSMASS and station PM₂.₅",
            )
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, ncol=2)
            fig.tight_layout()
            fig.savefig(output_folder / "Figure_5_monthly_OCSMASS_PM25_zscores.png",
                        dpi=400, bbox_inches="tight")
            plt.close(fig)


# =============================================================================
# 8. MAIN WORKFLOW
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if not CAMS_ZIP.exists():
        raise FileNotFoundError(f"CAMS ZIP not found:\n{CAMS_ZIP}")
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Master CSV not found:\n{MASTER_CSV}")

    print("\n1/7 Reading and standardizing CAMS-BD workbooks...")
    cams_raw = read_all_cams(CAMS_ZIP)
    print(f"Raw rows combined: {len(cams_raw):,}")

    print("\n2/7 Applying quality control...")
    cams_hourly, qc_report = apply_qc(cams_raw)
    print(f"Clean hourly rows retained for 2019-2024: {len(cams_hourly):,}")

    unmatched = sorted(s for s in cams_hourly["station"].unique() if s.startswith("UNMATCHED__"))
    if unmatched:
        unmatched_file = OUTPUT_FOLDER / "Unmatched_station_names.csv"
        (
            cams_hourly.loc[cams_hourly["station"].isin(unmatched), ["station_original", "station"]]
            .drop_duplicates().sort_values(["station", "station_original"])
            .to_csv(unmatched_file, index=False)
        )
        raise ValueError(
            "Some station names were not recognized. Review:\n"
            f"{unmatched_file}\nThen add their rules to canonical_station()."
        )

    print("\n3/7 Creating daily station table...")
    cams_daily = daily_aggregate(cams_hourly)
    print(f"Daily station rows: {len(cams_daily):,}")

    print("\n4/7 Reading the OCSMASS/TROPOMI/ERA5 master table...")
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    required_master = {"date", "grid_id", "latitude", "longitude", "OCSMASS_ug_m3"}
    missing_master = required_master.difference(master.columns)
    if missing_master:
        raise ValueError(f"Master CSV is missing required columns: {sorted(missing_master)}")
    master["date"] = pd.to_datetime(master["date"], errors="coerce")
    master = master.loc[master["date"].between(START_DATE, END_DATE)].copy()

    grid = (
        master[["grid_id", "latitude", "longitude"]]
        .drop_duplicates("grid_id").sort_values("grid_id").reset_index(drop=True)
    )
    print(f"Unique master grid cells: {len(grid):,}")
    if len(grid) != 68:
        warnings.warn(f"Expected 68 master grid cells but found {len(grid)}.")

    print("\n5/7 Matching stations to nearest MERRA-2 cells...")
    crosswalk = build_station_grid_crosswalk(
        sorted(cams_daily["station"].dropna().unique()), grid
    )
    cams_daily = cams_daily.merge(crosswalk, on="station", how="left", validate="many_to_one")

    # Avoid same-name collisions between station and master meteorology.
    station_rename = {
        "temperature_C": "temperature_C_station",
        "relative_humidity_percent": "relative_humidity_percent_station",
        "wind_speed_m_s": "wind_speed_m_s_station",
        "wind_direction_degrees": "wind_direction_degrees_station",
        "solar_radiation_W_m2": "solar_radiation_W_m2_station",
        "pressure_hPa": "pressure_hPa_station",
    }
    cams_daily_for_join = cams_daily.rename(columns=station_rename)

    matched = cams_daily_for_join.merge(
        master,
        on=["date", "grid_id"],
        how="left",
        validate="many_to_one",
        suffixes=("_station", "_master"),
    )

    print("\n6/7 Calculating statistics and figures...")
    statistics_table = calculate_statistics(matched)
    make_figures(matched, OUTPUT_FOLDER)

    print("\n7/7 Saving outputs...")
    # CSVs are used instead of Excel because the hourly table may exceed Excel's
    # row limit and CSV preserves a transparent, reproducible workflow.
    cams_hourly.to_csv(OUTPUT_FOLDER / "01_CAMS_hourly_clean_2019_2024.csv", index=False)
    cams_daily.to_csv(OUTPUT_FOLDER / "02_CAMS_daily_2019_2024.csv", index=False)
    crosswalk.to_csv(OUTPUT_FOLDER / "03_station_grid_crosswalk.csv", index=False)
    matched.to_csv(OUTPUT_FOLDER / "04_CAMS_master_daily_matched_2019_2024.csv", index=False)
    statistics_table.to_csv(OUTPUT_FOLDER / "05_external_consistency_statistics.csv", index=False)
    qc_report.to_csv(OUTPUT_FOLDER / "06_quality_control_report.csv", index=False)

    coverage = (
        cams_daily.groupby("station")
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            daily_rows=("date", "size"),
            valid_PM25_days=("PM25_ug_m3", "count"),
            valid_NO2_days=("NO2_ppb", "count"),
        )
        .reset_index()
    )
    coverage.to_csv(OUTPUT_FOLDER / "07_station_coverage_summary.csv", index=False)

    readme = f"""CAMS-BD external-consistency analysis

Analysis period: {START_DATE.date()} to {END_DATE.date()}
Daily completeness requirement: {MIN_HOURS_PER_DAY} valid hours
Station time basis: timestamps were treated as local Bangladesh civil time as exported.

Critical interpretation rules:
1. OCSMASS versus PM2.5 is external consistency, not direct organic-carbon validation.
2. TROPOMI column NO2 versus surface NO2 is temporal consistency; do not report
   bias/RMSE because the quantities have different units and vertical support.
3. ERA5 versus station temperature, RH, wind speed, and rainfall are agreement checks.
   Station rainfall agreement is disabled by default because the CAMS Rain field's
   accumulation/reset semantics and extreme values require further documentation.
4. Station coordinates were transcribed from Table 3 of the Bangladesh Department
   of Environment Air Quality Monthly Report for April 2025. The report displays
   decimal-degree digits with degree/minute/second-like punctuation; the digits
   were concatenated as decimal degrees because the printed minute/second groups
   are outside valid DMS ranges.
5. QC thresholds are documented at the top of the Python script and the number of
   removed values is recorded in 06_quality_control_report.csv.

Suggested dataset citation:
Ansari, N. W., & Rafi, A. H. (2026). CAMS-BD: Hourly Air Quality and
Meteorological Measurements Dataset (Version 1). Mendeley Data.
https://doi.org/10.17632/hwmzyh5w4w.1
"""
    (OUTPUT_FOLDER / "README_interpretation.txt").write_text(readme, encoding="utf-8")

    print("\nCOMPLETE")
    print(f"Results saved in:\n{OUTPUT_FOLDER}")
    print("\nStation coordinates: DoE April 2025 report, Table 3.")


if __name__ == "__main__":
    main()
