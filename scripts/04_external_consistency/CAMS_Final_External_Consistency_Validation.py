"""
Final, CAMS-BD external-consistency analysis.

This script starts from the previously created matched daily CSV. It does not
re-read the large hourly CAMS archive. Its primary purpose is to test whether
MERRA-2 OCSMASS varies consistently with independent surface PM2.5 measurements;
it is not a direct validation of organic carbon because CAMS-BD does not measure
OC/OA/OM.

Run in Spyder: edit PROJECT_FOLDER below, then Run File.
Run from a terminal (optional):
  python CAMS_Final_External_Consistency_Validation.py --input "...csv" --output "...folder"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------------------------------------------------------
# USER SETTINGS (only these paths normally need editing)
# -----------------------------------------------------------------------------
PROJECT_FOLDER = Path(os.environ.get("OA_PROJECT_FOLDER", str(Path.cwd())))
DEFAULT_RESULTS_FOLDER = PROJECT_FOLDER / "CAMS_External_Consistency_Results_2019_2024"
DEFAULT_INPUT_CSV = DEFAULT_RESULTS_FOLDER / "04_CAMS_master_daily_matched_2019_2024.csv"
DEFAULT_INPUT_ZIP = PROJECT_FOLDER / "CAMS_External_Consistency_Results_2019_2024.zip"
DEFAULT_OUTPUT_FOLDER = PROJECT_FOLDER / "CAMS_Final_Validation_Results_2019_2024"

BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 42
MIN_STATION_VALID_DAYS = 60
SUSPECT_PM25_VALUE = 985.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None,
                        help="Matched daily CSV or prior results ZIP")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def read_input(path: Path | None) -> tuple[pd.DataFrame, str]:
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.extend([DEFAULT_INPUT_CSV, DEFAULT_INPUT_ZIP])

    selected = next((p for p in candidates if p.exists()), None)
    if selected is None:
        raise FileNotFoundError(
            "Could not find the matched CAMS CSV or results ZIP.\n"
            f"Expected CSV: {DEFAULT_INPUT_CSV}\n"
            f"Expected ZIP: {DEFAULT_INPUT_ZIP}\n"
            "Edit PROJECT_FOLDER or pass --input."
        )

    if selected.suffix.lower() == ".csv":
        return pd.read_csv(selected, low_memory=False), str(selected)

    if selected.suffix.lower() == ".zip":
        with zipfile.ZipFile(selected) as zf:
            matches = [n for n in zf.namelist()
                       if n.endswith("04_CAMS_master_daily_matched_2019_2024.csv")]
            if not matches:
                raise FileNotFoundError("The ZIP does not contain the matched daily CSV.")
            with zf.open(matches[0]) as handle:
                return pd.read_csv(handle, low_memory=False), f"{selected}!{matches[0]}"

    raise ValueError("--input must be a CSV or ZIP file.")


def safe_corr(x: pd.Series | np.ndarray,
              y: pd.Series | np.ndarray,
              method: str) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    if method == "pearson":
        return float(stats.pearsonr(pair["x"], pair["y"]).statistic)
    return float(stats.spearmanr(pair["x"], pair["y"]).statistic)


def month_block_bootstrap(data: pd.DataFrame, x: str, y: str, method: str,
                          replicates: int, seed: int) -> tuple[float, float, int, int]:
    clean = data[["date", x, y]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    clean["block"] = clean["date"].dt.to_period("M").astype(str)
    blocks = clean["block"].unique()
    if len(clean) < 3 or len(blocks) < 2:
        return np.nan, np.nan, len(clean), len(blocks)

    arrays = {
        block: clean.loc[clean["block"].eq(block), [x, y]].to_numpy(float)
        for block in blocks
    }
    rng = np.random.default_rng(seed)
    estimates = np.full(replicates, np.nan)
    for i in range(replicates):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        sample = np.concatenate([arrays[b] for b in sampled], axis=0)
        estimates[i] = safe_corr(sample[:, 0], sample[:, 1], method)
    valid = estimates[np.isfinite(estimates)]
    if len(valid) < max(100, replicates // 2):
        return np.nan, np.nan, len(clean), len(blocks)
    low, high = np.percentile(valid, [2.5, 97.5])
    return float(low), float(high), len(clean), len(blocks)


def add_monthly_anomalies(df: pd.DataFrame, group: str,
                          columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["calendar_month"] = out["date"].dt.month
    for col in columns:
        if col in out:
            climatology = out.groupby([group, "calendar_month"])[col].transform("mean")
            out[f"{col}_anomaly"] = out[col] - climatology
    return out


def metric_row(data: pd.DataFrame, analysis: str, x: str, y: str,
               scale: str, replicates: int, seed: int) -> dict:
    clean = data[["date", x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    row = {
        "analysis": analysis,
        "scale": scale,
        "x": x,
        "y": y,
        "n": len(clean),
        "months": clean["date"].dt.to_period("M").nunique(),
    }
    for j, method in enumerate(["pearson", "spearman"]):
        row[f"{method}_r"] = safe_corr(clean[x], clean[y], method)
        low, high, _, _ = month_block_bootstrap(
            clean, x, y, method, replicates, seed + j * 100003
        )
        row[f"{method}_ci_low"] = low
        row[f"{method}_ci_high"] = high
    return row


def aggregate_grid_day(station_day: pd.DataFrame, pm25_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_columns = [
        "OCSMASS_ug_m3", "HCHO", "NO2", "temperature_C",
        "relative_humidity_percent", "wind_speed_m_s", "total_precipitation_mm",
        "grid_latitude", "grid_longitude",
    ]
    model_columns = [c for c in model_columns if c in station_day]
    conflicts = []
    for col in model_columns:
        counts = station_day.groupby(["grid_id", "date"])[col].nunique(dropna=True)
        bad = counts[counts > 1]
        if len(bad):
            conflicts.append(pd.DataFrame({
                "grid_id": bad.index.get_level_values(0),
                "date": bad.index.get_level_values(1),
                "variable": col,
                "unique_values": bad.values,
            }))

    aggregations: dict[str, tuple[str, str | callable]] = {
        "PM25_ug_m3": (pm25_column, "mean"),
        "NO2_ppb_surface": ("NO2_ppb", "mean"),
        "station_count": ("station", "nunique"),
        "stations": ("station", lambda x: " | ".join(sorted(set(x.dropna().astype(str))))),
    }
    for col in model_columns:
        aggregations[col] = (col, "first")

    grid_day = (
        station_day.groupby(["grid_id", "date"], as_index=False)
        .agg(**aggregations)
        .sort_values(["date", "grid_id"])
        .reset_index(drop=True)
    )
    conflict_df = pd.concat(conflicts, ignore_index=True) if conflicts else pd.DataFrame(
        columns=["grid_id", "date", "variable", "unique_values"]
    )
    return grid_day, conflict_df


def station_year_meteorology_qc(data: pd.DataFrame) -> pd.DataFrame:
    specifications = {
        "temperature_C_station": {"median": (15, 35), "sd": (1, 10), "p01": (5, math.inf), "p99": (-math.inf, 45)},
        "relative_humidity_percent_station": {"median": (30, 95), "sd": (2, 30), "p01": (5, math.inf), "p99": (-math.inf, 100)},
        "wind_speed_m_s_station": {"median": (0.05, 15), "sd": (0.05, 15), "p01": (0, math.inf), "p99": (-math.inf, 40)},
        "pressure_hPa_station": {"median": (950, 1050), "sd": (0.5, 30), "p01": (900, math.inf), "p99": (-math.inf, 1100)},
    }
    rows = []
    work = data.copy()
    work["year"] = work["date"].dt.year
    for (station, year), group in work.groupby(["station", "year"]):
        for variable, rules in specifications.items():
            values = pd.to_numeric(group.get(variable), errors="coerce").dropna()
            n = len(values)
            summary = {
                "station": station, "year": year, "variable": variable, "valid_days": n,
                "mean": values.mean() if n else np.nan,
                "median": values.median() if n else np.nan,
                "sd": values.std() if n > 1 else np.nan,
                "p01": values.quantile(0.01) if n else np.nan,
                "p99": values.quantile(0.99) if n else np.nan,
                "minimum": values.min() if n else np.nan,
                "maximum": values.max() if n else np.nan,
            }
            reasons = []
            if n < MIN_STATION_VALID_DAYS:
                reasons.append(f"fewer than {MIN_STATION_VALID_DAYS} valid days")
            for measure, (lower, upper) in rules.items():
                value = summary[measure]
                if not np.isfinite(value) or value < lower or value > upper:
                    reasons.append(f"{measure} outside [{lower}, {upper}]")
            summary["qualified"] = not reasons
            summary["exclusion_reasons"] = "; ".join(reasons)
            rows.append(summary)
    return pd.DataFrame(rows)


def plot_hexbin(data: pd.DataFrame, x: str, y: str, title: str,
                xlabel: str, ylabel: str, statistics: dict, destination: Path,
                one_to_one: bool = False) -> None:
    pair = data[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    # Statistics use every valid pair. Only the display is trimmed so a handful
    # of extreme observations cannot compress the dense central distribution.
    bounds = pair.quantile([0.005, 0.995])
    display = pair[
        pair[x].between(bounds.loc[0.005, x], bounds.loc[0.995, x]) &
        pair[y].between(bounds.loc[0.005, y], bounds.loc[0.995, y])
    ]
    fig, ax = plt.subplots(figsize=(8.2, 6.5))
    hb = ax.hexbin(display[x], display[y], gridsize=45, mincnt=1, bins="log", cmap="viridis")
    fig.colorbar(hb, ax=ax, label="log-scaled grid-day count")
    if len(pair) >= 3 and pair[x].nunique() > 1:
        slope, intercept = np.polyfit(pair[x], pair[y], 1)
        xx = np.linspace(pair[x].quantile(0.005), pair[x].quantile(0.995), 200)
        ax.plot(xx, slope * xx + intercept, color="#d62728", linewidth=2, label="Linear fit")
    if one_to_one:
        lo = min(pair[x].quantile(0.005), pair[y].quantile(0.005))
        hi = max(pair[x].quantile(0.995), pair[y].quantile(0.995))
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.5, label="1:1 line")
    text = (
        f"n = {statistics['n']:,}\n"
        f"Pearson r = {statistics['pearson_r']:.3f} "
        f"[{statistics['pearson_ci_low']:.3f}, {statistics['pearson_ci_high']:.3f}]\n"
        f"Spearman ρ = {statistics['spearman_r']:.3f} "
        f"[{statistics['spearman_ci_low']:.3f}, {statistics['spearman_ci_high']:.3f}]"
    )
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.text(0.5, 0.012,
             "Display limited to joint 0.5th–99.5th percentiles; statistics use all valid pairs",
             ha="center", va="bottom", fontsize=8, color="#555555")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(destination, dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_forest(results: pd.DataFrame, destination: Path) -> None:
    view = results.dropna(subset=["spearman_r", "spearman_ci_low", "spearman_ci_high"]).copy()
    view = view.sort_values("spearman_r")
    fig_height = max(6, 0.42 * len(view) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    y = np.arange(len(view))
    left = view["spearman_r"] - view["spearman_ci_low"]
    right = view["spearman_ci_high"] - view["spearman_r"]
    ax.errorbar(view["spearman_r"], y, xerr=np.vstack([left, right]), fmt="o",
                color="#1f77b4", ecolor="#607d8b", capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, labels=[f"{s} (n={n:,})" for s, n in zip(view.station, view.n)])
    ax.set_xlabel("Spearman ρ for calendar-month anomalies (95% month-block CI)")
    ax.set_title("Station-level consistency: MERRA-2 OCSMASS vs surface PM₂.₅")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=350, bbox_inches="tight")
    plt.close(fig)


def write_readme(destination: Path, source: str, audit: dict) -> None:
    text = f"""FINAL CAMS-BD EXTERNAL-CONSISTENCY ANALYSIS

Purpose
-------
This analysis assesses whether MERRA-2 surface organic-carbon mass concentration
(OCSMASS) co-varies with independent surface PM2.5 observations. CAMS-BD contains
no direct organic carbon, organic aerosol, or organic-mass measurements. Therefore,
these results are external consistency/robustness evidence, not direct validation
of OCSMASS.

Input
-----
{source}

Primary design
--------------
1. Daily station observations were already matched to the nearest MERRA-2 grid cell.
2. PM2.5 values exactly equal to 985 µg m-3 were treated as a suspected repeated
   instrument ceiling/status value and excluded from the primary analysis.
3. When multiple stations occupied the same MERRA-2 cell on the same date, their
   observations were averaged before pooled analyses. This prevents duplicating the
   same model value and reduces pseudoreplication.
4. Calendar-month anomalies were calculated within each grid cell (or within each
   station for station-level estimates), removing stable spatial differences and the
   mean seasonal cycle.
5. Pearson and Spearman correlations use 95% confidence intervals from a calendar-
   month block bootstrap ({audit['bootstrap_replicates']:,} replicates; seed
   {audit['seed']}). The block preserves all spatial and daily observations within
   a resampled month.
6. Sensitivity analyses retain all PM2.5 values and separately restrict PM2.5 to
   values below 500 µg m-3.
7. Station meteorology is screened independently for every station-year. Screened
   temperature comparisons are supplementary diagnostics only.

Key audit counts
----------------
Matched station-days: {audit['station_days']:,}
Unique grid-days: {audit['grid_days']:,}
Grid-days containing multiple stations: {audit['multi_station_grid_days']:,}
Suspected PM2.5=985 station-days removed: {audit['pm25_985_removed']:,}
Model-value conflicts within grid-day groups: {audit['model_conflicts']:,}

Interpretation guardrails
-------------------------
- PM2.5 is a total-particle metric, not an organic-carbon measurement.
- Correlation does not establish unbiased magnitude, causality, or source identity.
- Satellite tropospheric NO2 columns and surface NO2 have different vertical and
  temporal representativeness; their comparison is supplementary.
- Station coverage is uneven, station-to-grid separation is nonzero, and the CAMS
  observation record is shorter than the complete modeling period.
- Use the anomaly/grid-day result as the primary CAMS robustness result.
"""
    destination.write_text(text, encoding="utf-8")


def fmt_ci(row: pd.Series, method: str) -> str:
    return (f"{row[f'{method}_r']:.3f} (95% CI "
            f"{row[f'{method}_ci_low']:.3f} to {row[f'{method}_ci_high']:.3f})")


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    print("Reading matched CAMS daily data...")
    data, source = read_input(args.input)
    required = {"station", "date", "grid_id", "PM25_ug_m3", "OCSMASS_ug_m3"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["station", "date", "grid_id"]).copy()
    for col in data.columns:
        if col not in {"station", "date", "grid_id", "season", "coordinate_status"}:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0:
                data[col] = converted

    data["PM25_original_ug_m3"] = data["PM25_ug_m3"]
    data["PM25_suspected_985"] = np.isclose(
        data["PM25_original_ug_m3"], SUSPECT_PM25_VALUE, rtol=0, atol=1e-8
    )
    data["PM25_primary_ug_m3"] = data["PM25_original_ug_m3"].mask(data["PM25_suspected_985"])

    print("Aggregating stations to unique grid-days...")
    grid_primary, conflicts = aggregate_grid_day(data, "PM25_primary_ug_m3")
    grid_primary = add_monthly_anomalies(
        grid_primary, "grid_id", ["OCSMASS_ug_m3", "PM25_ug_m3", "NO2", "NO2_ppb_surface"]
    )
    grid_primary.to_csv(output / "01_grid_day_external_consistency.csv", index=False)
    conflicts.to_csv(output / "08_model_value_conflicts.csv", index=False)

    print("Calculating pooled block-bootstrap statistics...")
    primary_rows = [
        metric_row(grid_primary, "OCSMASS vs PM2.5", "OCSMASS_ug_m3", "PM25_ug_m3",
                   "raw grid-day", args.bootstrap, args.seed),
        metric_row(grid_primary, "OCSMASS vs PM2.5",
                   "OCSMASS_ug_m3_anomaly", "PM25_ug_m3_anomaly",
                   "within-grid calendar-month anomaly", args.bootstrap, args.seed + 10),
    ]
    if {"NO2", "NO2_ppb_surface"}.issubset(grid_primary.columns):
        primary_rows.extend([
            metric_row(grid_primary, "TROPOMI column NO2 vs surface NO2", "NO2", "NO2_ppb_surface",
                       "raw grid-day", args.bootstrap, args.seed + 20),
            metric_row(grid_primary, "TROPOMI column NO2 vs surface NO2",
                       "NO2_anomaly", "NO2_ppb_surface_anomaly",
                       "within-grid calendar-month anomaly", args.bootstrap, args.seed + 30),
        ])
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(output / "02_primary_bootstrap_results.csv", index=False)

    print("Calculating station-level estimates...")
    station_work = add_monthly_anomalies(
        data.rename(columns={"PM25_primary_ug_m3": "PM25_primary"}),
        "station", ["OCSMASS_ug_m3", "PM25_primary"]
    )
    station_rows = []
    for number, (station, group) in enumerate(station_work.groupby("station")):
        row = metric_row(group, "OCSMASS vs PM2.5", "OCSMASS_ug_m3_anomaly",
                         "PM25_primary_anomaly", "station calendar-month anomaly",
                         args.bootstrap, args.seed + 1000 + number * 97)
        row["station"] = station
        station_rows.append(row)
    station_results = pd.DataFrame(station_rows)
    ordered = ["station"] + [c for c in station_results if c != "station"]
    station_results = station_results[ordered].sort_values("spearman_r", ascending=False)
    station_results.to_csv(output / "03_station_level_PM25_results.csv", index=False)

    print("Running PM2.5 sensitivity analyses...")
    sensitivity_rows = []
    scenarios = {
        "primary: exact 985 excluded": data["PM25_primary_ug_m3"],
        "all reported values retained": data["PM25_original_ug_m3"],
        "conservative: PM2.5 < 500": data["PM25_original_ug_m3"].where(data["PM25_original_ug_m3"] < 500),
    }
    for index, (scenario, series) in enumerate(scenarios.items()):
        scenario_data = data.copy()
        scenario_data["PM25_scenario"] = series
        grid, _ = aggregate_grid_day(scenario_data, "PM25_scenario")
        grid = add_monthly_anomalies(grid, "grid_id", ["OCSMASS_ug_m3", "PM25_ug_m3"])
        for scale, x, y in [
            ("raw grid-day", "OCSMASS_ug_m3", "PM25_ug_m3"),
            ("within-grid calendar-month anomaly", "OCSMASS_ug_m3_anomaly", "PM25_ug_m3_anomaly"),
        ]:
            row = metric_row(grid, "OCSMASS vs PM2.5", x, y, scale,
                             args.bootstrap, args.seed + 5000 + index * 100 + (0 if scale.startswith("raw") else 1))
            row["scenario"] = scenario
            sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output / "04_PM25_sensitivity_results.csv", index=False)

    print("Screening station-year meteorology...")
    met_qc = station_year_meteorology_qc(data)
    met_qc.to_csv(output / "05_station_year_meteorology_QC.csv", index=False)
    temp_qualified = met_qc.loc[
        met_qc["variable"].eq("temperature_C_station") & met_qc["qualified"],
        ["station", "year"]
    ]
    temp = data.copy()
    temp["year"] = temp["date"].dt.year
    temp = temp.merge(temp_qualified.assign(temp_qualified=True), on=["station", "year"], how="left")
    temp = temp[temp["temp_qualified"].fillna(False)].copy()
    temp_grid = (
        temp.groupby(["grid_id", "date"], as_index=False)
        .agg(temperature_C_station=("temperature_C_station", "mean"),
             temperature_C_ERA5=("temperature_C", "first"),
             station_count=("station", "nunique"))
    )
    temp_row = metric_row(temp_grid, "ERA5 vs screened station temperature",
                          "temperature_C_ERA5", "temperature_C_station",
                          "screened grid-day", args.bootstrap, args.seed + 8000)
    temp_pair = temp_grid[["temperature_C_ERA5", "temperature_C_station"]].dropna()
    difference = temp_pair["temperature_C_ERA5"] - temp_pair["temperature_C_station"]
    temp_row.update({
        "bias_ERA5_minus_station_C": difference.mean(),
        "MAE_C": difference.abs().mean(),
        "RMSE_C": np.sqrt(np.mean(difference ** 2)) if len(difference) else np.nan,
        "qualified_station_years": len(temp_qualified),
    })
    met_results = pd.DataFrame([temp_row])
    met_results.to_csv(output / "06_screened_meteorology_results.csv", index=False)

    audit = {
        "input": source,
        "station_days": int(len(data)),
        "grid_days": int(len(grid_primary)),
        "multi_station_grid_days": int((grid_primary["station_count"] > 1).sum()),
        "pm25_985_removed": int(data["PM25_suspected_985"].sum()),
        "stations_with_pm25_985": sorted(data.loc[data["PM25_suspected_985"], "station"].dropna().unique().tolist()),
        "model_conflicts": int(len(conflicts)),
        "bootstrap_replicates": int(args.bootstrap),
        "seed": int(args.seed),
        "qualified_temperature_station_years": int(len(temp_qualified)),
    }
    pd.DataFrame([{
        "metric": key,
        "value": json.dumps(value) if isinstance(value, list) else value
    } for key, value in audit.items()]).to_csv(output / "07_analysis_audit.csv", index=False)

    print("Creating publication figures...")
    oa_anomaly = primary.loc[
        primary["analysis"].eq("OCSMASS vs PM2.5") &
        primary["scale"].eq("within-grid calendar-month anomaly")
    ].iloc[0].to_dict()
    plot_hexbin(
        grid_primary, "OCSMASS_ug_m3_anomaly", "PM25_ug_m3_anomaly",
        "External consistency after removing grid-specific seasonality",
        "MERRA-2 OCSMASS anomaly (µg m⁻³)", "Surface PM₂.₅ anomaly (µg m⁻³)",
        oa_anomaly, output / "Figure_1_GridDay_OCSMASS_PM25_Anomalies.png"
    )
    plot_forest(station_results, output / "Figure_2_Station_Spearman_Forest.png")
    plot_hexbin(
        temp_grid, "temperature_C_ERA5", "temperature_C_station",
        "Supplementary check: ERA5 vs screened station temperature",
        "ERA5 daily temperature (°C)", "Station daily temperature (°C)",
        temp_row, output / "Figure_3_Screened_ERA5_Temperature.png", one_to_one=True
    )
    no2_rows = primary.loc[
        primary["analysis"].eq("TROPOMI column NO2 vs surface NO2") &
        primary["scale"].eq("within-grid calendar-month anomaly")
    ]
    if len(no2_rows):
        plot_hexbin(
            grid_primary, "NO2_anomaly", "NO2_ppb_surface_anomaly",
            "Supplementary check: column vs surface NO₂ anomalies",
            "TROPOMI tropospheric-column NO₂ anomaly", "Surface NO₂ anomaly (ppb)",
            no2_rows.iloc[0].to_dict(), output / "Figure_S1_NO2_Anomalies.png"
        )

    write_readme(output / "README_methods_and_interpretation.txt", source, audit)

    oa_raw = primary.loc[
        primary["analysis"].eq("OCSMASS vs PM2.5") & primary["scale"].eq("raw grid-day")
    ].iloc[0]
    oa_anom = primary.loc[
        primary["analysis"].eq("OCSMASS vs PM2.5") &
        primary["scale"].eq("within-grid calendar-month anomaly")
    ].iloc[0]
    manuscript = f"""MANUSCRIPT-READY CAMS-BD TEXT

Methods — independent external-consistency assessment
Daily surface observations from Bangladesh continuous air-monitoring stations were
matched to the nearest MERRA-2 analytical grid cell. CAMS-BD does not report organic
carbon or organic aerosol; therefore, PM2.5 was used only as an independent, nonspecific
particle indicator and not as a direct validation target for OCSMASS. Repeated PM2.5
values of exactly 985 µg m-3 were treated as a suspected instrument ceiling or status
value and excluded from the primary analysis; analyses retaining all reported values
and restricting PM2.5 to <500 µg m-3 were used as sensitivity checks. When multiple
stations mapped to the same grid cell on the same day, observations were averaged so
that the corresponding model value was represented once. Calendar-month anomalies
were calculated separately within each grid cell to remove the mean seasonal cycle and
stable spatial contrasts. Pearson and Spearman correlations were summarized with 95%
confidence intervals from a {args.bootstrap:,}-replicate calendar-month block bootstrap.

Results — external consistency with surface PM2.5
Across {int(oa_raw['n']):,} unique matched grid-days, raw MERRA-2 OCSMASS and surface
PM2.5 were positively associated (Pearson r={fmt_ci(oa_raw, 'pearson')}; Spearman
rho={fmt_ci(oa_raw, 'spearman')}). After removing grid-specific calendar-month means,
the association was attenuated but remained positive across {int(oa_anom['n']):,}
grid-days (Pearson r={fmt_ci(oa_anom, 'pearson')}; Spearman rho={fmt_ci(oa_anom, 'spearman')}).
This pattern provides independent evidence that OCSMASS captures meaningful temporal
variation in particulate pollution, while the attenuation after seasonal adjustment
indicates that part of the raw agreement reflects shared spatial and seasonal structure.

Limitations and interpretation
The CAMS-BD comparison is an external-consistency test rather than a chemical-species
validation because PM2.5 includes inorganic aerosol, dust, sea salt, elemental carbon,
and water in addition to organic material. Monitor coverage was spatially and temporally
uneven, monitoring sites were separated from MERRA-2 cell centers, and point observations
and grid-cell averages have different representativeness. TROPOMI tropospheric-column
NO2 and surface NO2 are also not directly equivalent; that comparison should be reported
only as a supplementary diagnostic. Accordingly, the CAMS-BD results support the
plausibility and external consistency of the modeled OCSMASS field but do not establish
unbiased OCSMASS magnitude or source-specific causal attribution.

Suggested Figure 1 caption
Grid-day association between anomalies in MERRA-2 surface organic-carbon mass
concentration (OCSMASS) and independent surface PM2.5 observations over Bangladesh,
2019–2024. Anomalies were computed by subtracting each MERRA-2 grid cell's calendar-
month mean. Multiple monitors in the same grid cell were averaged by day, and suspected
PM2.5 ceiling/status values of exactly 985 µg m-3 were excluded. Confidence intervals
were estimated using a calendar-month block bootstrap. PM2.5 is used as a nonspecific
external consistency indicator, not a direct organic-carbon measurement.
"""
    (output / "Manuscript_ready_text.txt").write_text(manuscript, encoding="utf-8")

    script_path = Path(__file__).resolve()
    if script_path.parent != output:
        shutil.copy2(script_path, output / script_path.name)
    archive = shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)

    print("\nAnalysis complete.")
    print("Results folder:", output)
    print("ZIP archive:", archive)
    print("\nPrimary anomaly result:")
    print(pd.DataFrame([oa_anom]).to_string(index=False))


if __name__ == "__main__":
    main()
