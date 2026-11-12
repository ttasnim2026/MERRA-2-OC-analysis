import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_FOLDER = Path(os.environ.get("OA_PROJECT_FOLDER", str(Path.cwd())))
MERGED_FILE = Path(os.environ.get(
    "OA_CO_FIRE_MERGED_FILE",
    PROJECT_FOLDER
    / "CO_Fire_Integrated_Model_Results_2019_2024"
    / "Tables"
    / "01_OA_HCHO_NO2_ERA5_VIIRS_CO_merged_2019_2024.csv",
))
PREDICTION_FILE = Path(os.environ.get(
    "OA_SPATIAL_PREDICTIONS_FILE",
    PROJECT_FOLDER
    / "Spatial_Validation_Results_2019_2024"
    / "Tables"
    / "06_spatial_oof_predictions.csv.gz",
))
OUTPUT = Path(os.environ.get(
    "OA_OUTLIER_OUTPUT_FOLDER",
    PROJECT_FOLDER / "Outlier_Investigation_2022_MERRA2_OC",
))


def metrics(frame: pd.DataFrame) -> dict:
    observed = frame["OCSMASS_ug_m3"].to_numpy()
    predicted = frame["predicted_OCSMASS_ug_m3"].to_numpy()
    return {
        "n": len(frame),
        "MAE_ug_m3": mean_absolute_error(observed, predicted),
        "RMSE_ug_m3": mean_squared_error(observed, predicted) ** 0.5,
        "R2": r2_score(observed, predicted),
        "mean_error_pred_minus_obs_ug_m3": float(np.mean(predicted - observed)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    merged = pd.read_csv(MERGED_FILE, parse_dates=["date"])
    predictions = pd.read_csv(PREDICTION_FILE, parse_dates=["date"])
    predictions = predictions.loc[predictions["specification"].eq("Full model")].copy()

    join_columns = [
        "date",
        "grid_id",
        "season",
        "fire_count_25km",
        "fire_frp_MW_25km",
        "fire_count_50km",
        "fire_frp_MW_50km",
        "fire_count_50km_3day",
        "fire_frp_MW_50km_3day",
        "fire_any_50km_3day",
        "wind_speed_m_s",
        "boundary_layer_height_m",
        "total_precipitation_mm",
    ]
    predictions = predictions.merge(
        merged[join_columns], on=["date", "grid_id"], how="left", validate="one_to_one"
    )

    target = "OCSMASS_ug_m3"
    p999 = float(merged[target].quantile(0.999))
    maximum = merged.loc[merged[target].idxmax()].copy()
    max_mask = predictions["date"].eq(maximum["date"]) & predictions["grid_id"].eq(maximum["grid_id"])
    above_200 = predictions[target].gt(200.0)
    fire_extreme = predictions[target].gt(p999) & predictions["fire_any_50km_3day"].eq(1)

    strategies = {
        "All records": pd.Series(False, index=predictions.index),
        "Exclude maximum record only": max_mask,
        "Exclude records above 200 ug m-3": above_200,
        "Exclude fire-associated top 0.1%": fire_extreme,
    }
    subsets = {
        "All observations": pd.Series(True, index=predictions.index),
        "Pre-monsoon": predictions["season"].eq("pre_monsoon"),
        "Spatial block 5": predictions["spatial_block"].eq(5),
    }

    metric_rows = []
    for subset_name, subset_mask in subsets.items():
        for strategy_name, exclusion_mask in strategies.items():
            retained = predictions.loc[subset_mask & ~exclusion_mask]
            row = {"evaluation_subset": subset_name, "sensitivity_strategy": strategy_name}
            row.update(metrics(retained))
            row["n_excluded_within_subset"] = int((subset_mask & exclusion_mask).sum())
            metric_rows.append(row)
    sensitivity = pd.DataFrame(metric_rows)
    baseline = sensitivity.loc[sensitivity["sensitivity_strategy"].eq("All records")].set_index("evaluation_subset")
    sensitivity["RMSE_change_vs_all_ug_m3"] = sensitivity.apply(
        lambda row: row["RMSE_ug_m3"] - baseline.loc[row["evaluation_subset"], "RMSE_ug_m3"], axis=1
    )
    sensitivity["RMSE_percent_change_vs_all"] = sensitivity.apply(
        lambda row: 100.0
        * (row["RMSE_ug_m3"] / baseline.loc[row["evaluation_subset"], "RMSE_ug_m3"] - 1.0),
        axis=1,
    )
    sensitivity.to_csv(OUTPUT / "01_sensitivity_metrics.csv", index=False)

    flags = predictions.loc[fire_extreme].copy()
    flags["flag_definition"] = f"OCSMASS above dataset 99.9th percentile ({p999:.3f} ug m-3) and VIIRS fire exposure within 50 km over 3 days"
    flags.sort_values(target, ascending=False).to_csv(OUTPUT / "02_flagged_extreme_fire_records.csv", index=False)

    event_days = (
        merged.loc[merged[target].gt(200)]
        .groupby("date", as_index=False)
        .agg(
            records_above_200=(target, "size"),
            maximum_OCSMASS_ug_m3=(target, "max"),
            mean_OCSMASS_above_200_ug_m3=(target, "mean"),
            maximum_three_day_fire_count=("fire_count_50km_3day", "max"),
            maximum_three_day_FRP_MW=("fire_frp_MW_50km_3day", "max"),
        )
        .sort_values("maximum_OCSMASS_ug_m3", ascending=False)
    )
    event_days.to_csv(OUTPUT / "03_extreme_event_days.csv", index=False)

    annual_maxima = (
        merged.loc[merged.groupby("year")[target].idxmax(), [
            "year", "date", "grid_id", "latitude", "longitude", target,
            "fire_count_50km_3day", "fire_frp_MW_50km_3day"
        ]]
        .sort_values("year")
    )
    annual_maxima.to_csv(OUTPUT / "04_annual_maxima_and_fire_context.csv", index=False)

    window = merged.loc[
        merged["grid_id"].eq(maximum["grid_id"])
        & merged["date"].between(maximum["date"] - pd.Timedelta(days=7), maximum["date"] + pd.Timedelta(days=7)),
        [
            "date", "grid_id", target, "fire_count_50km", "fire_frp_MW_50km",
            "fire_count_50km_3day", "fire_frp_MW_50km_3day", "wind_speed_m_s",
            "boundary_layer_height_m", "total_precipitation_mm"
        ],
    ].copy()
    window.to_csv(OUTPUT / "05_maximum_event_time_window.csv", index=False)

    summary = pd.DataFrame(
        [
            ["Maximum OCSMASS", float(maximum[target]), "ug m-3"],
            ["Maximum date", maximum["date"].date().isoformat(), "UTC date"],
            ["Maximum grid", maximum["grid_id"], "MERRA-2 grid identifier"],
            ["Latitude", float(maximum["latitude"]), "degrees north"],
            ["Longitude", float(maximum["longitude"]), "degrees east"],
            ["Same-day 50 km fire count", int(maximum["fire_count_50km"]), "VIIRS detections"],
            ["Same-day 50 km FRP", float(maximum["fire_frp_MW_50km"]), "MW summed across detections"],
            ["Three-day 50 km fire count", int(maximum["fire_count_50km_3day"]), "VIIRS detections"],
            ["Three-day 50 km FRP", float(maximum["fire_frp_MW_50km_3day"]), "MW summed across detections"],
            ["Three-day fire-count percentile", 100.0 * merged["fire_count_50km_3day"].rank(method="max", pct=True).loc[maximum.name], "percentile"],
            ["Three-day FRP percentile", 100.0 * merged["fire_frp_MW_50km_3day"].rank(method="max", pct=True).loc[maximum.name], "percentile"],
            ["OCSMASS 99.9th percentile", p999, "ug m-3"],
            ["Records above 200", int(merged[target].gt(200).sum()), "records"],
            ["Fire-associated top 0.1% records", int(fire_extreme.sum()), "records"],
        ],
        columns=["diagnostic", "value", "unit_or_note"],
    )
    summary.to_csv(OUTPUT / "00_outlier_diagnostic_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(window["date"], window[target], color="#8c2d04", marker="o", linewidth=2.2, label="OCSMASS")
    ax2.bar(window["date"], window["fire_frp_MW_50km_3day"], color="#4c78a8", alpha=0.28, label="3-day VIIRS FRP")
    ax.axvline(maximum["date"], color="#333333", linestyle="--", linewidth=1)
    ax.set_title("Maximum event at 92.5 E, 23.0 N")
    ax.set_ylabel("OCSMASS (ug m-3)", color="#8c2d04")
    ax2.set_ylabel("Three-day VIIRS FRP (MW)", color="#4c78a8")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    plot_data = sensitivity.pivot(index="evaluation_subset", columns="sensitivity_strategy", values="RMSE_ug_m3")
    order = ["All observations", "Pre-monsoon", "Spatial block 5"]
    strategies_order = list(strategies)
    x = np.arange(len(order))
    width = 0.19
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for i, strategy in enumerate(strategies_order):
        ax.bar(x + (i - 1.5) * width, plot_data.loc[order, strategy], width, label=strategy, color=colors[i])
    ax.set_xticks(x, ["All", "Pre-monsoon", "Block 5"])
    ax.set_ylabel("Spatial OOF RMSE (ug m-3)")
    ax.set_title("Sensitivity of reported prediction error")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(OUTPUT / "Figure_outlier_event_and_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs written to {OUTPUT}")
    print(summary.to_string(index=False))
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
