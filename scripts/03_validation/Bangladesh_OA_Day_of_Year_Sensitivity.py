"""Rebuild the Table S16 comparison from the archived HGB settings."""

from pathlib import Path
from runpy import run_path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    scripts = Path(__file__).resolve().parents[1]
    primary = run_path(str(scripts / "02_primary_model" / "Bangladesh_OA_Leakage_Safe_Modeling.py"))
    validation = run_path(str(scripts / "03_validation" / "Bangladesh_OA_Robustness_and_Ablation.py"))
    if primary["QUICK_TEST"]:
        raise ValueError("Unset OA_QUICK_TEST before running the day-of-year comparison.")

    data = pd.read_csv(primary["INPUT_FILE"], parse_dates=["date"], low_memory=False)
    data["date"] = data["date"].dt.normalize()
    data["grid_id"] = data["grid_id"].astype(str).str.strip()
    primary["validate_input"](data)
    data = primary["prepare_satellite_retrievals"](data)
    day_features = {"day_of_year_sin", "day_of_year_cos"}
    records = []

    for year in (2022, 2023, 2024, 2025):
        train = data[data["year"].between(2019, year - 1)]
        test = data[data["year"].eq(year)]
        columns = primary["FEATURES"] if year == 2025 else validation["FULL_FEATURES"]
        feature_sets = {
            "Full model": columns,
            "Without day of year": [name for name in columns if name not in day_features],
        }

        for label, features in feature_sets.items():
            model = primary["build_hist_gradient_boosting"]()
            model.fit(train[features], train[primary["TARGET_LOG"]])
            metrics, _ = primary["calculate_metrics"](
                label, str(year), test[primary["TARGET_RAW"]], model.predict(test[features])
            )
            records.append({
                "Year": year,
                "Specification": label,
                "Log RMSE": metrics["RMSE_log"],
                "Raw RMSE": metrics["RMSE_ug_m3"],
                "Raw R2": metrics["R2_raw"],
            })

    results = pd.DataFrame(records)
    measures = ["Log RMSE", "Raw RMSE", "Raw R2"]
    expanding = results[results["Year"].lt(2025)].groupby(
        "Specification", sort=False, as_index=False
    )[measures].mean()
    expanding.insert(0, "Evaluation", "Expanding-year mean")
    final = results[results["Year"].eq(2025)].drop(columns="Year")
    final.insert(0, "Evaluation", "Independent 2025 test")
    summary = pd.concat([expanding, final], ignore_index=True)

    output = primary["PROJECT_FOLDER"] / "Day_of_Year_Sensitivity_Results"
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "day_of_year_by_year.csv", index=False)
    summary.to_csv(output / "day_of_year_summary.csv", index=False)

    chart = summary.pivot(index="Evaluation", columns="Specification", values="Log RMSE")
    ax = chart.plot.bar(rot=0, ylabel="Log RMSE", title="Sensitivity to day-of-year predictors")
    ax.set_xlabel("")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "day_of_year_log_rmse.png", dpi=300)
    plt.close(ax.figure)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
