# Bangladesh MERRA-2 organic carbon

Python scripts for *MERRA-2 Surface Organic Carbon over Bangladesh: Seasonality, Predictability, and Priorities for Monitoring*.

```bash
python -m pip install -r requirements.txt
```

Put `Bangladesh_OA_HCHO_NO2_ERA5_Master_2019_2025.csv` in the project folder. Run the scripts from that folder, or set `OA_PROJECT_FOLDER` to its path.

The 2025 test uses the locked histogram gradient boosting model reported in the paper.

1. Run `01_prepare` and `02_primary_model`, then the robustness, spatial block and research enhancement scripts in `03_validation`. The day-of-year script reruns the comparison in Table S16.
2. For CAMS-BD, put `CAMS-BD Hourly Air Quality and Meteorological Meas.zip` in the project folder. Run `CAMS_BD_External_Consistency_Analysis.py`, then `CAMS_Final_External_Consistency_Validation.py`.
3. For CO and fire, put `Bangladesh_TROPOMI_CO_Daily_MERRAgrid_2019_2024_FINAL.csv` in the project folder and the FIRMS archives in `FIRMS_VIIRS_2019_2024/`. Run `Bangladesh_VIIRS_Fire_Sensitivity_Analysis.py`, then `Bangladesh_CO_Fire_Integrated_Model_Comparison.py`.
4. Run `06_publication` for the main tables and plots. The optional outlier check needs both the spatial and CO/fire outputs.

Data and download scripts are not included. The paper lists the data sources.
