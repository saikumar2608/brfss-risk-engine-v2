# Model Card — Risk Engine V2

**Intended Use**: Individual-level chronic disease risk estimation for education and research.  
**Data Sources**: BRFSS 2019, 2020, 2021 Public Use Files.  
**Model**: XGBoost with class weighting and isotonic calibration.  
**Metrics**: AUROC, Recall at fixed alert rates, Calibration curve.  
**Limitations**: Self-reported data, model drift over time, not a diagnostic tool.
