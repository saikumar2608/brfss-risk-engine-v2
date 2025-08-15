Risk Engine V2

A Multi-Year Chronic Disease Risk Prediction Application using BRFSS Data

Risk Engine V2 is an upgraded machine learning–powered application designed to predict chronic disease risks (Diabetes, Hypertension, Heart Disease, and Obesity) based on individual lifestyle, demographic, and health factors from the CDC’s Behavioral Risk Factor Surveillance System (BRFSS). It delivers percentage-based risk estimates through a user-friendly Streamlit interface.

Project Overview

Models: XGBoost with class-weighted training for imbalanced targets

Targets: Diabetes, Hypertension, Heart Disease, Obesity

Data Sources: BRFSS 2019, 2020, 2021 (CDC)

Interface: Streamlit web app with dynamic inputs and contextual “About” section

Key Features:

Multi-year model training with cross-year validation

Feature importance via SHAP for transparency

Probability-based outputs for more informative feedback

Built-in definitions to ensure layperson understanding

Key Improvements in V2

Multi-year integration — Supports training on combined BRFSS years for better generalization.

Feature consistency — Unified feature naming (rename_dict) across datasets.

Enhanced interpretability — SHAP visualizations to explain predictions.

User experience upgrades — Clean, human-readable inputs and explanations in the UI.

Resilient deployment — Stable Streamlit Cloud release with pinned dependencies and asset checks.

Challenges & How We Solved Them
1. Age group bins problem

Issue: _AGEG5YR bins misaligned, producing illogical age–risk patterns.

Fix: Centralized mapping dictionary, rebuilt using pd.cut with precise boundaries, and validated with asserts on counts and min/max bins.

2. General health flipped

Issue: BRFSS GENHLTH scale (1=Excellent → 5=Poor) was reversed.

Fix: Explicit mapping so higher values indicate worse health; verified with correlation checks.

3. Smoking yes/no flipped

Issue: SMOKE100 (1=Yes, 2=No) was mistakenly inverted.

Fix: Boolean recode (1 → True, 2 → False) with sanity checks on prevalence rates.

4. Logic problems with BMI

Issue: Mismatch between dataset-calculated BMI and manual BMI from inputs (unit inconsistencies, off-by-one category edges).

Fix: Standardized formula and category definitions to match BRFSS exactly.

5. Sharp non-uniformness in SHAP outputs

Issue: Partial dependence sweeps had abrupt jumps.

Fix: Standardized categorical encodings, smoothed bins, validated on controlled synthetic sweeps.

6. Retraining BMI & HTN during UI fixes

Issue: Encoder/category order changes in UI broke old models.

Fix: Retrained BMI & HTN with unified preprocessing pipelines and locked category order.

7. Hypertension label gap across years

Issue: No consistent hypertension target in BRFSS 2020.

Fix: HTN trained on 2019 + 2021, with cross-year validation checks.

8. Cross-dataset comparability

Issue: CDC/WHO external features weren’t directly comparable to BRFSS definitions.

Fix: Stuck to BRFSS-native features for accuracy and comparability.

9. Model version & compatibility issues

Issue: XGBoost pickle load warnings, sklearn InconsistentVersionWarning.

Fix: Used save_model() for XGBoost, pinned dependency versions, rebuilt under consistent environment.

10. UI/UX cleanup

Issue: Raw category codes like yes(1) showing in UI.

Fix: Applied rename_dict and separate display labels from model codes.

11. Generalization across years

Issue: Performance dropped when predicting on a different BRFSS year.

Fix: Combined-year training for better distribution coverage, monitored calibration on holdout years.

12. Deployment friction

Issue: Local app worked; Streamlit Cloud failed due to missing files/dependency mismatches.

Fix: Added pre-deploy asset checks, pinned requirements.txt, and ensured all files bundled.

Tech Stack

Language: Python 3.10+

Libraries:

Data Processing: Pandas, NumPy

Modeling: XGBoost, scikit-learn

Interpretability: SHAP

UI: Streamlit

Deployment: Streamlit Cloud
