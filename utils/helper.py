# utils/helper.py
# Shared helpers for BRFSS Risk Engine v2

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------
# UI helpers
# ---------------------------
@contextmanager
def card():
    """Simple card-like container."""
    with st.container(border=True):
        yield

def muted(text: str):
    st.caption(text)

def _is_label_to_code(mapping: Dict[Any, Any]) -> bool:
    """Detect if mapping is {label->code} vs {code->label}."""
    # Heuristic: label strings on keys implies label->code
    return all(isinstance(k, str) for k in mapping.keys())

def coded_selectbox(label: str, mapping: Dict[Any, Any], key: str, index: int | None = None):
    """
    Show a selectbox with human labels and return the underlying code.
    Accepts either {code->label} or {label->code}.
    """
    if _is_label_to_code(mapping):
        labels = list(mapping.keys())
        codes  = list(mapping.values())
    else:
        codes  = list(mapping.keys())
        labels = list(mapping.values())

    default_index = 0 if index is None else index
    choice = st.selectbox(label, labels, index=default_index, key=key)
    if _is_label_to_code(mapping):
        return mapping[choice]
    # find code for chosen label
    return codes[labels.index(choice)]

def coded_radio(label: str, mapping: Dict[Any, Any], key: str, horizontal: bool = False):
    """
    Show a radio with human labels and return the underlying code.
    Accepts either {code->label} or {label->code}.
    """
    if _is_label_to_code(mapping):
        labels = list(mapping.keys())
        codes  = list(mapping.values())
    else:
        codes  = list(mapping.keys())
        labels = list(mapping.values())

    choice = st.radio(label, labels, key=key, horizontal=horizontal)
    if _is_label_to_code(mapping):
        return mapping[choice]
    return codes[labels.index(choice)]

def clinician_paragraph() -> str:
    return (
        "This tool estimates **probabilities** using models trained on BRFSS-like survey data. "
        "It’s built for **screening and awareness**, not diagnosis or treatment. "
        "We show local SHAP bars so you can see which inputs push the score **up** or **down** "
        "for this person. Outputs should be interpreted alongside vitals, labs, and history."
    )

# ---------------------------
# BMI helpers
# ---------------------------
def calc_bmi(weight: float, weight_unit: str, height_cm: float) -> float:
    """Compute BMI from weight (kg/lbs) and height in cm."""
    kg = weight if weight_unit == "kg" else weight * 0.45359237
    m  = height_cm / 100.0
    return kg / (m * m) if m > 0 else float("nan")

def interpret_bmi_standard(bmi: float) -> str:
    if math.isnan(bmi):
        return "Unknown"
    if bmi < 18.5: return "Underweight"
    if bmi < 25.0: return "Normal"
    if bmi < 30.0: return "Overweight"
    return "Obese"

def interpret_bmi_asian(bmi: float) -> str:
    if math.isnan(bmi):
        return "Unknown"
    if bmi < 18.5: return "Underweight"
    if bmi < 23.0: return "Normal"
    if bmi < 27.5: return "Overweight"
    return "Obese"

def plot_bmi_gauge(bmi: float):
    """Tiny horizontal gauge for BMI."""
    plt.figure(figsize=(6, 0.6))
    plt.axvspan(0,   18.5, alpha=0.2)
    plt.axvspan(18.5, 25,  alpha=0.2)
    plt.axvspan(25,  30,  alpha=0.2)
    plt.axvspan(30,  45,  alpha=0.2)
    plt.axvline(bmi, linewidth=3)
    plt.xlim(10, 45)
    plt.yticks([])
    plt.xlabel("BMI")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

def obesity_status_from_bmi(bmi: float, race_group_code: int) -> Dict[str, Any]:
    """Definition flag for obesity with Asian-specific cutoff."""
    asian_codes = {3}  # our mapping uses 3 for Asian
    cutoff = 25.0 if race_group_code in asian_codes else 30.0
    is_obese = (bmi >= cutoff)
    return {
        "is_obese": is_obese,
        "cutoff": cutoff,
        "reason": f"Obesity definition uses BMI ≥ {cutoff:.1f} (Asian-specific threshold if applicable)."
    }

# ---------------------------
# Age mapping
# ---------------------------
def map_age_to_group_13(age: int) -> int:
    """
    Map age in years to BRFSS _AGEG5YR style 13-bin code:
    1=18-24, 2=25-29, ..., 13=80+ (we compress some bands reasonably).
    """
    bands = [
        (18, 24), (25, 29), (30, 34), (35, 39), (40, 44),
        (45, 49), (50, 54), (55, 59), (60, 64), (65, 69),
        (70, 74), (75, 79), (80, 120),
    ]
    for i, (lo, hi) in enumerate(bands, start=1):
        if lo <= age <= hi:
            return i
    return 13

def age6_from_age13(age13: int) -> int:
    """
    Collapse 13-bin code to 6 broad groups used in some dashboards:
    1:18-24, 2:25-34, 3:35-44, 4:45-54, 5:55-64, 6:65+
    """
    if age13 <= 1: return 1
    if age13 <= 3: return 2
    if age13 <= 5: return 3
    if age13 <= 7: return 4
    if age13 <= 9: return 5
    return 6

# ---------------------------
# Model utils + SHAP
# ---------------------------
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Light touch: ensure numeric types where expected; fill nothing."""
    return df.copy()

def _model_feature_names(model) -> list[str] | None:
    """Try to recover feature names from the model."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # XGBClassifier sometimes exposes booster with feature names
    try:
        booster = model.get_booster()
        if booster and booster.feature_names:
            return list(booster.feature_names)
    except Exception:
        pass
    return None

def get_aligned_input(model, X: pd.DataFrame, want: Iterable[str] | None,
                      model_name: str = "", verbose: bool = False) -> pd.DataFrame:
    """
    Align input columns to what the model expects.
    - If 'want' is None, we try model.feature_names_in_ (or booster names).
    - Extra columns are dropped; missing columns are created as zeros.
    """
    X = X.copy()
    if want is None:
        want = _model_feature_names(model)
    if want is None:
        # Fall back: assume incoming X is already correct
        return X

    want = list(want)
    X = X.reindex(columns=want, fill_value=0)
    return X

def clamp_training_domains(X: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for any domain clamping; currently a no-op."""
    return X

def _unwrap_for_shap(model):
    """
    If the model is CalibratedClassifierCV, unwrap to the base estimator so SHAP
    explains the *booster*, not the calibrator.
    """
    if isinstance(model, CalibratedClassifierCV):
        # cv='prefit'
        if hasattr(model, "base_estimator") and model.base_estimator is not None:
            return model.base_estimator
        # cv=kfold (take first calibrated fold)
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            return model.calibrated_classifiers_[0].base_estimator
    return model

_EXPLAINERS: Dict[str, shap.Explainer] = {}

def get_shap_explainer(key: str, model):
    """
    Cache a SHAP explainer on the *unwrapped* model.
    We use model_output='raw' so values are on the log-odds scale for class=1.
    """
    base = _unwrap_for_shap(model)
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.Explainer(base, model_output="raw")
    return _EXPLAINERS[key]

def _shap_values_array(shap_out) -> np.ndarray:
    """
    Normalize SHAP output to a (n_features,) array for the positive class.
    """
    vals = shap_out.values
    if isinstance(vals, list):   # multiclass: choose positive class
        vals = vals[-1]
    # vals shape is (1, n_features) for single-row explanations
    return np.array(vals[0])

# ---------------------------
# SHAP plotting (per person)
# ---------------------------
def plot_shap_bar_tidy(model, X_df: pd.DataFrame, title: str, top: int = 10, ensure: list[str] | None = None):
    """
    Horizontal bar chart of per-person SHAP values (class=1, log-odds).
    Right=pushes risk up; Left=pushes risk down.
    """
    key = f"{model.__class__.__name__}_{title.lower().replace(' ', '_')}"
    explainer = get_shap_explainer(key, model)
    ex = explainer(X_df)
    vals = _shap_values_array(ex)
    names = np.array(X_df.columns)

    df = (
        pd.DataFrame({"feature": names, "impact": vals})
        .assign(abs_impact=lambda d: d["impact"].abs())
        .sort_values("abs_impact", ascending=False)
    )

    ensure = ensure or []
    top_df = df.head(top).copy()
    for feat in ensure:
        if feat in df["feature"].values and feat not in top_df["feature"].values:
            top_df = pd.concat([top_df, df[df["feature"] == feat]]).drop_duplicates("feature", keep="first")

    # Plot tidy horizontal bars
    plt.figure(figsize=(8, 5))
    order = list(top_df.index)[::-1]
    plt.barh(range(len(order)), top_df.loc[order, "impact"])
    plt.yticks(range(len(order)), top_df.loc[order, "feature"])
    plt.axvline(0, linewidth=1)
    plt.title(f"Feature Impact - {title}")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()
