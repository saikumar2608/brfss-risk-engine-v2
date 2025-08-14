# utils/helper.py
# ------------------------------------------------------------
# Small, human-readable helper layer used by Risk Engine v2.
# ------------------------------------------------------------

from __future__ import annotations

import math
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# UI helpers
# ============================================================

@contextmanager
def card():
    """Simple visual grouping—kept minimal on purpose."""
    with st.container(border=True):
        yield

def muted(text: str):
    st.caption(text)

def _reverse_lookup(mapping: Dict[int, str], label: str) -> int:
    for k, v in mapping.items():
        if v == label:
            return int(k)
    raise KeyError(f"Label {label!r} not found")

def coded_radio(
    label: str,
    mapping: Dict[int, str] | Dict[str, int],
    key: Optional[str] = None,
    horizontal: bool = False,
):
    """
    Radio that *shows friendly labels* but returns the code.
    Accepts either {code:int -> label:str} OR {label:str -> code:int}.
    """
    if mapping and isinstance(next(iter(mapping.keys())), int):
        # {code -> label}
        labels = list(mapping.values())
        default = labels[0]
        choice = st.radio(label, labels, key=key, horizontal=horizontal, index=labels.index(default))
        return _reverse_lookup(mapping, choice)
    else:
        # {label -> code}
        labels = list(mapping.keys())
        choice = st.radio(label, labels, key=key, horizontal=horizontal)
        return int(mapping[choice])

def coded_selectbox(label: str, mapping: Dict[int, str], key: Optional[str] = None):
    """Selectbox that shows labels and returns the code."""
    labels = list(mapping.values())
    choice = st.selectbox(label, labels, key=key)
    return _reverse_lookup(mapping, choice)


# ============================================================
# BMI utilities
# ============================================================

def calc_bmi(weight: float, weight_unit: str, height_cm: float) -> float:
    """BMI = kg / m^2. Supports kg/lbs in, converts to cm→m internally."""
    kg = weight if weight_unit == "kg" else weight * 0.45359237
    m = height_cm / 100.0
    if m <= 0:
        return float("nan")
    return float(round(kg / (m * m), 2))

def interpret_bmi_standard(bmi: float) -> str:
    if math.isnan(bmi):
        return "—"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obesity"

def interpret_bmi_asian(bmi: float) -> str:
    if math.isnan(bmi):
        return "—"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 23:
        return "Normal"
    if bmi < 27.5:
        return "At risk / Overweight"
    return "Obesity"

def plot_bmi_gauge(bmi: float):
    """Tiny horizontal gauge—fast and dependency-free."""
    fig, ax = plt.subplots(figsize=(6, 1.0))
    ax.set_xlim(10, 50)
    ax.set_ylim(0, 1)
    ax.barh(0.5, 8.5, left=10, alpha=0.25)   # <18.5
    ax.barh(0.5, 6.5, left=18.5, alpha=0.25) # 18.5–25
    ax.barh(0.5, 5,   left=25, alpha=0.25)   # 25–30
    ax.barh(0.5, 20,  left=30, alpha=0.25)   # 30–50
    ax.axvline(bmi, linewidth=2)
    ax.set_yticks([]); ax.set_xlabel("BMI")
    ax.text(bmi, 0.8, f"{bmi:.1f}", ha="center", va="bottom")
    st.pyplot(fig)
    plt.close(fig)

def obesity_status_from_bmi(bmi: float, race_group_code: int) -> dict:
    """
    'Asian' cutoff 25.0 (race code 3 in our map), others 30.0.
    Returns dict: {is_obese, cutoff, reason}
    """
    cutoff = 25.0 if int(race_group_code) == 3 else 30.0
    is_obese = float(bmi) >= cutoff
    reason = f"Race-aware cutoff used: {'Asian' if int(race_group_code)==3 else 'Standard'} ({cutoff:.1f})."
    return {"is_obese": is_obese, "cutoff": cutoff, "reason": reason}


# ============================================================
# Age mapping (BRFSS style)
# ============================================================

def map_age_to_group_13(age: int) -> int:
    """
    Maps age in years to BRFSS 13-level bin (1..13).
    (18–24)->1, (25–29)->2, ..., (80+)->13
    """
    a = max(18, min(int(age), 120))
    edges = [24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79]
    for i, hi in enumerate(edges, start=1):
        if a <= hi:
            return i
    return 13

def age6_from_age13(code13: int) -> int:
    """Rough mapping from 13-level to 6-level groups (1..6)."""
    c = int(code13)
    if c <= 2:  # 18–34
        return 1
    if c <= 4:  # 35–44
        return 2
    if c <= 6:  # 45–54
        return 3
    if c <= 8:  # 55–64
        return 4
    if c <= 10: # 65–74
        return 5
    return 6     # 75+


# ============================================================
# Model prep / alignment
# ============================================================

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light sanitation: cast numerics, coerce NaNs to 0 for model inputs.
    Keep column order as provided; alignment happens later.
    """
    out = df.copy()
    for c in out.columns:
        if c in ("bmi", "estimated_diabetes_duration"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(float)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out

def _expected_feature_list(model, fallback: Optional[List[str]]) -> Optional[List[str]]:
    if fallback:
        return list(fallback)
    # sklearn 1.3+ stores names here
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # otherwise try unwrapped
    names = _model_feature_names(model)
    return list(names) if names else None

def get_aligned_input(
    model,
    X: pd.DataFrame,
    features: Optional[Iterable[str]] = None,
    model_name: str = "",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Reindex to model's expected feature order. Unknowns dropped, missing filled with 0.
    """
    want = _expected_feature_list(model, list(features) if features else None)
    if want is None:
        # Fall back to whatever user provided (single-row case)
        return X.copy()
    aligned = X.reindex(columns=want).fillna(0)
    return aligned

def clamp_training_domains(X: pd.DataFrame) -> pd.DataFrame:
    """
    Very light sanity clamps so single out-of-range inputs don't explode SHAP.
    """
    out = X.copy()
    if "bmi" in out:
        out["bmi"] = out["bmi"].clip(10, 60)
    if "general_health" in out:
        out["general_health"] = out["general_health"].clip(1, 5)
    if "age_group_code" in out:
        out["age_group_code"] = out["age_group_code"].clip(1, 13)
    if "sex" in out:
        out["sex"] = out["sex"].clip(1, 2)
    if "recent_checkup" in out:
        out["recent_checkup"] = out["recent_checkup"].clip(0, 1)
    return out


# ============================================================
# SHAP utilities (with calibrated model unwrap)
# ============================================================

def _unwrap_estimator(model):
    """
    Return the real estimator inside wrappers like CalibratedClassifierCV
    across scikit-learn versions.
    """
    # Fitted CalibratedClassifierCV exposes a list of _CalibratedClassifier
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0]
        return getattr(inner, "estimator", getattr(inner, "base_estimator", model))

    # (pre-fit) attribute names vary by version
    if hasattr(model, "estimator"):
        return model.estimator
    if hasattr(model, "base_estimator"):
        return model.base_estimator

    return model

def _model_feature_names(model) -> Optional[List[str]]:
    base = _unwrap_estimator(model)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(base, "feature_names_in_"):
        return list(base.feature_names_in_)
    # try xgboost booster
    try:
        booster = getattr(base, "get_booster", lambda: None)()
        if booster is not None and getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    except Exception:
        pass
    return None

@lru_cache(maxsize=12)
def _cached_tree_explainer_key(key: str):
    # dummy cache key function; actual explainer cached by Streamlit below
    return key

@st.cache_resource(show_spinner=False)
def get_shap_explainer(key: str, model):
    """
    Cached SHAP explainer factory.
    - Unwraps calibrated models
    - Uses model_output='raw' so positive values push P(class=1) up
    """
    _ = _cached_tree_explainer_key(key)  # make key part of cache
    base = _unwrap_estimator(model)
    try:
        return shap.TreeExplainer(base, model_output="raw")
    except Exception:
        return shap.Explainer(base)

def _shap_values_array(explanation) -> np.ndarray:
    """
    Returns 1D SHAP values for POSITIVE class (index 1 if present).
    """
    vals = getattr(explanation, "values", None)
    if vals is None:
        return np.zeros(0)
    vals = np.array(vals)
    # (n, f) or (n, 2, f)
    if vals.ndim == 3 and vals.shape[1] == 2:
        vals = vals[:, 1, :]
    return vals[0]

def plot_shap_bar_tidy(model, X_df: pd.DataFrame, title: str, top: int = 10, ensure: Optional[List[str]] = None):
    """
    Per-person tidy SHAP bar (class=1). Ensures selected features are shown even if
    not in the top-k by absolute impact.
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

    order = list(top_df.index)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(order)), top_df.loc[order, "impact"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(top_df.loc[order, "feature"])
    ax.axvline(0, linewidth=1)
    ax.set_title(f"Feature Impact - {title}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# Clinician paragraph
# ============================================================

def clinician_paragraph() -> str:
    return (
        "This tool estimates **probability of condition** using gradient-boosted trees trained on "
        "BRFSS-like survey data. Predictions are **screening-oriented** and calibrated to population "
        "prevalence—not a diagnosis. Each score is paired with a per-person SHAP bar that shows which "
        "inputs **pushed the probability up or down** (to the right increases risk, to the left reduces it). "
        "Use alongside vitals, labs, and clinical judgment."
    )

