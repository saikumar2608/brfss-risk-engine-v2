# utils/helper.py

import streamlit as st
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
from contextlib import contextmanager

@contextmanager
def card():
    with st.container():
        st.markdown(
            """
            <style>
            .card {background: #ffffff; border: 1px solid #eee; border-radius: 12px; padding: 16px; margin: 6px 0;}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            yield
        finally:
            st.markdown("</div>", unsafe_allow_html=True)

def muted(text: str): st.caption(text)

def coded_selectbox(label: str, mapping: dict, key: str) -> int:
    reverse_map = {v: k for k, v in mapping.items()}
    if f"{key}_label" not in st.session_state:
        st.session_state[f"{key}_label"] = list(mapping.values())[0]
    selected_label = st.selectbox(
        label,
        options=list(mapping.values()),
        index=list(mapping.values()).index(st.session_state[f"{key}_label"]),
        key=f"{key}_ui",
    )
    st.session_state[f"{key}_label"] = selected_label
    return int(reverse_map[selected_label])

def coded_radio(label: str, mapping: dict, key: str, horizontal: bool = False) -> int:
    if f"{key}_label" not in st.session_state:
        st.session_state[f"{key}_label"] = list(mapping.keys())[0]
    choice = st.radio(label, options=list(mapping.keys()), key=f"{key}_radio", horizontal=horizontal)
    st.session_state[f"{key}_label"] = choice
    return int(mapping[choice])

def clinician_paragraph() -> str:
    return (
        "This tool estimates individual probability from survey-style inputs using "
        "gradient-boosted trees trained on BRFSS-like data. Predictions are calibrated "
        "to population prevalence where possible. The SHAP bars show each input’s local "
        "contribution to the score for that person. Use it to focus conversations on prevention, "
        "not to diagnose disease or replace clinical judgment."
    )

# ---- BMI ----
def calc_bmi(weight: float, w_unit: str, height_cm: float) -> float:
    if w_unit == "lbs":
        weight *= 0.453592
    return weight / ((height_cm / 100) ** 2)

def interpret_bmi_standard(bmi: float) -> str:
    if bmi < 18.5: return "Underweight"
    elif bmi < 25: return "Normal"
    elif bmi < 30: return "Overweight"
    else: return "Obese"

def interpret_bmi_asian(bmi: float) -> str:
    if bmi < 18.5: return "Underweight"
    elif bmi < 23: return "Normal (Asian)"
    elif bmi < 25: return "Overweight (Asian)"
    else: return "Obese (Asian)"

def plot_bmi_gauge(bmi: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 1.2))
    color = "green" if bmi < 25 else "orange" if bmi < 30 else "red"
    ax.barh([0], [bmi], color=color)
    ax.axvline(18.5, color="blue", linestyle="--", label="Underweight")
    ax.axvline(25,   color="green", linestyle="--", label="Normal")
    ax.axvline(30,   color="orange", linestyle="--", label="Overweight")
    ax.axvline(40,   color="red", linestyle="--", label="Obese")
    ax.set_xlim(10, max(40, bmi + 5))
    ax.set_yticks([])
    ax.set_title("BMI Gauge")
    ax.legend(loc="upper right", fontsize="small")
    st.pyplot(fig)

def obesity_status_from_bmi(bmi: float, race_group_code: int) -> dict:
    if bmi is None or pd.isna(bmi):
        return {"is_obese": None, "cutoff": None, "reason": "BMI missing"}
    ASIAN_CODE = 3
    cutoff = 25.0 if int(race_group_code) == ASIAN_CODE else 30.0
    return {
        "is_obese": bool(float(bmi) >= cutoff),
        "cutoff": cutoff,
        "reason": "Asian cutoff 25.0 used" if int(race_group_code) == ASIAN_CODE else "Standard cutoff 30.0 used",
    }

# ---- Age mapping ----
def map_age_to_group_13(age: int) -> int:
    if age <= 24: return 1
    if age <= 29: return 2
    if age <= 34: return 3
    if age <= 39: return 4
    if age <= 44: return 5
    if age <= 49: return 6
    if age <= 54: return 7
    if age <= 59: return 8
    if age <= 64: return 9
    if age <= 69: return 10
    if age <= 74: return 11
    if age <= 79: return 12
    return 13

def age6_from_age13(code13: int) -> int:
    if code13 == 1: return 1
    if code13 <= 3: return 2
    if code13 <= 5: return 3
    if code13 <= 7: return 4
    if code13 <= 9: return 5
    return 6

# ---- Alignment / clamping ----
def get_aligned_input(model, input_data: pd.DataFrame, feature_list=None, model_name: str = "", verbose: bool = True) -> pd.DataFrame:
    if feature_list is not None:
        expected = list(feature_list)
    elif hasattr(model, "feature_names_in_") and getattr(model, "feature_names_in_") is not None:
        expected = list(model.feature_names_in_)
    else:
        try:
            booster = model.get_booster()
            expected = list(booster.feature_names) if booster and booster.feature_names else list(input_data.columns)
        except Exception:
            expected = list(input_data.columns)

    X = input_data.reindex(columns=expected)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    missing_mask = X.isna().any(axis=0)
    X = X.fillna(0)

    if verbose:
        extra = [c for c in input_data.columns if c not in expected]
        missing = [c for c in expected if c not in input_data.columns or bool(missing_mask.get(c, False))]
        if extra or missing:
            st.warning(f"[{model_name}] Alignment → dropped extra={extra}, filled/cleaned missing={missing}")
    return X

def clamp_training_domains(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "age_group_code" in X: X["age_group_code"] = pd.to_numeric(X["age_group_code"], errors="coerce").clip(1, 13)
    if "income_level" in X:
        X["income_level"] = pd.to_numeric(X["income_level"], errors="coerce")
        X["income_level"] = X["income_level"].where(X["income_level"].isin([1,2,3,4,5,6,7,8,77,99]), 99)
    for c in ["race_group","education_level","marital_status"]:
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].where(X[c].isin([1,2,3,4,5,6,9]), 9)
    for c in ["ever_smoked","phys_activity","any_alcohol"]:
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].where(X[c].isin([1,2,7,9]), 9)
    for c in ["diabetes","heart_disease","depression","cost_barrier","recent_checkup","diabetes_duration_known"]:
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).clip(0, 1)
    if "hypertension" in X:
        X["hypertension"] = pd.to_numeric(X["hypertension"], errors="coerce")
        X["hypertension"] = X["hypertension"].where(X["hypertension"].isin([1,2]), 2)
    if "bmi" in X:
        X["bmi"] = pd.to_numeric(X["bmi"], errors="coerce").clip(10, 70)
    if "estimated_diabetes_duration" in X:
        X["estimated_diabetes_duration"] = pd.to_numeric(X["estimated_diabetes_duration"], errors="coerce").fillna(0)
    return X.fillna(0)

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sex" in df.columns:
        if df["sex"].dtype == object:
            df["sex"] = df["sex"].map({"Male": 1, "Female": 2})
        df.loc[~df["sex"].isin([1, 2]), "sex"] = np.nan

    bin_12 = ["ever_smoked","phys_activity","any_alcohol","hypertension"]
    for c in bin_12:
        if c in df.columns and df[c].dtype == object:
            if c in ["ever_smoked","phys_activity","any_alcohol"]:
                df[c] = df[c].map({"Yes": 1, "No": 2, "Don't know": 7, "Refused": 9})
            else:
                df[c] = df[c].map({"Yes": 1, "No": 2})

    bin_01 = ["diabetes","heart_disease","depression","cost_barrier","recent_checkup","diabetes_duration_known"]
    for c in bin_01:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].map({"Yes": 1, "No": 0})

    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").clip(lower=10.0, upper=70.0)

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    for c in df.columns:
        if c == "bmi":
            df[c] = df[c].astype("float32")
        else:
            if np.all(np.mod(df[c], 1) == 0):
                df[c] = df[c].astype("int32")
            else:
                df[c] = df[c].astype("float32")

    return df

# ---- SHAP support ----
def _shap_values_array(ex):
    vals = ex.values if hasattr(ex, "values") else np.array(ex)
    vals = np.array(vals)
    if vals.ndim == 1:
        return vals
    return vals[0]

@st.cache_resource(show_spinner=False)
def get_shap_explainer(model_key: str, _model):
    try:
        return shap.Explainer(_model, model_output="probability")
    except Exception:
        pass
    try:
        return shap.TreeExplainer(_model, feature_perturbation="interventional", model_output="probability")
    except Exception:
        pass
    return shap.TreeExplainer(_model, feature_perturbation="tree_path_dependent", model_output="raw")

def plot_shap_bar(model, input_data: pd.DataFrame, title: str, top: int = 10) -> None:
    key = f"{model.__class__.__name__}_{title.lower().replace(' ', '_')}"
    expl = get_shap_explainer(key, model)
    ex = expl(input_data)
    vals = _shap_values_array(ex)
    names = np.array(input_data.columns)

    df = pd.DataFrame({"feature": names, "impact": vals})
    df["abs_impact"] = df["impact"].abs()
    df = df.sort_values("abs_impact", ascending=False).head(top)

    plt.figure(figsize=(8, 5))
    order = df.index[::-1]
    plt.barh(range(len(order)), df.loc[order, "impact"])
    plt.yticks(range(len(order)), df.loc[order, "feature"])
    plt.axvline(0, linewidth=1)
    plt.title(f"Feature Impact - {title}")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()
