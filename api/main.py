# api/main.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

from utils.helper import (
    # ui helpers
    card, muted, coded_selectbox, coded_radio, clinician_paragraph,
    # bmi helpers
    calc_bmi, interpret_bmi_standard, interpret_bmi_asian, plot_bmi_gauge, obesity_status_from_bmi,
    # age mapping
    map_age_to_group_13, age6_from_age13,
    # model utilities
    preprocess_input, get_aligned_input, clamp_training_domains,
    get_shap_explainer, _shap_values_array,
)

CURRENT_YEAR = datetime.now().year

# ============================
# paths / model loaders
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_model(filename: str):
    # loading from /models
    return joblib.load(os.path.join(MODEL_DIR, filename))

# loading models
diabetes_model     = load_model("xgb_model_diabetes_v2.pkl")
hypertension_model = load_model("xgb_model_htn_v2.pkl")
heart_model        = load_model("xgb_model_heart_v2.pkl")

# obesity propensity (no BMI)
obesity_model_no_bmi     = load_model("xgb_model_obesity_no_bmi.pkl")
obesity_model_no_bmi_cal = load_model("xgb_model_obesity_no_bmi_cal.pkl")
with open(os.path.join(MODEL_DIR, "xgb_model_obesity_no_bmi_features.json"), "r") as f:
    FEATURES_OB_NO_BMI = json.load(f)

# ============================
# display mappings
# ============================
age_map = {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65+"}
race_map = {1: "White", 2: "Black/African American", 3: "Asian", 4: "Native American", 5: "Other", 9: "Unknown"}
education_map = {1: "Less than High School", 2: "High School Graduate", 3: "Some College", 4: "College Graduate", 5: "Postgraduate", 6: "Other", 9: "Unknown"}
income_map = {
    1: "<$15k", 2: "$15k-$25k", 3: "$25k-$35k", 4: "$35k-$50k",
    5: "$50k-$75k", 6: "$75k-$100k", 7: "$100k-$150k", 8: "$150k+",
    77: "Don't know", 99: "Refused"
}
marital_map = {1: "Married", 2: "Divorced", 3: "Widowed", 4: "Separated", 5: "Never Married", 6: "Other", 9: "Unknown"}
general_health_map = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
recent_checkup_map = {1: "Within 1 year", 2: "Within 2 years", 3: "Within 5 years", 4: "5+ years ago"}

# training encodings (human labels only in UI)
YES_NO_12 = {"Yes": 1, "No": 2}
YN_01     = {"Yes": 1, "No": 0}
YN_DK_RF  = {"Yes": 1, "No": 2, "Don't know": 7, "Refused": 9}

# ============================
# features (friendly names)
# ============================
FEATURES = {
    "diabetes": [
        "age_group_code","sex","race_group","education_level","income_level","marital_status",
        "ever_smoked","phys_activity","any_alcohol","obesity","heart_disease","hypertension",
        "bmi","general_health","depression","cost_barrier","recent_checkup"
    ],
    "hypertension": None,  # fall back to model feature names
    "heart": [
        "age_group_code","sex","race_group","education_level","income_level","marital_status",
        "ever_smoked","phys_activity","any_alcohol","diabetes","obesity","hypertension",
        "bmi","general_health","depression","cost_barrier","recent_checkup",
        "estimated_diabetes_duration","diabetes_duration_known",
    ],
}

# ============================
# SHAP plotting (per-person)
# ============================
def plot_shap_bar_tidy(model, X_df, title, top=10, ensure=None):
    # computing shap
    key = f"{model.__class__.__name__}_{title.lower().replace(' ', '_')}"
    ex = get_shap_explainer(key, model)(X_df)
    vals = _shap_values_array(ex)  # positive-class values via helper
    names = np.array(X_df.columns)

    df = (
        pd.DataFrame({"feature": names, "impact": vals})
        .assign(abs_impact=lambda d: d["impact"].abs())
        .sort_values("abs_impact", ascending=False)
    )

    # forcing some features to always show (e.g., age_group_code)
    ensure = ensure or []
    top_df = df.head(top).copy()
    for feat in ensure:
        if feat in df["feature"].values and feat not in top_df["feature"].values:
            top_df = pd.concat([top_df, df[df["feature"] == feat]]).drop_duplicates("feature", keep="first")

    # plotting a tidy horizontal bar
    plt.figure(figsize=(8, 5))
    order = list(top_df.index)[::-1]
    plt.barh(range(len(order)), top_df.loc[order, "impact"])
    plt.yticks(range(len(order)), top_df.loc[order, "feature"])
    plt.axvline(0, linewidth=1)
    plt.title(f"Feature Impact - {title}")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

# ============================
# small helper
# ============================
def show_status_and_skip(label, already_has, note=""):
    # skipping risk model if user already has the condition
    if already_has:
        st.info(f"{label}: already present. Skipping risk prediction. {note}")
        st.metric(label, "Present")
        return True
    return False

# ============================
# Home
# ============================
def home_page():
    st.header("BRFSS Risk Engine v2")

    with card():
        st.markdown("""
**Screening-first risk estimates** for Diabetes, Hypertension, Heart Disease, and Obesity.
Built on BRFSS-style inputs; explains each prediction with local SHAP.
        """)
        muted("Awareness tool — not for diagnosis or emergency use")

    with card():
        st.subheader("Get started")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("Start with BMI"):
                st.session_state["nav_page"] = "BMI Calculator"
                st.rerun()
        with c2:
            if st.button("Go to Risk Prediction"):
                st.session_state["nav_page"] = "Risk Prediction"
                st.rerun()
        with c3:
            if st.button("Read About"):
                st.session_state["nav_page"] = "About"
                st.rerun()

        if "bmi_standard" in st.session_state:
            st.caption(f"BMI captured: {st.session_state['bmi_standard']:.2f} (ready for Risk Prediction)")

    with card():
        st.subheader("What you’ll see")
        st.markdown("""
- Inputs aligned with BRFSS codes (labels simplified for humans)
- Per-person SHAP explanations
- Race-aware BMI cutoff for obesity definition
- Obesity propensity model that intentionally excludes BMI
        """)

    with card():
        st.subheader("Clinician note")
        st.markdown(clinician_paragraph())
    muted("Use alongside vitals, labs, and history. Calibrated to survey prevalence, not a clinic panel.")

# ============================
# BMI page
# ============================
def bmi_page():
    st.header("BMI Calculator")
    muted("Showing Standard and Asian cutoffs")

    with card():
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input("Weight", min_value=30.0, max_value=200.0, step=0.1)
            unit_weight = st.selectbox("Weight Unit", ["kg", "lbs"])
        with col2:
            unit_height = st.selectbox("Height Unit", ["cm", "ft+in"])
            if unit_height == "cm":
                height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.1)
            else:
                feet = st.number_input("Height (ft)", min_value=3, max_value=8, step=1)
                inches = st.number_input("Height (in)", min_value=0, max_value=11, step=1)
                height = (feet * 30.48) + (inches * 2.54)

        if st.button("Calculate BMI"):
            bmi = calc_bmi(weight, unit_weight, height)
            st.session_state["bmi_standard"] = round(bmi, 2)
            st.session_state["bmi_standard_category"] = interpret_bmi_standard(bmi)
            st.session_state["bmi_asian_category"] = interpret_bmi_asian(bmi)

    if "bmi_standard" in st.session_state:
        with card():
            bmi = st.session_state["bmi_standard"]
            st.success(f"Your BMI: {bmi:.2f}")
            st.caption(f"Standard BMI: {st.session_state['bmi_standard_category']}")
            st.caption(f"Asian BMI: {st.session_state['bmi_asian_category']}")
            plot_bmi_gauge(bmi)
            muted("Standard BMI is used for obesity status; propensity model excludes BMI.")
    return st.session_state.get("bmi_standard", None)

# ============================
# Risk page
# ============================
def risk_page(auto_bmi=None):
    st.header("Risk Prediction")

    # demographics
    with card():
        st.subheader("Demographics")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
        age_code13 = map_age_to_group_13(int(age))
        age6 = age6_from_age13(age_code13)
        st.info(f"BRFSS Age Group: {age_map[age6]}")

        sex = coded_radio("Sex", {"Male": 1, "Female": 2}, key="sex_radio", horizontal=True)

        col1, col2 = st.columns(2)
        with col1:
            race_group = coded_selectbox("Race Group", race_map, "race_group")
            education_level = coded_selectbox("Education Level", education_map, "education_level")
            income_level = coded_selectbox("Income Level", income_map, "income_level")
        with col2:
            marital_status = coded_selectbox("Marital Status", marital_map, "marital_status")
            general_health = coded_selectbox("General Health", general_health_map, "general_health")
            rc_ui = coded_selectbox("Recent Checkup", recent_checkup_map, "recent_checkup")
            recent_checkup_01 = 1 if rc_ui == 1 else 0

    # lifestyle & health
    with card():
        st.subheader("Lifestyle & Health")
        ever_smoked   = coded_radio("Ever Smoked?", YN_DK_RF, key="ever_smoked")
        phys_activity = coded_radio("Physical Activity?", YN_DK_RF, key="phys_activity")
        any_alcohol   = coded_radio("Any Alcohol?", YN_DK_RF, key="any_alcohol")

        diabetes          = coded_radio("Diabetes? (ever told by a professional)", YN_01, key="diabetes", horizontal=True)
        hypertension_code = coded_radio("Hypertension? (ever told high BP; not pregnancy-related)", YES_NO_12, key="hypertension", horizontal=True)
        heart_disease     = coded_radio("Heart Disease? (heart attack/MI or coronary heart disease/angina)", YN_01, key="heart_disease", horizontal=True)
        depression        = coded_radio("Depression?", YN_01, key="depression", horizontal=True)
        cost_barrier      = coded_radio("Cost Barrier to Care in past 12 months?", YN_01, key="cost_barrier", horizontal=True)

    # BMI section
    with card():
        st.subheader("BMI (Optional)")
        if "bmi_standard" in st.session_state:
            bmi = float(st.session_state["bmi_standard"])
            st.success(f"Using BMI (Standard): {bmi}")
            st.caption(f"Standard: {st.session_state['bmi_standard_category']} | Asian: {st.session_state['bmi_asian_category']}")
            plot_bmi_gauge(bmi)
        else:
            st.warning("No BMI calculated on the BMI page. Enter BMI manually or go to BMI page.")
            bmi = float(st.number_input("Enter BMI (Standard)", min_value=10.0, max_value=60.0,
                                        value=auto_bmi if auto_bmi else 25.0))

    # building model row
    row = {
        "age_group_code": int(age_code13),
        "sex": int(sex),
        "race_group": int(race_group),
        "education_level": int(education_level),
        "income_level": int(income_level),
        "marital_status": int(marital_status),
        "ever_smoked": int(ever_smoked),
        "phys_activity": int(phys_activity),
        "any_alcohol": int(any_alcohol),
        "diabetes": int(diabetes),
        "obesity": 0,  # placeholder for heart model only
        "hypertension": int(hypertension_code),
        "heart_disease": int(heart_disease),
        "bmi": float(bmi),
        "general_health": int(general_health),
        "depression": int(depression),
        "cost_barrier": int(cost_barrier),
        "recent_checkup": int(recent_checkup_01),
        "estimated_diabetes_duration": 0.0,
        "diabetes_duration_known": 0,
    }

    # capturing duration as years since dx (simple, direct)
    duration_years = int(st.slider("Years since diabetes diagnosis (0 if none/unknown)", 0, 30, 0))
    if diabetes == 1 and duration_years > 0:
        row["estimated_diabetes_duration"] = float(duration_years)
        row["diabetes_duration_known"] = 1

    X_raw = pd.DataFrame([row])

    # predict
    if st.button("Predict Risk"):
        try:
            # preprocessing + aligning quietly (no yellow lines)
            X_prep = preprocess_input(X_raw)

            X_diab  = clamp_training_domains(get_aligned_input(diabetes_model,     X_prep, FEATURES["diabetes"],     model_name="diabetes",     verbose=False))
            X_htn   = clamp_training_domains(get_aligned_input(hypertension_model, X_prep, FEATURES["hypertension"], model_name="hypertension", verbose=False))
            X_heart = clamp_training_domains(get_aligned_input(heart_model,        X_prep, FEATURES["heart"],        model_name="heart",        verbose=False))

            # reversing age bin for HTN if your training used that pattern
            if "age_group_code" in X_htn:
                X_htn["age_group_code"] = 14 - X_htn["age_group_code"]

            # ----------------- Diabetes -----------------
            if not show_status_and_skip("Diabetes", diabetes == 1):
                p = diabetes_model.predict_proba(X_diab)[0][1] * 100
                st.success(f"Diabetes Risk: {'Low (<1%)' if p < 1 else f'{p:.2f}%'}")
                with st.expander("Diabetes Feature Impact"):
                    plot_shap_bar_tidy(diabetes_model, X_diab, "Diabetes", ensure=["age_group_code"])
                    st.caption("Right = pushes risk up; Left = pushes risk down.")

            # ----------------- Hypertension -----------------
            if not show_status_and_skip("Hypertension", hypertension_code == 1):
                p = hypertension_model.predict_proba(X_htn)[0][1] * 100
                st.success(f"Hypertension Risk: {'Low (<1%)' if p < 1 else f'{p:.2f}%'}")
                with st.expander("Hypertension Feature Impact"):
                    plot_shap_bar_tidy(hypertension_model, X_htn, "Hypertension", ensure=["age_group_code"])
                    st.caption("Right = pushes risk up; Left = pushes risk down.")

            # ----------------- Heart Disease -----------------
            if not show_status_and_skip(
                "Heart Disease", heart_disease == 1,
                note="(BRFSS scope: heart attack/MI or coronary heart disease/angina)"
            ):
                p = heart_model.predict_proba(X_heart)[0][1] * 100
                st.success(f"Heart Disease Risk: {'Low (<1%)' if p < 1 else f'{p:.2f}%'}")
                with st.expander("Heart Feature Impact"):
                    plot_shap_bar_tidy(heart_model, X_heart, "Heart Disease", ensure=["age_group_code"])
                    st.caption("Right = pushes risk up; Left = pushes risk down.")

            # ----------------- Obesity (Definition + Propensity) -----------------
            st.markdown("---")
            st.subheader("Obesity")

            ob_det = obesity_status_from_bmi(bmi, race_group)
            st.caption(ob_det["reason"])
            if ob_det["is_obese"]:
                st.metric("Obesity (definition)", "Present")
                st.info(f"BMI {bmi:.1f} ≥ cutoff {ob_det['cutoff']:.1f}")
            else:
                st.metric("Obesity (definition)", "Absent")
                st.success(f"BMI {bmi:.1f} < cutoff {ob_det['cutoff']:.1f}")

            X_for_ob_nobmi = X_prep.reindex(columns=FEATURES_OB_NO_BMI).fillna(0)
            try:
                p_prop = float(obesity_model_no_bmi_cal.predict_proba(X_for_ob_nobmi)[0, 1])
            except Exception:
                p_prop = float(obesity_model_no_bmi.predict_proba(X_for_ob_nobmi)[0, 1])
            st.success(f"Obesity propensity (non-BMI factors): {p_prop*100:.1f}%")
            st.caption("This reflects lifestyle/demographics/comorbidities; BMI is not used.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================
# About page
# ============================
def about_page():
    st.header("About Risk Engine v2")

    with card():
        st.markdown("""
**What this app estimates**
- Diabetes
- Hypertension
- Heart Disease
- Obesity

**How to read the numbers**
- These are **probabilities** from models trained on BRFSS-like survey data.
- Use for **screening and awareness**, not for diagnosis or treatment decisions.
- Local SHAP bars explain **why this person’s score looks like this**.
        """)
        muted("Version 2 • Built with Streamlit & XGBoost")

    with card():
        st.subheader("Definitions (BRFSS scope)")
        st.markdown("""
- **Heart Disease**: Ever told by a professional you had a **heart attack (MI)** or **coronary heart disease/angina**.  
- **Hypertension**: Ever told you have **high blood pressure** (not pregnancy-related).  
- **Diabetes**: Ever told you have **diabetes** (type not separated here).  
- **Obesity (definition)**: BMI-based; **25.0** cutoff for Asian, **30.0** for others.  
- **Obesity (propensity)**: Probability from **non-BMI** factors (lifestyle, demographics, comorbidities).
        """)

    with card():
        st.subheader("Clinician-friendly justification")
        st.markdown(clinician_paragraph())

    with card():
        st.subheader("Learn more")
        st.markdown("""
- WHO Diabetes — https://www.who.int/news-room/fact-sheets/detail/diabetes  
- WHO Hypertension — https://www.who.int/news-room/fact-sheets/detail/hypertension  
- WHO Cardiovascular diseases — https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)  
- CDC BRFSS — https://www.cdc.gov/brfss/
        """)

# ============================
# Backward-compat shims
# ============================
def risk_tab(auto_bmi=None):
    # delegating to new page function
    return risk_page(auto_bmi=auto_bmi)

def bmi_tab():
    # delegating to new page function
    return bmi_page()

def about():
    # delegating to new page function
    return about_page()
