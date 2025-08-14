import sys
import os
import streamlit as st

# ---------- path setup ----------
# adding project root once to import api.main cleanly
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

# importing main safely
try:
    import api.main as main  # provides bmi_tab() and risk_tab()
except Exception as e:
    st.error(f"Failed to import api.main: {e}")
    st.stop()

# ---------- page config ----------
# configuring page only once across reruns
if "page_configured" not in st.session_state:
    st.set_page_config(page_title="Risk Engine v2", layout="wide")
    st.session_state["page_configured"] = True

# drawing title once
st.title("Risk Engine v2")

# ---------- navigation ----------
# replacing tabs with a simple sidebar radio
st.sidebar.title("Navigate")
page = st.sidebar.radio("Pages", ["BMI Calculator", "Risk Engine", "About"], index=1)

# ---------- BMI page ----------
if page == "BMI Calculator":
    st.subheader("BMI Calculator")
    st.caption("Entering weight and height, showing Standard & Asian cutoffs.")
    # running BMI workflow
    bmi_value = main.bmi_tab()
    # storing BMI for risk page handoff
    if bmi_value is not None:
        st.session_state["bmi_standard"] = float(bmi_value)
        st.success("BMI captured. Open Risk Engine to use it.")

# ---------- Risk page ----------
elif page == "Risk Engine":
    # pulling BMI from session if available
    auto_bmi = st.session_state.get("bmi_standard", None)
    main.risk_tab(auto_bmi=auto_bmi)

# ---------- About page ----------
else:
    st.subheader("About Risk Engine v2")
    st.markdown("""
**BRFSS Risk Engine v2** predicts risks for:
- Diabetes
- Hypertension
- Heart Disease
- Obesity

**What do these mean here (BRFSS scope)?**
- **Heart Disease**: ever told you had a **heart attack (MI)** or **coronary heart disease/angina**
- **Hypertension**: ever told you have **high blood pressure** (not pregnancy-related)
- **Diabetes**: ever told you have **diabetes** (type not separated in UI)
- **Obesity (definition)**: BMI cutoff **25.0** for Asian; **30.0** for others
- **Obesity (propensity)**: likelihood from **non-BMI** factors (lifestyle, demographics, comorbidities)

**Links**
- WHO Diabetes — https://www.who.int/news-room/fact-sheets/detail/diabetes  
- WHO Hypertension — https://www.who.int/news-room/fact-sheets/detail/hypertension  
- WHO Cardiovascular diseases — https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)  
- CDC BRFSS — https://www.cdc.gov/brfss/

This app provides educational risk estimates, not clinical diagnosis.
""")
    st.caption("Version 2 • Built with Streamlit & XGBoost")
