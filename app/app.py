import sys
import os
import streamlit as st

# ----- setting up import path -----
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

# ----- importing api.main safely -----
try:
    import api.main as main  # provides home_page, bmi_page, risk_page, about_page
except Exception as e:
    st.error(f"Failed to import api.main: {e}")
    st.stop()

# ----- configure page once -----
if "page_configured" not in st.session_state:
    st.set_page_config(page_title="Risk Engine v2", layout="wide")
    st.session_state["page_configured"] = True

st.title("Risk Engine v2")

# ----- apply pending redirects BEFORE nav widget -----
if "__nav_redirect" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("__nav_redirect")

# ----- sidebar navigation -----
pages = ["Home", "BMI Calculator", "Risk Prediction", "About"]
default_page = st.session_state.get("nav_page", "Risk Prediction")
if default_page not in pages:
    default_page = "Risk Prediction"

page = st.sidebar.radio(
    "Pages",
    pages,
    index=pages.index(default_page),
    key="nav_page",
)

# ----- routing -----
if page == "Home":
    main.home_page()

elif page == "BMI Calculator":
    st.subheader("BMI Calculator")
    st.caption("Calculating BMI and showing both Standard & Asian cutoffs.")
    bmi_value = main.bmi_page()
    if bmi_value is not None:
        st.session_state["bmi_standard"] = float(bmi_value)
        st.success("BMI captured. Open Risk Prediction to use it.")

elif page == "Risk Prediction":
    auto_bmi = st.session_state.get("bmi_standard", None)
    main.risk_page(auto_bmi=auto_bmi)

else:
    main.about_page()
