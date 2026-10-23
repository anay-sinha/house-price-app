# app.py
import streamlit as st, pandas as pd, numpy as np, joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

@st.cache_resource
def load():
    return joblib.load("model.joblib")

art = load(); model = art["model"]

st.title("🏠 House Price Predictor")
st.caption("XGBoost trained on the Ames Housing dataset")

with st.sidebar:
    st.header("House details")
    qual   = st.slider("Overall quality (1–10)", 1, 10, 6)
    sf     = st.slider("Total area (sq ft)", 500, 6000, 2000, step=50)
    baths  = st.slider("Total bathrooms", 1.0, 5.0, 2.0, step=0.5)
    garage = st.selectbox("Garage capacity (cars)", [0, 1, 2, 3, 4], index=2)
    age    = st.slider("House age (years)", 0, 120, 20)
    hood   = st.selectbox("Neighborhood", art["neighborhoods"])

row = pd.DataFrame([{"OverallQual": qual, "TotalSF": sf, "TotalBath": baths,
                     "GarageCars": garage, "HouseAge": age, "Neighborhood": hood}])

price = float(np.expm1(model.predict(row)[0]))
st.metric("Estimated sale price", f"${price:,.0f}")
st.caption(f"Typical error is roughly ±{12:.0f}% — this is an estimate, not an appraisal.")
