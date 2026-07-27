import streamlit as st

st.set_page_config(
    page_title="Student Health Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

home = st.Page(
    "modules/home.py",
    title="Home",
    icon="🏠",
    default=True
)

analytics = st.Page(
    "modules/analytic.py",
    title="Analytics",
    icon="📊"
)

prediction = st.Page(
    "modules/prediction.py",
    title="Health Risk Prediction",
    icon="🩺"
)

pg = st.navigation([
    home,
    analytics,
    prediction
])

pg.run()