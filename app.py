import streamlit as st

st.set_page_config(
    page_title="Student Health Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

home = st.Page(
    "pages/home.py",
    title="Home",
    icon="🏠",
    default=True
)

analytics = st.Page(
    "pages/analytic.py",
    title="Analytics",
    icon="📊"
)

prediction = st.Page(
    "pages/prediction.py",
    title="Health Risk Prediction",
    icon="🩺"
)

pg = st.navigation([
    home,
    analytics,
    prediction
])

pg.run()