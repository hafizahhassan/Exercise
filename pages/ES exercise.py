import streamlit as st

# Title for the app
st.title("Form with Two Buttons")

col1, col2 = st.columns(2)
submit_button = col1.button("MU, LAMDA")
clear_button = col2.button("MU + LAMDA")

