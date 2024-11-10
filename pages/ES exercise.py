import streamlit as st

# Title for the app
st.title("Form with Two Buttons")

col1, col3 = st.columns(2)
submit_button = col1.st.form_submit_button(label="Submit")
clear_button = col2.st.button("Clear Form")

