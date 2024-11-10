import streamlit as st

# Title for the app
st.title("Form with Two Buttons")

# Create a form
with st.form(key="my_form"):
    # Input fields inside the form
    name = st.text_input("Enter your name")
    age = st.number_input("Enter your age", min_value=0, step=1)

    # Add submit button
    submit_button = st.form_submit_button(label="Submit")
    
    # Add another button outside form for extra action
    clear_button = st.button("Clear Form")

# Display output based on submit button click
if submit_button:
    st.write(f"Name: {name}")
    st.write(f"Age: {age}")

# Action for the clear button
if clear_button:
    st.session_state["my_form"] = None  # Resets form values (Streamlit will handle clearing on reload)
