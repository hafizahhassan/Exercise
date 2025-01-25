import streamlit as st
import csv
import pandas as pd
import random
import numpy as np

##################################### CSS FOR DIVIDER ################################################################
# CSS for shimmering divider effect
# CSS for Button
st.markdown("""
<style>

@keyframes shimmer {
  0% {
    background-position: -1000px 0;
  }
  100% {
    background-position: 1000px 0;
  }
}

.shimmer-divider {
  height: 3px;
  background: linear-gradient(to right, #f2f2f2 0%, #FFD700 50%, #f2f2f2 100%);
  background-size: 1000px 100%;
  animation: shimmer 20s infinite linear;
  margin: 20px 0;
}

.stFormSubmitButton > button:first-child {
    background-color: #02ab21;
    color: white;
    font-size: 20px;
    width: 100%;
}
.stFormSubmitButton > button:nth-child(2) {
    background-color: #FFD700;
    color: white;
    font-size: 20px;
    width: 100%;
}
.stFormSubmitButton > button:hover {
    opacity: 0.8;
}

</style>
""", unsafe_allow_html=True)

#####################################################################################################################

st.title("OPTIMIZED EXAM TIMETABLE GENERATION USING FIREFLY ALGORITHM")

# Add a shimmering divider
st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

##################################### INSERT DATASET ###################################################################

st.subheader("D A T A")

def clear_output():
  # Clear all the file uploader widgets
  st.session_state.schedule_file = None
  st.session_state.courses_file = None
  st.session_state.timeslots_file = None
  st.session_state.classrooms_file = None

# Initialize session state for file uploaders if not already done
if 'schedule_file' not in st.session_state:
  st.session_state.schedule_file = None
if 'courses_file' not in st.session_state:
  st.session_state.courses_file = None
if 'timeslots_file' not in st.session_state:
  st.session_state.timeslots_file = None
if 'classrooms_file' not in st.session_state:
  st.session_state.classrooms_file = None

with st.form("Firefly_Algorithm"):
  # Upload the Dataset
  # File upload for schedule, courses, timeslots, and classrooms
  schedule_file = st.file_uploader("Upload Schedule Data", type=["csv"])
  
  courses_file = st.file_uploader("Upload Courses Data", type=["csv"])
  
  timeslots_file = st.file_uploader("Upload Timeslots Data", type=["csv"])
  
  classrooms_file = st.file_uploader("Upload Classrooms Data", type=["csv"])

  col1, col2, col3 = st.columns(3)
  with col2:
    Submit_Button = st.form_submit_button("Submit")
    Clear_Button = st.form_submit_button("Clear")

##################################### OUTPUT ###################################################################

# Add a shimmering divider
st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

st.subheader("O U T P U T")

if Submit_Button:
  st.write("Form submitted!")

if Clear_Button:
  clear_output()
  st.rerun()







