import streamlit as st
import csv
import pandas as pd
import random
import numpy as np

##################################### CSS FOR DIVIDER ################################################################
# CSS for shimmering divider effect
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
</style>
""", unsafe_allow_html=True)

#####################################################################################################################

st.title("OPTIMIZED EXAM TIMETABLE GENERATION USING FIREFLY ALGORITHM")

# Add a shimmering divider
st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

##################################### INSERT DATASET ###################################################################

st.subheader("D A T A")

# Upload the Dataset
# File upload for schedule, courses, timeslots, and classrooms
st.subheader("Upload Schedule CSV")
schedule_file = st.file_uploader("Upload Schedule Data", type=["csv"])

st.subheader("Upload Courses CSV")
courses_file = st.file_uploader("Upload Courses Data", type=["csv"])

st.subheader("Upload Timeslots CSV")
timeslots_file = st.file_uploader("Upload Timeslots Data", type=["csv"])

st.subheader("Upload Classrooms CSV")
classrooms_file = st.file_uploader("Upload Classrooms Data", type=["csv"])

##################################### OUTPUT ###################################################################

# Add a shimmering divider
st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

st.subheader("O U T P U T")
