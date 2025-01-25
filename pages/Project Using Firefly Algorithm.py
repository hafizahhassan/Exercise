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

st.subheader("I N S E R T    D A T A S E T")

# Add a shimmering divider
st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

##################################### OUTPUT ###################################################################

st.subheader("O U T P U T")

# Add a shimmering divider
st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)
