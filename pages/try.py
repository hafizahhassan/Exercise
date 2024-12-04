import streamlit as st
import csv
import requests # Import the requests module
import pandas as pd
import numpy as np
import random

st.title("T R Y")

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

st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

# Add a divider
st.divider()
st.header("This is a header with a colored divider", divider="red")
st.subheader("This is a subheader with a colored divider", divider="green")
