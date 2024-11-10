import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import random
import numpy as np
import statistics
import pandas as pd
import seaborn as sns
import streamlit as st

st.title("Exercise ES")

# Create input form for cities
with st.form("es_form"):
    st.write("Choose button you want to see the Output : ")

    col1, col2 = st.columns(2)
    TARGET = col1.st.text_input("Enter your name : ")
    submitButton2 = col2st.form_submit_button("Submit")
        
    # Button
    submitButton = st.form_submit_button("Submit")
    
st.write("O U T P U T")
