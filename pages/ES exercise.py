import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import random
import numpy as np
import statistics
import pandas as pd
import seaborn as sns
import streamlit as st

st.title("City Coordinates Input TSP")

# Create input form for cities
with st.form("es_form"):
    st.write("Enter up to 10 cities with their coordinates (x,y) in range 1-10 :")

    TARGET = st.text_input("Enter your name : ")
        
    # Button
    submitButton = st.form_submit_button("Submit")
    
st.write("O U T P U T")
