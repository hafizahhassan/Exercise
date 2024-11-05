import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import random
import numpy as np
import statistics
import pandas as pd
import seaborn as sns
import streamlit as st

st.title("City Coordinate Input TSP")

# Create input form for cities
with st.form("city_input_form"):
    city_coords = {}
    for i in range(1, 11):
        col1, col2, col3 = st.columns(3)    #Buat 3 column
        cities_names = col1.text_input(f"City {i}", f"City {i}")
        x = col2.number_input(f"X Coordinate for {cities_names}", min_value=1, max_value=10, step=1, key=f"x{i}")
        y = col3.number_input(f"Y Coordinate for {cities_names}", min_value=1, max_value=10, step=1, key=f"y{i}")
        city_coords[cities_names] = (x, y)
        
    # Button
    submitButton = st.form_submit_button("Submit")
    
if submitButton:
    st.write(city_coords)

st.write("Outside the form")
        












