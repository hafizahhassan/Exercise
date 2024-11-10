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
    for i in range(10):
        col1, col2, col3 = st.columns(3)    #Buat 3 column
        
        city_name = col1.text_input(f"City {i+1}") #value=f"City {i+1}"
        cities_names.append(city_name)
        
        city_x = col2.number_input(f"X Coordinate for City {i+1}", min_value=1, max_value=10, step=1)
        x.append(city_x)
        
        city_y = col3.number_input(f"Y Coordinate for City {i+1}", min_value=1, max_value=10, step=1)
        y.append(city_y)
        
    # Button
    submitButton = st.form_submit_button("Submit")
    
st.write("O U T P U T")
