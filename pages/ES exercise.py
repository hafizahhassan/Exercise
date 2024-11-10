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
        
    # Button
    submitButton = st.form_submit_button("Submit")
    
st.write("O U T P U T")
