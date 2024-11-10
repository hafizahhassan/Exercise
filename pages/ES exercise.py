import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
#from numpy import arange, exp, sqrt, cos, e, pi, meshgrid
from mpl_toolkits.mplot3d import Axes3D

# evolution strategy (mu + lambda) of the ackley objective function
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import argsort
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

st.title("ES Exercise")

# Create form untuk button
with st.form("es_form"):
    st.write("Choose button you want to see the Output :")
    for i in range(10):
        col1, col2 = st.columns(3)    #Buat 3 column
        
        submitButton = col1.form_submit_button("Submit")

        submitButton = col2.form_submit_button("Submit")
        
    # Button
    submitButton = col1.form_submit_button("Submit")
