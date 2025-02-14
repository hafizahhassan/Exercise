import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
#from numpy import arange, exp, sqrt, cos, e, pi, meshgrid
from mpl_toolkits.mplot3d import Axes3D

# ackley multimodal function
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from numpy import asarray

st.title("Ackley Multimodal Function")

# objective function
def objective(x, y):
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
 
# define range for input = x, y
r_min, r_max = -5.0, 5.0

# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)

# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)

# compute targets
results = objective(x, y)

# create a surface plot with the jet color scheme
fig = plt.figure()
#axis = figure.gca(projection='3d')
axis = fig.add_subplot(projection='3d')
axis.plot_surface(x, y, results, cmap='jet')

# show the plot
#pyplot.show()
st.pyplot(fig)
