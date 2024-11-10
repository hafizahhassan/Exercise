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

st.title("Exercise ES")

st.write("Choose button you want to see the Output :")

col1, col2 = st.columns(2)
button1 = col1.button("MU, LAMDA")
button2 = col2.button("MU + LAMDA")

st.write("O U T P U T")

if button1:
	#st.write("1")

	# objective function
	def objective(v):
		x, y = v
		return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
	
	# check if a point is within the bounds of the search
	def in_bounds(point, bounds):
		# enumerate all dimensions of the point
		for d in range(len(bounds)):
			# check if out of bounds for this dimension
			if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
				return False
		return True
	
	# evolution strategy (mu, lambda) algorithm
	def es_comma(objective, bounds, n_iter, step_size, mu, lam):
		best, best_eval = None, 1e+10
	
		n_children = int(lam / mu)        # calculate the number of children per parent
	
		# initial population
		population = list()
		for _ in range(lam):
			candidate = None
			while candidate is None or not in_bounds(candidate, bounds):
				candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
			population.append(candidate)
	
		# perform the search
		for epoch in range(n_iter):
			scores = [objective(c) for c in population]       # evaluate fitness for the population
			ranks = argsort(argsort(scores))                  # rank scores in ascending order
			selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]   # select the indexes for the top mu ranked solutions
		
			children = list()              # create children from parents
		
		for i in selected:
			if scores[i] < best_eval:    # check if this parent is the best solution ever seen
				best, best_eval = population[i], scores[i]
				st.write("%d, Best: f(%s) = %.5f" % (epoch, best, best_eval))
		
			for _ in range(n_children):  # create children for paren
				child = None
				while child is None or not in_bounds(child, bounds):
					child = population[i] + randn(len(bounds)) * step_size
				children.append(child)
		
			# replace population with children
			population = children
		return [best, best_eval]
	
	seed(1)               # seed the pseudorandom number generator
	
	bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]]) # define range for input
	
	n_iter = 5000         # define the total iterations
	
	step_size = 0.15      # define the maximum step size
	
	mu = 20               # number of parents selected
	
	lam = 100             # the number of children generated by parents
	
	# perform the evolution strategy (mu, lambda) search
	best, score = es_comma(objective, bounds, n_iter, step_size, mu, lam)
	
	st.write("\n-------------------------- mu, lamda --------------------------\n")
	st.write("DONE!")
	st.write("f(%s) = %f" % (best, score))
  
  if button2:
    st.write("2")
