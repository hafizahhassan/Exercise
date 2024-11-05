import matplotlib.pyplot as plt
from itertools import permutations, combinations
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Parameters
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Sidebar for user input
st.sidebar.header("Input Kota dan Koordinat")
city_names = []
city_x = []
city_y = []

for i in range(10):
    city_name = st.sidebar.text_input(f"Nama Kota {i+1}", f"Kota{i+1}")
    city_coord_x = st.sidebar.number_input(f"Koordinat X untuk {city_name}", value=0.0, format="%.2f")
    city_coord_y = st.sidebar.number_input(f"Koordinat Y untuk {city_name}", value=0.0, format="%.2f")
    
    city_names.append(city_name)
    city_x.append(city_coord_x)
    city_y.append(city_coord_y)

# Button to confirm inputs and plot
if st.sidebar.button("Submit"):

    # Coordinates dictionary
    city_coords = dict(zip(city_names, zip(city_x, city_y)))

    # Pastel color palette
    colors = sns.color_palette("pastel", len(city_names))

    # City icons
    city_icons = {name: chr(9812 + i) for i, name in enumerate(city_names)}  # Unique chess icons

    # Plot cities
    fig, ax = plt.subplots()
    ax.grid(False)

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons[city]
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                    textcoords='offset points')

        # Connect cities with opaque lines
        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

    # Genetic Algorithm Functions
    def initial_population(cities_list, n_population=250):
        population_perms = []
        possible_perms = list(permutations(cities_list))
        random_ids = random.sample(range(0, len(possible_perms)), n_population)

        for i in random_ids:
            population_perms.append(list(possible_perms[i]))

        return population_perms

    def dist_two_c
