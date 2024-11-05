
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import random
import numpy as np
import statistics
import pandas as pd
import seaborn as sns
import streamlit as st

for i in range(10):
    cities_names = st.text_input(f"City {i}", f"City {i}")
    x = st.number_input(f"X Coordinate for {cities_names}", min_value=1, max_value=10, step=1, key=f"x{i}")
    y = st.number_input(f"Y Coordinate for {cities_names}", min_value=1, max_value=10, step=1, key=f"y{i}")
    city_coords = dict(zip(cities_names, zip(x, y)))
    n_population = 250
    crossover_per = 0.8
    mutation_per = 0.2
    n_generations = 200
    
    # Pastel Pallete
    colors = sns.color_palette("pastel", len(cities_names))
    
    # City Icons
    city_icons = {
        {cities_names}: "♕",
        {cities_names}: "♖",
        {cities_names}: "♗",
        {cities_names}: "♘",
        {cities_names}: "♙",
        {cities_names}: "♔",
        {cities_names}: "♚",
        {cities_names}: "♛",
        {cities_names}: "♜",
        {cities_names}: "♝"
    }
    
    fig, ax = plt.subplots()
    
    ax.grid(False)  # Grid
    
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
    #plt.show()
    st.pyplot(fig)
