import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns
import streamlit as st
from itertools import permutations

# Set up Streamlit UI to input cities and their coordinates
st.title("Traveling Salesman Problem (TSP) - City Input")

# Input for city names and coordinates
cities_names = []
x_coords = []
y_coords = []

# Chess piece icons for cities
city_icons = [
    "♕", "♖", "♗", "♘", "♙", "♔", "♚", "♛", "♜", "♝"
]

# Allow user to input the cities and coordinates
for i in range(10):
    city_name = st.text_input(f"Enter the name of City {i + 1}:", key=f"city_{i}")
    if city_name:
        cities_names.append(city_name)
        x_coord = st.number_input(f"Enter the x-coordinate (longitude) for {city_name}:", key=f"x_{i}", step=0.1)
        y_coord = st.number_input(f"Enter the y-coordinate (latitude) for {city_name}:", key=f"y_{i}", step=0.1)
        x_coords.append(x_coord)
        y_coords.append(y_coord)

# Create a dictionary for city coordinates if cities are entered
if len(cities_names) == 10:
    city_coords = dict(zip(cities_names, zip(x_coords, y_coords)))

    # Set pastel color palette
    colors = sns.color_palette("pastel", len(cities_names))

    # Map each city to a chess icon
    city_icon_map = {city: city_icons[i] for i, city in enumerate(cities_names)}

    # Plotting the cities on a graph
    fig, ax = plt.subplots()
    ax.grid(False)  # Hide grid

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icon_map[city]  # Get the icon for the city
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                    textcoords='offset points')

        # Connect cities with opaque lines
        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    fig.set_size_inches(16, 12)

    # Display the plot in the Streamlit app
    st.pyplot(fig)
else:
    st.warning("Please enter names and coordinates for 10 cities.")
