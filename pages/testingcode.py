import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import random
import numpy as np
import statistics
import pandas as pd
import seaborn as sns
import streamlit as st

# Streamlit input fields for cities
st.title("Traveling Salesperson Problem (TSP) with Genetic Algorithm")
st.write("Enter the coordinates for 10 cities:")

# Create input fields for cities
user_cities = []
for i in range(10):
    city_name = st.text_input(f"City {i + 1}", value=f"City{i + 1}")
    x_coord = st.number_input("X Coordinate", min_value=1, max_value=10, step=1, key=f"x{i}")
    y_coord = st.number_input("Y Coordinate", min_value=1, max_value=10, step=1, key=f"y{i}")
    user_cities.append((city_name, x_coord, y_coord))

# Submit button to generate results
if st.button("Submit"):
    # Unpack cities into names, x, and y coordinates
    cities_names, x, y = zip(*user_cities)
    city_coords = dict(zip(cities_names, zip(x, y)))
    
    # Genetic Algorithm parameters
    n_population = 250
    crossover_per = 0.8
    mutation_per = 0.2
    n_generations = 200

    # Pastel palette for city colors
    colors = sns.color_palette("pastel", len(cities_names))

    # City icons for annotation
    city_icons = {
        1: "♕", 
        2: "♖", 
        3: "♗", 
        4: "♘", 
        5: "♙",
        6: "♔", 
        7: "♚",
        8: "♛", 
        9: "♜", 
        10: "♝"
    }
    
    # Visualization of cities and their connections
    fig, ax = plt.subplots()
    ax.grid(False)
    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons.get(city, "●")  # Default icon if city name isn't in predefined list
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                    textcoords='offset points')
        
        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

    # Genetic Algorithm functions here...

    # Example placeholder for final output visualization
    # You would run your GA here and display the results
    # Shortest path and minimum distance placeholders
    shortest_path = ["City1", "City2", "City3"]  # Replace with actual GA result
    minimum_distance = 123.45  # Replace with actual GA result

    st.write("Minimum Distance:", minimum_distance)
    st.write("Shortest Path:", shortest_path)
    
    # Plotting the shortest path
    x_shortest = [city_coords[city][0] for city in shortest_path] + [city_coords[shortest_path[0]][0]]
    y_shortest = [city_coords[city][1] for city in shortest_path] + [city_coords[shortest_path[0]][1]]

    fig, ax = plt.subplots()
    ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
    plt.legend()

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

    plt.title("TSP Best Route Using GA", fontsize=25, color="k")
    plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}\n"
                 f"{n_generations} Generations\n{n_population} Population Size\n"
                 f"{crossover_per} Crossover\n{mutation_per} Mutation", fontsize=18, y=1.047)

    for i, txt in enumerate(shortest_path):
        ax.annotate(str(i+1) + "- " + txt, (x_shortest[i], y_shortest[i]), fontsize=20)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)
