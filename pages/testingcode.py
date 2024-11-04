import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Pastel Pallete
colors = sns.color_palette("pastel")

# Streamlit form for user input
st.title("Traveling Salesman Problem (TSP) with Genetic Algorithm")

# Initialize empty lists for user input
cities_names = []
x_coords = []
y_coords = []

with st.form(key='city_form'):
    n_cities = st.number_input("Number of Cities", min_value=1, max_value=20, value=10)
    
    for i in range(n_cities):
        city_name = st.text_input(f"City {i + 1} Name", "")
        x_coord = st.number_input(f"City {i + 1} X Coordinate", value=0.0)
        y_coord = st.number_input(f"City {i + 1} Y Coordinate", value=0.0)
        
        if city_name:
            cities_names.append(city_name)
            x_coords.append(x_coord)
            y_coords.append(y_coord)
    
    submit_button = st.form_submit_button(label='Submit')

# If the form is submitted
if submit_button:
    if len(cities_names) < 2:
        st.warning("Please enter at least two cities.")
    else:
        city_coords = dict(zip(cities_names, zip(x_coords, y_coords)))
        n_population = 250
        crossover_per = 0.8
        mutation_per = 0.2
        n_generations = 200

        # City Icons
        city_icons = {name: "🏙️" for name in cities_names}  # Default icon

        # Plotting cities
        fig, ax = plt.subplots()
        ax.grid(False)  # Grid

        for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
            color = colors[i % len(colors)]
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

        # Genetic Algorithm functions should be defined here (not included for brevity)
        # Make sure to include functions like initial_population, dist_two_cities, total_dist_individual, etc.

        # Run the genetic algorithm
        best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

        total_dist_all_individuals = [total_dist_individual(ind) for ind in best_mixed_offspring]
        index_minimum = np.argmin(total_dist_all_individuals)
        minimum_distance = min(total_dist_all_individuals)

        st.write("Minimum Distance:", minimum_distance)
        shortest_path = best_mixed_offspring[index_minimum]
        st.write("Shortest Path:", shortest_path)

        # Prepare the shortest path for plotting
        x_shortest = [city_coords[city][0] for city in shortest_path] + [city_coords[shortest_path[0]][0]]
        y_shortest = [city_coords[city][1] for city in shortest_path] + [city_coords[shortest_path[0]][1]]

        fig, ax = plt.subplots()
        ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
        plt.legend()

        # Draw connections between cities
        for i in range(len(shortest_path) - 1):
            ax.plot([x_shortest[i], x_shortest[i + 1]], [y_shortest[i], y_shortest[i + 1]], 'k-', alpha=0.09, linewidth=1)

        plt.title("TSP Best Route Using GA", fontsize=25, color="k")
        str_params = f'\n{n_generations} Generations\n{n_population} Population Size\n{crossover_per} Crossover\n{mutation_per} Mutation'
        plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}{str_params}", fontsize=18, y=1.047)

        for i, txt in enumerate(shortest_path):
            ax.annotate(f"{i + 1} - {txt}", (x_shortest[i], y_shortest[i]), fontsize=20)

        fig.set_size_inches(16, 12)
        st.pyplot(fig)
