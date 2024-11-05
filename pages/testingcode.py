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

    def dist_two_cities(city_1, city_2):
        city_1_coords = city_coords[city_1]
        city_2_coords = city_coords[city_2]
        return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords)) ** 2))

    def total_dist_individual(individual):
        total_dist = 0
        for i in range(0, len(individual)):
            if i == len(individual) - 1:
                total_dist += dist_two_cities(individual[i], individual[0])
            else:
                total_dist += dist_two_cities(individual[i], individual[i + 1])
        return total_dist

    def fitness_prob(population):
        total_dist_all_individuals = []
        for i in range(0, len(population)):
            total_dist_all_individuals.append(total_dist_individual(population[i]))

        max_population_cost = max(total_dist_all_individuals)
        population_fitness = max_population_cost - np.array(total_dist_all_individuals)
        population_fitness_sum = sum(population_fitness)
        population_fitness_probs = population_fitness / population_fitness_sum
        return population_fitness_probs

    def roulette_wheel(population, fitness_probs):
        population_fitness_probs_cumsum = fitness_probs.cumsum()
        bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)
        selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
        return population[selected_individual_index]

    def crossover(parent_1, parent_2):
        n_cities_cut = len(city_names) - 1
        cut = round(random.uniform(1, n_cities_cut))
        offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1, offspring_2

    def mutation(offspring):
        n_cities_cut = len(city_names) - 1
        index_1, index_2 = random.sample(range(n_cities_cut + 1), 2)
        offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
        return offspring

    def run_ga(city_names, n_population, n_generations, crossover_per, mutation_per):
        population = initial_population(city_names, n_population)
        for gen in range(n_generations):
            fitness_probs = fitness_prob(population)
            parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
            offspring = []
            for i in range(0, len(parents), 2):
                off1, off2 = crossover(parents[i], parents[i + 1])
                if random.random() < mutation_per:
                    off1 = mutation(off1)
                if random.random() < mutation_per:
                    off2 = mutation(off2)
                offspring.extend([off1, off2])

            population = sorted(parents + offspring, key=total_dist_individual)[:n_population]
        return min(population, key=total_dist_individual)

    # Run genetic algorithm
    best_route = run_ga(city_names, n_population, n_generations, crossover_per, mutation_per)
    minimum_distance = total_dist_individual(best_route)

    # Plot the shortest path
    x_shortest = [city_coords[city][0] for city in best_route] + [city_coords[best_route[0]][0]]
    y_shortest = [city_coords[city][1] for city in best_route] + [city_coords[best_route[0]][1]]

    fig, ax = plt.subplots()
    ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
    for i, txt in enumerate(best_route):
        ax.annotate(f"{i+1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=12)
    plt.title(f"TSP Best Route Using GA\nTotal Distance: {round(minimum_distance, 3)}")
    fig.set_size_inches(16, 12)
    st.pyplot(fig)

    # Display results
    st.write("Jarak Terpendek:", round(minimum_distance, 3))
    st.write("Jalur Terpendek:", best_route)
