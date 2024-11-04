import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import random
import numpy as np
import statistics
import pandas as pd
import seaborn as sns
import streamlit as st

# Function to take city inputs from user
def get_city_data():
    cities_names = []
    x_coords = []
    y_coords = []

    with st.form("my_form"):
        st.write("Enter details for 10 cities:")
        for i in range(10):
            city_name = st.text_input(f"Enter name of city {i + 1}:", f"City{i + 1}")
            city_x = st.slider(f"Enter x-coordinate for {city_name} (1-10):", 1, 10, 5)
            city_y = st.slider(f"Enter y-coordinate for {city_name} (1-10):", 1, 10, 5)
            cities_names.append(city_name)
            x_coords.append(city_x)
            y_coords.append(city_y)
    
        return cities_names, x_coords, y_coords

# Collect user inputs
cities_names, x, y = get_city_data()
city_coords = dict(zip(cities_names, zip(x, y)))

# Parameters
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Pastel Palette for cities
colors = sns.color_palette("pastel", len(cities_names))

# City Distance Functions
def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords)) ** 2))

def total_dist_individual(individual):
    total_dist = 0
    for i in range(len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i + 1])
    return total_dist

# GA Functions
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_probs = population_fitness / population_fitness.sum()
    return population_fitness_probs

def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1)
    selected_individual_index = len(bool_prob_array[bool_prob_array]) - 1
    return population[selected_individual_index]

def crossover(parent_1, parent_2):
    n_cities_cut = len(cities_names) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

def mutation(offspring):
    n_cities_cut = len(cities_names) - 1
    index_1, index_2 = random.sample(range(n_cities_cut + 1), 2)
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    for _ in range(n_generations):
        fitness_probs = fitness_prob(population)
        parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]

        offspring = []
        for i in range(0, len(parents), 2):
            offspring_1, offspring_2 = crossover(parents[i], parents[i + 1])
            if random.random() < mutation_per:
                offspring_1 = mutation(offspring_1)
            if random.random() < mutation_per:
                offspring_2 = mutation(offspring_2)
            offspring.extend([offspring_1, offspring_2])

        mixed_population = parents + offspring
        fitness_probs = fitness_prob(mixed_population)
        best_indices = np.argsort(fitness_probs)[::-1][:n_population]
        population = [mixed_population[i] for i in best_indices]

    return population

# Display a submit button
if st.button("Submit and Run TSP"):
    # Run Genetic Algorithm
    best_population = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

    # Evaluate and Display Best Route
    total_distances = [total_dist_individual(ind) for ind in best_population]
    min_distance = min(total_distances)
    st.write("Minimum Distance:", min_distance)

    best_route = best_population[np.argmin(total_distances)]
    st.write("Shortest Path:", best_route)

    # Plot Best Route
    x_best = [city_coords[city][0] for city in best_route] + [city_coords[best_route[0]][0]]
    y_best = [city_coords[city][1] for city in best_route] + [city_coords[best_route[0]][1]]

    fig, ax = plt.subplots()
    ax.plot(x_best, y_best, '--go', label='Best Route', linewidth=2.5)
    plt.legend()
    for i, txt in enumerate(best_route):
        ax.annotate(f"{i + 1} - {txt}", (x_best[i], y_best[i]), fontsize=10)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

