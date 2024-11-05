import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns
import pandas as pd

# Pastel Pallete
colors = sns.color_palette("pastel", 10)

# City Icons (Customizable)
city_icons = {f"City {i+1}": f"♕" for i in range(10)}

# Function for generating initial population
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)

    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

# Distance between two cities
def dist_two_cities(city_1, city_2, city_coords):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

def total_dist_individual(individual, city_coords):
    total_dist = 0
    for i in range(0, len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0], city_coords)
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1], city_coords)
    return total_dist

def fitness_prob(population, city_coords):
    total_dist_all_individuals = [total_dist_individual(ind, city_coords) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_sum = sum(population_fitness)
    population_fitness_probs = population_fitness / population_fitness_sum
    return population_fitness_probs

# Roulette Wheel Selection
def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)
    selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
    return population[selected_individual_index]

# Crossover
def crossover(parent_1, parent_2):
    n_cities_cut = len(parent_1) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1, offspring_2 = [], []
    offspring_1 = parent_1[0:cut] + [city for city in parent_2 if city not in parent_1[0:cut]]
    offspring_2 = parent_2[0:cut] + [city for city in parent_1 if city not in parent_2[0:cut]]
    return offspring_1, offspring_2

# Mutation
def mutation(offspring):
    n_cities_cut = len(offspring) - 1
    index_1 = round(random.uniform(0, n_cities_cut))
    index_2 = round(random.uniform(0, n_cities_cut))
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

def run_ga(city_names, n_population, n_generations, crossover_per, mutation_per, city_coords):
    population = initial_population(city_names, n_population)
    fitness_probs = fitness_prob(population, city_coords)

    parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
    offspring_list = []

    for i in range(0, len(parents_list), 2):
        offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])
        if random.random() > (1 - mutation_per):
            offspring_1 = mutation(offspring_1)
        if random.random() > (1 - mutation_per):
            offspring_2 = mutation(offspring_2)
        offspring_list.extend([offspring_1, offspring_2])

    mixed_offspring = parents_list + offspring_list
    fitness_probs = fitness_prob(mixed_offspring, city_coords)
    sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
    best_fitness_indices = sorted_fitness_indices[0:n_population]

    best_mixed_offspring = [mixed_offspring[i] for i in best_fitness_indices]
    for i in range(n_generations):
        fitness_probs = fitness_prob(best_mixed_offspring, city_coords)
        parents_list = [roulette_wheel(best_mixed_offspring, fitness_probs) for _ in range(int(crossover_per * n_population))]
        offspring_list = []

        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])
            if random.random() > (1 - mutation_per):
                offspring_1 = mutation(offspring_1)
            if random.random() > (1 - mutation_per):
                offspring_2 = mutation(offspring_2)
            offspring_list.extend([offspring_1, offspring_2])

        mixed_offspring = parents_list + offspring_list
        fitness_probs = fitness_prob(mixed_offspring, city_coords)
        sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
        best_fitness_indices = sorted_fitness_indices[0:int(0.8 * n_population)]

        best_mixed_offspring = [mixed_offspring[i] for i in best_fitness_indices]
        old_population_indices = [random.randint(0, (n_population - 1)) for _ in range(int(0.2 * n_population))]
        best_mixed_offspring.extend([population[i] for i in old_population_indices])
        random.shuffle(best_mixed_offspring)

    return best_mixed_offspring

# Streamlit Interface
st.title("TSP GA Solver with User Input")

# Collecting city names and coordinates
city_names = []
city_coords = {}
for i in range(1, 11):
    city_name = st.text_input(f"Enter name for City {i}", f"City {i}")
    city_names.append(city_name)
    x_coord = st.number_input(f"Enter x-coordinate for {city_name}", min_value=1, max_value=10, value=1)
    y_coord = st.number_input(f"Enter y-coordinate for {city_name}", min_value=1, max_value=10, value=1)
    city_coords[city_name] = (x_coord, y_coord)

# User parameters for GA
n_population = st.slider("Population Size", 50, 500, 250)
n_generations = st.slider("Number of Generations", 50, 500, 200)
crossover_per = st.slider("Crossover Probability", 0.1, 1.0, 0.8)
mutation_per = st.slider("Mutation Probability", 0.01, 1.0, 0.2)

if st.button("Run GA"):
    best_mixed_offspring = run_ga(city_names, n_population, n_generations, crossover_per, mutation_per, city_coords)

    # Calculate total distances for the final generation
    total_dist_all_individuals = [total_dist_individual(ind, city_coords) for ind in best_mixed_offspring]
    index_minimum = np.argmin(total_dist_all_individuals)
    minimum_distance = min(total_dist_all_individuals)
    st.write(f"Minimum Distance: {minimum_distance}")

    shortest_path = best_mixed_offspring[index_minimum]
    st.write(f"Shortest Path: {shortest_path}")

    # Plotting the shortest path
    x_shortest = [city_coords[city][0] for city in shortest_path]
    y_shortest = [city_coords[city][1] for city in shortest_path]

    # Complete the loop
    x_shortest.append(x_shortest[0])
    y_shortest.append(y_shortest[0])

    fig, ax = plt.subplots()
    ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
    plt.legend()

    for i in range(10):
        for j in range(i + 1, 10):
            ax.plot([x_shortest[i], x_shortest[j]], [y_shortest[i], y_shortest[j]], 'k-', alpha=0.09, linewidth=1)

    plt.title("TSP Best Route Using GA")
    plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}", fontsize=18)
    for i, txt in enumerate(shortest_path):
        ax.annotate(f"{i+1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=20)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)
