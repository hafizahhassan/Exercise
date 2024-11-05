import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import streamlit as st
from itertools import permutations
import pandas as pd

# Pastel Palette for Colors
colors = sns.color_palette("pastel", 10)

# City Icons
city_icons = {
    "♕", "♖", "♗", "♘", "♙", "♔", "♚", "♛", "♜", "♝"
}

def get_city_input():
    # Getting 10 city names and coordinates (x, y) from the user
    city_names = []
    city_coords = {}

    for i in range(10):
        city_name = st.text_input(f"Enter name for city {i + 1}:", f"City_{i + 1}")
        city_names.append(city_name)

        x_coord = st.number_input(f"Enter x-coordinate for {city_name}:", min_value=-100, max_value=100, value=random.randint(-50, 50))
        y_coord = st.number_input(f"Enter y-coordinate for {city_name}:", min_value=-100, max_value=100, value=random.randint(-50, 50))
        
        city_coords[city_name] = (x_coord, y_coord)
    
    return city_names, city_coords

def plot_city_map(city_names, city_coords):
    fig, ax = plt.subplots()

    ax.grid(False)  # Grid off

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = list(city_icons)[i]
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')

        # Connect cities with opaque lines
        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

def initial_population(cities_list, n_population=250):
    """
    Generating initial population of cities randomly selected from all possible permutations
    of the given cities.
    """
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)

    for i in random_ids:
        population_perms.append(list(possible_perms[i]))

    return population_perms

def dist_two_cities(city_1, city_2, city_coords):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

def total_dist_individual(individual, city_coords):
    total_dist = 0
    for i in range(len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0], city_coords)
        else:
            total_dist += dist_two_cities(individual[i], individual[i + 1], city_coords)
    return total_dist

def fitness_prob(population, city_coords):
    total_dist_all_individuals = [total_dist_individual(individual, city_coords) for individual in population]
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
    n_cities_cut = len(parent_1) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1, offspring_2 = [], []

    offspring_1 = parent_1[:cut]
    offspring_1 += [city for city in parent_2 if city not in offspring_1]

    offspring_2 = parent_2[:cut]
    offspring_2 += [city for city in parent_1 if city not in offspring_2]

    return offspring_1, offspring_2

def mutation(offspring):
    n_cities_cut = len(offspring) - 1
    index_1, index_2 = round(random.uniform(0, n_cities_cut)), round(random.uniform(0, n_cities_cut))

    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

def run_ga(city_names, n_population, n_generations, crossover_per, mutation_per, city_coords):
    population = initial_population(city_names, n_population)
    fitness_probs = fitness_prob(population, city_coords)

    parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]

    offspring_list = []
    for i in range(0, len(parents_list), 2):
        offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i + 1])

        mutate_threashold = random.random()
        if mutate_threashold > (1 - mutation_per):
            offspring_1 = mutation(offspring_1)

        mutate_threashold = random.random()
        if mutate_threashold > (1 - mutation_per):
            offspring_2 = mutation(offspring_2)

        offspring_list.append(offspring_1)
        offspring_list.append(offspring_2)

    mixed_offspring = parents_list + offspring_list

    fitness_probs = fitness_prob(mixed_offspring, city_coords)
    sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
    best_fitness_indices = sorted_fitness_indices[:n_population]
    best_mixed_offspring = [mixed_offspring[i] for i in best_fitness_indices]

    for i in range(n_generations):
        fitness_probs = fitness_prob(best_mixed_offspring, city_coords)
        parents_list = [roulette_wheel(best_mixed_offspring, fitness_probs) for _ in range(int(crossover_per * n_population))]

        offspring_list = []
        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i + 1])

            mutate_threashold = random.random()
            if mutate_threashold > (1 - mutation_per):
                offspring_1 = mutation(offspring_1)

            mutate_threashold = random.random()
            if mutate_threashold > (1 - mutation_per):
                offspring_2 = mutation(offspring_2)

            offspring_list.append(offspring_1)
            offspring_list.append(offspring_2)

        mixed_offspring = parents_list + offspring_list
        fitness_probs = fitness_prob(mixed_offspring, city_coords)
        sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
        best_fitness_indices = sorted_fitness_indices[:int(0.8 * n_population)]

        best_mixed_offspring = [mixed_offspring[i] for i in best_fitness_indices]

        old_population_indices = [random.randint(0, n_population - 1) for _ in range(int(0.2 * n_population))]
        for i in old_population_indices:
            best_mixed_offspring.append(population[i])

        random.shuffle(best_mixed_offspring)

    return best_mixed_offspring

# Streamlit UI
st.title("Traveling Salesman Problem (TSP) with Genetic Algorithm")

# User input for city names and coordinates
city_names, city_coords = get_city_input()

# Plot city map
plot_city_map(city_names, city_coords)

# Run the genetic algorithm
best_mixed_offspring = run_ga(city_names, 250, 200, 0.8, 0.2, city_coords)

# Calculate the total distances of all individuals
total_dist_all_individuals = [total_dist_individual(individual, city_coords) for individual in best_mixed_offspring]
index_minimum = np.argmin(total_dist_all_individuals)
minimum_distance = min(total_dist_all_individuals)

# Display results
st.write(f"Minimum Distance: {minimum_distance}")
st.write(f"Shortest Path: {best_mixed_offspring[index_minimum]}")

# Plot the best route
shortest_path = best_mixed_offspring[index_minimum]
x_shortest = [city_coords[city][0] for city in shortest_path] + [city_coords[shortest_path[0]][0]]
y_shortest = [city_coords[city][1] for city in shortest_path] + [city_coords[shortest_path[0]][1]]

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i in range(len(x_shortest) - 1):
    ax.plot([x_shortest[i], x_shortest[i + 1]], [y_shortest[i], y_shortest[i + 1]], 'k-', alpha=0.09, linewidth=1)

plt.title(f"TSP Best Route Using GA\nTotal Distance: {round(minimum_distance, 3)}", fontsize=18)

for i, city in enumerate(shortest_path):
    ax.annotate(f"{i+1}- {city}", (x_shortest[i], y_shortest[i]), fontsize=12)

fig.set_size_inches(16, 12)
st.pyplot(fig)
