import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# User input for city names and coordinates
st.title("Traveling Salesman Problem with Genetic Algorithm")

# Number of cities
num_cities = st.number_input("Enter the number of cities:", min_value=2, max_value=20, value=10)

# Dynamic input for city names and coordinates
city_names = []
x = []
y = []

for i in range(num_cities):
    city_name = st.text_input(f"Enter name of city {i+1}:", f"City{i+1}")
    city_x = st.number_input(f"Enter x-coordinate for {city_name}:", value=float(i*2))
    city_y = st.number_input(f"Enter y-coordinate for {city_name}:", value=float(i*2 + 1))
    
    city_names.append(city_name)
    x.append(city_x)
    y.append(city_y)

# Generate the city_coords dictionary
city_coords = dict(zip(city_names, zip(x, y)))

# Genetic algorithm parameters
n_population = st.slider("Population Size", min_value=50, max_value=500, value=250)
crossover_per = st.slider("Crossover Percentage", min_value=0.0, max_value=1.0, value=0.8)
mutation_per = st.slider("Mutation Percentage", min_value=0.0, max_value=1.0, value=0.2)
n_generations = st.slider("Number of Generations", min_value=50, max_value=1000, value=200)

# Plot color palette and city icons
colors = sns.color_palette("pastel", len(city_names))
city_icons = ["♕", "♖", "♗", "♘", "♙", "♔", "♚", "♛", "♜", "♝"]
city_icons = {city_names[i]: city_icons[i % len(city_icons)] for i in range(len(city_names))}

# Plot cities and connections
fig, ax = plt.subplots()
ax.grid(False)

for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    icon = city_icons[city]
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')
    
    # Connect cities with opaque lines
    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

fig.set_size_inches(16, 12)
st.pyplot(fig)

# Helper functions
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
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

def total_dist_individual(individual):
    total_dist = 0
    for i in range(0, len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1])
    return total_dist

def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_sum = sum(population_fitness)
    return population_fitness / population_fitness_sum

def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1)
    selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
    return population[selected_individual_index]

def crossover(parent_1, parent_2):
    cut = round(random.uniform(1, len(city_names) - 1))
    offspring_1 = parent_1[0:cut] + [city for city in parent_2 if city not in parent_1[0:cut]]
    offspring_2 = parent_2[0:cut] + [city for city in parent_1 if city not in parent_2[0:cut]]
    return offspring_1, offspring_2

def mutation(offspring):
    index_1, index_2 = random.sample(range(len(city_names)), 2)
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    best_mixed_offspring = population

    for _ in range(n_generations):
        fitness_probs = fitness_prob(best_mixed_offspring)
        parents_list = [roulette_wheel(best_mixed_offspring, fitness_probs) for _ in range(int(crossover_per * n_population))]

        offspring_list = []
        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[(i + 1) % len(parents_list)])
            if random.random() < mutation_per:
                offspring_1 = mutation(offspring_1)
            if random.random() < mutation_per:
                offspring_2 = mutation(offspring_2)
            offspring_list += [offspring_1, offspring_2]

        mixed_offspring = parents_list + offspring_list
        fitness_probs = fitness_prob(mixed_offspring)
        sorted_indices = np.argsort(fitness_probs)[::-1]
        best_mixed_offspring = [mixed_offspring[i] for i in sorted_indices[:n_population]]

    return best_mixed_offspring

best_mixed_offspring = run_ga(city_names, n_population, n_generations, crossover_per, mutation_per)

# Calculate the minimum distance and plot the path
total_dist_all_individuals = [total_dist_individual(ind) for ind in best_mixed_offspring]
minimum_distance = min(total_dist_all_individuals)
st.write("Minimum Distance:", minimum_distance)

shortest_path = best_mixed_offspring[np.argmin(total_dist_all_individuals)]

x_shortest, y_shortest = zip(*[city_coords[city] for city in shortest_path])
x_shortest += (x_shortest[0],)
y_shortest += (y_shortest[0],)

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i in range(len(x)):
    for j in range(i + 1, len(x)):
        ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

plt.title("TSP Best Route Using GA", fontsize=25)
plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}", fontsize=18, y=1.047)

for i, txt in enumerate(shortest_path):
    ax.annotate(f"{i+1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=20)

fig.set_size_inches(16, 12)
st.pyplot(fig)

