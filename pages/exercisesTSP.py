import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from itertools import permutations
import random

# Streamlit inputs
st.title("Traveling Salesperson Problem (TSP) using Genetic Algorithm")
st.write("Input city names, and then generate a random TSP solution.")

# Input form for city names
city_names_input = st.text_area("Enter city names separated by commas", "Gliwice, Cairo, Rome, Krakow, Paris")
cities_names = [city.strip() for city in city_names_input.split(",")]

# User parameters
n_population = st.number_input("Population Size", min_value=50, max_value=500, value=250)
crossover_per = st.slider("Crossover Percentage", 0.0, 1.0, 0.8)
mutation_per = st.slider("Mutation Percentage", 0.0, 1.0, 0.2)
n_generations = st.number_input("Number of Generations", min_value=50, max_value=1000, value=200)

# City Coordinates and Colors
if st.button("Generate Coordinates and Calculate"):
    # Generate random coordinates
    x = np.random.uniform(0, 20, len(cities_names))
    y = np.random.uniform(0, 20, len(cities_names))
    city_coords = dict(zip(cities_names, zip(x, y)))

    # Colors and Icons
    colors = sns.color_palette("pastel", len(cities_names))
    city_icons = ["♕", "♖", "♗", "♘", "♙", "♔", "♚", "♛", "♜", "♝"]
    city_icons = dict(zip(cities_names, city_icons[:len(cities_names)]))

    # Plot city locations
    fig, ax = plt.subplots()
    ax.grid(False)
    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons[city]
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

    # Genetic Algorithm TSP Functions
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
        for i in range(len(individual)):
            total_dist += dist_two_cities(individual[i], individual[(i + 1) % len(individual)])
        return total_dist

    def fitness_prob(population):
        total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
        max_population_cost = max(total_dist_all_individuals)
        population_fitness = max_population_cost - np.array(total_dist_all_individuals)
        return population_fitness / population_fitness.sum()

    def roulette_wheel(population, fitness_probs):
        population_fitness_probs_cumsum = fitness_probs.cumsum()
        selected_individual_index = (population_fitness_probs_cumsum < np.random.uniform(0, 1)).sum() - 1
        return population[selected_individual_index]

    def crossover(parent_1, parent_2):
        cut = random.randint(1, len(cities_names) - 2)
        offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1, offspring_2

    def mutation(offspring):
        index_1, index_2 = random.sample(range(len(offspring)), 2)
        offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
        return offspring

    def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
        population = initial_population(cities_names, n_population)
        for _ in range(n_generations):
            fitness_probs = fitness_prob(population)
            parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
            offspring = []
            for i in range(0, len(parents), 2):
                off1, off2 = crossover(parents[i], parents[(i + 1) % len(parents)])
                offspring.extend([mutation(off1) if random.random() < mutation_per else off1,
                                  mutation(off2) if random.random() < mutation_per else off2])
            population = sorted(parents + offspring, key=total_dist_individual)[:n_population]
        return min(population, key=total_dist_individual)

    # Run Genetic Algorithm
    shortest_path = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
    minimum_distance = total_dist_individual(shortest_path)

    # Display result
    st.write(f"Minimum Distance: {round(minimum_distance, 3)}")
    st.write("Shortest Path:", " ➔ ".join(shortest_path))

    # Plotting the shortest path
    x_shortest, y_shortest = zip(*[city_coords[city] for city in shortest_path] + [city_coords[shortest_path[0]]])
    fig, ax = plt.subplots()
    ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
    plt.legend()
    plt.title("TSP Best Route Using GA", fontsize=25)
    plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}\n{n_generations} Generations, {n_population} Population Size", fontsize=18, y=1.047)
    fig.set_size_inches(16, 12)
    st.pyplot(fig)
