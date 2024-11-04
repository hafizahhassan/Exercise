import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import permutations
import seaborn as sns

# Define default settings for the genetic algorithm
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Pastel color palette
colors = sns.color_palette("pastel", 10)

# City Icons
city_icons = {
    1: "♕", 2: "♖", 3: "♗", 4: "♘", 5: "♙",
    6: "♔", 7: "♚", 8: "♛", 9: "♜", 10: "♝"
}

st.title("Traveling Salesperson Problem (TSP) with Genetic Algorithm")

# Create input form for cities
with st.form("city_input_form"):
    city_coords = {}
    for i in range(1, 11):
        col1, col2, col3 = st.columns(3)
        city_name = col1.text_input(f"City {i} Name", f"City_{i}")
        x_coord = col2.number_input(f"X Coordinate for {city_name}", min_value=1, max_value=10, step=1, key=f"x{i}")
        y_coord = col3.number_input(f"Y Coordinate for {city_name}", min_value=1, max_value=10, step=1, key=f"y{i}")
        city_coords[city_name] = (x_coord, y_coord)
    
    submit_button = st.form_submit_button(label="Submit")

# If the form is submitted, run the genetic algorithm
if submit_button:
    cities_names = list(city_coords.keys())
    
    # Define distance calculation functions
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
        total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
        max_population_cost = max(total_dist_all_individuals)
        population_fitness = max_population_cost - np.array(total_dist_all_individuals)
        population_fitness_sum = np.sum(population_fitness)
        population_fitness_probs = population_fitness / population_fitness_sum
        return population_fitness_probs

    def initial_population(cities_list, n_population=250):
        possible_perms = list(permutations(cities_list))
        population_perms = random.sample(possible_perms, n_population)
        return population_perms

    def roulette_wheel(population, fitness_probs):
        population_fitness_probs_cumsum = fitness_probs.cumsum()
        selected_index = np.searchsorted(population_fitness_probs_cumsum, np.random.rand())
        return population[selected_index]

    def crossover(parent_1, parent_2):
        cut = random.randint(1, len(cities_names) - 1)
        offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1, offspring_2

    def mutation(offspring):
        index_1, index_2 = random.sample(range(len(cities_names)), 2)
        offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
        return offspring

    def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
        population = initial_population(cities_names, n_population)
        for _ in range(n_generations):
            fitness_probs = fitness_prob(population)
            
            # Select parents
            num_parents = int(crossover_per * n_population)
            if num_parents % 2 != 0:
                num_parents -= 1  # Ensure an even number of parents
            parents = [roulette_wheel(population, fitness_probs) for _ in range(num_parents)]
            
            offspring = []
            for i in range(0, len(parents), 2):
                parent_1, parent_2 = parents[i], parents[i + 1]
                offspring_1, offspring_2 = crossover(parent_1, parent_2)
                
                # Apply mutation
                if random.random() < mutation_per:
                    offspring_1 = mutation(offspring_1)
                if random.random() < mutation_per:
                    offspring_2 = mutation(offspring_2)
                
                offspring.extend([offspring_1, offspring_2])
            
            # Update the population and keep the best individuals
            population = parents + offspring
            population = sorted(population, key=total_dist_individual)[:n_population]
        
    # Return the best route found
    return min(population, key=total_dist_individual)

    # Run GA and get the best route
    best_route = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
    minimum_distance = total_dist_individual(best_route)

    # Output results
    st.write(f"Minimum Distance: {minimum_distance}")
    st.write("Best Route:", " → ".join(best_route))

    # Visualization
    x_vals = [city_coords[city][0] for city in best_route] + [city_coords[best_route[0]][0]]
    y_vals = [city_coords[city][1] for city in best_route] + [city_coords[best_route[0]][1]]

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, '--go', label='Best Route', linewidth=2.5)
    plt.legend()

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons[i+1]
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=30, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                    textcoords='offset points')
    
    fig.set_size_inches(16, 12)
    st.pyplot(fig)
