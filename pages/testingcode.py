import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import permutations
import seaborn as sns

# Create city icons and color palette
city_icons = {f"City {i+1}": f"♕" for i in range(10)}
colors = sns.color_palette("pastel", 10)

# Function to calculate the Euclidean distance between two cities
def dist_two_cities(city_1, city_2, city_coords):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt((city_1_coords[0] - city_2_coords[0]) ** 2 + (city_1_coords[1] - city_2_coords[1]) ** 2)

# Total distance for an individual path (sum of distances between consecutive cities)
def total_distance(individual, city_coords):
    total_dist = 0
    for i in range(len(individual) - 1):
        total_dist += dist_two_cities(individual[i], individual[i+1], city_coords)
    total_dist += dist_two_cities(individual[-1], individual[0], city_coords)  # return to the starting city
    return total_dist

# Initialize population with random permutations of cities
def initial_population(cities, n_population=250):
    population = []
    all_permutations = list(permutations(cities))
    random_indices = random.sample(range(len(all_permutations)), n_population)
    for idx in random_indices:
        population.append(list(all_permutations[idx]))
    return population

# Roulette Wheel Selection (fitness proportionate selection)
def roulette_wheel(population, fitness_probs):
    cumsum_fitness = np.cumsum(fitness_probs)
    pick = random.random()
    for i, value in enumerate(cumsum_fitness):
        if value > pick:
            return population[i]
    return population[-1]  # Return the last individual if none is picked

# Crossover function (two-point crossover)
def crossover(parent_1, parent_2):
    size = len(parent_1)
    cut1, cut2 = sorted(random.sample(range(size), 2))
    offspring_1 = parent_1[:cut1] + [city for city in parent_2 if city not in parent_1[:cut1]]
    offspring_2 = parent_2[:cut1] + [city for city in parent_1 if city not in parent_2[:cut1]]
    return offspring_1, offspring_2

# Mutation function (swap mutation)
def mutation(offspring):
    idx1, idx2 = random.sample(range(len(offspring)), 2)
    offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]
    return offspring

# Calculate fitness probabilities for a population
def fitness_prob(population, city_coords):
    distances = [total_distance(ind, city_coords) for ind in population]
    max_distance = max(distances)
    fitness = [max_distance - dist for dist in distances]
    total_fitness = sum(fitness)
    return np.array(fitness) / total_fitness

# Main GA function
def run_ga(cities, city_coords, n_population, n_generations, crossover_rate, mutation_rate):
    population = initial_population(cities, n_population)
    for generation in range(n_generations):
        fitness_probs = fitness_prob(population, city_coords)

        # Selection: select parents based on fitness
        parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_rate * n_population))]
        
        # Crossover: generate offspring
        offspring = []
        for i in range(0, len(parents), 2):
            parent_1, parent_2 = parents[i], parents[i+1]
            off1, off2 = crossover(parent_1, parent_2)
            
            # Mutation: occasionally mutate offspring
            if random.random() < mutation_rate:
                off1 = mutation(off1)
            if random.random() < mutation_rate:
                off2 = mutation(off2)
                
            offspring.append(off1)
            offspring.append(off2)
        
        # Combine parents and offspring
        population = parents + offspring
        population = sorted(population, key=lambda ind: total_distance(ind, city_coords))
        population = population[:n_population]  # Keep the best n_population individuals

    # Return the best solution found
    best_individual = population[0]
    best_distance = total_distance(best_individual, city_coords)
    return best_individual, best_distance

# Streamlit UI elements for user input
st.title("TSP Genetic Algorithm Solver")

# Collect city names and coordinates from user input
city_names = []
city_coords = {}
for i in range(10):
    city_name = st.text_input(f"City {i+1}", f"City {i+1}")
    x_coord = st.number_input(f"X Coordinate", min_value=1, max_value=10, step=1, key=f"x{i}")
    y_coord = st.number_input(f"Y Coordinate", min_value=1, max_value=10, step=1, key=f"y{i}")
    city_names.append(city_name)
    city_coords[city_name] = (x_coord, y_coord)

# Hyperparameters for the genetic algorithm
#n_population = st.slider("Population Size", 50, 500, 100)
#n_generations = st.slider("Number of Generations", 50, 500, 200)
#crossover_rate = st.slider("Crossover Rate", 0.1, 1.0, 0.8)
#mutation_rate = st.slider("Mutation Rate", 0.01, 1.0, 0.2)
# Define default settings for the genetic algorithm
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Run GA when user clicks the button
if st.button("Run GA"):
    best_individual, best_distance = run_ga(city_names, city_coords, n_population, n_generations, crossover_rate, mutation_rate)
    
    # Display results
    st.write(f"Best Distance: {best_distance:.2f}")
    st.write("Best Path:", best_individual)
    
    # Plot the path
    x_values = [city_coords[city][0] for city in best_individual]
    y_values = [city_coords[city][1] for city in best_individual]
    
    # To make the path loop (start == end)
    x_values.append(x_values[0])
    y_values.append(y_values[0])

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, 'r-o', label='TSP Route', linewidth=2)
    ax.set_title(f"TSP Path with Total Distance {best_distance:.2f}")
    for i, city in enumerate(best_individual):
        ax.annotate(f"{city}", (x_values[i], y_values[i]), fontsize=10, ha='right')

    st.pyplot(fig)
