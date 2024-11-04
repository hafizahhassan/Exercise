import numpy as np
import random

# Sample distance matrix (could be replaced with real user input)
def create_distance_matrix(num_cities):
    return np.random.randint(10, 100, size=(num_cities, num_cities))

# Fitness function to calculate total route distance
def calculate_route_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)) + distance_matrix[route[-1], route[0]]

# Generate initial population
def generate_initial_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

# Selection function based on fitness
def select_parents(population, fitness_scores, num_parents):
    parents = np.array([x for _, x in sorted(zip(fitness_scores, population))])[:num_parents]
    return parents.tolist()

# Crossover between two parents
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = gene
    return child

# Mutation by swapping two cities
def mutate(route, mutation_rate=0.01):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route

# Main GA function
def genetic_algorithm(num_cities, pop_size=100, num_generations=500, mutation_rate=0.01, num_parents=20):
    distance_matrix = create_distance_matrix(num_cities)
    population = generate_initial_population(pop_size, num_cities)
    
    for generation in range(num_generations):
        # Calculate fitness scores
        fitness_scores = [calculate_route_distance(route, distance_matrix) for route in population]
        
        # Select parents
        parents = select_parents(population, fitness_scores, num_parents)
        
        # Generate offspring through crossover
        offspring = []
        while len(offspring) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            offspring.append(child)
        
        population = offspring
    
    # Select the best route from the final population
    fitness_scores = [calculate_route_distance(route, distance_matrix) for route in population]
    best_route = population[np.argmin(fitness_scores)]
    best_distance = min(fitness_scores)
    
    return best_route, best_distance

# Example Usage
num_cities = 10
best_route, best_distance = genetic_algorithm(num_cities)
print("Best Route:", best_route)
print("Best Distance:", best_distance)
