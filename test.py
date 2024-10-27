import streamlit as st
st.set_page_config(
    page_title="Genetic Algorithm"
)

st.header("Genetic Algorithm Exercise", divider="blue")

import random

# Define constants
GENES = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

# User inputs
target = st.text_input("Enter the target string:", value="hafizah")
mut_rate = st.slider("Set mutation rate:", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
pop_size = st.number_input("Set population size:", min_value=10, max_value=1000, value=500, step=10)

# Button to start the algorithm
if st.button("Run Genetic Algorithm"):

    # Genetic Algorithm functions
    def initialize_pop(TARGET):
        population = []
        for _ in range(pop_size):
            temp = [random.choice(GENES) for _ in range(len(TARGET))]
            population.append(temp)
        return population

    def fitness_cal(TARGET, chromo_from_pop):
        difference = sum(1 for tar_char, chromo_char in zip(TARGET, chromo_from_pop) if tar_char != chromo_char)
        return [chromo_from_pop, difference]

    def selection(population):
        sorted_pop = sorted(population, key=lambda x: x[1])
        return sorted_pop[:int(0.5 * pop_size)]

    def crossover(selected_chromo, CHROMO_LEN, population):
        offspring_cross = []
        for _ in range(pop_size):
            parent1 = random.choice(selected_chromo)[0]
            parent2 = random.choice(population[:int(pop_size * 0.5)])[0]
            crossover_point = random.randint(1, CHROMO_LEN - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            offspring_cross.append(child)
        return offspring_cross

    def mutate(offspring):
        mutated_offspring = []
        for arr in offspring:
            mutated = [random.choice(GENES) if random.random() < mut_rate else char for char in arr]
            mutated_offspring.append(mutated)
        return mutated_offspring

    def replace(new_gen, population):
        for i in range(len(population)):
            if population[i][1] > new_gen[i][1]:
                population[i] = new_gen[i]
        return population

    # Main function
    def main():
        initial_population = initialize_pop(target)
        found = False
        population = [fitness_cal(target, chromo) for chromo in initial_population]
        generation = 1

        while not found:
            selected = selection(population)
            crossovered = crossover(selected, len(target), population)
            mutated = mutate(crossovered)
            new_gen = [fitness_cal(target, chromo) for chromo in mutated]
            population = replace(new_gen, population)

            # Output
            st.write(f"Generation: {generation}, String: {''.join(population[0][0])}, Fitness: {population[0][1]}")

            if population[0][1] == 0:
                st.write("Target found!")
                st.write(f"String: {''.join(population[0][0])}, Generation: {generation}, Fitness: {population[0][1]}")
                found = True
            generation += 1

    # Run the main function
    main()
