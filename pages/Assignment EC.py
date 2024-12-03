import streamlit as st
import csv
import requests # Import the requests module
import pandas as pd
import numpy as np

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}

    try:
      response = requests.get(file_path)
      response.raise_for_status()

      # Decode the content as text and split into lines
      lines = response.text.splitlines()
      reader = csv.reader(lines)
      
      # Skip the header
      header = next(reader)

      for row in reader:
          program = row[0]
          ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
          program_ratings[program] = ratings
      return program_ratings

    except requests.exceptions.RequestException as e:
      st.write(f"Error fetching or processing CSV data: {e}")
      return None  # or raise the exception if you prefer

# Path to the CSV file
file_path = 'https://raw.githubusercontent.com/hafizahhassan/Exercise/refs/heads/main/pages/program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

# Print the result (you can also return or process it further)
for program, ratings in program_ratings_dict.items():
    st.write(f"'{program}': {ratings},")


import random

##################################### DEFINING PARAMETERS AND DATASET ################################################################
# Sample rating programs dataset for each time slot.
ratings = program_ratings_dict

GEN = 100
POP = 50
CO_R = 0.8
MUT_R = 0.2
EL_S = 2

all_programs = list(ratings.keys()) # all programs
all_time_slots = list(range(6, 24)) # time slots

######################################### DEFINING FUNCTIONS ########################################################################
# Defining fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# Initializing the population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules

# Selection
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule

# Calling the pop func.
all_possible_schedules = initialize_pop(all_programs, all_time_slots)

# Callin the schedule func.
best_schedule = finding_best_schedule(all_possible_schedules)


############################################# GENETIC ALGORITHM #############################################################################

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutating
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Calling the fitness func.
def evaluate_fitness(schedule):
    return fitness_function(schedule)

# Genetic algorithms with parameters
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):

    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitsm
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

##################################################### RESULTS ###################################################################################

# Brute Force
initial_best_schedule = finding_best_schedule(all_possible_schedules)

rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
genetic_schedule = genetic_algorithm(initial_best_schedule, generations=GEN, population_size=POP, elitism_size=EL_S)

final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

# Create a DataFrame for the schedule
schedule_data = {
    "Time Slot": [f"{all_time_slots[time_slot]:02d}:00" for time_slot in range(len(final_schedule))],
    "Program": final_schedule
}
schedule_df = pd.DataFrame(schedule_data)

# Function to color the background based on program number
def color_background(val):
    color = f'background-color: #{hash(val) % 0xFFFFFF:06x}'
    return color

# Apply styling to the DataFrame
styled_df = schedule_df.style.applymap(color_background, subset=['Program'])

# Display the table
st.write("Final Optimal Schedule:")
st.dataframe(styled_df, use_container_width=True, height=668)

st.write("Total Ratings:", fitness_function(final_schedule))


