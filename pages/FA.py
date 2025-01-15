import pandas as pd
import random
import numpy as np
#import matplotlib.pyplot as plt
import streamlit as st

# Function to load uploaded CSV file
def load_csv(file):
    return pd.read_csv(file)

# Streamlit UI
st.title("Optimized Exam Timetable Generation using Firefly Algorithm")

# File upload for schedule, courses, timeslots, and classrooms
st.subheader("Upload Schedule CSV")
schedule_file = st.file_uploader("Upload Schedule CSV", type=["csv"])

st.subheader("Upload Courses CSV")
courses_file = st.file_uploader("Upload Courses CSV", type=["csv"])

st.subheader("Upload Timeslots CSV")
timeslots_file = st.file_uploader("Upload Timeslots CSV", type=["csv"])

st.subheader("Upload Classrooms CSV")
classrooms_file = st.file_uploader("Upload Classrooms CSV", type=["csv"])

# Process the data if all files are uploaded
if schedule_file and courses_file and timeslots_file and classrooms_file:
    schedule_df = load_csv(schedule_file)
    courses_df = load_csv(courses_file)
    timeslots_df = load_csv(timeslots_file)
    classrooms_df = load_csv(classrooms_file)

    # Merge datasets
    exam_timetable = schedule_df.merge(courses_df, on='course_id').merge(timeslots_df, on='timeslot_id')

    # Select and rename relevant columns
    exam_timetable = exam_timetable[['course_name', 'day', 'start_time', 'end_time', 'classroom_id']]
    exam_timetable.columns = ['Subject', 'Date', 'Start Time', 'End Time', 'Venue']

    # Display the exam timetable
    st.write("Generated Exam Timetable:")
    st.dataframe(exam_timetable)

    # Parameters for Firefly Algorithm
    num_fireflies = 10
    num_generations = 50
    alpha = 0.2  # Randomness weight
    beta0 = 1.0  # Attraction constant
    gamma = 1.0  # Absorption coefficient

    # Initialize fireflies with random assignments of courses to timeslots and classrooms
    fireflies = []
    for _ in range(num_fireflies):
        firefly = []
        for idx, row in exam_timetable.iterrows():
            timeslot = random.choice(timeslots_df['timeslot_id'])
            classroom = random.choice(classrooms_df['classroom_id'])
            firefly.append((row['Subject'], timeslot, classroom))
        fireflies.append(firefly)

    # Fitness function
    def fitness(firefly):
        conflicts = 0
        timeslot_map = {}
        classroom_map = {}

        for course, timeslot, classroom in firefly:
            # Check for timeslot conflicts
            if timeslot in timeslot_map:
                conflicts += 1
            timeslot_map[timeslot] = timeslot_map.get(timeslot, 0) + 1

            # Check for classroom conflicts
            if classroom in classroom_map:
                conflicts += 1
            classroom_map[classroom] = classroom_map.get(classroom, 0) + 1

        return conflicts

    # Calculate attractiveness between fireflies
    def attractiveness(distance):
        return beta0 * np.exp(-gamma * distance**2)

    # Euclidean distance function for fireflies
    def euclidean_distance(firefly1, firefly2):
        distance = 0
        for i in range(len(firefly1)):
            distance += (timeslots_df['timeslot_id'].tolist().index(firefly1[i][1]) - \
                        timeslots_df['timeslot_id'].tolist().index(firefly2[i][1]))**2
            distance += (classrooms_df['classroom_id'].tolist().index(firefly1[i][2]) - \
                        classrooms_df['classroom_id'].tolist().index(firefly2[i][2]))**2
        return np.sqrt(distance)

    # Firefly Algorithm main loop
    fitness_trends = []
    for generation in range(num_generations):
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if fitness(fireflies[j]) < fitness(fireflies[i]):
                    dist = euclidean_distance(fireflies[i], fireflies[j])
                    beta = attractiveness(dist)

                    # Move firefly i towards firefly j
                    for k in range(len(fireflies[i])):
                        current_timeslot = timeslots_df['timeslot_id'].tolist().index(fireflies[i][k][1])
                        current_classroom = classrooms_df['classroom_id'].tolist().index(fireflies[i][k][2])

                        brighter_timeslot = timeslots_df['timeslot_id'].tolist().index(fireflies[j][k][1])
                        brighter_classroom = classrooms_df['classroom_id'].tolist().index(fireflies[j][k][2])

                        new_timeslot = int(current_timeslot + beta * (brighter_timeslot - current_timeslot) + \
                                          alpha * (random.random() - 0.5))
                        new_classroom = int(current_classroom + beta * (brighter_classroom - current_classroom) + \
                                           alpha * (random.random() - 0.5))

                        new_timeslot = np.clip(new_timeslot, 0, len(timeslots_df) - 1)
                        new_classroom = np.clip(new_classroom, 0, len(classrooms_df) - 1)

                        fireflies[i][k] = (fireflies[i][k][0], \
                                           timeslots_df['timeslot_id'].iloc[new_timeslot], \
                                           classrooms_df['classroom_id'].iloc[new_classroom])

        # Track the best solution
        best_fitness = min([fitness(f) for f in fireflies])
        fitness_trends.append(best_fitness)
        st.write(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    # Output the optimal timetable
    st.subheader("Optimal Exam Timetable")
    best_firefly = min(fireflies, key=fitness)
    optimal_timetable = []
    for course, timeslot, classroom in best_firefly:
        optimal_timetable.append({
            'Subject': course,
            'Timeslot': timeslot,
            'Classroom': classroom
        })

    optimal_timetable_df = pd.DataFrame(optimal_timetable)
    st.write("Optimal Timetable Data:")
    st.dataframe(optimal_timetable_df)

    # Merge with timeslots_df and classrooms_df to get details
    final_timetable = pd.merge(optimal_timetable_df, timeslots_df, left_on='Timeslot', right_on='timeslot_id')
    final_timetable = pd.merge(final_timetable, classrooms_df, left_on='Classroom', right_on='classroom_id')

    # Select and rename columns for the final table
    final_timetable = final_timetable[['Subject', 'day', 'start_time', 'end_time', 'building_name', 'room_number']]
    final_timetable.columns = ['Subject', 'Day', 'Start Time', 'End Time', 'Building', 'Room']

    # Display the final exam timetable in a table format
    st.subheader("Final Exam Timetable")
    st.dataframe(final_timetable)

    # Plot the fitness trend over generations
    st.subheader("Fitness Trend Over Generations")
    plt.figure(figsize=(12, 6))
    plt.plot(fitness_trends, label="Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Fitness Trends Over Generations")
    plt.legend()
    st.pyplot()

else:
    st.write("Please upload all the required CSV files.")

