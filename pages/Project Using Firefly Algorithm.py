import streamlit as st
import csv
import pandas as pd
import random
import numpy as np

##################################### CSS FOR DIVIDER ################################################################
# CSS for shimmering divider effect
# CSS for Button
st.markdown("""
<style>

@keyframes shimmer {
  0% {
    background-position: -1000px 0;
  }
  100% {
    background-position: 1000px 0;
  }
}

.shimmer-divider {
  height: 3px;
  background: linear-gradient(to right, #f2f2f2 0%, #FFD700 50%, #f2f2f2 100%);
  background-size: 1000px 100%;
  animation: shimmer 20s infinite linear;
  margin: 20px 0;
}

.stFormSubmitButton > button {
    background-color: #02ab21;
    color: white;
    font-size: 20px;
    width: 100%;
}
.stFormSubmitButton > button:hover {
    background-color: #027a18;
    color: white;
}

</style>
""", unsafe_allow_html=True)

#####################################################################################################################

st.title("OPTIMIZATION EXAM SCHEDULING USING FIREFLY ALGORITHM")

# Add a shimmering divider
st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

##################################### INSERT DATASET ###################################################################

st.subheader("U P L O A D &nbsp;&nbsp; D A T A")

def clear_output():
  # Clear all the file uploader widgets
  st.session_state.schedule_file = None
  st.session_state.courses_file = None
  st.session_state.timeslots_file = None
  st.session_state.classrooms_file = None

# Initialize session state for file uploaders if not already done
if 'schedule_file' not in st.session_state:
  st.session_state.schedule_file = None
if 'courses_file' not in st.session_state:
  st.session_state.courses_file = None
if 'timeslots_file' not in st.session_state:
  st.session_state.timeslots_file = None
if 'classrooms_file' not in st.session_state:
  st.session_state.classrooms_file = None

with st.form("Firefly_Algorithm"):
  # Upload the Dataset
  # File upload for schedule, courses, timeslots, and classrooms
  schedule_file = st.file_uploader("Upload Schedule CSV", type=["csv"])
  courses_file = st.file_uploader("Upload Courses CSV", type=["csv"])
  timeslots_file = st.file_uploader("Upload Timeslots CSV", type=["csv"])
  classrooms_file = st.file_uploader("Upload Classrooms CSV", type=["csv"])

  col1, col2, col3 = st.columns(3)
  with col2:
    Submit_Button = st.form_submit_button("Submit")
    Clear_Button = st.form_submit_button("Clear")

##################################### OUTPUT ###################################################################

# Add a shimmering divider
st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

st.subheader("O U T P U T")

if Submit_Button:
  if schedule_file and courses_file and timeslots_file and classrooms_file:
        schedule = load_csv(schedule_file)
        courses = load_csv(courses_file)
        timeslots = load_csv(timeslots_file)
        classrooms = load_csv(classrooms_file)
    
        # Parameters for Firefly Algorithm
        num_fireflies = 10
        num_iterations = 100
        gamma = 1.0  # Light absorption coefficient
        beta0 = 2.0  # Attraction coefficient base value
        alpha = 0.2  # Randomization coefficient
      
        # Initialize fireflies
        fireflies = []
        for _ in range(num_fireflies):
            firefly = [(random.choice(courses), random.choice(instructors),
                        random.choice(classrooms), random.choice(timeslots))
                       for _ in range(len(courses))]
            fireflies.append(firefly)
        
        # Fitness function
        def fitness(firefly):
            instructor_conflicts = len(firefly) - len(set((c[1], c[3]) for c in firefly))
            room_conflicts = len(firefly) - len(set((c[2], c[3]) for c in firefly))
            return instructor_conflicts + room_conflicts
        
        # Move firefly i towards firefly j
        def move_firefly(firefly_i, firefly_j, beta):
            new_firefly = firefly_i[:]
            for k in range(len(firefly_i)):
                if random.random() < beta:
                    new_firefly[k] = firefly_j[k]
            return new_firefly
    
        # Main Firefly Algorithm
        for iteration in range(num_iterations):
            fitness_values = [fitness(f) for f in fireflies]
            for i in range(num_fireflies):
                for j in range(num_fireflies):
                    if fitness_values[j] < fitness_values[i]:  # Brighter fireflies attract dimmer ones
                        distance = np.linalg.norm(
                            [courses.index(fireflies[i][k][0]) - courses.index(fireflies[j][k][0]) for k in range(len(courses))]
                        )
                        beta = beta0 * np.exp(-gamma * distance**2)
                        fireflies[i] = move_firefly(fireflies[i], fireflies[j], beta)
                        # Random perturbation
                        if random.random() < alpha:
                            fireflies[i] = [(random.choice(courses), random.choice(instructors),
                                             random.choice(classrooms), random.choice(timeslots))
                                            for _ in range(len(courses))]
        
            # Update global best
            best_index = np.argmin(fitness_values)
            best_firefly = fireflies[best_index]
            best_fitness = fitness_values[best_index]
        
            st.write(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")
        
        # Output the best schedule
        st.write("Best Schedule :")
        for course in best_firefly:
            st.dataframe("Course {course[0]} - Instructor {course[1]} - Room {course[2]} - Timeslot {course[3]}")
  else:
    st.write("Please upload all the required CSV files.")

if Clear_Button:
  clear_output()
  st.rerun()







