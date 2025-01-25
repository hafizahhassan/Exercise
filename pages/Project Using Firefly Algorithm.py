import streamlit as st
import csv
import pandas as pd
import random
import numpy as np

# Function to load uploaded CSV file
def load_csv(file):
    return pd.read_csv(file)

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

def clear_output():
  # Clear all the file uploader widgets
  st.session_state.schedule_file = None
  st.session_state.courses_file = None
  st.session_state.instructors_file = None    
  st.session_state.timeslots_file = None
  st.session_state.classrooms_file = None

# Initialize session state for file uploaders if not already done
if 'schedule_file' not in st.session_state:
  st.session_state.schedule_file = None
if 'courses_file' not in st.session_state:
  st.session_state.courses_file = None
if 'instructors_file' not in st.session_state:
  st.session_state.instructors_file = None
if 'timeslots_file' not in st.session_state:
  st.session_state.timeslots_file = None
if 'classrooms_file' not in st.session_state:
  st.session_state.classrooms_file = None

st.subheader("U P L O A D &nbsp;&nbsp; D A T A")

with st.form("Firefly_Algorithm"):
  # Upload the Dataset
  # File upload for schedule, courses, timeslots, and classrooms
  schedule_file = st.file_uploader("Upload Schedule CSV", type=["csv"])
  courses_file = st.file_uploader("Upload Courses CSV", type=["csv"])
  instructors_file = st.file_uploader("Upload Instructors CSV", type=["csv"])  
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
  if schedule_file and courses_file and instructors_file and timeslots_file and classrooms_file:
        schedule = load_csv(schedule_file)
        courses = load_csv(courses_file)
        instructors = load_csv(instructors_file)
        timeslots = load_csv(timeslots_file)
        classrooms = load_csv(classrooms_file)

        # Inspect the DataFrame
        #st.write("Courses DataFrame:")
        #st.write(courses.head())
        #st.write("Column Names in Courses:")
        #st.write(courses.columns)

        #st.write("Column Names in Instructors:")
        #st.write(instructors.columns)

        #st.write("Column Names in Timeslots:")
        #st.write(timeslots.columns)

        #st.write("Column Names in Classroom:")
        #st.write(classrooms.columns)
    
        # Parameters for Firefly Algorithm
        num_fireflies = 10
        num_iterations = 100
        gamma = 0.3  # Light absorption coefficient
        beta0 = 1.0  # Attraction coefficient base value
        alpha = 0.1  # Randomization coefficient

        courses_list = courses['course_name'].tolist()  # Replace 'Course' with the actual column name
        instructors_list = instructors['first_name'].tolist()  # Replace 'Instructor' with the column name
        classrooms_list = classrooms['building_name'].tolist()  # Replace 'Classroom' with the column name
        timeslots_list = timeslots['day'].tolist()  # Replace 'Timeslot' with the column name
      
        # Initialize fireflies
        fireflies = []
        for _ in range(num_fireflies):
            firefly = [
                        (
                            random.choice(courses_list), 
                            random.choice(instructors_list), 
                            random.choice(classrooms_list), 
                            random.choice(timeslots_list)
                        ) 
                        for _ in range(len(courses_list))
                      ]
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
                            [courses_list.index(fireflies[i][k][0]) - courses_list.index(fireflies[j][k][0]) for k in range(len(courses_list))]
                        )
                        beta = beta0 * np.exp(-gamma * distance**2)
                        fireflies[i] = move_firefly(fireflies[i], fireflies[j], beta)
                        # Random perturbation
                        if random.random() < alpha:
                            fireflies[i] = [(random.choice(courses_list), random.choice(instructors_list),
                                             random.choice(classrooms_list), random.choice(timeslots_list))
                                            for _ in range(len(courses_list))]
        
            # Update global best
            best_index = np.argmin(fitness_values)
            best_firefly = fireflies[best_index]
            best_fitness = fitness_values[best_index]
        
            st.write(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

        st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)
        st.subheader("B E S T &nbsp;&nbsp; S C H E D U L I N G")

        # Assuming best_schedule_df is your DataFrame
        def color_index(x):
            return ['color: blue' if x.name == x.index.name else '' for _ in x]

         # Output the best schedule
        best_schedule_df = pd.DataFrame(best_firefly, columns=["Course", "Instructor", "Room", "Timeslot"])
        styled_df = best_schedule_df.style.apply(color_index, axis=0)
      
        st.dataframe(styled_df, width=800, height=1400)
      
  else:
    st.write("Please upload all the required CSV files.")

if Clear_Button:
  clear_output()
  st.rerun()







