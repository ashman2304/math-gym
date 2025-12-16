import streamlit as st
from math_engine import MathEngine

# Initialize session state to keep track of the current problem
if 'engine' not in st.session_state:
    st.session_state.engine = MathEngine()
if 'current_problem' not in st.session_state:
    st.session_state.current_problem = None
if 'show_solution' not in st.session_state:
    st.session_state.show_solution = False

def new_problem():
    st.session_state.current_problem = st.session_state.engine.get_random_problem()
    st.session_state.show_solution = False

# Title
st.title("Daily Maths Gym")
st.markdown("### Master's Level Training")
st.divider()

# Generate first problem if none exists
if st.session_state.current_problem is None:
    new_problem()

# Display Problem
problem = st.session_state.current_problem
st.info(f"**Topic:** {problem['topic']} - {problem['type']}")
    
 # st.markdown is smart: it renders text as text, and $$...$$ as math!
st.markdown(problem['problem'])


# Interaction
col1, col2 = st.columns(2)

with col1:
    if st.button("Reveal Solution"):
        st.session_state.show_solution = True

with col2:
    if st.button("New Problem"):
        new_problem()
        st.rerun()

# Display Solution
if st.session_state.show_solution:
    st.divider()
    st.success("Solution")
    # Change this line too:
    st.markdown(problem['solution'])

### streamlit run app.py
### ctrl + C to stop