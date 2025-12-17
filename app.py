import streamlit as st
from math_engine import MathEngine

# 1. Initialize Session State
if 'engine' not in st.session_state:
    st.session_state.engine = MathEngine()

if 'current_problem' not in st.session_state:
    st.session_state.current_problem = None

if 'show_solution' not in st.session_state:
    st.session_state.show_solution = False

# 2. Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
# We get the list of topics directly from the engine's keys, plus "Mix"
topic_options = ["Mix"] + list(st.session_state.engine.topic_map.keys())
selected_topic = st.sidebar.selectbox("Choose Problem Set:", topic_options)

def new_problem():
    """Fetch a new problem based on the selected topic"""
    # We pass the 'selected_topic' to the engine now!
    st.session_state.current_problem = st.session_state.engine.get_random_problem(selected_topic)
    st.session_state.show_solution = False

# 3. Main Page Layout
st.title("🧮 Daily Maths Gym")
st.markdown("### Master's Level Training")
st.divider()

# Logic: If no problem exists OR if the user switched topics, generate a new one
# (Optional: You can remove the 'switched topics' auto-refresh if you prefer manual button presses)
if st.session_state.current_problem is None:
    new_problem()

problem = st.session_state.current_problem

# 4. Display Problem
st.info(f"**Topic:** {problem['topic']} - {problem['type']}")
st.markdown(problem['problem'])

# 5. Interaction Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Reveal Solution"):
        st.session_state.show_solution = True

with col2:
    if st.button("New Problem"):
        new_problem()
        st.rerun()

# 6. Display Solution
if st.session_state.show_solution:
    st.divider()
    st.success("Solution")
    st.markdown(problem['solution'])

### streamlit run app.py
### ctrl + C to stop