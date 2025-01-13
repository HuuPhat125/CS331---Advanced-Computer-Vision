import streamlit as st
import json
import random
import os
# Load data from JSON files
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Display the question with LaTeX support
def display_question(question_data, images_dir = './data/images'):
    st.write(f"**Question ID:** {question_data['questionId']}")
    st.write(f"**Question:** {question_data['question']}")
    st.write(f"**Options:** {question_data['options']}")
    st.write(f"**Answer:** {question_data['answer']}")
    st.write(f"**Explanation:** {question_data['explaination']}")
    st.write(f"**Subject:** {question_data['subject']}")
    st.write(f"**Grade:** {question_data['grade']}")
    if 'image' in question_data:
        st.image(os.path.join(images_dir, question_data['image']))

# Main app
st.title("Question Viewer")

# Sidebar for dataset and question selection
with st.sidebar:
    st.header("Options")
    dataset_choice = st.selectbox("Choose a dataset", ["train", "test", "dev"])
    question_id = st.text_input("Enter a Question ID or leave blank for random questions")
    num_questions = st.number_input("Number of random questions", min_value=1, max_value=None, value=1)

# Load the selected dataset
if dataset_choice == "train":
    data = load_data('./data/data_train.json')
elif dataset_choice == "test":
    data = load_data('./data/data_test.json')
else:
    data = load_data('./data/data_dev.json')

# Display the selected question(s)
if question_id:
    # Display specific question
    question = next((q for q in data if q['questionId'] == question_id), None)
    if question:
        display_question(question)
    else:
        st.error("Question ID not found!")
else:
    # Display random questions
    random_questions = random.sample(data, num_questions)
    for question in random_questions:
        display_question(question)
        st.write("---")
