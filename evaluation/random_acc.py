import json
import random

# Hàm để load dữ liệu từ file JSON
def load_questions_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        questions = json.load(file)
    return questions

# Hàm để thực hiện random đáp án
def random_answer(questions):
    correct_count = 0
    for question in questions:
        random_choice = random.choice(['D','A', 'B', 'C'])
        if random_choice == question['answer']:
            correct_count += 1
    return correct_count / len(questions)

def average_accuracy(questions, n_trials=1000):
    total_accuracy = 0
    for _ in range(n_trials):
        accuracy = random_answer(questions)
        total_accuracy += accuracy
    return total_accuracy / n_trials

file_path = '../data/data_test.json'  # Đường dẫn tới file JSON của bạn
questions = load_questions_from_json(file_path)

# Chạy thử với 1000 lần random
n_trials = 10
accuracy = average_accuracy(questions, n_trials)
print(f'Độ chính xác trung bình sau {n_trials} lần random: {accuracy:.2%}')
