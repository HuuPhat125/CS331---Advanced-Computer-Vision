import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import re
import google.generativeai as genai
import os
import glob
import time
from model.gemini_api_key import api_key

def get_gemini_response(query, response):
    print(f'Using gemini API to extract answer from {response}')
    prompt = f"""Từ nội dung trong phần phản hồi hãy trích xuất lựa chọn đúng cho câu hỏi, chỉ trả lời chữ cái in hoa trước lựa chọn đúng (A hoặc B hoặc C hoặc D) nếu trong phản hồi không chứa đáp án thì trả lời là N/A\n
                    Câu hỏi: {query}\n
                    Phản hồi: {response}\n
                    Câu trả lời được trích xuất:
                """
    model = genai.GenerativeModel(model_name='gemini-1.5-flash',
                                  system_instruction="Nhiệm vụ của bạn là trích xuất lựa chọn đúng cho câu hỏi từ phản hồi")

    for email, key in api_key.items():
        try:
            genai.configure(api_key=key)
            response = model.generate_content(prompt)
            response = r'{}'.format(response.text.strip()).replace('\n', '')
            response = re.sub(r'[^a-zA-Z/]', '', response)
            time.sleep(1)
            return response, 'gemini'
        except Exception as e:
            print(f"Error with API key for {email}: {str(e)}. Trying next key.")
    
    print("All API keys failed. Unable to get response.")
    return 'N/A', None

def remove_last_non_alphanumeric(text):
    # Remove the last character if it's not a letter (including Vietnamese letters) or a number
    pattern = r'^(.*[a-zA-Z0-9\u00C0-\u1EF9]).*$'
    return re.sub(pattern, r'\1', text)

def extract_answer(input_file, output_file):
    # Kiểm tra xem file output đã tồn tại chưa
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_data = existing_data.copy()  # Copy existing data để giữ nguyên dữ liệu cũ

    for key, value in data.items():
        # Kiểm tra xem câu đã có predict hợp lệ chưa
        if key in existing_data and existing_data[key].get('predict') in ['A', 'B', 'C', 'D', 'N/A'] and existing_data[key].get('extraction_method') not in [None, ""]:
            print(f'{key}: Extracted, continue')
            continue  # Bỏ qua câu này nếu đã có predict hợp lệ

        response = value.get('response', '').lower().replace('*', '').replace('\n', '')
        options = {
            'A': value.get('option_A', '').strip(),
            'B': value.get('option_B', '').strip(),
            'C': value.get('option_C', '').strip(),
            'D': value.get('option_D', '').strip()
        }
        predict = 'N/A'
        method = ''

        if 'rất tiếc' in response:
            predict = 'N/A'
            method = 'rule base'
        # elif len(response.strip() == 1):
        #     predict = response.upper()
        #     method = 'rule base'
        else:
            answer_pattern = r'(?i)(?:đáp án[: ]*|đáp án đúng là[: ]*|đáp án là[: ]*|đáp án cuối cùng là[: ]*|đáp án đúng[: ]*|the answer is[: ]*|trả lời[: ]*)([abcd])(?=[. ]|$)' #chọn[: ]*|
            match = re.search(answer_pattern, response, re.IGNORECASE)
            if match:
                predict = match.group(1).upper()
                method = 'rule base 1'
            else:
                for option in options:
                    option = option.replace(r"'", '')
                    option = remove_last_non_alphanumeric(option)
                    if option.lower() in response:
                        predict = next((char for char in option if char in 'ABCD'), None)
                        method = 'rule base 2'
                        break

                if predict == 'N/A':
                    for option in options:
                        option = option.replace(r"'", '')
                        option_label = next((char for char in option if char in 'ABCD'), None)
                        if option_label is None:
                            continue

                        label_pos = option.find(option_label)
                        if label_pos == -1:
                            continue

                        option_text = option[label_pos+1:].strip().lower()
                        option_text = remove_last_non_alphanumeric(option_text)
                        if option_text in response:
                            predict = option_label
                            method = 'rule base 3'

                if response.strip().upper() in ['A', 'B', 'C', 'D']:
                    predict = response.strip().upper()
                    method = 'rule base'

                if predict == 'N/A':
                    query = f"Câu hỏi: {value.get('question', '')}\nCác lựa chọn:\n" + "\n".join(options)
                    try:
                        predict, method = get_gemini_response(query, response)
                        if predict not in ['A', 'B', 'C', 'D']:
                            predict = 'N/A'
                    except Exception as e:
                        print(f'sleeping {e}')
                        time.sleep(30)

        print(f"Key: {key}, Predict: {predict}, Method: {method}")

        updated_data[key] = value
        updated_data[key]['predict'] = predict
        updated_data[key]['extraction_method'] = method

    # Ghi dữ liệu đã cập nhật vào file output một lần
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)


def process_all_json_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_file in glob.glob(os.path.join(input_dir, '**', '*.json'), recursive=True):

        relative_path = os.path.relpath(input_file, input_dir)
        output_file = os.path.join(output_dir, relative_path)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print(f"Processing: {input_file}")
        extract_answer(input_file, output_file)
        print(f"Completed: {output_file}")

    

# input_dirs= ['../result/PaliGemma']

# output_dirs=['../prediction/PaliGemma']
# for i in range (len(input_dirs)):
#     process_all_json_files(input_dirs[i],  output_dirs[i])
extract_answer(r'D:\DSClub\result\PaliGemma\paligemma-3b-448-ft_ops_0.json', r'D:\DSClub\prediction\PaliGemma\paligemma-3b-448-ft_ops_0.json')