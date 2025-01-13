"""
Reference: https://github.com/JaidedAI/EasyOCR 
"""

import os
import easyocr
import json
import tqdm
import random
import sys

sys.path.append('../')

from evaluation.utils import *
# def load_data(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = [json.loads(line) for line in f]
#     return data

def perform_ocr(reader, image_dir, data):
    results = {
        "model": "easyocr",
        "url": "https://github.com/JaidedAI/EasyOCR",
        "version": "1.1.8",
        "date": "2023-04-06",
        "texts": {}
    }
    pids = list(data.keys())
    # print(pids[:10])
    for pid in tqdm.tqdm(pids):
        image_file = os.path.join(image_dir, data[pid]["image"])
        try:
            result = reader.readtext(image_file, detail = 0)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            result = []

        results["texts"][pid] = str(result)
        print(f'{pid}: {result}')
    return results

def save_results(output_file, results):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    data_dir = '../data'
    data_file = "data_test.json"  
    image_dir = "../data/images"  
    output_file = "../data/texts/ocrs_easyocr.json"  

    data = read_json(os.path.join(data_dir, data_file))

    # random.seed(42)
    # data = random.sample(data, 10)
    data = {str(i): sample for i, sample in enumerate(data)}
    
    first_10_items = dict(list(data.items())[:10])
    print(first_10_items)

    print("Number of questions:", len(data))

    reader = easyocr.Reader(['vi'])  # Initialize the OCR reader

    results = perform_ocr(reader, image_dir, data)
    save_results(output_file, results)
    print("OCR process completed and results saved.")
