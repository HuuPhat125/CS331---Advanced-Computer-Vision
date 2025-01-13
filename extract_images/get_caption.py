import os
import json
import tqdm
import argparse

import sys
sys.path.append('../')
from evaluation.utils import *

from model import gemini

def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True

def save_results(output_file, results):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../data/images')
    parser.add_argument('--output_dir', type=str, default='../data/texts')
    parser.add_argument('--output_file', type=str, default='captions_gemini.json')
    parser.add_argument('--model', type=str, default='gemini', help='model name')
    parser.add_argument('--key', type=str, default='', help='key for gemini api')
    args = parser.parse_args()

    data_file = "data_test.json"

    data = json.load(open(os.path.join('../data', data_file), 'r'))
    
    data = {str(i): sample for i, sample in enumerate(data)}
    pids = list(data.keys())
    print("number of images: ", len(pids))

    # output file
    output_file = os.path.join(args.output_dir, args.output_file)
    
    # Load existing results or create new results structure
    if os.path.exists(output_file):
        print("Loading existing results...")
        with open(output_file, 'r') as f:
            results = json.load(f)
    else:
        results = {
            "model": "gemini",
            "url": "https://gemini.google.com/",
            "date": "2024-7-23",
            "texts": {}
        }

    # Identify pids with empty responses
    test_pids = [pid for pid in pids if pid not in results['texts'] or results['texts'][pid] == ""]
    print("\nNumber of images to process: ", len(test_pids))

    # build model
    query = "Mô tả nội dung chi tiết của hình ảnh hoặc hình vẽ bao gồm bảng số liệu, ký hiệu, cảnh, đồ vật, mối quan hệ và bất kỳ văn bản nào hiện diện. Phải mô tả bằng Tiếng Việt"
    
    if args.model == 'gemini':
        if args.key == '':
            print("Loading key from environment variable")
            key = os.environ['_GEMINI_API_KEY']
        else:
            key = args.key
        model = gemini.Gemini_Model(key)

    for pid in tqdm.tqdm(test_pids):
        image_path = os.path.join(args.image_dir, data[pid]["image"])

        print(f"Generating response for {pid}-{image_path}...")
        try:
            response = model.get_response(query, image_path)
            results['texts'][pid] = response
            save_results(output_file, results)
        except Exception as e:
            print(e)