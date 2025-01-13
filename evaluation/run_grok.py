import argparse
import random
from tqdm import tqdm
import sys
import os
import time
from typing import List, Dict, Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils import *
from build_query import create_query_data
from model import grok, gemini_api_key

class GrokWithKeyRotation:
    def __init__(self, model_name="grok-2-1212"):
        self.current_key_index = 0
        self.api_keys = self._load_api_keys()
        self.model_name = model_name

    def _load_api_keys(self) -> List[str]:
        """Load API keys from imported gemini_api_key module"""
        try:
            from model.gemini_api_key import api_key
            return list(api_key.values())
        except Exception as e:
            print(f"Error loading API keys: {str(e)}")
            return []

    def _get_next_key(self) -> str:
        """Get the next API key in rotation"""
        if not self.api_keys:
            raise ValueError("No API keys available")
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key

    def get_response(self, instructions: List[str], image_paths: Optional[List[str]]) -> List[str]:
        """Get responses for a batch of queries, handling both text and images"""
        responses = []
        for instruction, image_path in zip(instructions, image_paths):
            try:
                key = self._get_next_key()
                # Initialize chatbot with the current key
                chatbot = grok.ChatbotClient(
                    model=self.model_name,
                    api_key=key
                )


                # Get response from the model
                try:
                    response = chatbot.chat(instruction, image_path)
                    responses.append(response)
                    time.sleep(10)  # Avoid rate limits
                except Exception as e:
                    print(f"Chat error: {str(e)}")
                    responses.append(f"Response Error: {str(e)}")

            except Exception as e:
                print(f"Error getting response: {str(e)}")
                responses.append(f"Response Error: {str(e)}")

        return responses

def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    # Input arguments
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='data_test.json')
    parser.add_argument('--output_dir', type=str, default='../result/grok')
    parser.add_argument('--output_file', type=str, default='output_grok.json')
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_gemini.json')
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')
    parser.add_argument('--context_file', type=str, default='../data/texts/top1_context.json')
    parser.add_argument('--shot_num', type=int, default=0)
    parser.add_argument('--use_caption', action='store_true')
    parser.add_argument('--use_ocr', action='store_true')
    parser.add_argument('--has_image', action='store_true')
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--model_name', type=str, default='grok-2-vision-1212')

    args = parser.parse_args()

    # Load and process data
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)
    data = {str(i): sample for i, sample in enumerate(data)}

    # Create query data
    query_data = create_query_data(
        data, args,
        caption_data=read_json(args.caption_file)["texts"] if args.use_caption else {},
        ocr_data=read_json(args.ocr_file)["texts"] if args.use_ocr else {},
        context_data=read_json(args.context_file) if args.use_context else {}
    )

    # Initialize output
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)
    results = read_json(output_file) if os.path.exists(output_file) else {}

    # Initialize model
    model = GrokWithKeyRotation(model_name=args.model_name)

    # Process test cases
    test_pids = list(data.keys())
    if not args.rerun:
        skip_pids = [pid for pid in test_pids if pid in results and verify_response(results[pid].get('response'))]
        test_pids = [pid for pid in test_pids if pid not in skip_pids]

    if args.check:
        test_pids = random.sample(test_pids, min(1, len(test_pids)))
    print(f'{len(test_pids)=}')
    # Process in batches
    for i in tqdm(range(0, len(test_pids), args.batch_size)):
        batch_pids = test_pids[i:i+args.batch_size]
        batch_instructions = [query_data[pid] for pid in batch_pids]
        batch_image_paths = [
            os.path.join(args.data_dir, 'images', data[pid]['image'])
            for pid in batch_pids
        ]

        try:
            batch_responses = model.get_response(batch_instructions, batch_image_paths)

            for pid, instruction, response in zip(batch_pids, batch_instructions, batch_responses):
                results[pid] = {
                    **data[pid],
                    'query': instruction,
                    'response': response
                }

            # Save results periodically
            save_json(results, output_file)

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            for pid in batch_pids:
                if pid not in results:
                    results[pid] = {'error': str(e)}

if __name__ == '__main__':
    main()
