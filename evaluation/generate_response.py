# (https://github.com/lupantech/MathVista)
import os
import io
import time
import argparse
import random
from tqdm import tqdm
import torch
import sys

sys.path.append('../')
import sys
import os

# Lấy đường dẫn tuyệt đối đến thư mục gốc của dự án (DSCLUB)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, project_root)

# print(sys.path)

# Thêm thư mục chứa llava vào sys.path để sử dụng cho mô hình LaVy
# llava_path = os.path.join(project_root, 'model', 'LaVy')
# sys.path.insert(0, llava_path)

#Dùng cho LLaVa, nó bị trùng với cái trên, nên chỉ để 1 trong 2 cái
# sys.path.append('/data1/npl/ds/model/LLaVA') (do dùng llava onevision nên comment, tại # vì nó sử dụng thư viện từ pip)


from utils import *
from build_query import create_query_data
# phoGPT, qwen2vl, uraLlama, LaVy, vistral, gemini, vinaLlama, vintern,interlm, internvl, 
from model import qwen2math, llava_onevision, paligema
# from run_llava import LLaVa_E

def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='data_test.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../result/phoGPT')
    parser.add_argument('--output_file', type=str, default='output_phoGPT_4B_Chat.json')
    # model
    parser.add_argument('--model', type=str, default='PhoGPT-4B-Chat', help='llm engine', choices=[
        'PhoGPT-4B-Chat',
        'PhoGPT-7B5',
        'ura-llama-7b',
        'ura-llama-13b',
        'ura-llama-70b',
        'lavy',
        'vistral',
        'gemini',
        'vinallama',
        'vintern',
        'internlm-xcomposer2-vl-7b',
        'llava-v1.6-34b',
        'InternVL-Chat-V1-2-Plus',
        'Qwen2-VL-7B-Instruct',
        'Qwen2-Math-72B-Instruct',
        'llava-onevision-qwen2-7b-ov',
        'paligemma-3b-ft-science-qa-448',
        'paligemma-3b-mix-448'])
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_gemini.json')
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')
    parser.add_argument('--context_file', type=str, default='../data/texts/top1_context.json') 
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    parser.add_argument('--use_context', action='store_true', help='use context data')

    parser.add_argument('--check', action='store_true', help='checking for debug')

    # other settings
    parser.add_argument('--batch_size', type=int, default=16, help='Kích thước batch để thực hiện inference')
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    
    parser.add_argument('--has_image', action='store_true', help='has image')

    #run lavy
    parser.add_argument('--model_base', type=str, default='../model/LaVy/checkpoints/model_base')
    parser.add_argument('--model_path', type=str, default='../model/LaVy/checkpoints/model_path')


    args = parser.parse_args()

    # load data
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)

    # Select 10 random samples from the data
    # random.seed(42)
    # data_samples = random.sample(data, 10)


    data = {str(i): sample for i, sample in enumerate(data)}

    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            print(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        print("\nCreating new query...")
        # load caption
        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                print(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    print("Caption data loaded.")
                except:
                    print("Caption data not found!! Please Check.")
        # load ocr
        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                print(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    print("OCR data loaded.")
                except:
                    print("OCR data not found!! Please Check.")
        context_data = {}
        if args.use_context:
            context_file = args.context_file
            if os.path.exists(context_file):
                print(f"Reading {context_file}...")
                try:
                    context_data = read_json(context_file)
                    print("Context data loaded.")
                except:
                    print("context data not found!! Please Check.")
        # create query
        query_data = create_query_data(data, args, caption_data, ocr_data, context_data)
        print(len(query_data))
    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        results = read_json(output_file)
    else:
        results = {}

    # load model
    print(f"\nLoading {args.model}...")
    if args.model == 'PhoGPT-4B-Chat':
        model = phoGPT.PhoGPT_Model(
            model_path="vinai/PhoGPT-4B-Chat"
        )
    elif args.model == 'PhoGPT-7B5':
        model = phoGPT.PhoGPT_Model(
            model_path="../model/checkpoints/PhoGPT-7B5"
        )
    elif args.model == 'ura-llama-7b':
        model = uraLlama.URA_Llama_Model(model_path="ura-hcmut/ura-llama-7b")
    elif args.model == 'ura-llama-13b':
        model = uraLlama.URA_Llama_Model(model_path="ura-hcmut/ura-llama-13b")
    elif args.model == 'ura-llama-70b':
        model = uraLlama.URA_Llama_Model(model_path="ura-hcmut/ura-llama-70b")
    elif args.model == 'lavy':
        model = LaVy.LaVyModel(model_path=args.model_path, model_base=args.model_base, model_name='llava_lora')
    elif args.model == 'vistral':
        model = vistral.Vistral_Model()
    elif args.model == 'gemini':
        model = gemini.Gemini_Model(args.key)
    elif args.model == 'vinallama':
        model = vinaLlama.VinaLlama(model_path="vilm/vinallama-7b-chat", max_new_tokens=768)
    elif args.model == 'vintern':
        model = vintern.Vintern_Model()
    elif args.model == 'internlm-xcomposer2-vl-7b':
        model = interlm.InterLM_Model(model_path=f'internlm/{args.model}')
    elif args.model == 'llava-v1.6-34b':
        model = LLaVa_E(model_path=f"liuhaotian/{args.model}")
    elif args.model == 'InternVL-Chat-V1-2-Plus':
        model = internvl.InternVL()
    elif args.model == 'Qwen2-VL-7B-Instruct':
        model = qwen2vl.Qwen2VL()
    elif args.model == 'Qwen2-Math-72B-Instruct':
        model = qwen2math.Qwen2Math()
    elif args.model == 'llava-onevision-qwen2-7b-ov':
        model = llava_onevision.LlavaOneVision()
    elif args.model == 'paligemma-3b-mix-448': #google/paligemma-3b-ft-science-qa-448
        model = paligema.PaliGemma(model_id=f'google/{args.model}')
    elif args.model == 'paligemma-3b-ft-science-qa-448':
        model = paligema.PaliGemma(model_id=f'google/{args.model}')
    else:
        raise ValueError(f"Model {args.model} not recognized.")
    print(f"Model loaded.")

    # build final test pid list
    test_pids = list(data.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    skip_pids = []
    if not args.rerun:
        print("\nRemoving problems with existing valid response...")
        for pid in test_pids:
            # print(f"Checking {pid}...")
            if pid in results and 'response' in results[pid]:
                response = results[pid]['response']
                if verify_response(response):
                    print(f"Valid response found for {pid}.")
                    skip_pids.append(pid)
    else:
        print("\nRerun answer extraction for all problems...")

    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    if args.check:
        test_pids = random.sample(test_pids, min(5, len(test_pids)))
    print("Number of test problems to run:", len(test_pids))
    # print(test_pids)

    batch_size = args.batch_size  
    for i in tqdm(range(0, len(test_pids), batch_size)):
        batch_pids = test_pids[i:i+batch_size]
        batch_instructions = [query_data[pid] for pid in batch_pids]
        batch_image_paths = [os.path.join(args.data_dir, 'images', data[pid]['image']) for pid in batch_pids]
        
        # if args.debug:
        #     print("--------------------------------------------------------------")
        #     print(f"\nGenerating responses for batch {i//batch_size + 1}...")
        #     for pid, instruction, image_path in zip(batch_pids, batch_instructions, batch_image_paths):
        #         print(f"PID: {pid}")
        #         print(f"Instruction: {instruction}")
        #         print(f"Image path: {image_path}")
        #         print("--------------------------------------------------------------")

        try:
            if args.check:
                print(f'batch_instructions: {batch_instructions}')
                print(f'batch_image_paths: {batch_image_paths}')
            batch_responses = model.get_response(batch_instructions, batch_image_paths)
            if args.model == 'gemini':
                batch_responses = [batch_responses]
            if args.debug:
                print(f'check before loop: {batch_responses}')
            for pid, instruction, response in zip(batch_pids, batch_instructions, batch_responses):
                problem = data[pid]
                results[pid] = problem
                results[pid]['query'] = instruction
                results[pid]['response'] = response
                
                if args.debug:
                    print(f"\n#Query for {pid}: \n{instruction}")
                    print(f"\n#Response for {pid}: \n{response}")
        
        except Exception as e:
            print(f"Error in processing batch: {str(e)}")
            for pid in batch_pids:
                if pid not in results:
                    results[pid] = {}
                results[pid]['error'] = str(e)

        try:
            print(f"Saving results to {output_file}...")
            save_json(results, output_file)
            print(f"Results saved.")
        except Exception as e:
            print(f"Error in saving {output_file}: {str(e)}")