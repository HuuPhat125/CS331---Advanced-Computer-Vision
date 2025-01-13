import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
sys.path.append('../')
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images
from torch.utils.data import Dataset, DataLoader

from PIL import Image

current_dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir_path
while not os.path.isdir(os.path.join(project_root, '.git')):  # hoặc một thư mục đặc trưng khác của dự án
    project_root = os.path.dirname(project_root)
    if project_root == '/':  # Đảm bảo không đi quá root của hệ thống
        raise Exception("Không tìm thấy thư mục gốc của dự án.")
class LlavaDataset(Dataset):
    def __init__(self, json_path, tokenizer, image_processor, model_config):
        self.model_config = model_config
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
            
        image_path = os.path.join(project_root, 'data', 'images', item['image'])
        question = item['question']
        options = item['options']
        answer = item['answer']
        
        qs = 'Câu hỏi:\n' + r'{}'.format(question) + "\n" + 'Các lựa chọn\n' + r'{}'.format(str(options)) + '\nHãy chọn đáp án đúng và giải thích'
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return prompt, input_ids, image_tensor, image.size, item

def collate_fn(batch):
    prompts, input_ids, image_tensors, image_sizes, items = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return prompts, input_ids, image_tensors, image_sizes, items

def create_data_loader(json_path, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = LlavaDataset(json_path, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = 'llava_lora' if args.pretrain == 'false' else 'llava'
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(args.file, tokenizer, image_processor, model.config)

    results = {}

    for idx, (prompt, input_ids, image_tensor, image_sizes, items) in enumerate(tqdm(data_loader)):
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens)

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        item = items[0]  # Since batch_size is 1
        result = {
            "questionId": item['questionId'],
            "question": item['question'],
            "options": str(item['options']),  # Convert string representation of list to actual list
            "answer": item['answer'],
            "explaination": item['explaination'],
            "subject": item['subject'],
            "grade": item['grade'],
            "image": item['image'],
            "query": prompt,
            "response": output
        }
        print(result)
        results[str(idx)] = result

        # Save results to JSON file
        output_file = os.path.join(project_root, 'result', 'Lavy', 'output_lavy.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {output_file}")
    print(f"All results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--pretrain", default='false', type=str)
    args = parser.parse_args()

    eval_model(args)

