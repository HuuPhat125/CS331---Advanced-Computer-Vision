import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
from typing import List
from torch.amp import autocast

class Qwen2Math:
    def __init__(self, model_name: str = "Qwen/Qwen2-Math-72B-Instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_response(self, queries: List[str], image_paths: List[str]) -> List[str]:
        """Generate responses for a batch of queries and corresponding images."""
        
        # Create a batch of prompts using the queries
        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            for query in queries
        ]
        
        texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        
        with autocast('cuda'):
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return responses

# qwen_model = Qwen2Math()

# queries = [
#     "Tìm giá trị của $x$ thỏa mãn phương trình $4x+5 = 6x+7$.",
#     "Giải phương trình $2x + 3 = 4x - 5$."
# ]

# image_paths = ["", ""]

# responses = qwen_model.get_response(queries, image_paths)

# for response in responses:
#     print(response)
