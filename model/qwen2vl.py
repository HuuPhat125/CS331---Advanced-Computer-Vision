from PIL import Image
import requests
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import List

class Qwen2VL:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        # Load model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_image(self, image_path: str):
        if image_path.startswith("http://") or image_path.startswith("https://"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        resized_image = image.resize((max(28, image.width), max(28, image.height)))
        return resized_image
    
    def get_response(self, queries: List[str], image_paths: List[str]) -> List[str]:
        # Create conversation structure for each query and image pair
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                }
            ]
            for query in queries
        ]
        
        # Prepare text prompts and images for batch processing
        texts = [
            self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            for conversation in conversations
        ]
        
        # Load images in batch
        images_batch = [self.load_image(image_path) for image_path in image_paths]
        
        # Preprocess inputs for the model
        inputs = self.processor(
            text=texts, 
            images=images_batch, 
            padding=True, 
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # Perform batch inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated text responses
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        return output_texts

# # Example usage
# qwen_model = Qwen2VL()

# # Batch queries and image paths
# queries = [
#     "có thể suy ra được gì từ bức ảnh này <image>",
#     "có thể suy ra được gì từ bức ảnh này <image>"
# ]

# image_paths = [
#     r"/data1/npl/ds/data/images/train_16469.png",
#     r"/data1/npl/ds/data/images/train_16470.png"
# ]

# # Get responses for the batch
# responses = qwen_model.get_response(queries, image_paths)
# for response in responses:
#     print(response)
