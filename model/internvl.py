import math
import torch
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image

class InternVL:
    def __init__(self, model_name='InternVL-Chat-V1-2-Plus', model_path="OpenGVLab/InternVL-Chat-V1-2-Plus"):
        self.device_map = self.split_model(model_name)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device_map
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {'InternVL-Chat-V1-2': 60, 'InternVL-Chat-V1-2-Plus': 60}[model_name]
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    def get_response(self, queries, image_paths, generation_config=dict(max_new_tokens=1024, do_sample=True)):
        responses = []
        for query, image_path in zip(queries, image_paths):
            # Load and preprocess image
            image = Image.open(image_path).resize((448, 448))
            pixel_values = self.image_processor(images=image, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()

            # Generate response
            response = self.model.chat(self.tokenizer, pixel_values, query, generation_config)
            responses.append(response)
        return responses

