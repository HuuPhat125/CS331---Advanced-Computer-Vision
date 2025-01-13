import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

class Vintern_Model:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_path="5CD-AI/Vintern-1B-v2", device="cuda", dtype=torch.bfloat16, input_size=448, max_num=12):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.input_size = input_size
        self.max_num = max_num

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def build_transform(self):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * self.input_size * self.input_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, self.max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= self.max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height)

        target_width = self.input_size * target_aspect_ratio[0]
        target_height = self.input_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // self.input_size)) * self.input_size,
                (i // (target_width // self.input_size)) * self.input_size,
                ((i % (target_width // self.input_size)) + 1) * self.input_size,
                ((i // (target_width // self.input_size)) + 1) * self.input_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self.input_size, self.input_size))
            processed_images.append(thumbnail_img)
        
        return processed_images

    def load_image(self, image_file):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform()
        images = self.dynamic_preprocess(image, use_thumbnail=True)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values.to(self.dtype).to(self.device)

    def get_response(self, question, image_file, generation_config=None, history=None):
        if generation_config is None:
            generation_config = dict(max_new_tokens=1024, do_sample=True, num_beams=3, repetition_penalty=1)

        pixel_values = self.load_image(image_file[0])
        response, new_history = self.model.chat(self.tokenizer, pixel_values, question[0], generation_config, history=history, return_history=True)
        return [response]

