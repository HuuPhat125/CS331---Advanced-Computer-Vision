import torch
import os
import sys
import requests
from PIL import Image
from io import BytesIO
import re

# Đảm bảo đường dẫn tới thư mục LLaVa đã được thêm vào sys.path
sys.path.append('/data1/npl/ds/model/LLaVA')

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

class LLaVa_E:
    def __init__(self, model_path="liuhaotian/llava-v1.6-34b", model_base=None):
        self.model_path = model_path
        self.model_base = model_base
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = self.load_model()

    def load_model(self):
        disable_torch_init()
        return load_pretrained_model(
            self.model_path, self.model_base, self.model_name
        )

    def load_image(self, image_file):
        if os.path.isdir(image_file):
            raise ValueError(f"{image_file} is a directory, expected a file path.")
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image


    def load_images(self, image_files):
        images = [self.load_image(image_file) for image_file in image_files]
        return images

    def get_response(self, queries, image_paths):
        responses = []
        # print(f'len(queries): {len(queries)}')
        # print(f'len(image_paths): {len(image_paths)}')
        images = self.load_images(image_paths)
        image_sizes = [image.size for image in images]
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        for query in queries:
            qs = query
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if self.model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if self.model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            if "llama-2" in self.model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in self.model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in self.model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in self.model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in self.model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=512,
                    use_cache=True,
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            responses.append(outputs)
        return responses

