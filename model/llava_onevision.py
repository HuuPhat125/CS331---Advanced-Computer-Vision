from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import requests
import copy
import torch
import warnings

warnings.filterwarnings("ignore")

class LlavaOneVision:
    def __init__(self, pretrained_model="lmms-lab/llava-onevision-qwen2-7b-ov", model_name="llava_qwen", device="cuda", device_map="auto"):
        """
        Khởi tạo class LlavaOneVision, load mô hình và tokenizer.
        """
        # Load tokenizer, model, image processor
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            pretrained_model, None, model_name, device_map=device_map
        )
        self.device = device
        self.model.eval()  # Set model to evaluation mode
    
    def get_response(self, queries, image_paths):
        
        image_path = image_paths[0]
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)

        # Xử lý hình ảnh
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        # Chuẩn bị conversation template
        conv_template = "qwen_1_5"  # Đảm bảo sử dụng đúng template
        conv = copy.deepcopy(conv_templates[conv_template])

        query=queries[0]
        if "<image>" in query:
            question = query.replace("<image>", DEFAULT_IMAGE_TOKEN)
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + query

        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Tokenize câu hỏi và hình ảnh
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size]

        # Generate response từ mô hình
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
        )

        # Decode output
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return [text_outputs[0]]


# Ví dụ sử dụng:
if __name__ == "__main__":
    # Khởi tạo class LlavaOneVision
    llava = LlavaOneVision()

    # Đường dẫn hình ảnh và câu hỏi
    image_url = ["https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"]
    question = ["What is shown in this image?"]

    # Lấy câu trả lời
    response = llava.get_response(question, image_url)
    for res in response:
        print(res)
