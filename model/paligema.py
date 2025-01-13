from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import requests
import os

class PaliGemma:
    def __init__(self, model_id="google/paligemma-3b-ft-science-qa-448", device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        
        # Load model and processor
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            revision="bfloat16",
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_id)

    def load_image(self, image_path: str) -> Image:
        # Check if image_path is a URL or a local file path
        if image_path.startswith("http://") or image_path.startswith("https://"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        elif os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            raise ValueError("Invalid image path: Must be a valid URL or local file path.")
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def clean_query(self, query: str) -> str:
        return query.replace("<image>", "").strip()

    def get_response(self, query, image_path):
        # Clean the query by removing the <image> tag
        cleaned_query = self.clean_query(query[0])
        
        # Load and process the image
        image = self.load_image(image_path[0])
        
        # Prepare the model inputs
        model_inputs = self.processor(text=cleaned_query, images=image, return_tensors="pt").to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        # Perform inference
        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]  # Extract the new tokens (generation)
            decoded = self.processor.decode(generation, skip_special_tokens=True)

        return [decoded]

# # Example usage
# if __name__ == "__main__":
#     model = PaliGemma()
#     query = ["Bộ dụng cụ như hình vẽ có thể dùng để thu khí nào sau đây? <image> \n\nA. H2.\n B. NH3.\n C. CH4.\n D. Cl2.\n\n Hãy chọn đáp án đúng"]  # Describe the image in Vietnamese
#     # For a local image path
#     image_url = [r"/data1/npl/ds/data/images/test_1.png"]

#     response = model.get_response(query, image_url)
#     print(response)
