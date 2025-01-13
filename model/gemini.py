import os
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import PIL.Image

def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True

class Gemini_Model:
    def __init__(self, key, model_name="gemini-1.5-flash", patience=1, sleep_time=5):
        print("!!Gemini Key:", key)
        self.patience = patience
        self.sleep_time = sleep_time
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name=model_name)

    def get_response(self, query, image_path):
        patience = self.patience
        while patience > 0:
            patience -= 1
            try:
                assert os.path.exists(image_path[0])
                img = PIL.Image.open(image_path[0])
                response = self.model.generate_content([query[0], img])
                response_text = response.text.strip()
                if verify_response(response_text):
                    return response_text
                else:
                    print(response_text)
            except Exception as e:
                print(e)
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        return ""
    
    def generate(self, query):
        patience = self.patience
        while patience > 0:
            patience -= 1
            try:
                response = self.model.generate_content(query)  
                return response.text.strip()
            except Exception as e:
                print(e)
                if self.sleep_time > 0:
                        time.sleep(self.sleep_time)
        return ""
               
#python generate_response.py --output_dir ../result/gemini --output_file gemini-1.5-flash.json --model gemini --key AIzaSyA7Ears0vAUcJiDbHJ6Stfd7oa0ru540Ww --debug --batch_size 1

#python generate_response.py --output_dir ../result/gemini --output_file gemini-1.5-flash_context_1.json --model gemini --key AIzaSyDNSs2GPcCCswR_f4Wt5DuYlvfx-2_2Iqs --batch_size 1 --use_context --has_image

#vators anni: AIzaSyAmxe0fNYfXA1DJGx2UdcZO57b5zJeBVjk
#uit: AIzaSyDNSs2GPcCCswR_f4Wt5DuYlvfx-2_2Iqs
#phatpm77: AIzaSyAXAltc9elYbUZCWG01XuPpPQwBcUbdM_0