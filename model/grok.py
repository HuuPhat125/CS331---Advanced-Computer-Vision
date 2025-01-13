import os
import base64
from openai import OpenAI
import time
class ChatbotClient:
    def __init__(self, model, api_key):
        """
        Initializes the chatbot client with the API key and model name.

        Args:
            model (str): The name of the model to use.
            api_key (str): The API key for authentication.
        """
        self.model = model
        self.api_key = api_key
        self.system_message = "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."

        if not self.api_key:
            raise ValueError("API key not provided. Please ensure the API key is set correctly.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1",
        )

    def _encode_image(self, image_path):
        """
        Encodes an image to a base64 string.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The base64-encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

    def chat(self, user_message, image_path=None):
        """
        Sends a chat message to the model and retrieves the response.

        Args:
            user_message (str): The user's input message.
            image_path (str, optional): The path to an image file. Defaults to None.

        Returns:
            str: The chatbot's response content.
        """
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": {"type": "text", "text": user_message}},
        ]

        if image_path:
            base64_image = self._encode_image(image_path)
            image_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high",
                },
            }
            messages[1]["content"] = [image_message, {"type": "text", "text": user_message}]

        time.sleep(10)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,  # Set to True if streaming responses are required
            temperature=0.01,
        )

        return completion.choices[0].message.content

# # Example usage:
# if __name__ == "__main__":
#     api_key = os.getenv("XAI_API_KEY")
#     chatbot = ChatbotClient(model="grok-2-vision-1212", api_key=api_key)

#     user_message = (
#         "Câu hỏi: Mô hình mô tả quá trình tạo liên kết hóa học sau 
#         đây phù hợp với xu hướng tạo liên kết hóa học của nguyên tử nào? <image_1>\n"
#         "Các lựa chọn:\nA. Aluminium.\nB. Nitrogen.\nC. Phosphorus.\nD. Oxygen.\n"
#         "Dựa vào nội dung câu hỏi và nội dung trong bức ảnh hãy chọn đáp án đúng và giải thích bằng Tiếng Việt"
#     )
    
#     image_path = "path/to/your/image.jpg"  # Replace with the actual image path
#     response = chatbot.chat(user_message=user_message, image_path=image_path)
#     print(response)
