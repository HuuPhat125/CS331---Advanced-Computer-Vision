import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class VinaLlama:
    def __init__(self, model_path="vilm/vinallama-7b-chat", device="cuda", max_new_tokens=100):
        """
        Khởi tạo lớp SimpleGPTModel với các tham số cấu hình mô hình.

        :param model_path: Đường dẫn hoặc tên mô hình trên Hugging Face Hub.
        :param device: Thiết bị để tải mô hình, ví dụ: "cuda" hoặc "cpu".
        :param max_new_tokens: Số lượng token tối đa cho phần sinh của mô hình.
        """
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        # Tải mô hình và tokenizer từ Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        self.model.eval()  # Đặt mô hình vào chế độ đánh giá

    def get_response(self, user_inputs, image_paths=None):
        """
        Sinh phản hồi từ mô hình dựa trên đầu vào của người dùng theo batch.

        :param user_inputs: Danh sách các chuỗi văn bản đầu vào từ người dùng.
        :return: Danh sách các chuỗi văn bản đầu ra từ mô hình.
        """
        prompts = [f"system\nBạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.\nuser\n{input_text}\nassistant\n" for input_text in user_inputs]
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Sinh văn bản từ mô hình
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens
        )
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        responses = [response.split("assistant\n")[1].strip() if "assistant\n" in response else response for response in responses]
        
        return responses