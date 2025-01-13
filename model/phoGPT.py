#PhoGPT
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

class PhoGPT_Model:
    def __init__(self, model_path="vinai/PhoGPT-4B-Chat", device="cuda", dtype=torch.bfloat16, temperature=1.0, top_k=50, top_p=0.9, max_new_tokens=1024):
        """
        Khởi tạo lớp PhoGPT4_Model với các tham số cấu hình mô hình.

        :param model_path: Đường dẫn hoặc tên mô hình trên Hugging Face Hub.
        :param device: Thiết bị để tải mô hình, ví dụ: "cuda" hoặc "cpu".
        :param dtype: Kiểu dữ liệu của mô hình, ví dụ: torch.bfloat16 hoặc torch.float16.
        :param temperature: Tham số điều chỉnh độ sáng tạo của mô hình.
        :param top_k: Tham số điều chỉnh việc chọn từ, giới hạn top-k.
        :param top_p: Tham số điều chỉnh việc chọn từ, chọn từ có xác suất tích lũy top-p.
        :param max_new_tokens: Số lượng token tối đa cho phần sinh của mô hình.
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        # Tải cấu hình và mô hình từ Hugging Face
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.config.init_device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, config=self.config, torch_dtype=dtype, trust_remote_code=True).to(device)
        self.model.eval()  # Đặt mô hình vào chế độ đánh giá
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        self.PROMPT_TEMPLATE = "### Câu hỏi: Trả lời câu hỏi sau:\n {instruction}\n### Trả lời:"

    def get_response(self, instructions, image_paths=None):
        """
        Trả lời các câu hỏi dựa trên hướng dẫn đầu vào theo batch.
        :param instructions: List các hướng dẫn hoặc câu hỏi mà bạn muốn mô hình trả lời.
        :param image_paths: List các đường dẫn ảnh tương ứng (nếu có).
        :return: List các phản hồi từ mô hình.
        """
        input_prompts = [self.PROMPT_TEMPLATE.format_map({"instruction": instr}) for instr in instructions]
        inputs = self.tokenizer(input_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        outputs = self.model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [response.split("### Trả lời:")[1].strip() if "### Trả lời:" in response else response for response in responses]
        return responses
