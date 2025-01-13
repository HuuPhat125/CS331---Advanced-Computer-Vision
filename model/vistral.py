import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Vistral_Model:
    def __init__(self, model_path="Viet-Mistral/Vistral-7B-Chat", device="cuda", dtype=torch.bfloat16, temperature=0.1, top_k=40, top_p=0.95, max_new_tokens=768):
        """
        Initialize the Vistral_Model class with model configuration parameters.

        :param model_path: Path or name of the model on Hugging Face Hub.
        :param device: Device to load the model on, e.g., "cuda" or "cpu".
        :param dtype: Data type of the model, e.g., torch.bfloat16 or torch.float16.
        :param temperature: Parameter to adjust the model's creativity.
        :param top_k: Parameter to adjust word selection, limiting to top-k.
        :param top_p: Parameter to adjust word selection, choosing words with cumulative top-p probability.
        :param max_new_tokens: Maximum number of tokens for the model's generation.
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        # Load tokenizer and model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            use_cache=True,
        )
        self.model.eval()  # Set the model to evaluation mode

        self.system_prompt = """Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.
        """


    def get_response(self, instructions, image_paths=None):
        """
        Respond to questions based on the input instructions in batches.
        :param instructions: List of instructions or questions you want the model to respond to.
        :param image_paths: List of image paths (if applicable).
        :return: List of responses from the model.
        """
        conversation = [{"role": "system", "content": self.system_prompt}]
        conversation.append({"role": "user", "content": instructions[0]})
        
        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt", padding=True).to(self.model.device)
        
        out_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            repetition_penalty=1.05,
        )
        
        responses = self.tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()        
        return [responses]
