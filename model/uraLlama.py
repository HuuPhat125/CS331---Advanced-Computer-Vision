import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class URA_Llama_Model:
    def __init__(self, model_path="ura-hcmut/ura-llama-7b", temperature=1.0, max_new_tokens=1024, top_k=1, repetition_penalty=1.1):
        """
        Initializes the URA_Llama_Model class with model configuration parameters.

        :param model_path: Path or model name on Hugging Face Hub.
        :param temperature: Parameter to control the creativity of the model.
        :param max_new_tokens: Maximum number of tokens for model generation.
        :param top_k: Parameter to control word selection, limiting top-k words.
        :param repetition_penalty: Parameter to penalize repetitions in the generated text.
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto"
        )
        self.model.config.pretraining_tp = 1
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task='text-generation',
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty
        )

        self.query_template = "[INST] <<SYS>>\nBạn là một trợ lý thông minh.\n<</SYS>>\n\nHãy trả lời câu hỏi sau.\nCâu hỏi: {query}\nTrả lời: [/INST]"

    def get_response(self, query, image_path=None):
        """
        Generate a response for the given query using the model pipeline.

        :param query: The input query to generate a response for.
        :return: The generated response.
        """
        formatted_query = self.query_template.format(query=query)
        # Tokenize the input
        inputs = self.tokenizer(formatted_query, return_tensors="pt").to(self.model.device)
        
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated tokens
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response part (after "[/INST]")
        response = generated_text.split("[/INST]")[-1].strip()
        
        return response
