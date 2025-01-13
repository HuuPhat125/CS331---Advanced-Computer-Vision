import torch
from transformers import AutoModel, AutoTokenizer

class InterLM_Model:
    def __init__(self, model_path='internlm/internlm-xcomposer2-vl-7b', device='cuda'):
       
        self.device = device
        torch.set_grad_enabled(False)

        # Initialize model and tokenizer
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    def process_query(self, query):
        
        if '<image>' in query:
            return query.replace('<image>', '<ImageHere>')
        else:
            return f'<ImageHere>{query}'
        
    def get_response(self, queries, image_paths):

        responses = []

        for query, image_path in zip(queries, image_paths):
            full_query = self.process_query(query)
            
            with torch.amp.autocast('cuda'):
                response, *_ = self.model.chat(self.tokenizer, query=full_query, image=image_path, history=[], do_sample=False)
            
            responses.append(response)

        return responses