import re
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer

def div_paragraph(paragraph, book = None):
    arr = paragraph.split('\n')
    sto = []
    cur = ""
    
    if 'Hóa' in book or 'Sinh' in book:
        pattern = r'\*\*(I{1,3}|IV|V|VI|VII|VIII|IX|X|A-Z).*\*\*'
    elif 'Lí' in book or 'Lý' in book or 'Hình' in book or 'Số' in book or 'Tích' in book:
        pattern = r'\b.*\d+\. .+'
    
    for sentence in arr:
        matches = re.findall(pattern, sentence.strip())
        if matches:
            sto.append(cur)
            cur = sentence
        else:
            cur += '\n' + sentence
    
    sto.append(cur)
    return sto

def check(name):
    pattern = r'\b(Thực hành|Ôn tập|Luyện tập)\b'
    return re.search(pattern, name, re.IGNORECASE)

def retrieval_part(arr_sto, question, topk = 1):
    tokenized_part = [ViTokenizer.tokenize(part).split() for part in arr_sto]
    tokenized_question = ViTokenizer.tokenize(question).split()
    subPart_bm25 = BM25Okapi(tokenized_part)
    scores = subPart_bm25.get_scores(tokenized_question)
    
    ranked_indices = scores.argsort()[::-1]
    
    ranked_part = [arr_sto[i] for i in ranked_indices]
    
    return "\n".join(ranked_part[0:topk])

def getPart(score_docs, textbook, question, topk=1):
    ranked_indices = score_docs.argsort()[::-1]
    idx = 0
    while check(textbook[ranked_indices[idx]]['lesson']) == 1:
        idx += 1
    topDoc = textbook[ranked_indices[idx]]
    
    return retrieval_part(div_paragraph(topDoc['content'][0], topDoc['book']), question, topk)

class RetrievalModel():
    def __init__(self, docs):
        
        token_docs = [ViTokenizer.tokenize(doc).split() for doc in docs]
        self.model = BM25Okapi(token_docs)
        
    def retrieve(self, query):
        token_query = ViTokenizer.tokenize(query).split()
        return self.model.get_scores(token_query)