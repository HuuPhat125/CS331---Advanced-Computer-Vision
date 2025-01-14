import os
from PIL import Image
import numpy as np
from preprocess_paragraph import *
import torch
import json
import numpy as np

def getContext(question, textbook, retrieval_model):
    retrieval_docs = []
    retrieval_scores = retrieval_model.retrieve(question)
    retrieval_docs.append(getPart(retrieval_scores, textbook, question))
    
    return retrieval_docs

def convert(sto, sgk_path = 'E:\DemoCS331\data_sgk.json'):
    with open(sgk_path, 'r', encoding='utf-8') as f:
        textbook = json.load(f)
    docs = [lecture['content'][0] for lecture in textbook]

    retrieval_model = RetrievalModel(docs)

    question = sto['question']

    contexts = getContext(question, textbook, retrieval_model)

    candidate_answers = {
        'option_A': [sto['option_A']],
        'option_B': [sto['option_B']],
        'option_C': [sto['option_C']],
        'option_D': [sto['option_D']]
    }

    return {'images': [sto['image']],
            'contexts': contexts,
            'queries': [sto['question']],
            'candidateAnswers': candidate_answers,
            'labels': torch.tensor([0], dtype=torch.long)}


def predictCNN(test_data, model):
    answer = np.random.randint(0, 5)
    return answer