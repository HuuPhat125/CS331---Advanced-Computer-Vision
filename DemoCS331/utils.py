import os
from PIL import Image
import numpy as np
from preprocess_paragraph import *
import torch

labels = ['A', 'B', 'C', 'D']

def load_labels(data):
    
    y_target = [encodeLabels(data[i]['answer'], labels) for i in range(len(data))]
    y_target = torch.tensor(y_target, dtype=torch.long)
    return y_target

def readImage(data, image_folder: str):
    img_sto = []
    
    for i in range(len(data)):
        image_path = data[i]['image']
        image = Image.open(os.path.join(image_folder, image_path)).convert('RGB')
        img_sto.append(image)

    return img_sto

def readContext(data, textbook, retrieval_model):
    
    retrieval_docs = []
    queries = [row['question'] for row in data]
    for query in queries:
        retrieval_scores = retrieval_model.retrieve(query)
        retrieval_docs.append(getPart(retrieval_scores, textbook, query))
    
    return retrieval_docs

def readQuery(data):
    queries = [row['question'] for row in data]
    return queries

def readCandidateAnswers(data):
    candidate_answers = {
        'option_A': [],
        'option_B': [],
        'option_C': [],
        'option_D': []
    }
    
    for i in range(len(data)):
        for key in data[i].keys():
            if 'option' in key:
                candidate_answers[key].append(data[i][key])

    return candidate_answers

def readData(data, image_folder, textbook, retrieval_model):
    image_sto = readImage(data, image_folder)
    contexts = readContext(data, textbook, retrieval_model)
    queries = readQuery(data)
    candidateAnswers = readCandidateAnswers(data)
    labels = load_labels(data)

    return {'images': image_sto,
            'contexts': contexts,
            'queries': queries,
            'candidateAnswers': candidateAnswers,
            'labels': labels}

def encodeLabels(y, labels):
    return np.argmax([y == label for label in labels])

def load_data(data, textbook, image_folder):

    docs = [lecture['content'][0] for lecture in textbook]
    retrieval_model = RetrievalModel(docs)

    sto = readData(data, image_folder, textbook, retrieval_model)
    
    return sto