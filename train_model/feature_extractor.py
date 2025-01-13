
import copy
from tqdm import tqdm
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from preprocess_paragraph import *
import torch
from torch.utils.data import TensorDataset, DataLoader

def extractImageFeatures(img_sto, model, device, batch_size=32):
    sto_image = None
    vit = model
    preprocessing = ViT_B_16_Weights.DEFAULT.transforms()

    for i in tqdm(range(0, len(img_sto), batch_size)):
        batch_images = img_sto[i:min(i + batch_size, len(img_sto))]
        batch_tensors = []

        for x in batch_images:

            img = preprocessing(x)
            img = img.unsqueeze(0).to(device)
            batch_tensors.append(img)

        sto_image_batch = torch.cat(batch_tensors, dim=0)

        with torch.no_grad():
            feats = vit._process_input(sto_image_batch)

            batch_class_token = vit.class_token.expand(sto_image_batch.shape[0], -1, -1)
            feats = torch.cat([batch_class_token, feats], dim=1)
            feats = vit.encoder(feats) 

            avg_feats = feats[:, 1:].mean(dim=1)

        if sto_image is None:
            sto_image = avg_feats
        else:
            sto_image = torch.cat((sto_image, avg_feats), dim=0)

    return sto_image.cpu().reshape(len(img_sto), -1)

def extractTextFeatures(texts, token, model, device, batch_size=32):
    
    texts_features = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:min(i + batch_size, len(texts))]
        tokenized_texts = token(batch_texts, padding='max_length', truncation=True, max_length=256, return_tensors='pt').input_ids.to(device)

        with torch.no_grad():
            outputs = model(tokenized_texts)
            last_hidden_state = outputs.last_hidden_state
            texts_features.append(last_hidden_state.mean(dim=1).cpu())

    return torch.cat(texts_features, dim=0)

def extractCandidateAnswers(candidate_answers, token, model, device, batch_size=32):
    candidate_features = {}

    for key in candidate_answers.keys():
        candidate_features[key] = []
               
    for key in candidate_answers.keys():
        candidate_features[key] = extractTextFeatures(candidate_answers[key], token, model, device)
    return candidate_features

## Baseline method
def combine_all_feature(images_features, queries_features, contexts_features, candidates_features, target):
    combined_features = torch.cat((images_features, queries_features, contexts_features), dim=1)
    q_list = []
    for key in candidates_features.keys():
        q_list.append(candidates_features[key])

    dim_kv_input = combined_features.shape[-1]
    dim_q_input = q_list.shape[-1]
    
    dataset = TensorDataset(combined_features, q_list, target)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    feature = {
        'combined_features': combined_features,
        'q_list': q_list,
        'dim_kv_input': dim_kv_input,
        'dim_q_input': dim_q_input,
        'data_loader': data_loader
    }

    print("Images vector shape:", images_features.shape)
    print("Queries vector shape:", queries_features.shape)
    print("Context vector shape:", contexts_features.shape)

    for key in candidates_features:
        print(f"Candidate vector {key}:", candidates_features[key].shape)
    
    return feature
    
class FeatureExtractor:
    def __init__(self, stoModel):
        
        self.imageModel = stoModel['imageModel']
        
        if 'textModel' in stoModel:
            self.queryModel = stoModel['textModel']
            self.queryToken = stoModel['textToken']

            self.contextModel = stoModel['textModel']
            self.contextToken = stoModel['textToken']

            self.candidateModel = stoModel['textModel']
            self.candidateToken = stoModel['textToken']
        else:
            self.queryModel = stoModel['queryModel']
            self.queryToken = stoModel['queryToken']

            self.contextModel = stoModel['contextModel']
            self.queryToken = stoModel['contextToken']

            self.candidateModel = stoModel['candidateModel']
            self.candidateToken = stoModel['candidateToken']

    def extractFeature(self, sto, target, device, type):
        images_features = extractImageFeatures(sto['images'], self.imageModel, device=device)
        queries_features = extractTextFeatures(sto['queries'], self.queryToken, self.queryModel, device=device)
        contexts_features = extractTextFeatures(sto['contexts'], self.queryToken, self.contextModel, device=device)
        candidates_features = extractCandidateAnswers(sto['candidateAnswers'], self.candidateToken, self.candidateModel, device=device)

        if type == 1:
            feature = combine_all_feature(images_features, queries_features, contexts_features, candidates_features, target)
        return feature

        
class FeatureExtractorOCR:
    def __init__(self, stoModel):
        
        if 'textModel' in stoModel:
            self.queryModel = stoModel['textModel']
            self.queryToken = stoModel['textToken']

            self.contextModel = stoModel['textModel']
            self.contextToken = stoModel['textToken']

            self.candidateModel = stoModel['textModel']
            self.candidateToken = stoModel['textToken']

            self.ocrModel = stoModel['textModel']
            self.ocrToken = stoModel['textToken']
        else:
            self.queryModel = stoModel['queryModel']
            self.queryToken = stoModel['queryToken']

            self.contextModel = stoModel['contextModel']
            self.queryToken = stoModel['contextToken']

            self.candidateModel = stoModel['candidateModel']
            self.candidateToken = stoModel['candidateToken']

            self.ocrModel = stoModel['ocrModel']
            self.ocrToken = stoModel['ocrToken']

    def extractFeature(self, sto, target, device, type):
        ocr_features = extractTextFeatures(sto['ocr'], self.ocrToken, self.ocrModel, device=device)
        queries_features = extractTextFeatures(sto['queries'], self.queryToken, self.queryModel, device=device)
        contexts_features = extractTextFeatures(sto['contexts'], self.queryToken, self.contextModel, device=device)
        candidates_features = extractTextFeatures(sto['candidateAnswers'], self.candidateToken, self.candidateModel, device=device)

        if type == 1:
            feature = combine_all_feature(ocr_features, queries_features, contexts_features, candidates_features, target)
        return feature
        

