import torch, torchvision
from tqdm import tqdm
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from preprocess_paragraph import *
import torch
from torch.utils.data import TensorDataset, DataLoader

def extractImageFeatures(img_sto, model, model_type, device, batch_size=32):
    sto_image = None

    if model_type == 'vit':
        preprocessing = ViT_B_16_Weights.DEFAULT.transforms()
    elif model_type in ['resnet', 'resnext']:
        if model_type == 'resnet':
            preprocessing = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
        elif model_type == 'resnext':
            preprocessing = torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT.transforms()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    for i in tqdm(range(0, len(img_sto), batch_size)):
        batch_images = img_sto[i:min(i + batch_size, len(img_sto))]
        batch_tensors = []

        for x in batch_images:
            img = preprocessing(x)
            img = img.unsqueeze(0).to(device)
            batch_tensors.append(img)

        sto_image_batch = torch.cat(batch_tensors, dim=0)

        with torch.no_grad():
            if model_type == 'vit':
                feats = model._process_input(sto_image_batch)
                batch_class_token = model.class_token.expand(sto_image_batch.shape[0], -1, -1)
                feats = torch.cat([batch_class_token, feats], dim=1)
                feats = model.encoder(feats)
                avg_feats = feats[:, 1:].mean(dim=1)
            elif model_type in ['resnet', 'resnext']:
                feats = model(sto_image_batch)
                avg_feats = feats

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

def extractContextFeatures(retrieval_docs, token, model, device, batch_size=32):
    
    contexts_features = []
    for i in tqdm(range(0, len(retrieval_docs), batch_size)):
        batch_retrieval_docs = retrieval_docs[i:min(i + batch_size, len(retrieval_docs))]
        tokenized_docs = token(batch_retrieval_docs, padding='max_length', truncation=True, max_length=256, return_tensors='pt').input_ids.to(device)

        with torch.no_grad():
            outputs = model(tokenized_docs)
            last_hidden_state = outputs.last_hidden_state
            contexts_features.append(last_hidden_state.mean(dim=1).cpu())

    return torch.cat(contexts_features, dim=0)

def extractQueryFeatures(queries, token, model, device, batch_size=32):
    query_features = []
    
    for i in tqdm(range(0, len(queries), batch_size)):
        batch_queries = queries[i:min(i + batch_size, len(queries))]
        input_ids = token(batch_queries, padding='max_length', truncation=True, max_length=256, return_tensors='pt').input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            last_hidden_state = outputs.last_hidden_state
            query_features.append(last_hidden_state.mean(dim=1).cpu())

    return torch.cat(query_features, dim=0)

def extractCandidateAnswers(candidate_answers, token, model, device, batch_size=32):
    candidate_features = {}

    for key in candidate_answers.keys():
        candidate_features[key] = []
               
    for key in candidate_answers.keys():
        for i in tqdm(range(0, len(candidate_answers[key]), batch_size)):
            can_answers = candidate_answers[key][i:min(i + batch_size, len(candidate_answers[key]))]
            input_ids = token(can_answers, padding='max_length', truncation=True, max_length=256, return_tensors='pt').input_ids.to(device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                last_hidden_state = outputs.last_hidden_state
                candidate_features[key].append(last_hidden_state.mean(dim=1).cpu())
        candidate_features[key] = torch.cat(candidate_features[key], dim=0)
    return candidate_features

def normalize_features(*features):
    normalized_features = []
    for feature in features:
        mean = feature.mean(dim=-1, keepdim=True)
        std = feature.std(dim=-1, keepdim=True)
        
        normalized_feature = (feature - mean) / (std + 1e-7)
        normalized_features.append(normalized_feature)
    
    return normalized_features

def combine_all_feature(images_features, queries_features, contexts_features, candidates_features, target):

    images_features, queries_features, contexts_features = normalize_features(
        images_features, queries_features, contexts_features
    )
    
    normalized_candidates = {}
    for key, value in candidates_features.items():
        normalized_candidates[key] = normalize_features(value)[0]
        print(normalized_candidates[key].shape)

    combined_features = torch.cat((images_features, queries_features, contexts_features), dim=1)
    q_list = []
    for key in normalized_candidates.keys():
        q_list.append(normalized_candidates[key])
    q_list = torch.stack(q_list, dim=1)

    dim_kv_input = combined_features.shape[-1]
    dim_q_input = q_list.shape[-1]
    
    dataset = TensorDataset(combined_features, q_list, target)
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size=32)

    feature = {
        'combined_features': combined_features,
        'q_list': q_list,
        'dim_kv_input': dim_kv_input,
        'dim_q_input': dim_q_input,
        'data_loader': data_loader
    }
    
    return feature
    
class FeatureExtractor:
    def __init__(self, stoModel):
        
        self.imageModel = stoModel['imageModel']
        self.typeImageModel = stoModel['typeImageModel']
        
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
            self.contextToken = stoModel['contextToken']

            self.candidateModel = stoModel['candidateModel']
            self.candidateToken = stoModel['candidateToken']

    def extractFeature(self, sto, target, device, type):
        images_features = extractImageFeatures(sto['images'], self.imageModel, self.typeImageModel, device=device)
        queries_features = extractTextFeatures(sto['queries'], self.queryToken, self.queryModel, device=device)
        contexts_features = extractTextFeatures(sto['contexts'], self.contextToken, self.contextModel, device=device)
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
        candidates_features = extractCandidateAnswers(sto['candidateAnswers'], self.candidateToken, self.candidateModel, device=device)

        if type == 1:
            feature = combine_all_feature(ocr_features, queries_features, contexts_features, candidates_features, target)
        return feature

        

