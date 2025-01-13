from preprocess_paragraph import *
from utils import *
from feature_extractor import FeatureExtractor, FeatureExtractorOCR
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
import joblib
import copy
from tqdm import tqdm
from models.baseline_model import BaselineModel
from sklearn.metrics import accuracy_score
import argparse


def extractDataset(sto, target, mode, device):
    
    ### Where to set feature extractor
    nameTextModel = 'xlm-roberta-large'
    textModel = AutoModel.from_pretrained(nameTextModel).to(device)
    tokenizer = AutoTokenizer.from_pretrained(nameTextModel)

    imageModel = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)

    encoder = {
        'imageModel': imageModel,
        'textModel': textModel,
        'textToken': tokenizer
    }

    if mode == 'normal':
        extractor = FeatureExtractor(encoder)
    elif mode == 'OCR':
        extractor = FeatureExtractorOCR(encoder)

    feature = extractor.extractFeature(sto, target, device, 1)
    return feature


def training(model_name, train_data, dev_data, device, num_epochs = 100, learning_rate = 0.001, mode_feature='normal'):
    
    y_train = train_data['labels']
    y_dev = dev_data['labels']

    trainFeature = extractDataset(train_data, target=y_train, mode=mode_feature, device=device)
    devFeature = extractDataset(dev_data, target=y_dev, mode=mode_feature, device=device)

    if model_name == 'baseline':
        dim_kv_input = trainFeature['dim_kv_input']
        dim_q_input = trainFeature['dim_q_input']

        model = BaselineModel(dim_kv_input, dim_q_input, n_head=4).to(device)

    train_loader = trainFeature['data_loader']
    dev_loader = devFeature['data_loader']

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, threshold=0.0001, verbose=True)
    train_losses, dev_losses = [], []
    
    for epoch in tqdm(range(num_epochs)):
        running_train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            k, q_list, labels = data
            v = copy.deepcopy(k)
            
            k, v = k.to(device), v.to(device)
            q_list = q_list.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(q_list, k, v)

            loss = criterion(outputs, labels)
            scheduler.step(loss)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_dev_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dev_loader:
                k, q_list, labels = data
                v = copy.deepcopy(k)
                
                k, v = k.to(device), v.to(device)
                q_list = q_list.to(device)
                labels = labels.to(device)
                
                outputs = model(q_list, k, v)
                loss = criterion(outputs, labels)
                running_dev_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_dev_loss = running_dev_loss / len(dev_loader)
        dev_losses.append(epoch_dev_loss)
        dev_accuracy = 100 * correct / total
        
        # Scheduler step
        scheduler.step(epoch_dev_loss)

        
        if epoch % 10 == 9:
            print(f"[Epoch {epoch + 1}] Train Loss: {epoch_train_loss:.3f}, Dev Loss: {epoch_dev_loss:.3f}, Dev Accuracy: {dev_accuracy:.2f}%")

    
    return model, train_losses, dev_losses

def predict(model, test_data, device, mode_feature='normal'):

    y_test = test_data['labels']
    testFeature = extractDataset(test_data, y_test, device, mode_feature)

    
    test_loader = testFeature['data_loader']

    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            k, q_list, labels = data
            v = copy.deepcopy(k)

            k, v = k.to(device), v.to(device)
            q_list = q_list.to(device)

            outputs = model(q_list, k, v)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    print("Accuracy on test set:", accuracy_score(y_test, predictions))
    return predictions


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Tốc độ học (learning rate). Mặc định: 0.001")
    parser.add_argument("--num_epochs", type=int, default=100, help="Số epoch cần train. Mặc định: 100")
    parser.add_argument("--fea_extract", type=str, default='normal', help="Mode to extract feature")
    args = parser.parse_args()

    LR = args.learning_rate
    EPOCHS = args.num_epochs
    MODE_FEATURE_EXTRACT = args.fea_extract
    
    path = r'..\data'
    image_folder = r'..\data\images'
    ano_folder = r'..\Modified_data'

    train_path = os.path.join(ano_folder, 'data_train.json')
    dev_path = os.path.join(ano_folder, 'data_dev.json')
    test_path = os.path.join(ano_folder, 'data_test.json')
    sgk_path = os.path.join(path, 'data_sgk.json')

    with open(sgk_path, 'r', encoding='utf-8') as f:
        textbook = json.load(f)

    with open(train_path,'r', encoding='utf-8') as f:
        train_data = load_data(json.load(f), textbook, image_folder)

    with open(dev_path,'r', encoding='utf-8') as f:
        dev_data = load_data(json.load(f), textbook, image_folder)

    with open(test_path,'r', encoding='utf-8') as f:
        test_data = load_data(json.load(f), textbook, image_folder)

    ## read ocr
    text_folder = r'../text-data-mmmu'
    name_file = 'ocrs_easyocr_'

    def ocr_to_list(dct):
        arr = []
        for i in range(len(dct)):
            arr.append(dct[str(i)])
        return arr

    with open(os.path.join(text_folder, name_file + 'train.json'), 'r') as f:
        train_ocr = ocr_to_list(json.load(f)['texts'])
    with open(os.path.join(text_folder, name_file + 'dev.json'), 'r') as f:
        dev_ocr = ocr_to_list(json.load(f)['texts'])
    with open(os.path.join(text_folder, name_file + 'test.json'), 'r') as f:
        test_ocr = ocr_to_list(json.load(f)['texts'])
    train_data['ocr'] = train_ocr
    dev_data['ocr'] = dev_ocr
    test_data['ocr'] = test_ocr

    model, train_losses, dev_losses  = training('baseline', train_data, dev_data, textbook, image_folder, device, num_epochs=EPOCHS, learning_rate=LR, mode_feature=MODE_FEATURE_EXTRACT)
    
    
    plt.plot(train_losses)
    plt.plot(dev_losses)

    y_predict = predict(model, test_data, mode_feature=MODE_FEATURE_EXTRACT, device=device)
    joblib.dump(model, 'model.pkl')
    with open("predict.txt", 'w') as f:
        for value in y_predict:
            f.write(f"{value}\n")

if __name__ == "__main__":
    main()
