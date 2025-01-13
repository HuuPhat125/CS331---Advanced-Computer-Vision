from preprocess_paragraph import *
from utils import *
from feature_extractor import FeatureExtractor, FeatureExtractorOCR
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModel, AutoTokenizer
import torch, torchvision
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
from sklearn.utils import class_weight


def extractDataset(sto, target, device, image_model, is_ocr):
    
    nameTextModel = 'xlm-roberta-large'
    textModel = AutoModel.from_pretrained(nameTextModel).to(device)
    tokenizer = AutoTokenizer.from_pretrained(nameTextModel)

    if image_model == 'vit':
        imageModel = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)
    elif image_model == 'resnext':
        imageModel = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT).to(device)
    elif image_model == 'resnet':
        imageModel = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).to(device)
    else:
        raise ValueError(f"Unsupported image model: {image_model}")

    encoder = {
        'imageModel': imageModel,
        'typeImageModel': image_model,
        'textModel': textModel,
        'textToken': tokenizer
    }

    if is_ocr:
        extrator = FeatureExtractorOCR
    else:
        extractor = FeatureExtractor(encoder)

    feature = extractor.extractFeature(sto, target, device, 1)
    return feature


def training(model_name, train_data, dev_data, device, image_model, is_ocr, num_epochs=100, learning_rate=0.000001, early_stopping_patience=20):
    # Extract features and prepare data
    y_train = train_data['labels']
    y_dev = dev_data['labels']
    trainFeature = extractDataset(train_data, y_train, device, image_model, is_ocr)
    devFeature = extractDataset(dev_data, y_dev, device, image_model, is_ocr)

    
    # Model selection with more flexible approach
    try:
        if model_name == 'baseline':
            dim_kv_input = trainFeature['dim_kv_input']
            dim_q_input = trainFeature['dim_q_input']
            print(f"kv: {dim_kv_input}, q: {dim_q_input}")
            model = BaselineModel(dim_kv_input, dim_q_input, n_head=4).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    except Exception as e:
        print(f"Model initialization error: {e}")
        return None, None, None

    # Data loaders
    train_loader = trainFeature['data_loader']
    dev_loader = devFeature['data_loader']

    print(f"Train Loader: {len(train_loader)}")
    print(f"Dev Loader: {len(dev_loader)}")
    
    # Loss and optimization
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',    # Keyword argument đầu tiên
        classes=np.unique(y_train), # Danh sách các lớp duy nhất
        y=y_train.numpy()           # Dữ liệu nhãn (chuyển sang numpy nếu dùng PyTorch Tensor)
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, threshold=1e-5, verbose=True)

    # Training tracking
    train_losses, dev_losses = [], []
    best_dev_accuracy = 0
    best_model = None
    total_train = 0
    train_accuracy = 0
    correct_train = 0
    early_stopping_counter = 0

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            k, q_list, labels = data
            
            v = k.clone()
            
            # Move to device
            k, v = k.to(device), v.to(device)
            q_list = q_list.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(q_list, k, v)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
                
            running_train_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_accuracy = 100 * correct_train / total_train

        model.eval()
        running_dev_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in dev_loader:
                k, q_list, labels = data
                v = k.clone()
                
                k, v = k.to(device), v.to(device)
                q_list = q_list.to(device)
                labels = labels.to(device)
                
                outputs = model(q_list, k, v)
                loss = criterion(outputs, labels)
                running_dev_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_dev_loss = running_dev_loss / len(dev_loader)
        dev_accuracy = 100 * correct / total
        dev_losses.append(epoch_dev_loss)

        scheduler.step(epoch_dev_loss)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_model = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        print(f"[Epoch {epoch + 1}] Train Loss: {epoch_train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}%"
              f"Dev Loss: {epoch_dev_loss:.3f}, "
              f"Dev Accuracy: {dev_accuracy:.2f}%")

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model:
        model.load_state_dict(best_model)

    return model, train_losses, dev_losses


def predict(model, test_data, device, image_model, is_ocr):

    y_test = test_data['labels']
    testFeature = extractDataset(test_data, y_test, device, image_model, is_ocr)

    
    test_loader = testFeature['data_loader']
    print(f"Test Loader: {len(test_loader)}")
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            k, q_list, labels = data
            v = copy.deepcopy(k)

            k, v = k.to(device), v.to(device)
            q_list = q_list.to(device)

            
            outputs = model(q_list, k, v)
            print(outputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    print("Accuracy on test set:", accuracy_score(y_test, predictions))
    return predictions


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_model", type=str, default="resnet", help="Chọn image model.")
    parser.add_argument("--is_ocr", type=bool, default=0, help="Có sử dụng feature ocr không.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Tốc độ học (learning rate). Mặc định: 0.001")
    parser.add_argument("--num_epochs", type=int, default=100, help="Số epoch cần train. Mặc định: 100")
    args = parser.parse_args()

    image_model = args.image_model
    is_ocr = args.is_ocr
    LR = args.learning_rate
    EPOCHS = args.num_epochs
    
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

    model, train_losses, dev_losses  = training('baseline', train_data, dev_data, textbook, image_folder, device, image_model, is_ocr, num_epochs=EPOCHS, learning_rate=LR)
    

    y_predict = predict(model, test_data, device, image_model, is_ocr)
    joblib.dump(model, 'model.pkl')
    with open("predict.txt", 'w') as f:
        for value in y_predict:
            f.write(f"{value}\n")

if __name__ == "__main__":
    main()
