import streamlit as st
from PIL import Image
import os
import torch
from run import *
from read_app import *
from models.baseline_model import BaselineModel
import joblib
import torchvision

def convertIDtoLabel(id):
    true_labels = ['A', 'B', 'C', 'D']
    return true_labels[id]

def get_model(image_model):

    if image_model == 'vit':
        imageModel = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)
    elif image_model == 'resnext':
        imageModel = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT).to(device)
    elif image_model == 'resnet':
        imageModel = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).to(device)
    else:
        raise ValueError(f"Unsupported image model: {image_model}")


    nameTextModel = 'xlm-roberta-large'
    textModel = AutoModel.from_pretrained(nameTextModel).to(device)
    tokenizer = AutoTokenizer.from_pretrained(nameTextModel)

    encoder = {
        'imageModel': imageModel,
        'typeImageModel': image_model,
        'textModel': textModel,
        'textToken': tokenizer
    }

    return encoder

def extractDatasetApp(encoder, sto, target, device, mode = 'normal'):

    if mode == 'normal':
        extractor = FeatureExtractor(encoder)
    elif mode == 'OCR':
        extractor = FeatureExtractorOCR(encoder)

    feature = extractor.extractFeature(sto, target, device, 1)
    return feature

def predictApp(encoder, model, test_data, device):

    y_test = test_data['labels']
    testFeature = extractDatasetApp(encoder, test_data, y_test, device)

    
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
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    print("Accuracy on test set:", accuracy_score(y_test, predictions))
    return predictions

def run_predict(sto, model, encoder):
    
    sgk_path = 'data_sgk.json'
    newsto = convert(sto, sgk_path)
    
    pred = predictApp(encoder, model, newsto, device)
    answer = convertIDtoLabel(pred[0])  
    return answer


# Thiết lập giao diện
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
st.set_page_config(layout="wide")
st.title("Demo Final Project CS331")

# Nhập câu hỏi
st.subheader("Chọn mô hình để trích xuất đặc trưng hình ảnh")
model_type = st.selectbox(
    "Chọn mô hình bạn muốn sử dụng:",
    ["vit", "resnet", "resnext"]
)

encoder = get_model(model_type)
model_path=os.path.join(model_type, 'model.pth')

if model_type == 'vit':
    model = BaselineModel(dim_kv_input=2816, dim_q_input=1024)
else:
    model = BaselineModel(dim_kv_input=3048, dim_q_input=1024)

model.load_state_dict(torch.load(model_path))

model.to(device)
model.eval()

st.subheader("Nhập câu hỏi")
question = st.text_input("")

# Chia cột cho giao diện
col1, col2 = st.columns(2)

temp_folder = "temp_images"
os.makedirs(temp_folder, exist_ok=True)

# Cột 1: Upload hình ảnh và hiển thị hình ảnh
image_path = None
image = None
with col1:
    st.subheader("Upload hình ảnh của câu hỏi")
    uploaded_file = st.file_uploader("Chọn file ảnh", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Save the uploaded image to a temporary folder
        image_path = os.path.join(temp_folder, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Reload the image using Image.open and convert to RGB
        image = Image.open(image_path).convert("RGB")
        st.image(image, caption="Plot hình ảnh", use_column_width=False)

# Cột 2: Nhập các tùy chọn (A, B, C, D)
with col2:
    st.subheader("Nhập 4 option để lựa chọn")
    option_a = st.text_input("A.")
    option_b = st.text_input("B.")
    option_c = st.text_input("C.")
    option_d = st.text_input("D.")

sto = None
if question and image and option_a and option_b and option_c and option_d:
    sto = {
        'question': question,
        'option_A': option_a,
        'option_B': option_b,
        'option_C': option_c,
        'option_D': option_d,
        'image': image
    }


# Phần đáp án
if st.button("Lời giải của mô hình:"):
    if sto:
        st.subheader("Đáp án của câu hỏi là:")

        answer = run_predict(sto, model, encoder)
        if answer == 'A':
            text_answer = option_a
        elif answer == 'B':
            text_answer = option_b
        elif answer == 'C':
            text_answer = option_c
        else:
            text_answer = option_d

        st.write(text_answer)
    else:
        st.warning("Vui lòng điền đầy đủ thông tin.")
