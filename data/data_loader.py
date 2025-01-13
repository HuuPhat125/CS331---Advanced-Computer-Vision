import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class QuestionDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=256):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        options = eval(item['options'])  # Chuyển chuỗi thành danh sách
        answer = item['answer']
        image_path = item['image']  #

        input_text = question + " " + " ".join(options)

        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Chuyển đổi đáp án thành chỉ mục
        labels = ['A', 'B', 'C', 'D']
        label = labels.index(answer)

        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Bỏ đi kích thước không cần thiết
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_dataloader(json_file, tokenizer, batch_size=8, shuffle=True):
    dataset = QuestionDataset(json_file, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = get_dataloader('data/train.json', tokenizer)
test_loader = get_dataloader('data/test.json', tokenizer)
dev_loader = get_dataloader('data/dev.json', tokenizer)
