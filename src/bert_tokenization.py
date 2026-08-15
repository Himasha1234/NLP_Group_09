import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

print("1. Loading dataset...")
df = pd.read_csv('cleaned_WELFake_with_sentiment.csv').dropna(subset=['clean_text'])

print("2. Initializing WordPiece BertTokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("3. Preparing DataLoader sample...")
dataset = FakeNewsDataset(df['clean_text'][:100], df['label'][:100], tokenizer)
loader = DataLoader(dataset, batch_size=16)

for batch in loader:
    print("Batch Input IDs shape:", batch['input_ids'].shape)
    print("Batch Attention Mask shape:", batch['attention_mask'].shape)
    print("Batch Labels shape:", batch['labels'].shape)
    break

print("\nWordPiece Tokenization Pipeline verified successfully!")