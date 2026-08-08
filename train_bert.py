import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification 
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("1. Loading dataset...")
df = pd.read_csv('cleaned_WELFake_with_sentiment.csv').dropna(subset=['clean_text'])

# Split data into Train and Validation sets (using a sample for fast initial fine-tuning)
train_df, val_df = train_test_split(df.sample(n=5000, random_state=42), test_size=0.2, random_state=42)

print("2. Tokenizing data...")
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
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = FakeNewsDataset(train_df['clean_text'], train_df['label'], tokenizer)
val_dataset = FakeNewsDataset(val_df['clean_text'], val_df['label'], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

print("3. Initializing BERT Model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

print("4. Starting Fine-Tuning (2 Epochs)...")
model.train()
for epoch in range(2):
    total_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

print("\n5. Evaluating Fine-Tuned BERT Model...")
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels.numpy())

print(f"\nBERT Accuracy: {accuracy_score(true_labels, predictions) * 100:.2f}%\n")
print(classification_report(true_labels, predictions, target_names=['Real (0)', 'Fake (1)']))