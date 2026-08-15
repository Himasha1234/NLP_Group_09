import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('data/cleaned_data.csv') # හෝ WELFake_Dataset.csv

df['text'] = df['text'].fillna('')
X = df['text']
y = df['label']

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Logistic Regression & Vectorizer
print("Training Logistic Regression & Vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)

y_pred_lr = lr_model.predict(X_test_vec)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

# 4. Train CNN Model & Tokenizer
print("Training CNN Model...")
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

cnn_model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(X_train_pad, y_train, epochs=2, batch_size=64, validation_data=(X_test_pad, y_test))

# Evaluate CNN on Test Data
loss, cnn_accuracy = cnn_model.evaluate(X_test_pad, y_test, verbose=0)
print(f"CNN Model Accuracy: {cnn_accuracy * 100:.2f}%")

# 5. Save All Models, Vectorizer, Tokenizer
os.makedirs('models', exist_ok=True)
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
cnn_model.save('models/cnn_model.keras')
joblib.dump(tokenizer, 'models/tokenizer.pkl')
print("All models, vectorizers, and tokenizers saved successfully!")

# 6. Save Training Results to training_results.txt
results_text = f"""=== Fake News Detection Model Training Results ===
Dataset: cleaned_data.csv
Test Size: 20%
--------------------------------------------------
1. Logistic Regression & TF-IDF:
   - Accuracy: {lr_accuracy * 100:.2f}%

   Classification Report:
{classification_report(y_test, y_pred_lr)}

--------------------------------------------------
2. Convolutional Neural Network (CNN):
   - Test Accuracy: {cnn_accuracy * 100:.2f}%
   - Test Loss: {loss:.4f}
"""

with open('models/training_results.txt', 'w', encoding='utf-8') as f:
    f.write(results_text)

print("Training results saved successfully to models/training_results.txt!")