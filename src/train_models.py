import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# CNN Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train_lr_cnn():
    print("1. Loading dataset for Logistic Regression & CNN...")
    df = pd.read_csv('data/cleaned_data.csv')
    df['text'] = df['text'].fillna('')
    
    X = df['text']
    y = 1 - df['label'] # Invert labels if Real/Fake are flipped

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("2. Training Logistic Regression with balanced class weight...")
    vectorizer = joblib.load('models/vectorizer.pkl')
    X_train_vec = vectorizer.transform(X_train)
    
    lr_model = LogisticRegression(class_weight='balanced', random_state=42)
    lr_model.fit(X_train_vec, y_train)
    
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
    print("Logistic Regression model saved.")

    print("3. Training CNN Model...")
    max_words, max_len = 10000, 200
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post')
    
    cnn_model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_pad, y_train, epochs=2, batch_size=64, validation_split=0.1)
    
    cnn_model.save('models/cnn_model.keras')
    joblib.dump(tokenizer, 'models/tokenizer.pkl')
    print("[SUCCESS] CNN model saved.")

if __name__ == "__main__":
    train_lr_cnn()