import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_svm():
    print("1. Loading dataset...")
    data_path = os.path.join('data', 'WELFake_cleaned.csv')
    
    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'WELFake_Dataset.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'cleaned_data.csv')
        
    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found in data/ folder.")
        return
        
    df = pd.read_csv(data_path)
    
    # Automatically detect the correct text column name
    text_col = None
    for col in ['text_clean_ml', 'clean_text', 'text_clean', 'text']:
        if col in df.columns:
            text_col = col
            break
            
    if text_col is None:
        print(f"[Error] Text column not found. Available columns: {list(df.columns)}")
        return
        
    print(f"Using text column: '{text_col}'")
    df[text_col] = df[text_col].fillna('')
    df = df.dropna(subset=[text_col, 'label'])
    
    X = df[text_col]
    y = df['label']
    
    print("2. Splitting dataset into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("3. Vectorizing text using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print("4. Training LinearSVC model...")
    svm_model = LinearSVC(random_state=42, max_iter=10000)
    svm_model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    preds = svm_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"SVM Training Completed! Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    
    # Save model and vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(svm_model, os.path.join('models', 'svm_model.pkl'))
    joblib.dump(tfidf, os.path.join('models', 'vectorizer.pkl'))
        
    print("[SUCCESS] SVM model and vectorizer saved to 'models/' folder!")

if __name__ == "__main__":
    train_and_save_svm()