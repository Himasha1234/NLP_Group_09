import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def train_svm():
    print("1. Loading dataset for SVM...")
    df = pd.read_csv('data/cleaned_data.csv') 
    df['text'] = df['text'].fillna('')
    
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("2. Vectorizing text with optimized N-grams & min_df...")
    # min_df=3 මඟින් එක පාරකට වඩා වැඩි වාර ගණනක් යෙදුණු අර්ථවත් වචන පමණක් ගනී
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), stop_words='english', min_df=3)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    print("3. Training SVM with balanced class weight...")
    # Real නිව්ස් වෙනුවෙන් ප්‍රමුඛතාවය සුළු වශයෙන් ඉහළ නැංවීමට class_weight සැකසීම
    svm_model = LinearSVC(class_weight='balanced', random_state=42, max_iter=10000, C=0.8)
    svm_model.fit(X_train_vec, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(svm_model, 'models/svm_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print("[SUCCESS] SVM model & Vectorizer retrained successfully.")

if __name__ == "__main__":
    train_svm()