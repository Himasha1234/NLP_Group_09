import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.sparse import hstack

print("1. Loading dataset with sentiment features...")
df = pd.read_csv('cleaned_WELFake_with_sentiment.csv')

# Drop any potential NaN values in clean_text
df = df.dropna(subset=['clean_text'])

print("2. Splitting dataset into train and test sets (80/20)...")
X_text = df['clean_text']
X_sentiment = df[['sentiment_compound', 'sentiment_pos', 'sentiment_neu', 'sentiment_neg']].values
y = df['label']

X_text_train, X_text_test, X_sent_train, X_sent_test, y_train, y_test = train_test_split(
    X_text, X_sentiment, y, test_size=0.2, random_state=42, stratify=y
)

print("3. Extracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_text_train)
X_test_tfidf = tfidf.transform(X_text_test)

print("4. Combining TF-IDF and Sentiment features...")
X_train_combined = hstack([X_train_tfidf, X_sent_train])
X_test_combined = hstack([X_test_tfidf, X_sent_test])

print("5. Training Support Vector Machine (LinearSVC)...")
svm_model = LinearSVC(max_iter=2000, random_state=42)
svm_model.fit(X_train_combined, y_train)

print("6. Evaluating Model Performance...\n")
y_pred = svm_model.predict(X_test_combined)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))