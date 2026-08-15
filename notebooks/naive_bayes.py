import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load Cleaned Dataset
print("Loading cleaned data...")
df = pd.read_csv(r'C:\Users\ushan\Desktop\NLP_Group_09\data\WELFake_cleaned.csv')

# Drop any potential missing values in the ML column
df = df.dropna(subset=['text_clean_ml']).reset_index(drop=True)

# 2. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean_ml'], 
    df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

# 3. Vectorization (TF-IDF)
print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. Train Multinomial Naive Bayes
print("Training Multinomial Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# 5. Evaluate Model
y_pred = nb_model.predict(X_test_tfidf)
y_proba = nb_model.predict_proba(X_test_tfidf)[:, 1]

print("\n--- Multinomial Naive Bayes Results ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)']))