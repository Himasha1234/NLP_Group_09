import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print("--- Step 1: Loading Dataset ---")
# Load dataset
df = pd.read_csv('data/WELFake_Dataset.csv')

# Drop missing values and combine text columns
df = df.dropna(subset=['text', 'label'])
X = df['text']
y = df['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Step 2: Vectorizing Text Data ---")
# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\n--- Step 3: Training Naive Bayes Model ---")
# Initialize and train Naive Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

print("\n--- Step 4: Evaluating Model Performance ---")
# Predict and evaluate
predictions = nb_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print(f"\nNaive Bayes Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))