import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('./data/cleaned_data.csv')
df = df.dropna(subset=['text'])

# 2.(Training & Testing)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 3. Text to Numbers (Vectorization)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Logistic Regression Training
logreg = LogisticRegression()
logreg.fit(X_train_vec, y_train)

# 5. Accuracy check
predictions = logreg.predict(X_test_vec)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, predictions)}")

# 6. save the model

joblib.dump(logreg, './models/logistic_regression.pkl')