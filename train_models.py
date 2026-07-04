import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Data loading and cleaning
data = pd.read_csv('data/cleaned_data.csv')

# Data cleaning: drop rows with missing text and convert text to string
data = data.dropna(subset=['text'])
data['text'] = data['text'].astype(str)

# 2. Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text']) 
y = data['label']

# 3. Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Logistic Regression Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model on the test set
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions)}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# CNN Model 
cnn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN model
cnn_model.fit(X_train.toarray(), y_train, epochs=5, batch_size=32, validation_data=(X_test.toarray(), y_test))

import joblib
# Logistic Regression 
joblib.dump(model, 'models/logistic_regression_model.pkl')
# CNN Model
cnn_model.save('models/cnn_model.keras')
print("Both models saved successfully!")