import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import os

# 1. පිරිසිදු කළ ඩේටාසෙට් එක ලෝඩ් කරගැනීම
print("Loading cleaned dataset...")
data = pd.read_csv('data/cleaned_data.csv')

data = data.dropna(subset=['text', 'label'])
data['text'] = data['text'].astype(str)

# 2. Vectorization (TfidfVectorizer මඟින් වචන සංඛ්‍යාත්මක අගයන්ට හැරවීම)
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text']) 
y = data['label']

# 3. Data splitting (Train සහ Test ලෙස දත්ත වෙන් කිරීම - 80% සහ 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Logistic Regression Model Training
print("Training Logistic Regression Model...")
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, predictions)}")

# 5. CNN Model Training (Deep Learning)
print("Training CNN Model...")
cnn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn_model.fit(X_train.toarray(), y_train, epochs=5, batch_size=32, validation_data=(X_test.toarray(), y_test))

# 6. මෝඩල්ස් සුරක්ෂිත කර ෆෝල්ඩරයක සේව් කිරීම
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/logistic_regression_model.pkl')
cnn_model.save('models/cnn_model.keras')
print("Both models trained and saved successfully in 'models/' folder!")

# Accuracy අගයන් ටෙක්ස්ට් ෆයිල් එකක සේව් කරගැනීම
results_content = f"""=== Model Training Results ===
Logistic Regression Accuracy: {accuracy_score(y_test, predictions):.4f}
CNN Model Final Accuracy: {cnn_model.history.history['accuracy'][-1]:.4f}
CNN Model Final Loss: {cnn_model.history.history['loss'][-1]:.4f}
"""

os.makedirs('models', exist_ok=True)
with open('models/training_results.txt', 'w') as f:
    f.write(results_content)

print("Results saved to 'models/training_results.txt' successfully!")