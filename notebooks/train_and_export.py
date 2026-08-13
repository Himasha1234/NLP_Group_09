import os
import joblib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D

# ප්‍රොජෙක්ට් රූට් එකේ models ෆෝල්ඩරය පාවිච්චි කිරීම වඩාත් සුදුසුය
os.makedirs('models', exist_ok=True)

print("Loading data...")
df = pd.read_csv('data/WELFake_cleaned.csv').dropna(subset=['text_clean_ml', 'text_clean_dl'])

# -----------------------------------------------------------------------------
# 1. TRAIN & SAVE NAIVE BAYES
# -----------------------------------------------------------------------------
print("Fitting TF-IDF and Naive Bayes...")
X_train_ml, _, y_train_ml, _ = train_test_split(
    df['text_clean_ml'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train_ml)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train_ml)

joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print("[SUCCESS] Naive Bayes artifacts saved in 'models/'!")

# -----------------------------------------------------------------------------
# 2. TRAIN & SAVE LSTM
# -----------------------------------------------------------------------------
print("\nTraining LSTM network...")
MAX_VOCAB = 20000
MAX_LEN = 300

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text_clean_dl'])

sequences = tokenizer.texts_to_sequences(df['text_clean_dl'])
X_padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y_dl = df['label'].values

X_train_dl, _, y_train_dl, _ = train_test_split(
    X_padded, y_dl, test_size=0.2, random_state=42, stratify=y_dl
)

model = Sequential([
    Embedding(input_dim=MAX_VOCAB, output_dim=128, input_length=MAX_LEN),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_dl, y_train_dl, epochs=3, batch_size=64, validation_split=0.1, verbose=1)

model.save('models/lstm_model.keras')
joblib.dump(tokenizer, 'models/lstm_tokenizer.pkl')
print("[SUCCESS] LSTM artifacts saved in 'models/'!")