import pandas as pd
import re
import os
import nltk

# NLTK රිසෝස් නිවැරදිව ඩවුන්ලෝඩ් කරගැනීම
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

print("Loading raw dataset...")
data = pd.read_csv('data/WELFake_Dataset.csv')

print("Cleaning and preprocessing data (Tokenization, Lemmatization, Stopwords)...")

# හිස් පේළි සහ නෝ-ස්ට්‍රින්ග් දත්ත ඉවත් කිරීම
data = data.dropna(subset=['text', 'label'])
data['text'] = data['text'].astype(str)

data = data[data['text'].str.strip() != '']
data = data[data['text'].str.lower() != 'nan']

lemmatizer = WordNetLemmatizer()

# Stopwords ලබා ගැනීම
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

def preprocess_text(text):
    text = str(text)
    
    # URLs සහ විශේෂ අක්ෂර ඉවත් කිරීම
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Lowercase කිරීම
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stop words ඉවත් කිරීම සහ Lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    
    return " ".join(cleaned_tokens)

print("Processing text data (This may take a moment)...")
data['cleaned_text'] = data['text'].apply(preprocess_text)

data = data[data['cleaned_text'].str.strip() != '']

cleaned_df = pd.DataFrame({
    'text': data['cleaned_text'],
    'label': data['label']
})

os.makedirs('data', exist_ok=True)
cleaned_df.to_csv('data/cleaned_data.csv', index=False)
print("Advanced preprocessing completed successfully! Saved to 'data/cleaned_data.csv'.")