import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import pandas as pd
from tqdm import tqdm
tqdm.pandas() # Enable progress_apply for pandas
import re
import os
import nltk

# 1. Download necessary NLTK resources (if not already downloaded)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

print("Loading raw dataset...")
data = pd.read_csv('data/WELFake_Dataset.csv')

print("Cleaning and preprocessing data (Tokenization, Lemmatization, Stopwords)...")

# 1. Drop rows with missing 'text' or 'label' and ensure 'text' is a string
data = data.dropna(subset=['text', 'label'])
data['text'] = data['text'].astype(str)

data = data[data['text'].str.strip() != '']
data = data[data['text'].str.lower() != 'nan']

lemmatizer = WordNetLemmatizer()

# 2. Define stop words (with a fallback in case of download issues)
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    # Fallback stop words list if NLTK stopwords are not available
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

def preprocess_text(text):
    # Ensure the input is a string
    text = str(text)
    
    # 1. Remove URLs and non-alphabetic characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Tokenization 
    tokens = word_tokenize(text)
    
    # 4. Lemmatization and Stopword Removal
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    
    return " ".join(cleaned_tokens)

print("Processing text data (This may take a moment)...")
data['cleaned_text'] = data['text'].progress_apply(preprocess_text)

# Remove rows where the cleaned text is empty after preprocessing
data = data[data['cleaned_text'].str.strip() != '']

# Save the cleaned dataset to a new CSV file for model training
cleaned_df = pd.DataFrame({
    'text': data['cleaned_text'],
    'label': data['label']
})

os.makedirs('data', exist_ok=True)
cleaned_df.to_csv('data/cleaned_data.csv', index=False)
print("Advanced preprocessing completed successfully! Saved to 'data/cleaned_data.csv'.")