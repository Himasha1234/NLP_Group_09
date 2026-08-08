import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stop-words data if not already present
nltk.download('stopwords')

# -----------------------------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------------------------
# Replace 'data/WELFake_Dataset.csv' with your actual file path
print("Loading dataset...")
df = pd.read_csv('data/WELFake_Dataset.csv')

print(f"Initial shape: {df.shape}")

# -----------------------------------------------------------------------------
# 2. STRUCTURAL CLEANING (Missing Values & Duplicates)
# -----------------------------------------------------------------------------
# Drop missing values in the text and title columns
df = df.dropna(subset=['text', 'title']).copy()

# Remove exact duplicates based on the text column
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)

print(f"Shape after removing nulls and duplicates: {df.shape}")

# -----------------------------------------------------------------------------
# 3. TEXT CLEANING FUNCTIONS
# -----------------------------------------------------------------------------
# Set up standard English stop-words
stop_words = set(stopwords.words('english'))

def clean_text_base(text):
    """
    Base cleaning for all NLP tasks:
    - Lowercasing
    - Stripping URLs, HTML tags, emails, special chars, numbers, and excess spaces.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emails and user handles (@user)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    
    # Remove non-alphanumeric characters and numbers (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Collapse multiple whitespaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text):
    """
    Removes stop-words from cleaned text (For ML / WordClouds / N-Grams).
    """
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# -----------------------------------------------------------------------------
# 4. APPLY CLEANING PIPELINES
# -----------------------------------------------------------------------------
print("Cleaning text for Deep Learning (LSTM)...")
# Base cleaning (retains stop-words for LSTM sequence modeling)
df['text_clean_dl'] = df['text'].apply(clean_text_base)

print("Cleaning text for Machine Learning (Naive Bayes / EDA)...")
# Remove stop-words for Naive Bayes, WordClouds, and N-gram analysis
df['text_clean_ml'] = df['text_clean_dl'].apply(remove_stopwords)

# Remove any rows where text became empty after cleaning
df = df[df['text_clean_ml'].str.strip().astype(bool)].reset_index(drop=True)

# -----------------------------------------------------------------------------
# 5. INSPECT & SAVE CLEANED DATA
# -----------------------------------------------------------------------------
print("\nSample Processed Results:")
print("Original Text:", df['text'].iloc[0][:100])
print("\nCleaned DL Text (LSTM):", df['text_clean_dl'].iloc[0][:100])
print("\nCleaned ML Text (Naive Bayes):", df['text_clean_ml'].iloc[0][:100])

# Save processed dataset into the data folder
output_path = 'data/WELFake_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved successfully to '{output_path}'!")