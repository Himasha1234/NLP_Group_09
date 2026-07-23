import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stop words data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print("1. Loading WELFake dataset...")
# Load dataset
df = pd.read_csv(r'c:\Users\chamo\OneDrive\Desktop\NLP_Group\data\WELFake_Dataset.csv')

print("2. Handling missing values and duplicates...")
# Handle missing text
df.dropna(subset=['text'], inplace=True)
df['title'] = df['title'].fillna('')

# Combine title and text into one single column
df['combined_text'] = df['title'] + " " + df['text']

# Remove duplicate entries
df.drop_duplicates(subset=['combined_text'], inplace=True)

print("3. Cleaning text (Lowercasing, removing URLs & punctuation)...")
def clean_text(text):
    text = text.lower()                                    # Lowercase
    text = re.sub(r'<.*?>', '', text)                      # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)     # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)               # Keep only letters/spaces
    text = re.sub(r'\s+', ' ', text).strip()               # Remove extra spaces
    return text

df['clean_text'] = df['combined_text'].apply(clean_text)

# Path for SVM (Remove stop words)
print("4. Preparing clean text for SVM model...")
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

df['svm_text'] = df['clean_text'].apply(remove_stopwords)

print("5. Saving processed data...")
# Save output to a new file
df.to_csv('cleaned_WELFake_Dataset.csv', index=False)

print("\n Done! Generated 'cleaned_WELFake_Dataset.csv'")
