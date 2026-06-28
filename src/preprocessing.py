import pandas as pd
import re
import nltk
from tqdm import tqdm          
tqdm.pandas()                  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



def clean_text(text):
    """
    Cleans the input text by removing HTML tags, special characters,
    converting to lowercase, and performing lemmatization.
    """
    # 1. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Convert to lowercase
    text = text.lower()
    
    # 4. Perform Lemmatization and remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords.words('english')]
    
    return " ".join(words)

def process_dataset(input_path, output_path):
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    df['text'] = df['text'].fillna('')
    df['text'] = df['text'].astype(str)
    print("Cleaning text, please wait...")
    df['cleaned_text'] = df['text'].progress_apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing completed! Cleaned data saved to '{output_path}'")
    """
    Reads the dataset, applies the cleaning function to the 'text' column,
    and saves the result to a new CSV file.
    """
    print("Starting data preprocessing...")
    
    # Load the dataset
    df = pd.read_csv(input_path)
    df = pd.read_csv(input_path)
    
    # Apply cleaning function
    df['cleaned_text'] = df['text'].apply(clean_text)
    print("Cleaning text, please wait...")
    df['cleaned_text'] = df['text'].progress_apply(clean_text)
    
    # Save the processed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessing completed! Cleaned data saved to '{output_path}'")

# Execute the process
if __name__ == "__main__":
    # Ensure the input file path matches your local setup
    process_dataset('data/WELFake_Dataset.csv', 'data/cleaned_data.csv')
    